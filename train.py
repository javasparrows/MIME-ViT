import os
import argparse
import importlib.util
import shutil
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, MultiStepLR, CosineAnnealingLR
import albumentations as A
from model_v5_seg.dataset_segmentation import CSVSegmentationDataset, custom_collate
from model_v5_seg.utils_csv import train
from model_v5_seg.loss import FocalLoss, WeightedFocalLoss, Focal_MultiLabel_Loss, CrossEntropyFBetaLoss
from model_v5_seg.models import UNet, MIMEViT
from model_v5.transforms import BreastShift

# Parse config file path from command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True, help="config file path")
args = parser.parse_args()

LOG_DIR = f"result_dir/results/result_{args.config.split('/')[1].split('_')[1]}"
print(f'LOG_DIR = {LOG_DIR}')

# Import the config file as a python module
spec = importlib.util.spec_from_file_location("config", args.config)
cfg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cfg)
cfg = cfg.Config  # Get the Config class

# 古いLOG_DIRを削除
if os.path.isdir(LOG_DIR) == True:
    shutil.rmtree(LOG_DIR)
    print(f'Directory {LOG_DIR} has been deleted.')
    
#     user_input = input(f"Are you sure you want to delete the directory {LOG_DIR}? (y/N): ")
#     if user_input.lower() == 'y':
#         shutil.rmtree(LOG_DIR)
#         print(f'Directory {LOG_DIR} has been deleted.')
#     else:
#         print('Operation cancelled.')
# else:
#     print(f'Directory {LOG_DIR} does not exist.')

class UnsharpMask(A.ImageOnlyTransform):
    def __init__(self, blur_size, blur_sigma, amount, threshold, always_apply=False, p=1.0):
        super(UnsharpMask, self).__init__(always_apply, p)
        self.blur_size = blur_size
        self.blur_sigma = blur_sigma
        self.amount = amount
        self.threshold = threshold

    def apply(self, img, **params):
        blurred = cv2.GaussianBlur(img, (self.blur_size, self.blur_size), self.blur_sigma)
        sharpened = cv2.addWeighted(img, 1.0 + self.amount, blurred, -self.amount, gamma=self.threshold)
        return sharpened

    def get_transform_init_args_names(self):
        return ("blur_size", "blur_sigma", "amount", "threshold")

class ExpandDims(A.ImageOnlyTransform):
    def __init__(self, always_apply=False, p=1.0):
        super(ExpandDims, self).__init__(always_apply, p)

    def apply(self, img, **params):
        return np.expand_dims(img, axis=-1)

    def get_transform_init_args_names(self):
        return ()


def get_transforms(cfg, train=True):
    """
    ElasticTransform: これは、乳房組織の自然な変形をシミュレートします。ただし、パラメータは適切に調整する必要があります。
    GaussianNoise: マンモグラフィ画像には時折ノイズが含まれるため、ガウスノイズを追加することでモデルのロバスト性を向上させることができます。
    GridDistortion: この変換は局所的な歪みを画像に追加します。これは、乳房の圧迫による形状の変化を模倣する可能性があります。
    """
    transforms_list = []
    if cfg.AUGMENTATION.CLAHE:
        # clip_limit, tile_grid_size = 4.0, (8, 8)
        # clip_limit, tile_grid_size = 2.0, (8, 8)
        clip_limit, tile_grid_size = 2.0, (16, 16)
        transforms_list.append(A.CLAHE(clip_limit=clip_limit, tile_grid_size=tile_grid_size, always_apply=True, p=1.0))
        
    if train:
        if cfg.AUGMENTATION.HFLIP:
            transforms_list.append(A.HorizontalFlip(p=0.5))
        if cfg.AUGMENTATION.VFLIP:
            transforms_list.append(A.VerticalFlip(p=0.5))
        if cfg.AUGMENTATION.ELASTIC:
            transforms_list.append(A.ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03))
        if cfg.AUGMENTATION.GAUSS_NOISE:
            transforms_list.append(A.GaussNoise(var_limit=(10.0, 50.0), p=0.5))
        if cfg.AUGMENTATION.GRID_DISTORTION:
            transforms_list.append(A.GridDistortion(p=0.5))
        if cfg.AUGMENTATION.ROTATE:
            transforms_list.append(A.Rotate(p=0.5, limit=cfg.AUGMENTATION.ROTATE.LIMIT))
        if cfg.AUGMENTATION.RANDOMSIZEDCROP:
            transforms_list.append(A.RandomSizedCrop(min_max_height=(1600, 2000), height=2048, width=2048, p=0.5))
            transforms_list.append(A.OneOf([
                    A.PadIfNeeded(min_height=2048, min_width=2048, always_apply=True),
                    A.Resize(height=2048, width=2048)
                ], p=1.0))
        if cfg.AUGMENTATION.RESIZE:
            transforms_list.append(A.Resize(height=2294, width=2294))
        if cfg.AUGMENTATION.EQUALIZE:
            transforms_list.append(A.Equalize(mode='cv', by_channels=False, p=1.0))
        if cfg.AUGMENTATION.UNSHARP_MASK:
            transforms_list.append(UnsharpMask(blur_size=5, blur_sigma=1.5, amount=1.5, threshold=0))
            transforms_list.append(ExpandDims())
        if cfg.AUGMENTATION.BREASTSHIFT:
            transforms_list.append(BreastShift(p=0.5, shift_limit=0.25)),
    return A.Compose(transforms_list)

train_transform = get_transforms(cfg, train=True)
val_transform = get_transforms(cfg, train=False)
print(train_transform)
print(val_transform)
print()
print(f'REMOVE_MASK = {cfg.GENERAL.REMOVE_MASK}')

print(f'csv_random_seed = {cfg.GENERAL.CSV_RANDOM_SEED}')
csv_dir = f'/home/yukik/Work/Tohoku/original/convert_dataset/split_{cfg.GENERAL.CSV_RANDOM_SEED}'
train_csv_path = 'train.csv'
val_csv_path = 'test.csv'

train_dataset = CSVSegmentationDataset(csv_path=os.path.join(csv_dir, train_csv_path),
                                    transform=train_transform,
                                    remove_mask=cfg.GENERAL.REMOVE_MASK)
val_dataset = CSVSegmentationDataset(csv_path=os.path.join(csv_dir, val_csv_path),
                                    transform=val_transform,
                                    remove_mask=cfg.GENERAL.REMOVE_MASK)

train_loader = DataLoader(train_dataset, batch_size=cfg.GENERAL.BATCH_SIZE, shuffle=True, num_workers=cfg.GENERAL.NUM_WORKERS, collate_fn=custom_collate)
val_loader = DataLoader(val_dataset, batch_size=cfg.GENERAL.BATCH_SIZE, shuffle=False, num_workers=cfg.GENERAL.NUM_WORKERS, collate_fn=custom_collate)

print('Train dataset size:', len(train_dataset))
print('Validation dataset size:', len(val_dataset))

# モデルの定義
print(cfg.MODEL.NAME)
if cfg.MODEL.NAME == 'UNet':
    model = UNet(n_channels=1, n_classes=3)
elif cfg.MODEL.NAME == 'MIMEViT':
    model = MIMEViT(n_classes=3)

# パラメータカウント
params = 0
for p in model.parameters():
    if p.requires_grad:
        params += p.numel()
        
print(f'The Number of params = {params/10**6}M')

print(f'Loss: FocalLoss')
criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
# criterion = FocalLoss(alpha=1.0, gamma=2.0, reduction='mean')

device = torch.device(cfg.GENERAL.DEVICE if torch.cuda.is_available() else 'cpu')
print(f'device = {device}')
model = model.to(device)

if cfg.OPTIMIZER.NAME == 'Adam':
    print(f'Optimizer: {cfg.OPTIMIZER.NAME}')
    optimizer = optim.Adam(model.parameters(), lr=cfg.GENERAL.LR)
elif cfg.OPTIMIZER.NAME == 'AdamW':
    optimizer = optim.AdamW(model.parameters(), lr=cfg.GENERAL.LR, weight_decay=0.01)

print('Scheduler: ', cfg.SCHEDULER.NAME)
if cfg.SCHEDULER.NAME == 'StepLR':
    scheduler = StepLR(optimizer, step_size=cfg.SCHEDULER.STEP_SIZE, gamma=cfg.SCHEDULER.GAMMA)
elif cfg.SCHEDULER.NAME == 'ReduceLROnPlateau':
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.1)
elif cfg.SCHEDULER.NAME == 'MultiStepLR':
    scheduler = MultiStepLR(optimizer, milestones=cfg.SCHEDULER.MILESTONES, gamma=cfg.SCHEDULER.GAMMA)
elif cfg.SCHEDULER.NAME == 'CosineAnnealingLR':
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.SCHEDULER.T_MAX, eta_min=cfg.SCHEDULER.ETA_MIN)
else:
    scheduler = None

fold_num = 1
train(cfg,
    fold_num,
    model, 
    device, 
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    cfg.GENERAL.EPOCHS,
    LOG_DIR,
    metric=cfg.GENERAL.METRIC)