import os
import shutil
import pickle
import argparse
import importlib.util
import torch
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
from model_v5_cv.dataset_segmentation import CSVSegmentationDataset
from model_v5_cv.model_concat_2294 import HIPT as HIPT_2294
from model_v5_cv.model_concat_2048 import HIPT as HIPT_2048
from model_v5_cv.utils_csv import test_with_nomask, save_prediction_images, compute_ensemble_scores, make_filename_from_dir_path
import albumentations as A
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Parse config file path from command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True, help="config file path")
args = parser.parse_args()

# Import the config file as a python module
spec = importlib.util.spec_from_file_location("config", args.config)
cfg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cfg)
cfg = cfg.Config  # Get the Config class

log_dir = cfg.GENERAL.LOG_DIR
date = log_dir[-8:]


print(cfg.MODEL.NAME)
if cfg.MODEL.NAME == 'HIPT_2294':
    model = HIPT_2294(num_classes=cfg.GENERAL.NUM_CLASSES, pretrained=cfg.MODEL.PRETRAINED,)
elif cfg.MODEL.NAME == 'HIPT_2048':
    model = HIPT_2048(num_classes=cfg.GENERAL.NUM_CLASSES, pretrained=cfg.MODEL.PRETRAINED,)

# DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
DEVICE = 'cpu'
print(f'device = {DEVICE}')

os.makedirs('result_dir/pickles_with_nomask', exist_ok=True)

transforms_list = []
if cfg.AUGMENTATION.CLAHE:
        transforms_list.append(A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), always_apply=True, p=1.0))
test_transform = A.Compose(transforms_list)

# Define test data loader
test_csv_path = f'/home/yukik/Work/Tohoku/original/convert_dataset/split_{cfg.GENERAL.CSV_RANDOM_SEED}/test.csv'
test_dataset = CSVSegmentationDataset(csv_path=test_csv_path,
                                      transform=test_transform)


dir_pickle = f'{log_dir}/pickles'
os.makedirs(dir_pickle, exist_ok=True)
verbose = False
leave = True

ensemble_results = {
    "hasmask_true": {"ious": [], "dices": [], "masks_pred": []},
    "hasmask_false": {"ious": [], "dices": [], "masks_pred": []}
}

df = pd.read_csv(f'/home/yukik/Work/Tohoku/original/convert_dataset/split_{cfg.GENERAL.CSV_RANDOM_SEED}/test.csv')

test_dir_path_hasmask_true = df[df['annotation'] == True]['save_dir'].values.tolist()
test_dir_path_hasmask_false = df[df['annotation'] == False]['save_dir'].values.tolist()

fold_num = 1
test_result_path = f'{dir_pickle}/test_results_fold{fold_num}.pkl'

if os.path.exists(test_result_path):
    with open(test_result_path, 'rb') as f:
        test_results = pickle.load(f)
else:
    path = glob(f'{log_dir}/fold{fold_num}/weights/*.pth')[0]
    model = model.to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=torch.device(DEVICE)))

    SAVE_DIR = f'{log_dir}/test_images_cv/fold{fold_num}'
    if os.path.exists(SAVE_DIR):
        shutil.rmtree(SAVE_DIR)
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    test_results = test_with_nomask(model, test_dataset, test_dir_path_hasmask_true, test_dir_path_hasmask_false, DEVICE, log_dir, fold_num, verbose=verbose, leave=leave)
    save_path = f'{dir_pickle}/test_results_fold{fold_num}.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump(test_results, f)
    print(f'Test results saved to {save_path}')

for mask_status, results in ensemble_results.items():
    results["dices"].extend(test_results[mask_status]["dices"])
    results["masks_pred"].extend(test_results[mask_status]["masks_pred"])

# Displaying the metrics
for mask_status in ["hasmask_true", "hasmask_false"]:
    print(f"For {mask_status.replace('_', ' ')}:")
    print(f"Dice mean: {np.nanmean(test_results[mask_status]['dices']):.4f} | Dice std: {np.nanstd(test_results[mask_status]['dices']):.4f}\n")
    print(f'IoU mean: {np.nanmean(test_results[mask_status]["ious"]):.4f} | IoU std: {np.nanstd(test_results[mask_status]["ious"]):.4f}\n')
print()