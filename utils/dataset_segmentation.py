import os
import cv2
import json
import torch
import numpy as np
import pandas as pd
from PIL import Image
from skimage.draw import polygon
from pycocotools.coco import COCO
from torchvision import transforms
from torch.utils.data import Dataset
import albumentations as A
from shapely.geometry import Polygon, box
import math
import torch

class CSVSegmentationDataset(Dataset):
    def __init__(self, 
                 csv_path,
                 num_channels=1,
                 overlap_threshold=0.2,
                 num_classes=1,
                 transform=None,
                 remove_mask=False,):
        
        self.data = pd.read_csv(csv_path)
        self.num_channels = num_channels
        self.overlap_threshold = overlap_threshold
        self.num_classes = num_classes
        self.transform = transform
        self.transform_patch = transforms.ToTensor()
        self.normalize = A.Normalize(mean=0.5, std=0.5)
        if remove_mask:
            self.data = self.data[self.data['annotation'] == True]

    def __len__(self):
        return len(self.data)

    def apply_transforms(self, image, masks=None):
        if masks is not None:
            augmented = self.transform(image=image, masks=masks)
            image = augmented['image']
            masks = augmented['masks']
        else:
            augmented = self.transform(image=image)
            image = augmented['image']
        return image, masks

    def create_mask_from_points(self, points, H, W, cls_value):
        pts = np.array([[int(point['x']), int(point['y'])] for point in points])

        mask = np.zeros((H, W), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], cls_value)
        return mask
    
    def create_mask_from_annos(self, annos, H, W):
        # Initialize an empty mask
        final_mask = np.zeros((H, W), dtype=np.uint8)

        labels = []
        for i, anno in enumerate(annos):
            cls = anno['label'].replace(' ', '')
            labels.append(cls)
            if cls == 'calc':
                cls_value = 2
            else:
                cls_value = 1
            
            # Create mask for the current annotation
            mask = self.create_mask_from_points(anno['cgPoints'], H, W, cls_value)
            
            # Combine the mask with the final mask
            final_mask = np.maximum(final_mask, mask)
        
        return final_mask, labels
    
    def create_combined_mask(self, row, json_path, H, W):
        if isinstance(row['labels'], str):
            with open(json_path, 'r') as f:
                annos = json.load(f)
            combined_mask, labels = self.create_mask_from_annos(annos, H, W)
        else:
            combined_mask = np.zeros((H, W), dtype=np.uint8)
            labels = []
        return combined_mask, labels

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        save_dir = row['save_dir']
        img_path = os.path.join(save_dir, 'image.png')
        image = Image.open(img_path).convert('L')
        image = np.array(image)[:,:,None]
        H, W = image.shape[:2]

        json_path = os.path.join(save_dir, 'AnnotationFile.Json')
        combined_mask, labels = self.create_combined_mask(row, json_path, H, W)

        image, mask = self.apply_transforms(image, masks=[combined_mask])

        # image = self.normalize(image=image)['image']
        image = image / 255.0
        image = image.transpose(2, 0, 1)
        image = torch.tensor(image, dtype=torch.float32)
        mask = torch.tensor(mask[0], dtype=torch.long)

        return image, mask, img_path, labels
    
    def get_image_and_masks(self, dir_path):
        row = self.data[self.data['save_dir'] == dir_path]
        save_dir = row['save_dir'].values[0]
        img_path = os.path.join(save_dir, 'image.png')
        image = Image.open(img_path).convert('L')
        image = np.array(image)[:,:,None]
        H, W = image.shape[:2]
        
        json_path = os.path.join(save_dir, 'AnnotationFile.Json')
        labels = row['labels'].values[0]
        if isinstance(labels, str):
            with open(json_path, 'r') as f:
                annos = json.load(f)
            masks = []
            category_ids = []
            for i, anno in enumerate(annos):
                masks.append(self.create_mask_from_points(anno['cgPoints'], [H, W]))
                category_ids.append(anno['label'].replace(' ', ''))
        else:
            masks = [np.zeros((H, W), dtype=np.uint8)]
            category_ids = [-1]
        
        image, masks_transformed = self.apply_transforms(image, masks=masks)
        
        # 正規化
        image = self.normalize(image=image)['image']
        image = image.transpose(2, 0, 1)
        
        masks_return = [self.mask_to_label(mask, H, W) for mask in masks_transformed]
        masks_return = [(mask * 1).astype(np.int32) for mask in masks_return]
        
        image = torch.from_numpy(image).float().unsqueeze(0)
        masks_return = [torch.from_numpy(mask) for mask in masks_return]
        
        return image, masks_return, category_ids
    
def custom_collate(batch):
    images, masks, img_paths, labels_list = zip(*batch)
    images = torch.stack(images)
    masks = torch.stack(masks)
    return images, masks, img_paths, labels_list
