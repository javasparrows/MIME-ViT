import os
import csv
import cv2
import pickle
import torch
from torch import Tensor
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from PIL import Image
from sklearn.metrics import classification_report, accuracy_score
from torchvision.utils import make_grid
from torchmetrics.functional import jaccard_index
import torch.nn.functional as F
from collections import Counter
from model_v5_seg.pcgrad import PCGrad
from torch.cuda.amp import autocast, GradScaler

def train(cfg, fold_num, model, device, train_loader, val_loader, criterion, optimizer, scheduler, epochs, root_path, metric='v1'):
    model.train()  # Set model to training mode

    best_score = 0.0
    root_path += f'/fold{fold_num}'
    os.makedirs(f'{root_path}/weights', exist_ok=True)
    os.makedirs(f'{root_path}/logs', exist_ok=True)
    
    if cfg.OPTIMIZER.IF_PCGRAD:
        print('Use PCGrad for optimizer.')
        optimizer_pcgrad = PCGrad(optimizer)
    
    multiclass = cfg.GENERAL.NUM_CLASSES > 1
    
    with open(f'{root_path}/logs/training_log.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "lr", "loss"])

        scaler = GradScaler()
        for epoch in range(epochs):  # Number of epochs
            loss_list = []
            cls_loss_list = []
            seg_loss_list = []
            
            accumulation_steps = 16
            optimizer.zero_grad()  # Reset optimizer gradients to zero

            for i, (images, true_masks, _, _) in enumerate(tqdm(train_loader)):
                true_masks = torch.squeeze(true_masks, dim=1).long()
                
                images = images.to(device)
                true_masks = true_masks.to(device)

                with autocast():  # Use autocast context
                    masks_pred = model(images)

                    cls_loss = criterion(masks_pred, true_masks.long())
                    seg_loss = dice_loss(
                        F.softmax(masks_pred, dim=1).float(),
                        F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                        multiclass=True
                    )

                    loss = cls_loss + seg_loss
                    cls_loss_list.append(cls_loss.item())
                    seg_loss_list.append(seg_loss.item())
                    loss_list.append(loss.item())
                    
                    loss = loss / accumulation_steps  # Gradient scaling

                scaler.scale(loss).backward()  # Accumulate scaled gradients

                if (i + 1) % accumulation_steps == 0:  # Update gradients every accumulation_steps
                    optimizer.step()
                    optimizer.zero_grad()  # Reset gradients
                # optimizer.step()

                loss_list.append(loss.item())

            lr = optimizer.param_groups[0]['lr']
            loss_mean = np.mean(loss_list)
            cls_loss_mean = np.mean(cls_loss_list)
            seg_loss_mean = np.mean(seg_loss_list)
            print(f"\nEpoch {epoch} | lr: {lr:.6f} | Loss: {loss_mean:.4f} | cls_loss: {cls_loss_mean:.4f} | seg_loss: {seg_loss_mean:.4f}")
            writer.writerow([epoch, round(lr, 6), round(loss_mean, 3)])
            
            # Add this line to flush the file after each epoch
            f.flush()
            
            # Update scheduler
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(loss_mean)  # Update learning rate
                else:
                    scheduler.step()  # Update learning rate


            # Evaluation loop
            model.eval()
            total = 0
            dice_scores = []
            iou_scores = []

            with torch.no_grad():
                for i, (images, true_masks, _, _) in enumerate(tqdm(val_loader)):
                    images = images.to(device)
                    true_masks = true_masks.to(device)

                    with autocast():
                        masks_pred = model(images)
                    
                    dice_score_batch = dice_coeff(
                        F.softmax(masks_pred, dim=1).float(),
                        F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                    ).cpu().numpy()
                    dice_scores.append(dice_score_batch)
                    
                    masks_pred = masks_pred.reshape(-1)
                    true_masks = true_masks.reshape(-1)

                    predicted = torch.where(torch.sigmoid(masks_pred.squeeze()) > 0.5, 1, 0)
                    total += true_masks.size(0)
                    
            # Print the mean iou score over all batches
            mean_dice_score = np.mean(dice_scores)
            
            save_score(epoch, mean_dice_score, root_path)
            
            print(f"Mean Dice score: {mean_dice_score:.4f}")
            if mean_dice_score > best_score:
                print(f'\033[1;31mNew best mean Dice score: {mean_dice_score:.4f}. Saving model weights...\033[0m')
                best_score = mean_dice_score
                save_model(model, epoch, root_path)

def train_eachmask(cfg, fold_num, model, device, train_loader, val_loader, criterion, optimizer, scheduler, epochs, root_path, metric='v1'):
    model.train()  # Set model to training mode

    best_score = 0.0
    root_path += f'/fold{fold_num}'
    os.makedirs(f'{root_path}/weights', exist_ok=True)
    os.makedirs(f'{root_path}/logs', exist_ok=True)
    
    if cfg.OPTIMIZER.IF_PCGRAD:
        print('Use PCGrad for optimizer.')
        optimizer_pcgrad = PCGrad(optimizer)
    
    multiclass = cfg.GENERAL.NUM_CLASSES > 1
    
    with open(f'{root_path}/logs/training_log.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "lr", "loss"])

        for epoch in range(epochs):
            loss_list = []
            cls_loss_list = []
            seg_loss_list = []
            for images, true_masks_list in train_loader:
                for true_masks in true_masks_list:
                    images = images.to(device)
                    true_masks = true_masks.to(device)
                    # print(f'images.shape: {images.shape}')
                    # print(f'true_masks.shape: {true_masks.shape}')

                    optimizer.zero_grad()
                    masks_pred = model(images)
                    # print(f'masks_pred.shape: {masks_pred.shape}')

                    cls_loss = criterion(masks_pred, true_masks.float())
                    seg_loss = dice_loss(torch.sigmoid(masks_pred), true_masks.float(), multiclass=multiclass, weight_for_empty_mask=1.0)
                    cls_loss_list.append(cls_loss.item())
                    seg_loss_list.append(seg_loss.item())
                    loss = cls_loss + seg_loss
                    
                    if cfg.OPTIMIZER.IF_PCGRAD:
                        losses = [cls_loss, seg_loss]
                        optimizer_pcgrad.pc_backward(losses)
                    else:    
                        loss.backward()

                    optimizer.step()

                    loss_list.append(loss.item())

            lr = optimizer.param_groups[0]['lr']
            loss_mean = np.mean(loss_list)
            cls_loss_mean = np.mean(cls_loss_list)
            seg_loss_mean = np.mean(seg_loss_list)
            print(f"\nEpoch {epoch} | lr: {lr:.6f} | Loss: {loss_mean:.4f} | cls_loss: {cls_loss_mean:.4f} | seg_loss: {seg_loss_mean:.4f}")
            writer.writerow([epoch, round(lr, 6), round(loss_mean, 3)])
            
            # Add this line to flush the file after each epoch
            f.flush()
            
            # Update scheduler
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(loss_mean)  # Update learning rate
                else:
                    scheduler.step()  # Update learning rate


            # Evaluation loop
            model.eval()
            total = 0
            all_true_masks = []
            all_predictions = []
            dice_scores = []
            iou_scores = []

            with torch.no_grad():
                for images, true_masks_list in val_loader:
                    for true_masks in true_masks_list:
                        images = images.to(device)
                        true_masks = true_masks.to(device)

                        masks_pred = model(images)
                        
                        # Calculate and store the iou score
                        dice_score_batch = dice_coeff(torch.sigmoid(masks_pred), true_masks.float()).cpu().numpy()
                        dice_scores.append(dice_score_batch)
                        
                        iou_score_batch = iou_score(masks_pred, true_masks)
                        iou_scores.append(iou_score_batch)
                        
                        masks_pred = masks_pred.reshape(-1)
                        true_masks = true_masks.reshape(-1)

                        predicted = torch.where(torch.sigmoid(masks_pred.squeeze()) > 0.5, 1, 0)
                        total += true_masks.size(0)

                        all_true_masks.extend(true_masks.cpu().numpy())
                        all_predictions.extend(predicted.cpu().numpy())
                        
            # Count the occurrences of each label in all_true_masks and all_predictions
            label_counts = Counter(all_true_masks)
            prediction_counts = Counter(all_predictions)

            print("Label counts:", label_counts)
            print("Prediction counts:", prediction_counts)

            # Print the mean iou score over all batches
            mean_dice_score = np.mean(dice_scores)
            mean_iou_score = np.mean(iou_scores)
            
            save_score(epoch, mean_dice_score, root_path)
            
            print(f"Mean Dice score: {mean_dice_score:.4f}")
            if mean_dice_score > best_score:
                print(f'\033[1;31mNew best mean Dice score: {mean_dice_score:.4f}. Saving model weights...\033[0m')
                best_score = mean_dice_score
                save_model(model, epoch, root_path)

def save_score(epoch, mean_dice_score, root_path):
    with open(f'{root_path}/logs/metrics_log.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        if f.tell() == 0:
            writer.writerow(["epoch", "mean_iou", "mean_dice"])
        writer.writerow([epoch, round(mean_dice_score, 4)])

def dice_score(masks_pred, true_masks, th=0.5):
    masks_pred_bin = torch.where(torch.sigmoid(masks_pred.squeeze()) > th, 1, 0)
    intersection = (masks_pred_bin * true_masks.squeeze()).sum()
    dice = (2. * intersection) / (masks_pred_bin.sum() + true_masks.squeeze().sum())
    return dice.item()

def get_image_and_masks(dataset, filename):
    img_info_list = [img_info for img_info in dataset.coco.imgs.values() if img_info['file_name'] == filename]
    img_ids = [img_info['id'] for img_info in img_info_list]
    if not img_ids:
        raise ValueError(f"No image found with filename: {filename}")

    img_id = img_ids[0]        
    img_info = dataset.coco.loadImgs(img_id)[0]
    img_path = os.path.join(dataset.root_dir, img_info['file_name'])
    
    image = Image.open(img_path).convert('L')
    image = np.array(image)[:,:,None]  # Add a dummy channel
    H, W = image.shape[:2]
    
    ann_ids = dataset.coco.getAnnIds(imgIds=img_id)
    anns = dataset.coco.loadAnns(ann_ids)
    masks = [dataset.coco.annToMask(ann) for ann in anns]
    class_names = [ann['category_id'] for ann in anns]

    if not masks:
        masks = [np.zeros((H, W), dtype=np.uint8)]
        class_names = [-1]
    
    image, masks_transformed = dataset.apply_transforms(image, masks=masks)
    
    # Normalization
    image = dataset.normalize(image=image)['image']
    image = image.transpose(2, 0, 1)
    
    masks_return = [dataset.mask_to_label(mask, H, W) for mask in masks_transformed]
    masks_return = [(mask * 1).astype(np.int32) for mask in masks_return]
    
    image = torch.from_numpy(image).float().unsqueeze(0)
    masks_return = [torch.from_numpy(mask) for mask in masks_return]
    
    return image, masks_return, class_names


def test(model, test_dataset, test_dir_path_hasmask_true, device, verbose=False, leave=True):
    model.eval()

    test_results = {
        'filenames': [],
        'class_names': [],
        'ious': [],
        'dices': [],
        'true_masks': [],
        'masks_pred': [],
    }

    with torch.no_grad():
        for idx, filename in enumerate(tqdm(test_dir_path_hasmask_true, leave=leave)):
            image, true_masks, class_names = get_image_and_masks(test_dataset, filename)
            image = image.to(device)
            
            for true_mask, class_name in zip(true_masks, class_names):
                true_mask = true_mask.to(device)

                masks_pred = model(image)
                iou = iou_score(masks_pred, true_mask)
                dice = dice_score(masks_pred, true_mask)

                if verbose:
                    print(f"Dice score: {dice:.4f}")

                test_results['filenames'].append(filename)
                test_results['class_names'].append(class_name)
                test_results['ious'].append(iou)
                test_results['dices'].append(dice)
                test_results['true_masks'].append(true_mask)
                test_results['masks_pred'].append(masks_pred)

    return test_results

def test_with_nomask(model, test_dataset, test_dir_path_hasmask_true, test_dir_path_hasmask_false, device, verbose=False, leave=True):
    model.eval()

    test_results = {
        "hasmask_true": {
            'dir_name': [],
            'class_names': [],
            'ious': [],
            'dices': [],
            'true_masks': [],
            'masks_pred': [],
            'hasmask': []
        },
        "hasmask_false": {
            'dir_name': [],
            'class_names': [],
            'ious': [],
            'dices': [],
            'true_masks': [],
            'masks_pred': [],
            'hasmask': []
        }
    }

    with torch.no_grad():
        for mask_status, dir_paths in [("hasmask_true", test_dir_path_hasmask_true), ("hasmask_false", test_dir_path_hasmask_false)]:
            for idx, dir_path in enumerate(tqdm(dir_paths, leave=leave)):
                image, true_masks, class_names = test_dataset.get_image_and_masks(dir_path)
                image = image.to(device)
                masks_pred = model(image)

                for true_mask, class_name in zip(true_masks, class_names):
                    true_mask = true_mask.to(device)

                    if mask_status == "hasmask_true":
                        th = 0.2
                        iou = iou_bounding_score(masks_pred, true_mask, th=th)
                        dice = dice_score(masks_pred, true_mask, th=th)
                    else:
                        dice = background_dice_score(masks_pred)
                        iou = dice

                    if verbose:
                        print(f"Dice score: {dice:.4f}")

                    result, _ = make_filename_from_dir_path(dir_path)
                    test_results[mask_status]['dir_name'].append(result)
                    test_results[mask_status]['ious'].append(iou)
                    test_results[mask_status]['dices'].append(dice)
                    test_results[mask_status]['true_masks'].append(true_mask)
                    test_results[mask_status]['masks_pred'].append(masks_pred)
                    test_results[mask_status]['hasmask'].append(True if mask_status == "hasmask_true" else False)
                    test_results[mask_status]['class_names'].append(class_name)

    return test_results

def background_dice_score(predicted_mask, threshold=0.5, alpha=0.0):
    TP = torch.sum(predicted_mask <= threshold).item()
    FP = torch.sum(predicted_mask > threshold).item()
    return (2.0 * TP) / (2.0 * TP + FP + alpha * FP)

def make_filename_from_dir_path(dir_path):
    base_name = os.path.basename(dir_path)
    parent_name = os.path.basename(os.path.dirname(dir_path))
    
    result = f"{parent_name}_{base_name}"
    patient_id = parent_name
    return result, patient_id

def compute_ensemble_scores(true_masks, ensemble_masks_preds_mean, mask_status, th=0.5):
    ensemble_ious = []
    ensemble_dices = []
    
    for idx, masks_pred in enumerate(ensemble_masks_preds_mean):
        true_mask_tensor = torch.Tensor(true_masks[idx])  # Convert the mask to tensor if it's not already
        
        if mask_status == "hasmask_true":
            iou = iou_score(masks_pred.unsqueeze(0), true_mask_tensor.unsqueeze(0), th=th)
            dice = dice_score(masks_pred.unsqueeze(0), true_mask_tensor.unsqueeze(0), th=th)
        else:
            dice = background_dice_score(masks_pred)
            iou = dice
            # # Proposed accuracy computation for hasmask=False
            # predicted_mask_area = torch.sum(masks_pred > th).item()
            # if predicted_mask_area == 0:
            #     iou, dice = 1.0, 1.0
            # else:
            #     iou, dice = 0.0, 0.0
                
        ensemble_ious.append(iou)
        ensemble_dices.append(dice)
        
    return ensemble_ious, ensemble_dices


def iou_score(masks_pred, true_masks, th=0.5):
    masks_pred = torch.where(torch.sigmoid(masks_pred.squeeze()) > th, 1, 0)
    iou = jaccard_index(masks_pred, true_masks.squeeze(), task="binary", num_classes=1)
    return iou.item()

import torch

def get_bounding_box(mask):
    """Get bounding box from mask"""
    rows = torch.any(mask, dim=1)
    cols = torch.any(mask, dim=0)
    
    if rows.any() and cols.any():
        rmin, rmax = torch.where(rows)[0][[0, -1]]
        cmin, cmax = torch.where(cols)[0][[0, -1]]
        return rmin, rmax, cmin, cmax
    else:
        return None

def compute_iou(box1, box2):
    """Calculate IoU of two bounding boxes"""
    rmin1, rmax1, cmin1, cmax1 = box1
    rmin2, rmax2, cmin2, cmax2 = box2

    # Intersection area coordinates
    rmin_int = max(rmin1, rmin2)
    rmax_int = min(rmax1, rmax2)
    cmin_int = max(cmin1, cmin2)
    cmax_int = min(cmax1, cmax2)

    # Intersection area
    area_int = max(0, rmax_int - rmin_int + 1) * max(0, cmax_int - cmin_int + 1)

    # Area of each bounding box
    area_box1 = (rmax1 - rmin1 + 1) * (cmax1 - cmin1 + 1)
    area_box2 = (rmax2 - rmin2 + 1) * (cmax2 - cmin2 + 1)

    # IoU calculation
    iou = area_int / (area_box1 + area_box2 - area_int)
    return iou

def iou_bounding_score(masks_pred, true_masks, th=0.5):
    masks_pred = torch.where(torch.sigmoid(masks_pred.squeeze()) > th, 1, 0)
    
    box_pred = get_bounding_box(masks_pred)
    box_true = get_bounding_box(true_masks)
    
    if box_pred is None or box_true is None:
        return 0.0  # If mask is empty, set IoU score to 0
    
    iou = compute_iou(box_pred, box_true)
    return iou

def save_model(model, epoch, root_path):
    # Delete all previous model weights
    weights_path = os.path.join(root_path, 'weights')
    files_in_directory = os.listdir(weights_path)
    filtered_files = [file for file in files_in_directory if file.endswith(".pth")]
    for file in filtered_files:
        path_to_file = os.path.join(weights_path, file)
        os.remove(path_to_file)

    # Save new best model weights
    torch.save(model.state_dict(), f'{weights_path}/best_model_{epoch+1}epoch.pth')


def to_numpy(tensor):
    return tensor.detach().cpu().numpy()

def to_pil(image):
    return Image.fromarray(np.uint8(image*255))

def to_numpy_image(img):
    return np.array(img)


def save_prediction_images(image_id, image, true_masks, masks_pred, category_id, save_dir, dice, verbose):
    # Make sure save directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    image = image.squeeze().cpu().numpy()
    masks_pred = masks_pred.squeeze().cpu().numpy()
    true_masks = true_masks.cpu().numpy()
    
    masks_pred = torch.sigmoid(torch.from_numpy(masks_pred))

    if verbose:
        print(f'dice={dice:.4f} {image_id}')
    if len(masks_pred.shape) < 3:
        masks_pred = masks_pred.unsqueeze(0)
        
    masks_pred = masks_pred.numpy()

    # Resize masks to match the original image using nearest neighbor interpolation
    masks_pred_resized = [cv2.resize(mask, (2294, 2294), interpolation=cv2.INTER_NEAREST) for mask in masks_pred]
    true_masks_resized = [cv2.resize(mask, (2294, 2294), interpolation=cv2.INTER_NEAREST) for mask in true_masks]

    # Threshold masks
    masks_pred_thresholded = [np.where(mask > 0.5, 1, 0) for mask in masks_pred_resized]

    # Define colors for each class
    colors_true = [[0, 0.8, 0, 0.6]]  # Soft green with alpha=0.6 for true masks
    colors_pred = [[1, 0.5, 0, 0.6]]  # Soft orange with alpha=0.6 for predicted masks

    # Create overlays for true and predicted masks
    overlay_true = np.zeros((2294, 2294, 4))
    for i, mask in enumerate(true_masks_resized):
        overlay_true[mask == 1] = colors_true[i]

    overlay_pred = np.zeros((2294, 2294, 4))
    for i, mask in enumerate(masks_pred_thresholded):
        overlay_pred[mask == 1] = colors_pred[i]

    # Plot
    fig, axs = plt.subplots(1, 3, figsize=(16, 6))

    # Original Image
    axs[0].imshow(image, cmap='gray')
    axs[0].set_title("Original Image")
    axs[0].axis('off')

    # Original Image with Ground Truth
    axs[1].imshow(image, cmap='gray')
    axs[1].imshow(overlay_true)
    axs[1].set_title("Original Image with Ground Truth")
    axs[1].axis('off')

    # Original Image with Predicted Mask
    axs[2].imshow(image, cmap='gray')
    axs[2].imshow(overlay_pred)
    axs[2].set_title("Original Image with Predicted Mask")
    axs[2].axis('off')
    
    # Save the figure
    figure_save_path = os.path.join(save_dir, f"{image_id}_dice={dice:.4f}_{category_id}.png")
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.1, hspace=0.1)
    plt.savefig(figure_save_path)
    plt.close(fig)

def flatten_masks_pred_and_labels(masks_pred, labels):
    # masks_pred shape: [batch_size, num_patches, num_classes] --> [batch_size*num_patches, num_classes]
    # labels shape: [batch_size, num_patches] --> [batch_size*num_patches]
    batch_size, num_patches, num_classes = masks_pred.shape
    masks_pred_flattened = masks_pred.reshape(batch_size * num_patches, num_classes)
    labels_flattened = labels.view(-1)
    return masks_pred_flattened, labels_flattened


def print_custom_metrics(all_labels, all_predictions, epoch, root_path):
    (metrics, components) = calculate_custom_metrics(all_labels, all_predictions)
    accuracy, sensitivity, specificity, precision, f1_score = metrics
    true_positive, true_negative, false_positive, false_negative = components

    with open(f'{root_path}/logs/metrics_log.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        if f.tell() == 0:
            writer.writerow(["epoch", "accuracy", "sensitivity", "specificity", "precision", "f1_score", "TP", "TN", "FP", "FN"])
        writer.writerow([epoch, 
                         round(accuracy, 3) if accuracy is not None else 'N/A', 
                         round(sensitivity, 3) if sensitivity is not None else 'N/A',
                         round(specificity, 3) if specificity is not None else 'N/A',
                         round(precision, 3) if precision is not None else 'N/A', 
                         round(f1_score, 3) if f1_score is not None else 'N/A', 
                         true_positive, 
                         true_negative, 
                         false_positive, 
                         false_negative])
    
    # Print metrics with color coding
    print("\033[1;31m" +  # Set color to red
            "Accuracy: " + (str(round(accuracy, 3)) if accuracy is not None else 'N/A') + "\033[1;37m" +
            " (TP + TN / Total = " + str(true_positive) + " + " + str(true_negative) + " / " + str(true_positive + false_positive + false_negative + true_negative) + ")" +
            "\033[0m"  # Reset color
            )
    print("\033[1;32m" +  # Set color to green
            "Sensitivity: " + (str(round(sensitivity, 3)) if sensitivity is not None else 'N/A') + "\033[1;37m" +
            " (TP / (TP + FN) = " + str(true_positive) + " / (" + str(true_positive) + " + " + str(false_negative) + "))" +
            "\033[0m"  # Reset color
            )
    print("\033[1;33m" +  # Set color to yellow
            "Specificity: " + (str(round(specificity, 3)) if specificity is not None else 'N/A') + "\033[1;37m" +
            " (TN / (TN + FP) = " + str(true_negative) + " / (" + str(true_negative) + " + " + str(false_positive) + "))" +
            "\033[0m"  # Reset color
            )
    print("\033[1;34m" +  # Set color to blue
            "Precision: " + (str(round(precision, 3)) if precision is not None else 'N/A') + "\033[1;37m" +
            " (TP / (TP + FP) = " + str(true_positive) + " / (" + str(true_positive) + " + " + str(false_positive) + "))" +
            "\033[0m"  # Reset color
            )
    print("\033[1;35m" +  # Set color to magenta
            "F1 Score: " + (str(round(f1_score, 3)) if f1_score is not None else 'N/A') + "\033[1;37m" +
            " (2 * ((Precision * Sensitivity) / (Precision + Sensitivity)) = 2 * ((" + (str(round(precision, 3)) if precision is not None else 'N/A') + " * " + (str(round(sensitivity, 3)) if sensitivity is not None else 'N/A') + ") / (" + (str(round(precision, 3)) if precision is not None else 'N/A') + " + " + (str(round(sensitivity, 3)) if sensitivity is not None else 'N/A') + ")))" +
            "\033[0m"  # Reset color
            )
    return f1_score



def calculate_custom_metrics(labels, predictions):
    # Initialize counters
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0

    for label, prediction in zip(labels, predictions):
        if label == 1:
            if prediction == 1:
                true_positive += 1
            else:
                false_negative += 1
        else:
            if prediction == 0:
                true_negative += 1
            else:
                false_positive += 1

    # Calculate metrics
    accuracy = (true_positive + true_negative) / (true_positive + false_positive + false_negative + true_negative)
    sensitivity = true_positive / (true_positive + false_negative) if true_positive + false_negative != 0 else None  # Recall for disease positive (label 1)
    specificity = true_negative / (true_negative + false_positive) if true_negative + false_positive != 0 else None  # Correctly identified negative cases (label 0)

    # Additional metrics
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive != 0 else None  # Correctly predicted positive cases (label 1)
    f1_score = 2 * ((precision * sensitivity) / (precision + sensitivity)) if precision is not None and sensitivity is not None and (precision + sensitivity) != 0 else None
    return (accuracy, sensitivity, specificity, precision, f1_score), (true_positive, true_negative, false_positive, false_negative)


def calculate_metrics(all_labels, all_predictions):
    accuracy = accuracy_score(all_labels, all_predictions)
    classification_metrics = classification_report(all_labels, all_predictions, output_dict=True, zero_division=0)
    current_f1_score = classification_metrics['macro avg']['f1-score']
    
    return accuracy, classification_metrics, current_f1_score


def print_metrics(epoch, accuracy, classification_metrics):
    print(f"Epoch {epoch+1}, Validation Accuracy: {accuracy:.2f}")
    print("Classification Report: ")

    if isinstance(classification_metrics, dict):
        for label, metrics in classification_metrics.items():
            if label not in ['accuracy', 'macro avg', 'weighted avg']:
                print(f"Label {label}:")
                print(f"    Precision: {metrics['precision']:.2f}")
                print(f"    Recall: {metrics['recall']:.2f}")
                print(f"    F1-score: {metrics['f1-score']:.2f}")
                print(f"    Support: {metrics['support']}")

        print("Overall:")
        print(f"    Accuracy: {classification_metrics['accuracy']:.2f}")
        print(f"    Macro avg precision: {classification_metrics['macro avg']['precision']:.2f}")
        print(f"    Macro avg recall: {classification_metrics['macro avg']['recall']:.2f}")
        print(f"    Macro avg f1-score: {classification_metrics['macro avg']['f1-score']:.2f}")
    else:
        print(classification_metrics)


def imshow(img):
    """
    Display an image.
    
    Args:
    img (torch.Tensor): The image to be displayed. Should be a PyTorch Tensor.
    """
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))  # transpose the tensor to match image dimension ordering
    plt.show()


def display_reconstructed_images(dataset, idx=0, save_dir='./saved_images'):
    """
    Display reconstructed images from patches, along with the original and labeled images.
    
    Args:
    dataset (COCOPatchDataset): The dataset to draw images from.
    idx (int, optional): The index of the image in the dataset to be displayed.
    save_dir (str, optional): The directory to save the images.
    """
    # Load original image
    original_image_path = dataset.get_image_path(idx)
    original_image_filename = os.path.splitext(os.path.basename(original_image_path))[0]
    
    # Make sure save directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    patches, labels = dataset[idx]  # get patches and labels for given image

    # Load original image
    original_image_path = dataset.get_image_path(idx)
    original_image = Image.open(original_image_path)
    original_image = np.array(original_image)
    original_image = original_image / 255.0  # normalize to [0,1] range

    # Prepare figure
    fig, axs = plt.subplots(2, len(dataset.patch_sizes) + 1, figsize=(16, 8))
    
    # Display original image
    axs[0, 0].imshow(original_image, cmap='gray')
    axs[0, 0].set_title("Original Image")
    axs[0, 0].axis('off')

    # Display original image with bounding box
    bbox_image = (original_image * 255).astype(np.uint8)  # Convert back to [0, 255] range for drawing bounding box
    bbox_image = dataset.draw_bboxes(bbox_image, idx)
    axs[1, 0].imshow(bbox_image)
    axs[1, 0].set_title("Original Image with Bboxes")
    axs[1, 0].axis('off')

    # reconstruct and display images for each patch size
    for i, (patch_size, patches_per_size, label_per_size) in enumerate(zip(dataset.patch_sizes, patches, labels)):
        num_patches = patches_per_size.shape[0]
        num_patches_side = int(np.sqrt(num_patches))  # Assuming square image

        # Rearrange patches into a grid
        patches_grid = patches_per_size.view(num_patches_side, num_patches_side, 3, patch_size, patch_size)

        # Transpose to match the original order
        patches_grid = patches_grid.transpose(0, 1)

        # reconstruct image
        reconstructed_image = make_grid(patches_grid.reshape(-1, 3, patch_size, patch_size), nrow=num_patches_side)
        reconstructed_image = np.transpose(reconstructed_image.numpy(), (1, 2, 0))  # Convert to numpy array and rearrange dimensions
        reconstructed_image = (reconstructed_image + 1) / 2  # Scale to [0, 1] range

        # Clip values to [0, 1] range and Display reconstructed image
        reconstructed_image = np.clip(reconstructed_image, 0, 1)
        axs[0, i+1].imshow(reconstructed_image)
        axs[0, i+1].set_title(f"Reconstructed Image (Patch Size: {patch_size})")
        axs[0, i+1].axis('off')

        # Label patches and Display labeled image
        labeled_image = reconstructed_image.copy()
        cmap = plt.get_cmap("tab10")  # Get the colormap
        
        # Transform the label tensor to a list for easy access
        label_list = label_per_size.tolist()
        
        # Apply the labels to the reconstructed image
        for j in range(num_patches_side):
            for k in range(num_patches_side):
                labeled_image[j*patch_size:(j+1)*patch_size, k*patch_size:(k+1)*patch_size, :] *= cmap(label_list[j*num_patches_side+k])[:3]
                
        axs[1, i+1].imshow(labeled_image)  # Display the labeled image
        axs[1, i+1].set_title(f"Labeled Image (Patch Size: {patch_size})")
        axs[1, i+1].axis('off')

    # plt.show()  # Display the plot
    # Save image
    plt.savefig(f'{save_dir}/image_{original_image_filename}.png')  # Save the image with the original image file name
    plt.close()  # Close the plot

def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()


def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all classes
    return dice_coeff(input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)


def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)


def weighted_custom_multiclass_dice_loss(input: Tensor, target: Tensor, class_weights):
    # Convert class_weights to a Tensor if it's a list
    if isinstance(class_weights, list):
        class_weights = torch.tensor(class_weights)

    # Ensure class_weights is on the same device as the input
    class_weights = class_weights.to(input.device)

    # Get the classes present in the target
    classes_present = target.unique()

    total_loss = 0
    for class_id in classes_present:
        # Convert class_id to long type
        class_id = class_id.long()

        # For each class present, compute the binary Dice loss
        binary_input = (input == class_id).float()
        binary_target = (target == class_id).float()

        # Compute binary Dice coefficient and loss
        dice_coeff_ = dice_coeff(binary_input, binary_target, reduce_batch_first=True)
        class_loss = 1 - dice_coeff_

        # Weight the loss by the class's weight
        class_loss *= class_weights[class_id]

        # Accumulate loss
        total_loss += class_loss

    # Average loss over classes present
    average_loss = total_loss / classes_present.numel()
    return average_loss
