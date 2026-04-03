"""Visualization utilities for training curves and predictions."""

import albumentations as A
import matplotlib.pyplot as plt
import torch
from albumentations.pytorch import ToTensorV2

from datagen import NinePatchCoords, NinePatchGeneratorV2
from model import NinePatchRegressor, compute_region_iou


def plot_training_history(history):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['val_loss'], label='Val')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(history['val_iou'])
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('IoU')
    axes[1].set_title('Validation IoU')
    axes[1].grid(True)

    axes[2].plot(history['val_mae'])
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('MAE (pixels)')
    axes[2].set_title('Validation MAE')
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()


def visualize_predictions(model, generator, device, n_samples=6, stage=4):
    model.eval()

    normalize = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    fig, axes = plt.subplots(n_samples, 2, figsize=(12, 4*n_samples))

    for i in range(n_samples):
        img_arr, gt_coords = generator.generate_sample(curriculum_stage=stage)

        img_tensor = normalize(image=img_arr)['image'].unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(img_tensor).cpu().numpy()[0]

        pred_coords = NinePatchCoords.from_array(pred)

        h, w = img_arr.shape[:2]
        gt_px = gt_coords.to_pixels(w, h)
        pred_px = pred_coords.to_pixels(w, h)

        axes[i, 0].imshow(img_arr)
        axes[i, 0].axvline(gt_px.stretch_left, color='green', lw=2, ls='--')
        axes[i, 0].axvline(gt_px.stretch_right, color='green', lw=2)
        axes[i, 0].axhline(gt_px.stretch_top, color='green', lw=2, ls='--')
        axes[i, 0].axhline(gt_px.stretch_bottom, color='green', lw=2)
        axes[i, 0].set_title('Ground Truth')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(img_arr)
        axes[i, 1].axvline(pred_px.stretch_left, color='red', lw=2, ls='--')
        axes[i, 1].axvline(pred_px.stretch_right, color='red', lw=2)
        axes[i, 1].axhline(pred_px.stretch_top, color='red', lw=2, ls='--')
        axes[i, 1].axhline(pred_px.stretch_bottom, color='red', lw=2)

        iou = compute_region_iou(
            torch.tensor(pred).unsqueeze(0),
            torch.tensor(gt_coords.to_array()).unsqueeze(0)
        ).item()
        axes[i, 1].set_title(f'Prediction (IoU: {iou:.3f})')
        axes[i, 1].axis('off')

    plt.tight_layout()
    plt.show()
