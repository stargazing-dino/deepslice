# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "torch", "timm", "albumentations", "pillow", "numpy", "tqdm",
# ]
# ///
"""
Training script for 9-patch coordinate regression.

Usage (standalone):
    python train.py --checkpoint-dir ./checkpoints --data-dir ./data

Usage (Colab):
    !python train.py --checkpoint-dir /content/drive/MyDrive/9patch_checkpoints_v2 --data-dir /content/9patch_data_v2
"""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from datagen import NinePatchGeneratorV2, generate_and_cache_dataset
from dataset import NinePatchDataset
from model import NinePatchRegressor, compute_region_iou, compute_coord_error


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Trainer:
    def __init__(self, model, device, checkpoint_dir):
        self.model = model.to(device)
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)

        self.criterion = nn.SmoothL1Loss()
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        self.scaler = GradScaler()

        self.best_iou = 0.0
        self.history = {'train_loss': [], 'val_loss': [], 'val_iou': [], 'val_mae': []}

    def train_epoch(self, train_loader, scheduler=None):
        self.model.train()
        total_loss = 0

        pbar = tqdm(train_loader, desc='Training')
        for images, targets in pbar:
            images = images.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()

            with autocast(device_type='cuda'):
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if scheduler:
                scheduler.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        return total_loss / len(train_loader)

    @torch.no_grad()
    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        all_ious = []
        all_errors = []

        for images, targets in tqdm(val_loader, desc='Validating'):
            images = images.to(self.device)
            targets = targets.to(self.device)

            with autocast(device_type='cuda'):
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)

            total_loss += loss.item()

            ious = compute_region_iou(outputs, targets)
            all_ious.append(ious.cpu())

            errors = compute_coord_error(outputs.cpu(), targets.cpu())
            all_errors.append(errors)

        avg_loss = total_loss / len(val_loader)
        avg_iou = torch.cat(all_ious).mean().item()
        avg_mae = np.mean([e['mae_overall'] for e in all_errors])

        return {
            'loss': avg_loss,
            'iou': avg_iou,
            'mae_px': avg_mae,
        }

    def save_checkpoint(self, epoch, val_metrics, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_metrics': val_metrics,
            'history': self.history,
        }

        path = self.checkpoint_dir / f'checkpoint_epoch{epoch}.pth'
        torch.save(checkpoint, path)

        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"  Saved new best model (IoU: {val_metrics['iou']:.4f})")

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint.get('history', self.history)
        self.best_iou = checkpoint.get('val_metrics', {}).get('iou', 0.0)
        return checkpoint['epoch']


def train_with_curriculum(trainer, generator, data_cache_dir,
                          stages_config, batch_size=32):
    global_epoch = 0

    for stage_cfg in stages_config:
        stage = stage_cfg['stage']
        n_epochs = stage_cfg['epochs']

        print(f"\n{'='*60}")
        print(f"CURRICULUM STAGE {stage} - {n_epochs} epochs")
        print(f"{'='*60}")

        train_dir = generate_and_cache_dataset(
            generator, stage_cfg['train_samples'], stage, data_cache_dir, 'train'
        )
        val_dir = generate_and_cache_dataset(
            generator, stage_cfg['val_samples'], stage, data_cache_dir, 'val'
        )

        train_dataset = NinePatchDataset(train_dir, augment=True)
        val_dataset = NinePatchDataset(val_dir, augment=False)

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=2, pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, num_workers=2
        )

        for epoch in range(n_epochs):
            global_epoch += 1
            print(f"\nEpoch {global_epoch} (Stage {stage}, epoch {epoch+1}/{n_epochs})")

            scheduler = OneCycleLR(
                trainer.optimizer,
                max_lr=3e-4,
                epochs=1,
                steps_per_epoch=len(train_loader)
            )

            train_loss = trainer.train_epoch(train_loader, scheduler)
            val_metrics = trainer.validate(val_loader)

            trainer.history['train_loss'].append(train_loss)
            trainer.history['val_loss'].append(val_metrics['loss'])
            trainer.history['val_iou'].append(val_metrics['iou'])
            trainer.history['val_mae'].append(val_metrics['mae_px'])

            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_metrics['loss']:.4f}, IoU: {val_metrics['iou']:.4f}, MAE: {val_metrics['mae_px']:.1f}px")

            is_best = val_metrics['iou'] > trainer.best_iou
            if is_best:
                trainer.best_iou = val_metrics['iou']

            if global_epoch % 5 == 0 or is_best:
                trainer.save_checkpoint(global_epoch, val_metrics, is_best)

    return trainer


STAGES_CONFIG = [
    {'stage': 1, 'epochs': 8, 'train_samples': 5000, 'val_samples': 500},
    {'stage': 2, 'epochs': 12, 'train_samples': 8000, 'val_samples': 800},
    {'stage': 3, 'epochs': 15, 'train_samples': 10000, 'val_samples': 1000},
    {'stage': 4, 'epochs': 15, 'train_samples': 12000, 'val_samples': 1200},
]


def main():
    parser = argparse.ArgumentParser(description="Train 9-patch coordinate regressor")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--resume", type=Path, help="Resume from checkpoint")
    args = parser.parse_args()

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.data_dir.mkdir(parents=True, exist_ok=True)

    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    generator = NinePatchGeneratorV2(output_size=(224, 224))
    model = NinePatchRegressor('efficientnet_b0', pretrained=True)
    trainer = Trainer(model, device, args.checkpoint_dir)

    if args.resume:
        epoch = trainer.load_checkpoint(args.resume)
        print(f"Resumed from epoch {epoch}")

    train_with_curriculum(trainer, generator, args.data_dir,
                          STAGES_CONFIG, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
