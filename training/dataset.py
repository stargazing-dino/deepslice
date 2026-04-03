"""Dataset class with coordinate-safe augmentations."""

import json
import random
from pathlib import Path

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset


class NinePatchDataset(Dataset):
    """Dataset with coordinate-safe augmentations."""

    def __init__(self, data_dir: Path, augment: bool = True):
        self.data_dir = Path(data_dir)

        with open(self.data_dir / 'metadata.json') as f:
            meta = json.load(f)

        self.coords = [np.array(c, dtype=np.float32) for c in meta['coords']]
        self.n_samples = meta['n_samples']
        self.augment = augment

        self.color_aug = A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=0.3),
            A.GaussNoise(var_limit=(5, 30), p=0.3),
        ])

        self.normalize = A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        img_path = self.data_dir / f'{idx:05d}.png'
        img = np.array(Image.open(img_path).convert('RGB'))

        coords = self.coords[idx].copy()

        if self.augment:
            img = self.color_aug(image=img)['image']

            if random.random() > 0.5:
                img = np.fliplr(img).copy()
                coords[0], coords[1] = 1.0 - coords[1], 1.0 - coords[0]

            if random.random() > 0.5:
                img = np.flipud(img).copy()
                coords[2], coords[3] = 1.0 - coords[3], 1.0 - coords[2]

        img = self.normalize(image=img)['image']

        return img, torch.tensor(coords, dtype=torch.float32)
