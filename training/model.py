"""9-patch coordinate regression model and metrics."""

import numpy as np
import timm
import torch
import torch.nn as nn


class NinePatchRegressor(nn.Module):
    """Predicts 4 normalized 9-patch coordinates from an image."""

    def __init__(self, backbone: str = 'efficientnet_b0', pretrained: bool = True):
        super().__init__()

        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,
            global_pool='avg'
        )

        feat_dim = self.backbone.num_features
        print(f"Backbone: {backbone}, feature dim: {feat_dim}")

        self.regressor = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.regressor(features)


def compute_region_iou(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_left, pred_right, pred_top, pred_bottom = pred.unbind(dim=1)
    tgt_left, tgt_right, tgt_top, tgt_bottom = target.unbind(dim=1)

    inter_left = torch.max(pred_left, tgt_left)
    inter_right = torch.min(pred_right, tgt_right)
    inter_top = torch.max(pred_top, tgt_top)
    inter_bottom = torch.min(pred_bottom, tgt_bottom)

    inter_w = (inter_right - inter_left).clamp(min=0)
    inter_h = (inter_bottom - inter_top).clamp(min=0)
    inter_area = inter_w * inter_h

    pred_area = (pred_right - pred_left) * (pred_bottom - pred_top)
    tgt_area = (tgt_right - tgt_left) * (tgt_bottom - tgt_top)
    union_area = pred_area + tgt_area - inter_area

    iou = inter_area / (union_area + 1e-7)
    return iou


def compute_coord_error(pred: torch.Tensor, target: torch.Tensor, img_size: int = 224) -> dict:
    pred_px = pred * img_size
    tgt_px = target * img_size

    abs_err = (pred_px - tgt_px).abs()

    return {
        'mae_overall': abs_err.mean().item(),
        'mae_left': abs_err[:, 0].mean().item(),
        'mae_right': abs_err[:, 1].mean().item(),
        'mae_top': abs_err[:, 2].mean().item(),
        'mae_bottom': abs_err[:, 3].mean().item(),
        'max_err': abs_err.max().item(),
    }
