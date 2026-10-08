"""
Controlled model-to-model Grad-CAM + VLM explainability comparison.

Compares:
    EYE:
        EfficientNet-B0 (best non-proposed)
        MCFA-Net V2 (proposed)

    SKIN:
        EfficientNetV2-S (best non-proposed)
        CSTF-Net (proposed)

IMPORTANT:
- The SAME test image is passed to both competing models.
- Each model uses its OWN architecture-specific Grad-CAM target layers.
- For multi-branch models, branch CAMs and a branch-weighted fused CAM are saved.
- VLM receives the original image + the model's own CAM using the same prompt.
- This script does NOT claim that a VLM is ground truth.

First test:
    python src/compare_explainability.py --eye-image "path/to/eye.jpg" --skin-image "path/to/skin.jpg"

Optional VLM:
    python src/compare_explainability.py --eye-image "..." --skin-image "..." --vlm

If you only want Grad-CAM:
    python src/compare_explainability.py --eye-image "..." --skin-image "..."
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

EYE_BASELINE_CHECKPOINT = (
    PROJECT_ROOT / "outputs" / "eye_combined" / "best_efficientnet_b0.pth"
)
EYE_PROPOSED_CHECKPOINT = (
    PROJECT_ROOT / "outputs" / "eye_mcfa_v2" / "best_model.pth"
)

SKIN_BASELINE_CHECKPOINT = (
    PROJECT_ROOT / "outputs" / "skin_efficientnet_v2s" / "best_model.pth"
)
SKIN_PROPOSED_CHECKPOINT = (
    PROJECT_ROOT / "outputs" / "skin_cstf" / "best_model.pth"
)

OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "explainability"


# ============================================================
# CONFIG
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EYE_IMAGE_SIZE = 224
SKIN_BASELINE_IMAGE_SIZE = 384
SKIN_PROPOSED_IMAGE_SIZE = 224

IMAGENET_MEAN = torch.tensor(
    [0.485, 0.456, 0.406], dtype=torch.float32
).view(1, 3, 1, 1)

IMAGENET_STD = torch.tensor(
    [0.229, 0.224, 0.225], dtype=torch.float32
).view(1, 3, 1, 1)

EYE_CLASSES = [
    "Blepharitis",
    "Cataract",
    "Cherry_Eye",
    "Conjunctivitis",
    "Corneal_Sequestrum",
    "Corneal_Ulcer",
    "Glaucoma",
    "Health",
    "Non_ulcerative",
]

SKIN_CLASSES = [
    "Bacterial_dermatosis",
    "demodicosis",
    "Dermatitis",
    "Flea Allergy",
    "Fungal_infections",
    "hotspot",
    "Hypersensitivity",
    "Hypersensitivity_alergic_dermatosis",
    "mange",
    "ringworm",
    "Scabies",
]


# ============================================================
# GENERAL UTILITIES
# ============================================================

def load_image(path: Path) -> Image.Image:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found:\n{path}")
    return Image.open(path).convert("RGB")


def load_checkpoint(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found:\n{path}")
    return torch.load(path, map_location=DEVICE, weights_only=False)


def get_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        raise RuntimeError("Checkpoint is not a dictionary.")

    for key in ("model_state_dict", "state_dict"):
        if key in checkpoint:
            return checkpoint[key]

    if all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
        return checkpoint

    raise RuntimeError("Could not find a model state_dict in checkpoint.")


def clean_state_dict(state_dict):
    cleaned = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[7:]
        cleaned[key] = value
    return cleaned


def get_checkpoint_classes(checkpoint, fallback):
    if isinstance(checkpoint, dict):
        for key in ("class_names", "classes"):
            if key in checkpoint and isinstance(checkpoint[key], (list, tuple)):
                return list(checkpoint[key])
    return list(fallback)


def preprocess_rgb(image: Image.Image, size: int) -> torch.Tensor:
    image = image.resize((size, size), Image.Resampling.BILINEAR)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return tensor


def normalize_imagenet(x):
    mean = IMAGENET_MEAN.to(device=x.device, dtype=x.dtype)
    std = IMAGENET_STD.to(device=x.device, dtype=x.dtype)
    return (x - mean) / std


def resize_cam(cam: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.resize(cam.astype(np.float32), (width, height))


def normalize_cam(cam: np.ndarray) -> np.ndarray:
    cam = np.maximum(cam, 0.0)
    maximum = float(cam.max())
    if maximum > 1e-8:
        cam = cam / maximum
    return np.clip(cam, 0.0, 1.0)


def save_overlay(
    image: Image.Image,
    cam: np.ndarray,
    output_path: Path,
    title: str,
):
    original = np.asarray(image.convert("RGB"), dtype=np.uint8)
    h, w = original.shape[:2]

    cam = resize_cam(cam, w, h)
    cam = normalize_cam(cam)

    heat = (cam * 255.0).astype(np.uint8)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)

    overlay = (
        0.55 * original.astype(np.float32)
        + 0.45 * heat.astype(np.float32)
    )
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    left = cv2.cvtColor(original, cv2.COLOR_RGB2BGR)
    right = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)

    bar_h = 60
    left = cv2.copyMakeBorder(
        left, bar_h, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )
    right = cv2.copyMakeBorder(
        right, bar_h, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )

    cv2.putText(
        left, "ORIGINAL", (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA
    )
    cv2.putText(
        right, title, (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA
    )

    combined = cv2.hconcat([left, right])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), combined)


def save_side_by_side(
    image_a: Path,
    image_b: Path,
    output_path: Path,
    label_a: str,
    label_b: str,
):
    a = cv2.imread(str(image_a))
    b = cv2.imread(str(image_b))

    if a is None or b is None:
        return

    if a.shape[0] != b.shape[0]:
        b = cv2.resize(b, (a.shape[1], a.shape[0]))

    bar_h = 55
    a = cv2.copyMakeBorder(
        a, bar_h, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )
    b = cv2.copyMakeBorder(
        b, bar_h, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )

    cv2.putText(
        a, label_a, (15, 37),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA
    )
    cv2.putText(
        b, label_b, (15, 37),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), cv2.hconcat([a, b]))


# ============================================================
# COLOR CONVERSIONS
# ============================================================

def rgb_to_hsv(x):
    x = x.clamp(0.0, 1.0)

    r = x[:, 0]
    g = x[:, 1]
    b = x[:, 2]

    max_val = torch.max(x, dim=1).values
    min_val = torch.min(x, dim=1).values
    delta = max_val - min_val

    hue = torch.zeros_like(max_val)
    nonzero = delta > 1e-6

    mask = nonzero & (max_val == r)
    hue[mask] = ((g[mask] - b[mask]) / delta[mask]) % 6.0

    mask = nonzero & (max_val == g)
    hue[mask] = ((b[mask] - r[mask]) / delta[mask]) + 2.0

    mask = nonzero & (max_val == b)
    hue[mask] = ((r[mask] - g[mask]) / delta[mask]) + 4.0

    hue = hue / 6.0

    saturation = torch.zeros_like(max_val)
    nonzero_max = max_val > 1e-6
    saturation[nonzero_max] = (
        delta[nonzero_max] / max_val[nonzero_max]
    )

    return torch.stack([hue, saturation, max_val], dim=1)


def rgb_to_ycbcr(x):
    r = x[:, 0:1]
    g = x[:, 1:2]
    b = x[:, 2:3]

    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 0.5
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 0.5

    return torch.cat([y, cb, cr], dim=1).clamp(0.0, 1.0)


def sobel_texture(x):
    gray = (
        0.299 * x[:, 0:1]
        + 0.587 * x[:, 1:2]
        + 0.114 * x[:, 2:3]
    )

    sobel_x = torch.tensor(
        [[[-1.0, 0.0, 1.0],
          [-2.0, 0.0, 2.0],
          [-1.0, 0.0, 1.0]]],
        device=x.device,
        dtype=x.dtype,
    ).unsqueeze(0)

    sobel_y = torch.tensor(
        [[[-1.0, -2.0, -1.0],
          [0.0, 0.0, 0.0],
          [1.0, 2.0, 1.0]]],
        device=x.device,
        dtype=x.dtype,
    ).unsqueeze(0)

    grad_x = F.conv2d(gray, sobel_x, padding=1)
    grad_y = F.conv2d(gray, sobel_y, padding=1)

    magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)
    batch_max = magnitude.amax(dim=(2, 3), keepdim=True)

    return magnitude / (batch_max + 1e-6)


# ============================================================
# GENERIC GRAD-CAM
# ============================================================

class GradCAMHook:
    """
    Architecture-independent Grad-CAM hook.

    The hook is attached BEFORE the forward pass.
    """

    def __init__(self, target_layer: nn.Module):
        self.activations = None
        self.gradients = None
        self.handle = target_layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module, inputs, output):
        self.activations = output

        if isinstance(output, torch.Tensor) and output.requires_grad:
            output.register_hook(self._save_gradient)

    def _save_gradient(self, gradient):
        self.gradients = gradient

    def generate(self, output: torch.Tensor, class_index: int) -> np.ndarray:
        if self.activations is None:
            raise RuntimeError("Grad-CAM activation was not captured.")

        score = output[:, class_index].sum()

        self.gradients = None
        score.backward(retain_graph=True)

        if self.gradients is None:
            raise RuntimeError("Grad-CAM gradient was not captured.")

        activations = self.activations
        gradients = self.gradients

        if activations.ndim != 4 or gradients.ndim != 4:
            raise RuntimeError(
                f"Grad-CAM target must be [B,C,H,W], got "
                f"{tuple(activations.shape)}"
            )

        weights = gradients.mean(dim=(2, 3), keepdim=True)

        cam = (weights * activations).sum(dim=1)
        cam = F.relu(cam)

        cam = cam[0].detach().float().cpu().numpy()
        return normalize_cam(cam)

    def remove(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


# ============================================================
# EFFICIENTNET-B0 EYE
# ============================================================

def build_eye_efficientnet(num_classes: int):
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


# ============================================================
# MCFA-NET V2
# Exact architecture matching train_eye_mcfa_v2.py
# ============================================================

def freeze_early_features(backbone, n_blocks=4):
    n_blocks = max(0, min(n_blocks, len(backbone.features)))
    for idx, block in enumerate(backbone.features):
        requires_grad = idx >= n_blocks
        for param in block.parameters():
            param.requires_grad = requires_grad


def make_mcfa_efficientnet():
    backbone = models.efficientnet_b0(weights=None)
    feature_dim = backbone.classifier[-1].in_features
    backbone.classifier = nn.Identity()
    freeze_early_features(backbone, 4)
    return backbone, feature_dim


def normalize_with_stats(x, mean, std):
    mean = mean.to(device=x.device, dtype=x.dtype)
    std = std.to(device=x.device, dtype=x.dtype)
    return (x - mean) / std


RGB_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
RGB_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
HSV_MEAN = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
HSV_STD = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
YCBCR_MEAN = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
YCBCR_STD = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)


class MCFA_Net_V2(nn.Module):
    def __init__(
        self,
        num_classes,
        token_dim=128,
        num_heads=4,
        num_layers=2,
        dropout=0.25,
    ):
        super().__init__()

        self.rgb_backbone, feature_dim = make_mcfa_efficientnet()
        self.hsv_backbone, _ = make_mcfa_efficientnet()
        self.ycbcr_backbone, _ = make_mcfa_efficientnet()

        self.rgb_projection = nn.Sequential(
            nn.Linear(feature_dim, token_dim),
            nn.LayerNorm(token_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.hsv_projection = nn.Sequential(
            nn.Linear(feature_dim, token_dim),
            nn.LayerNorm(token_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.ycbcr_projection = nn.Sequential(
            nn.Linear(feature_dim, token_dim),
            nn.LayerNorm(token_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.branch_gate = nn.Sequential(
            nn.Linear(token_dim * 3, token_dim),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(token_dim, 3),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=num_heads,
            dim_feedforward=token_dim * 4,
            dropout=0.15,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.cross_color_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.fusion_score = nn.Sequential(
            nn.Linear(token_dim, token_dim // 2),
            nn.GELU(),
            nn.Linear(token_dim // 2, 1),
        )

        self.rgb_residual = nn.Sequential(
            nn.Linear(token_dim, token_dim),
            nn.LayerNorm(token_dim),
        )

        self.residual_gate = nn.Sequential(
            nn.Linear(token_dim * 2, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, token_dim),
            nn.Sigmoid(),
        )

        self.fusion_norm = nn.LayerNorm(token_dim)

        self.classifier = nn.Sequential(
            nn.Linear(token_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.35),
            nn.Linear(256, num_classes),
        )

    def forward(self, x, return_branch_weights=False):
        rgb = x
        hsv = rgb_to_hsv(x)
        ycbcr = rgb_to_ycbcr(x)

        rgb = normalize_with_stats(rgb, RGB_MEAN, RGB_STD)
        hsv = normalize_with_stats(hsv, HSV_MEAN, HSV_STD)
        ycbcr = normalize_with_stats(ycbcr, YCBCR_MEAN, YCBCR_STD)

        rgb_features = self.rgb_backbone(rgb)
        hsv_features = self.hsv_backbone(hsv)
        ycbcr_features = self.ycbcr_backbone(ycbcr)

        rgb_token = self.rgb_projection(rgb_features)
        hsv_token = self.hsv_projection(hsv_features)
        ycbcr_token = self.ycbcr_projection(ycbcr_features)

        tokens = torch.stack(
            [rgb_token, hsv_token, ycbcr_token], dim=1
        )

        combined = torch.cat(
            [rgb_token, hsv_token, ycbcr_token], dim=1
        )

        branch_logits = self.branch_gate(combined)
        branch_weights = torch.softmax(branch_logits, dim=1)

        gated_tokens = tokens * branch_weights.unsqueeze(-1)

        attended = self.cross_color_transformer(gated_tokens)

        fusion_scores = self.fusion_score(attended)
        fusion_weights = torch.softmax(fusion_scores, dim=1)

        fused_color = (attended * fusion_weights).sum(dim=1)

        rgb_residual = self.rgb_residual(rgb_token)

        fusion_input = torch.cat(
            [fused_color, rgb_residual], dim=1
        )

        residual_gate = self.residual_gate(fusion_input)

        fused = (
            residual_gate * fused_color
            + (1.0 - residual_gate) * rgb_residual
        )

        fused = self.fusion_norm(fused)
        logits = self.classifier(fused)

        if return_branch_weights:
            return logits, {
                "branch_weights": branch_weights,
                "fusion_weights": fusion_weights.squeeze(-1),
                "residual_gate_mean": residual_gate.mean(dim=1),
            }

        return logits


# ============================================================
# CSTF-NET
# Exact architecture matching train_skin_cstf.py
# ============================================================

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.depthwise = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=3, stride=stride, padding=1,
            groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=1, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)


class ColorBranch(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            DepthwiseSeparableConv(32, 64, stride=2),
            DepthwiseSeparableConv(64, 128, stride=2),
            DepthwiseSeparableConv(128, 256, stride=2),
            DepthwiseSeparableConv(256, 256, stride=2),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.15),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.projection(x)


class TextureBranch(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            DepthwiseSeparableConv(32, 64, stride=2),
            DepthwiseSeparableConv(64, 128, stride=2),
            DepthwiseSeparableConv(128, 256, stride=2),
            DepthwiseSeparableConv(256, 256, stride=2),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.15),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.projection(x)


class CSTFNet(nn.Module):
    def __init__(self, num_classes, pretrained=False):
        super().__init__()

        # Checkpoint loading does not need ImageNet weights.
        rgb_model = models.efficientnet_b0(weights=None)

        rgb_features = rgb_model.classifier[1].in_features
        rgb_model.classifier = nn.Identity()

        self.rgb_branch = rgb_model

        self.rgb_projection = nn.Sequential(
            nn.Linear(rgb_features, 128),
            nn.GELU(),
            nn.Dropout(0.15),
        )

        self.color_branch = ColorBranch()
        self.texture_branch = TextureBranch()

        self.branch_gate = nn.Sequential(
            nn.Linear(128 * 3, 128),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(128, 3),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128,
            nhead=4,
            dim_feedforward=512,
            dropout=0.15,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2,
        )

        self.fusion_gate = nn.Sequential(
            nn.Linear(128 * 2, 128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.Sigmoid(),
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(128),
            nn.Linear(128, 256),
            nn.GELU(),
            nn.Dropout(0.30),
            nn.Linear(256, num_classes),
        )

    def forward(self, x, return_features=False):
        rgb = self.rgb_branch(x)
        rgb = self.rgb_projection(rgb)

        hsv = rgb_to_hsv(x)
        color = self.color_branch(hsv)

        texture_map = sobel_texture(x)
        texture = self.texture_branch(texture_map)

        tokens = torch.stack([rgb, color, texture], dim=1)

        combined = torch.cat([rgb, color, texture], dim=1)
        gate_logits = self.branch_gate(combined)
        branch_weights = torch.softmax(gate_logits, dim=1)

        weighted_tokens = tokens * branch_weights.unsqueeze(-1)

        transformed = self.transformer(weighted_tokens)

        transformed_global = transformed.mean(dim=1)
        original_global = weighted_tokens.mean(dim=1)

        fusion_input = torch.cat(
            [original_global, transformed_global], dim=1
        )

        fusion_gate = self.fusion_gate(fusion_input)

        fused = (
            fusion_gate * transformed_global
            + (1.0 - fusion_gate) * original_global
        )

        logits = self.classifier(fused)

        if return_features:
            return {
                "logits": logits,
                "rgb": rgb,
                "color": color,
                "texture": texture,
                "tokens": tokens,
                "weighted_tokens": weighted_tokens,
                "transformed": transformed,
                "fused": fused,
                "branch_weights": branch_weights,
            }

        return logits


# ============================================================
# MODEL LOADING
# ============================================================

def load_model(
    checkpoint_path: Path,
    model: nn.Module,
):
    checkpoint = load_checkpoint(checkpoint_path)
    state_dict = clean_state_dict(get_state_dict(checkpoint))

    missing, unexpected = model.load_state_dict(
        state_dict, strict=False
    )

    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint mismatch for {checkpoint_path}\n"
            f"Missing keys: {missing}\n"
            f"Unexpected keys: {unexpected}"
        )

    model = model.to(DEVICE)
    model.eval()

    return model, get_checkpoint_classes(checkpoint, [])


# ============================================================
# PREDICTION
# ============================================================

def predict_from_logits(logits, classes, top_k=3):
    probabilities = torch.softmax(logits, dim=1)[0]

    values, indices = torch.topk(
        probabilities,
        k=min(top_k, len(classes))
    )

    results = []

    for value, index in zip(values.detach().cpu(), indices.detach().cpu()):
        idx = int(index.item())
        results.append({
            "index": idx,
            "class": classes[idx],
            "confidence": float(value.item()),
        })

    return results


# ============================================================
# EYE: EFFICIENTNET
# ============================================================

def run_eye_efficientnet(image, output_dir):
    model, checkpoint_classes = load_model(
        EYE_BASELINE_CHECKPOINT,
        build_eye_efficientnet(len(EYE_CLASSES)),
    )

    classes = checkpoint_classes or EYE_CLASSES

    tensor = preprocess_rgb(image, EYE_IMAGE_SIZE).to(DEVICE)
    tensor = normalize_imagenet(tensor)

    target_layer = model.features[-1]
    cam = GradCAMHook(target_layer)

    model.zero_grad(set_to_none=True)

    logits = model(tensor)
    predictions = predict_from_logits(logits, classes)
    target_class = predictions[0]["index"]

    heatmap = cam.generate(logits, target_class)
    cam.remove()

    overlay_path = output_dir / "efficientnet_b0_gradcam.png"
    save_overlay(
        image,
        heatmap,
        overlay_path,
        f"EfficientNet-B0 | {predictions[0]['class']} "
        f"{predictions[0]['confidence']:.1%}",
    )

    result = {
        "model": "EfficientNet-B0",
        "role": "best_non_proposed",
        "prediction": predictions[0],
        "top_k": predictions,
        "cam_path": str(overlay_path),
    }

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


# ============================================================
# EYE: MCFA V2
# ============================================================

def run_eye_mcfa(image, output_dir):
    model, checkpoint_classes = load_model(
        EYE_PROPOSED_CHECKPOINT,
        MCFA_Net_V2(len(EYE_CLASSES)),
    )

    classes = checkpoint_classes or EYE_CLASSES

    tensor = preprocess_rgb(image, EYE_IMAGE_SIZE).to(DEVICE)

    # Four architecture-specific hooks:
    # RGB / HSV / YCbCr final convolutional features.
    rgb_cam = GradCAMHook(model.rgb_backbone.features[-1])
    hsv_cam = GradCAMHook(model.hsv_backbone.features[-1])
    ycbcr_cam = GradCAMHook(model.ycbcr_backbone.features[-1])

    model.zero_grad(set_to_none=True)

    logits, info = model(
        tensor,
        return_branch_weights=True
    )

    predictions = predict_from_logits(logits, classes)
    target_class = predictions[0]["index"]

    rgb_heat = rgb_cam.generate(logits, target_class)
    hsv_heat = hsv_cam.generate(logits, target_class)
    ycbcr_heat = ycbcr_cam.generate(logits, target_class)

    # Branch gate is part of the actual MCFA decision path.
    branch_weights = (
        info["branch_weights"][0]
        .detach()
        .float()
        .cpu()
        .numpy()
    )

    # Fuse branch CAMs using the learned branch weights.
    fused = (
        branch_weights[0] * rgb_heat
        + branch_weights[1] * hsv_heat
        + branch_weights[2] * ycbcr_heat
    )
    fused = normalize_cam(fused)

    rgb_cam.remove()
    hsv_cam.remove()
    ycbcr_cam.remove()

    branch_names = ["rgb", "hsv", "ycbcr"]

    branch_paths = {}

    for name, heat in zip(
        branch_names,
        [rgb_heat, hsv_heat, ycbcr_heat]
    ):
        path = output_dir / f"mcfa_v2_{name}_gradcam.png"
        save_overlay(
            image,
            heat,
            path,
            f"MCFA V2 {name.upper()} | "
            f"{predictions[0]['class']} "
            f"{predictions[0]['confidence']:.1%}",
        )
        branch_paths[name] = str(path)

    fused_path = output_dir / "mcfa_v2_fused_gradcam.png"

    save_overlay(
        image,
        fused,
        fused_path,
        f"MCFA-Net V2 FUSED | {predictions[0]['class']} "
        f"{predictions[0]['confidence']:.1%}",
    )

    result = {
        "model": "MCFA-Net V2",
        "role": "proposed",
        "prediction": predictions[0],
        "top_k": predictions,
        "cam_paths": branch_paths,
        "fused_cam_path": str(fused_path),
        "branch_weights": {
            "rgb": float(branch_weights[0]),
            "hsv": float(branch_weights[1]),
            "ycbcr": float(branch_weights[2]),
        },
    }

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


# ============================================================
# SKIN: EFFICIENTNETV2-S
# ============================================================

def build_skin_efficientnet_v2s(num_classes):
    model = models.efficientnet_v2_s(weights=None)

    in_features = model.classifier[1].in_features

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes),
    )

    return model


def run_skin_efficientnet_v2s(image, output_dir):
    model, checkpoint_classes = load_model(
        SKIN_BASELINE_CHECKPOINT,
        build_skin_efficientnet_v2s(len(SKIN_CLASSES)),
    )

    classes = checkpoint_classes or SKIN_CLASSES

    tensor = preprocess_rgb(
        image,
        SKIN_BASELINE_IMAGE_SIZE
    ).to(DEVICE)

    tensor = normalize_imagenet(tensor)

    target_layer = model.features[-1]
    cam = GradCAMHook(target_layer)

    model.zero_grad(set_to_none=True)

    logits = model(tensor)

    predictions = predict_from_logits(logits, classes)
    target_class = predictions[0]["index"]

    heatmap = cam.generate(logits, target_class)
    cam.remove()

    path = output_dir / "efficientnetv2s_gradcam.png"

    save_overlay(
        image,
        heatmap,
        path,
        f"EfficientNetV2-S | {predictions[0]['class']} "
        f"{predictions[0]['confidence']:.1%}",
    )

    result = {
        "model": "EfficientNetV2-S",
        "role": "best_non_proposed",
        "prediction": predictions[0],
        "top_k": predictions,
        "cam_path": str(path),
    }

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


# ============================================================
# SKIN: CSTF
# ============================================================

def run_skin_cstf(image, output_dir):
    model, checkpoint_classes = load_model(
        SKIN_PROPOSED_CHECKPOINT,
        CSTFNet(len(SKIN_CLASSES)),
    )

    classes = checkpoint_classes or SKIN_CLASSES

    tensor = preprocess_rgb(
        image,
        SKIN_PROPOSED_IMAGE_SIZE
    ).to(DEVICE)

    # CSTF has three different feature branches.
    rgb_cam = GradCAMHook(model.rgb_branch.features[-1])
    color_cam = GradCAMHook(model.color_branch.features[-1])
    texture_cam = GradCAMHook(model.texture_branch.features[-1])

    model.zero_grad(set_to_none=True)

    outputs = model(
        tensor,
        return_features=True
    )

    logits = outputs["logits"]

    predictions = predict_from_logits(logits, classes)
    target_class = predictions[0]["index"]

    rgb_heat = rgb_cam.generate(logits, target_class)
    color_heat = color_cam.generate(logits, target_class)
    texture_heat = texture_cam.generate(logits, target_class)

    branch_weights = (
        outputs["branch_weights"][0]
        .detach()
        .float()
        .cpu()
        .numpy()
    )

    fused = (
        branch_weights[0] * rgb_heat
        + branch_weights[1] * color_heat
        + branch_weights[2] * texture_heat
    )
    fused = normalize_cam(fused)

    rgb_cam.remove()
    color_cam.remove()
    texture_cam.remove()

    branch_names = ["rgb", "hsv", "texture"]
    branch_heats = [rgb_heat, color_heat, texture_heat]

    branch_paths = {}

    for name, heat in zip(branch_names, branch_heats):
        path = output_dir / f"cstf_{name}_gradcam.png"

        save_overlay(
            image,
            heat,
            path,
            f"CSTF {name.upper()} | "
            f"{predictions[0]['class']} "
            f"{predictions[0]['confidence']:.1%}",
        )

        branch_paths[name] = str(path)

    fused_path = output_dir / "cstf_fused_gradcam.png"

    save_overlay(
        image,
        fused,
        fused_path,
        f"CSTF-Net FUSED | {predictions[0]['class']} "
        f"{predictions[0]['confidence']:.1%}",
    )

    result = {
        "model": "CSTF-Net",
        "role": "proposed",
        "prediction": predictions[0],
        "top_k": predictions,
        "cam_paths": branch_paths,
        "fused_cam_path": str(fused_path),
        "branch_weights": {
            "rgb": float(branch_weights[0]),
            "hsv": float(branch_weights[1]),
            "texture": float(branch_weights[2]),
        },
    }

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


# ============================================================
# VLM
# ============================================================

VLM_MODEL_ID = os.environ.get(
    "PET_VLM_MODEL",
    "Qwen/Qwen2.5-VL-3B-Instruct"
)


def run_vlm(
    original_path: Path,
    cam_path: Path,
    model_name: str,
    prediction: Dict,
    modality: str,
):
    """
    Run Qwen2.5-VL using the SAME prompt template for every model.

    Requires:
        transformers
        accelerate
        bitsandbytes
        qwen-vl-utils
    """

    try:
        from transformers import (
            Qwen2_5_VLForConditionalGeneration,
            AutoProcessor,
            BitsAndBytesConfig,
        )
        from qwen_vl_utils import process_vision_info
    except ImportError as exc:
        raise RuntimeError(
            "VLM dependencies are missing. Install your existing Qwen-VL "
            "dependencies before using --vlm."
        ) from exc

    print(f"\nLoading VLM: {VLM_MODEL_ID}")

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        VLM_MODEL_ID,
        quantization_config=quant_config,
        device_map="auto",
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
    )

    processor = AutoProcessor.from_pretrained(
        VLM_MODEL_ID,
        min_pixels=256 * 28 * 28,
        max_pixels=768 * 28 * 28,
    )

    prompt = f"""
You are evaluating a computer-vision explainability result for a
{modality} image.

Model:
{model_name}

Predicted class:
{prediction["class"]}

Confidence:
{prediction["confidence"]:.4f}

The first image is the original animal image.
The second image is this model's Grad-CAM overlay.

Assess only visible evidence.

Return a concise structured explanation with:
1. Visible observation
2. Anatomical/skin region highlighted by Grad-CAM
3. Whether the highlighted region appears relevant to the predicted class
4. Whether the activation appears focused, partially focused, or diffuse
5. Important uncertainty/caution

Do NOT diagnose the animal.
Do NOT invent symptoms, history, treatment, or clinical facts that are
not visible.
Do NOT assume the Grad-CAM proves causal localization.
"""

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": str(original_path),
                },
                {
                    "type": "image",
                    "image": str(cam_path),
                },
                {
                    "type": "text",
                    "text": prompt,
                },
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    inputs = inputs.to(vlm.device)

    generated_ids = vlm.generate(
        **inputs,
        max_new_tokens=350,
    )

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(
            inputs.input_ids,
            generated_ids
        )
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    del inputs
    del generated_ids
    del vlm
    del processor
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return output_text


# ============================================================
# SAVE RESULT
# ============================================================

def save_json(data, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def run_eye(image_path: Path, use_vlm: bool):
    image = load_image(image_path)

    out = OUTPUT_ROOT / "eye" / image_path.stem
    out.mkdir(parents=True, exist_ok=True)

    # Save exact original used in comparison.
    original_copy = out / "original.jpg"
    image.save(original_copy, quality=95)

    print("\n" + "=" * 75)
    print("EYE EXPLAINABILITY COMPARISON")
    print("=" * 75)

    baseline = run_eye_efficientnet(image, out)

    # Release baseline before loading the much larger MCFA model.
    proposed = run_eye_mcfa(image, out)

    side_by_side = out / "baseline_vs_mcfa.png"
    save_side_by_side(
        Path(baseline["cam_path"]),
        Path(proposed["fused_cam_path"]),
        side_by_side,
        "EfficientNet-B0",
        "MCFA-Net V2",
    )

    result = {
        "modality": "eye",
        "image": str(image_path),
        "models": [baseline, proposed],
    }

    if use_vlm:
        print("\nRunning VLM for EfficientNet-B0...")
        baseline_vlm = run_vlm(
            original_copy,
            Path(baseline["cam_path"]),
            baseline["model"],
            baseline["prediction"],
            "eye",
        )

        result["models"][0]["vlm_explanation"] = baseline_vlm

        print("\nRunning VLM for MCFA-Net V2...")
        proposed_vlm = run_vlm(
            original_copy,
            Path(proposed["fused_cam_path"]),
            proposed["model"],
            proposed["prediction"],
            "eye",
        )

        result["models"][1]["vlm_explanation"] = proposed_vlm

        (out / "vlm_efficientnet.txt").write_text(
            baseline_vlm, encoding="utf-8"
        )
        (out / "vlm_mcfa_v2.txt").write_text(
            proposed_vlm, encoding="utf-8"
        )

    save_json(result, out / "comparison.json")

    return result


def run_skin(image_path: Path, use_vlm: bool):
    image = load_image(image_path)

    out = OUTPUT_ROOT / "skin" / image_path.stem
    out.mkdir(parents=True, exist_ok=True)

    original_copy = out / "original.jpg"
    image.save(original_copy, quality=95)

    print("\n" + "=" * 75)
    print("SKIN EXPLAINABILITY COMPARISON")
    print("=" * 75)

    baseline = run_skin_efficientnet_v2s(image, out)

    proposed = run_skin_cstf(image, out)

    side_by_side = out / "baseline_vs_cstf.png"
    save_side_by_side(
        Path(baseline["cam_path"]),
        Path(proposed["fused_cam_path"]),
        side_by_side,
        "EfficientNetV2-S",
        "CSTF-Net",
    )

    result = {
        "modality": "skin",
        "image": str(image_path),
        "models": [baseline, proposed],
    }

    if use_vlm:
        print("\nRunning VLM for EfficientNetV2-S...")
        baseline_vlm = run_vlm(
            original_copy,
            Path(baseline["cam_path"]),
            baseline["model"],
            baseline["prediction"],
            "skin",
        )

        result["models"][0]["vlm_explanation"] = baseline_vlm

        print("\nRunning VLM for CSTF-Net...")
        proposed_vlm = run_vlm(
            original_copy,
            Path(proposed["fused_cam_path"]),
            proposed["model"],
            proposed["prediction"],
            "skin",
        )

        result["models"][1]["vlm_explanation"] = proposed_vlm

        (out / "vlm_efficientnetv2s.txt").write_text(
            baseline_vlm, encoding="utf-8"
        )
        (out / "vlm_cstf.txt").write_text(
            proposed_vlm, encoding="utf-8"
        )

    save_json(result, out / "comparison.json")

    return result


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Controlled Grad-CAM + VLM model comparison."
    )

    parser.add_argument(
        "--eye-image",
        type=str,
        default=None,
        help="Path to one eye test image.",
    )

    parser.add_argument(
        "--skin-image",
        type=str,
        default=None,
        help="Path to one skin test image.",
    )

    parser.add_argument(
        "--vlm",
        action="store_true",
        help="Run Qwen2.5-VL after Grad-CAM generation.",
    )

    args = parser.parse_args()

    print("=" * 75)
    print("CONTROLLED MODEL EXPLAINABILITY COMPARISON")
    print("=" * 75)
    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Output: {OUTPUT_ROOT}")
    print("=" * 75)

    if not args.eye_image and not args.skin_image:
        parser.error(
            "Provide at least --eye-image or --skin-image."
        )

    results = {}

    if args.eye_image:
        results["eye"] = run_eye(
            Path(args.eye_image),
            args.vlm,
        )

    # Ensure the eye models are released before skin models.
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if args.skin_image:
        results["skin"] = run_skin(
            Path(args.skin_image),
            args.vlm,
        )

    save_json(
        results,
        OUTPUT_ROOT / "latest_comparison.json",
    )

    print("\n" + "=" * 75)
    print("COMPARISON COMPLETE")
    print("=" * 75)
    print(f"Results: {OUTPUT_ROOT}")
    print("=" * 75)


if __name__ == "__main__":
    main()
