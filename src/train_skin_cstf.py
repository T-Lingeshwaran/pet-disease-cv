# src/train_skin_cstf.py
"""
CSTF-Net
Color-Spatial-Texture Fusion Network

Proposed skin-condition classification architecture.

Branches:
    1. RGB semantic branch       -> pretrained EfficientNet-B0
    2. HSV color branch          -> lightweight CNN
    3. Texture branch            -> Sobel gradient + lightweight CNN

Fusion:
    -> learnable branch gating
    -> cross-feature Transformer
    -> gated residual fusion
    -> 11-class classifier

Dataset:
    data/skin_coat/combined_clean

Outputs:
    outputs/skin_cstf/
"""

import os
import json
import random
import copy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = "data/skin_coat/combined_clean"
OUTPUT_DIR = "outputs/skin_cstf"

IMAGE_SIZE = 224
BATCH_SIZE = 8

NUM_EPOCHS = 50
PATIENCE = 7

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.05

NUM_WORKERS = 0
SEED = 42

USE_AMP = True


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


set_seed(SEED)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# DEVICE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 70)
print("CSTF-NET")
print("Color-Spatial-Texture Fusion Network")
print("=" * 70)

print(f"Device        : {device}")

if torch.cuda.is_available():
    print(f"GPU           : {torch.cuda.get_device_name(0)}")
    print(
        f"VRAM          : "
        f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
    )

print(f"Dataset       : {DATA_DIR}")
print(f"Image size    : {IMAGE_SIZE}")
print(f"Batch size    : {BATCH_SIZE}")
print(f"Epochs        : {NUM_EPOCHS}")
print(f"Learning rate : {LEARNING_RATE}")
print("=" * 70)


# ============================================================
# DIFFERENTIABLE RGB -> HSV
# ============================================================

def rgb_to_hsv(x):
    """
    Differentiable RGB -> HSV conversion.

    Input:
        x: [B, 3, H, W], values in [0, 1]

    Output:
        HSV tensor [B, 3, H, W]
    """

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
    hue[mask] = (
        ((g[mask] - b[mask]) / delta[mask]) % 6.0
    )

    mask = nonzero & (max_val == g)
    hue[mask] = (
        ((b[mask] - r[mask]) / delta[mask]) + 2.0
    )

    mask = nonzero & (max_val == b)
    hue[mask] = (
        ((r[mask] - g[mask]) / delta[mask]) + 4.0
    )

    hue = hue / 6.0

    saturation = torch.zeros_like(max_val)

    nonzero_max = max_val > 1e-6

    saturation[nonzero_max] = (
        delta[nonzero_max] / max_val[nonzero_max]
    )

    value = max_val

    return torch.stack(
        [hue, saturation, value],
        dim=1
    )


# ============================================================
# SOBEL TEXTURE EXTRACTION
# ============================================================

def sobel_texture(x):
    """
    Creates a texture/edge representation using Sobel filters.

    Input:
        RGB tensor [B,3,H,W]

    Output:
        Gradient magnitude [B,1,H,W]
    """

    # Convert RGB to grayscale.
    gray = (
        0.299 * x[:, 0:1]
        + 0.587 * x[:, 1:2]
        + 0.114 * x[:, 2:3]
    )

    sobel_x = torch.tensor(
        [
            [-1.0, 0.0, 1.0],
            [-2.0, 0.0, 2.0],
            [-1.0, 0.0, 1.0],
        ],
        device=x.device,
        dtype=x.dtype,
    ).view(1, 1, 3, 3)

    sobel_y = torch.tensor(
        [
            [-1.0, -2.0, -1.0],
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 1.0],
        ],
        device=x.device,
        dtype=x.dtype,
    ).view(1, 1, 3, 3)

    grad_x = F.conv2d(
        gray,
        sobel_x,
        padding=1
    )

    grad_y = F.conv2d(
        gray,
        sobel_y,
        padding=1
    )

    magnitude = torch.sqrt(
        grad_x ** 2
        + grad_y ** 2
        + 1e-6
    )

    # Normalize each image independently.
    batch_max = magnitude.amax(
        dim=(2, 3),
        keepdim=True
    )

    magnitude = magnitude / (
        batch_max + 1e-6
    )

    return magnitude


# ============================================================
# LIGHTWEIGHT CNN BLOCK
# ============================================================

class DepthwiseSeparableConv(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1
    ):
        super().__init__()

        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_channels,
            bias=False,
        )

        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=False,
        )

        self.bn = nn.BatchNorm2d(
            out_channels
        )

        self.act = nn.GELU()

    def forward(self, x):

        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.act(x)

        return x


# ============================================================
# COLOR BRANCH
# ============================================================

class ColorBranch(nn.Module):
    """
    HSV color representation -> 128-D feature.
    """

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(
                3,
                32,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),

            nn.BatchNorm2d(32),
            nn.GELU(),

            DepthwiseSeparableConv(
                32,
                64,
                stride=2
            ),

            DepthwiseSeparableConv(
                64,
                128,
                stride=2
            ),

            DepthwiseSeparableConv(
                128,
                256,
                stride=2
            ),

            DepthwiseSeparableConv(
                256,
                256,
                stride=2
            ),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.projection = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                256,
                128
            ),

            nn.GELU(),

            nn.Dropout(0.15)
        )

    def forward(self, x):

        x = self.features(x)

        x = self.pool(x)

        x = self.projection(x)

        return x


# ============================================================
# TEXTURE BRANCH
# ============================================================

class TextureBranch(nn.Module):
    """
    Sobel texture representation -> 128-D feature.
    """

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(
                1,
                32,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),

            nn.BatchNorm2d(32),
            nn.GELU(),

            DepthwiseSeparableConv(
                32,
                64,
                stride=2
            ),

            DepthwiseSeparableConv(
                64,
                128,
                stride=2
            ),

            DepthwiseSeparableConv(
                128,
                256,
                stride=2
            ),

            DepthwiseSeparableConv(
                256,
                256,
                stride=2
            ),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.projection = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                256,
                128
            ),

            nn.GELU(),

            nn.Dropout(0.15)
        )

    def forward(self, x):

        x = self.features(x)

        x = self.pool(x)

        x = self.projection(x)

        return x


# ============================================================
# CSTF-NET
# ============================================================

class CSTFNet(nn.Module):

    def __init__(
        self,
        num_classes,
        pretrained=True
    ):

        super().__init__()

        print("\nBuilding CSTF-Net...")

        # ----------------------------------------------------
        # RGB semantic branch
        # ----------------------------------------------------

        weights = (
            models.EfficientNet_B0_Weights.DEFAULT
            if pretrained
            else None
        )

        rgb_model = models.efficientnet_b0(
            weights=weights
        )

        rgb_features = (
            rgb_model.classifier[1].in_features
        )

        rgb_model.classifier = nn.Identity()

        self.rgb_branch = rgb_model

        self.rgb_projection = nn.Sequential(

            nn.Linear(
                rgb_features,
                128
            ),

            nn.GELU(),

            nn.Dropout(0.15)
        )

        # ----------------------------------------------------
        # HSV color branch
        # ----------------------------------------------------

        self.color_branch = ColorBranch()

        # ----------------------------------------------------
        # Texture branch
        # ----------------------------------------------------

        self.texture_branch = TextureBranch()

        # ----------------------------------------------------
        # Learnable branch reliability gate
        # ----------------------------------------------------

        self.branch_gate = nn.Sequential(

            nn.Linear(
                128 * 3,
                128
            ),

            nn.GELU(),

            nn.Dropout(0.15),

            nn.Linear(
                128,
                3
            )
        )

        # ----------------------------------------------------
        # Cross-feature Transformer
        # ----------------------------------------------------

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
            num_layers=2
        )

        # ----------------------------------------------------
        # Learnable fusion gate
        # ----------------------------------------------------

        self.fusion_gate = nn.Sequential(

            nn.Linear(
                128 * 2,
                128
            ),

            nn.GELU(),

            nn.Linear(
                128,
                128
            ),

            nn.Sigmoid()
        )

        # ----------------------------------------------------
        # Final classifier
        # ----------------------------------------------------

        self.classifier = nn.Sequential(

            nn.LayerNorm(128),

            nn.Linear(
                128,
                256
            ),

            nn.GELU(),

            nn.Dropout(0.30),

            nn.Linear(
                256,
                num_classes
            )
        )

    def forward(
        self,
        x,
        return_features=False
    ):

        # ====================================================
        # RGB branch
        # ====================================================

        rgb = self.rgb_branch(x)

        rgb = self.rgb_projection(rgb)

        # ====================================================
        # HSV branch
        # ====================================================

        hsv = rgb_to_hsv(x)

        color = self.color_branch(hsv)

        # ====================================================
        # Texture branch
        # ====================================================

        texture_map = sobel_texture(x)

        texture = self.texture_branch(
            texture_map
        )

        # ====================================================
        # Stack branch features
        # ====================================================

        tokens = torch.stack(
            [
                rgb,
                color,
                texture
            ],
            dim=1
        )

        # Shape:
        # [batch, 3, 128]

        # ====================================================
        # Branch reliability gating
        # ====================================================

        combined = torch.cat(
            [
                rgb,
                color,
                texture
            ],
            dim=1
        )

        gate_logits = self.branch_gate(
            combined
        )

        branch_weights = torch.softmax(
            gate_logits,
            dim=1
        )

        # [B,3] -> [B,3,1]

        weighted_tokens = (
            tokens
            * branch_weights.unsqueeze(-1)
        )

        # ====================================================
        # Cross-feature Transformer
        # ====================================================

        transformed = self.transformer(
            weighted_tokens
        )

        # ====================================================
        # Attention-weighted aggregation
        # ====================================================

        transformed_global = transformed.mean(
            dim=1
        )

        original_global = weighted_tokens.mean(
            dim=1
        )

        # ====================================================
        # Gated residual fusion
        # ====================================================

        fusion_input = torch.cat(
            [
                original_global,
                transformed_global
            ],
            dim=1
        )

        fusion_gate = self.fusion_gate(
            fusion_input
        )

        fused = (
            fusion_gate * transformed_global
            + (1.0 - fusion_gate) * original_global
        )

        # ====================================================
        # Classification
        # ====================================================

        logits = self.classifier(
            fused
        )

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
# DATASET
# ============================================================

train_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    transforms.RandomRotation(
        12
    ),

    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15,
        saturation=0.15,
        hue=0.03,
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],
        std=[
            0.229,
            0.224,
            0.225
        ],
    ),
])


eval_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],
        std=[
            0.229,
            0.224,
            0.225
        ],
    ),
])


print("\nLoading datasets...")

train_dataset = datasets.ImageFolder(
    os.path.join(DATA_DIR, "train"),
    transform=train_transform
)

valid_dataset = datasets.ImageFolder(
    os.path.join(DATA_DIR, "valid"),
    transform=eval_transform
)

test_dataset = datasets.ImageFolder(
    os.path.join(DATA_DIR, "test"),
    transform=eval_transform
)

class_names = train_dataset.classes

num_classes = len(class_names)

print(f"Classes       : {num_classes}")

for i, name in enumerate(class_names):
    print(f"  {i}: {name}")

print(f"\nTrain images  : {len(train_dataset)}")
print(f"Valid images  : {len(valid_dataset)}")
print(f"Test images   : {len(test_dataset)}")


# ============================================================
# CHECK CLASS MAPPING
# ============================================================

if valid_dataset.class_to_idx != train_dataset.class_to_idx:
    raise RuntimeError(
        "Validation class mapping does not match training."
    )

if test_dataset.class_to_idx != train_dataset.class_to_idx:
    raise RuntimeError(
        "Test class mapping does not match training."
    )


# ============================================================
# CLASS WEIGHTS
# ============================================================

train_targets = np.array(
    train_dataset.targets
)

class_counts = np.bincount(
    train_targets,
    minlength=num_classes
)

print("\nClass distribution:")

for i, count in enumerate(class_counts):
    print(
        f"{class_names[i]:45s}: {count}"
    )


# Square-root inverse-frequency weighting.
class_weights = 1.0 / np.sqrt(
    class_counts.astype(np.float32)
    + 1e-8
)

# Normalize mean to 1.
class_weights = (
    class_weights
    / class_weights.mean()
)

class_weights_tensor = torch.tensor(
    class_weights,
    dtype=torch.float32,
    device=device
)

print("\nClass weights:")

for i, weight in enumerate(class_weights):
    print(
        f"{class_names[i]:45s}: {weight:.4f}"
    )


# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
)


# ============================================================
# MODEL
# ============================================================

model = CSTFNet(
    num_classes=num_classes,
    pretrained=True
)

model = model.to(device)


# ============================================================
# PARAMETER COUNT
# ============================================================

total_params = sum(
    p.numel()
    for p in model.parameters()
)

trainable_params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print("\nModel parameters:")
print(
    f"Total parameters     : "
    f"{total_params / 1e6:.2f} M"
)

print(
    f"Trainable parameters : "
    f"{trainable_params / 1e6:.2f} M"
)


# ============================================================
# LOSS
# ============================================================

criterion = nn.CrossEntropyLoss(
    weight=class_weights_tensor,
    label_smoothing=LABEL_SMOOTHING
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


# ============================================================
# LR SCHEDULER
# ============================================================

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=2,
    min_lr=1e-7
)


# ============================================================
# AMP
# ============================================================

amp_enabled = (
    USE_AMP
    and torch.cuda.is_available()
)

scaler = torch.amp.GradScaler(
    "cuda",
    enabled=amp_enabled
)


# ============================================================
# TRAINING FUNCTION
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    scaler
):

    model.train()

    running_loss = 0.0

    all_targets = []
    all_predictions = []

    for images, targets in loader:

        images = images.to(
            device,
            non_blocking=True
        )

        targets = targets.to(
            device,
            non_blocking=True
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        with torch.amp.autocast(
            device_type=device.type,
            enabled=amp_enabled
        ):

            outputs = model(images)

            loss = criterion(
                outputs,
                targets
            )

        scaler.scale(loss).backward()

        scaler.step(optimizer)

        scaler.update()

        running_loss += (
            loss.item()
            * images.size(0)
        )

        predictions = outputs.argmax(
            dim=1
        )

        all_targets.extend(
            targets.detach().cpu().numpy()
        )

        all_predictions.extend(
            predictions.detach().cpu().numpy()
        )

    epoch_loss = (
        running_loss
        / len(loader.dataset)
    )

    accuracy = accuracy_score(
        all_targets,
        all_predictions
    )

    macro_f1 = f1_score(
        all_targets,
        all_predictions,
        average="macro",
        zero_division=0
    )

    macro_recall = recall_score(
        all_targets,
        all_predictions,
        average="macro",
        zero_division=0
    )

    return (
        epoch_loss,
        accuracy,
        macro_f1,
        macro_recall
    )


# ============================================================
# VALIDATION FUNCTION
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion
):

    model.eval()

    running_loss = 0.0

    all_targets = []
    all_predictions = []

    for images, targets in loader:

        images = images.to(
            device,
            non_blocking=True
        )

        targets = targets.to(
            device,
            non_blocking=True
        )

        with torch.amp.autocast(
            device_type=device.type,
            enabled=amp_enabled
        ):

            outputs = model(images)

            loss = criterion(
                outputs,
                targets
            )

        running_loss += (
            loss.item()
            * images.size(0)
        )

        predictions = outputs.argmax(
            dim=1
        )

        all_targets.extend(
            targets.cpu().numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

    epoch_loss = (
        running_loss
        / len(loader.dataset)
    )

    accuracy = accuracy_score(
        all_targets,
        all_predictions
    )

    macro_f1 = f1_score(
        all_targets,
        all_predictions,
        average="macro",
        zero_division=0
    )

    macro_recall = recall_score(
        all_targets,
        all_predictions,
        average="macro",
        zero_division=0
    )

    macro_precision = precision_score(
        all_targets,
        all_predictions,
        average="macro",
        zero_division=0
    )

    return (
        epoch_loss,
        accuracy,
        macro_precision,
        macro_recall,
        macro_f1
    )


# ============================================================
# TRAINING LOOP
# ============================================================

history = {
    "train_loss": [],
    "train_accuracy": [],
    "train_macro_f1": [],
    "train_macro_recall": [],

    "val_loss": [],
    "val_accuracy": [],
    "val_macro_precision": [],
    "val_macro_recall": [],
    "val_macro_f1": [],

    "learning_rate": [],
}


best_val_f1 = -1.0

best_epoch = 0

best_state = None

epochs_without_improvement = 0


print("\n" + "=" * 70)
print("STARTING TRAINING")
print("=" * 70)


for epoch in range(1, NUM_EPOCHS + 1):

    current_lr = optimizer.param_groups[0]["lr"]

    print(
        f"\nEpoch {epoch}/{NUM_EPOCHS}"
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    train_loss, train_acc, train_f1, train_recall = (
        train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            scaler
        )
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    (
        val_loss,
        val_acc,
        val_precision,
        val_recall,
        val_f1
    ) = evaluate(
        model,
        valid_loader,
        criterion
    )

    # --------------------------------------------------------
    # Scheduler
    # --------------------------------------------------------

    scheduler.step(
        val_f1
    )

    # --------------------------------------------------------
    # History
    # --------------------------------------------------------

    history["train_loss"].append(
        train_loss
    )

    history["train_accuracy"].append(
        train_acc
    )

    history["train_macro_f1"].append(
        train_f1
    )

    history["train_macro_recall"].append(
        train_recall
    )

    history["val_loss"].append(
        val_loss
    )

    history["val_accuracy"].append(
        val_acc
    )

    history["val_macro_precision"].append(
        val_precision
    )

    history["val_macro_recall"].append(
        val_recall
    )

    history["val_macro_f1"].append(
        val_f1
    )

    history["learning_rate"].append(
        current_lr
    )

    # --------------------------------------------------------
    # Print metrics
    # --------------------------------------------------------

    print(
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_acc:.4f} | "
        f"Train F1: {train_f1:.4f}"
    )

    print(
        f"Val Loss:   {val_loss:.4f} | "
        f"Val Acc:   {val_acc:.4f} | "
        f"Val F1:    {val_f1:.4f} | "
        f"Val Recall:{val_recall:.4f}"
    )

    print(
        f"LR: {current_lr:.2e}"
    )

    # --------------------------------------------------------
    # Save best model
    # --------------------------------------------------------

    if val_f1 > best_val_f1:

        best_val_f1 = val_f1

        best_epoch = epoch

        best_state = copy.deepcopy(
            model.state_dict()
        )

        checkpoint = {
            "model_state_dict": best_state,
            "class_names": class_names,
            "class_to_idx": train_dataset.class_to_idx,
            "num_classes": num_classes,
            "image_size": IMAGE_SIZE,
            "architecture": "CSTF-Net",
            "best_val_macro_f1": best_val_f1,
            "best_epoch": best_epoch,
        }

        torch.save(
            checkpoint,
            os.path.join(
                OUTPUT_DIR,
                "best_model.pth"
            )
        )

        print(
            f"✓ Best model saved "
            f"(Val Macro-F1: {best_val_f1:.4f})"
        )

        epochs_without_improvement = 0

    else:

        epochs_without_improvement += 1

        print(
            f"No improvement "
            f"({epochs_without_improvement}/{PATIENCE})"
        )

    # --------------------------------------------------------
    # Early stopping
    # --------------------------------------------------------

    if epochs_without_improvement >= PATIENCE:

        print(
            f"\nEarly stopping triggered "
            f"after {PATIENCE} epochs without improvement."
        )

        break


# ============================================================
# RESTORE BEST MODEL
# ============================================================

if best_state is not None:

    model.load_state_dict(
        best_state
    )


print("\n" + "=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print(
    f"Best epoch          : {best_epoch}"
)

print(
    f"Best validation F1  : {best_val_f1:.4f}"
)


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

with open(
    os.path.join(
        OUTPUT_DIR,
        "training_history.json"
    ),
    "w"
) as f:

    json.dump(
        history,
        f,
        indent=2
    )


# ============================================================
# PLOT TRAINING CURVES
# ============================================================

epochs = range(
    1,
    len(history["train_loss"]) + 1
)


plt.figure(figsize=(8, 5))

plt.plot(
    epochs,
    history["train_loss"],
    label="Train Loss"
)

plt.plot(
    epochs,
    history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("CSTF-Net Training and Validation Loss")
plt.legend()
plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "loss_curve.png"
    ),
    dpi=200
)

plt.close()


plt.figure(figsize=(8, 5))

plt.plot(
    epochs,
    history["train_macro_f1"],
    label="Train Macro-F1"
)

plt.plot(
    epochs,
    history["val_macro_f1"],
    label="Validation Macro-F1"
)

plt.xlabel("Epoch")
plt.ylabel("Macro-F1")
plt.title("CSTF-Net Macro-F1")
plt.legend()
plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "macro_f1_curve.png"
    ),
    dpi=200
)

plt.close()


# ============================================================
# ACCURACY CURVE
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    epochs,
    history["train_accuracy"],
    label="Train Accuracy"
)

plt.plot(
    epochs,
    history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("CSTF-Net Training and Validation Accuracy")
plt.legend()
plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "accuracy_curve.png"
    ),
    dpi=200
)

plt.close()


# ============================================================
# TEST EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("TEST EVALUATION")
print("=" * 70)


model.eval()

all_targets = []
all_predictions = []
all_probabilities = []


with torch.no_grad():

    for images, targets in test_loader:

        images = images.to(
            device,
            non_blocking=True
        )

        with torch.amp.autocast(
            device_type=device.type,
            enabled=amp_enabled
        ):

            outputs = model(
                images
            )

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        predictions = outputs.argmax(
            dim=1
        )

        all_targets.extend(
            targets.numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_probabilities.extend(
            probabilities.cpu().numpy()
        )


# ============================================================
# TEST METRICS
# ============================================================

test_accuracy = accuracy_score(
    all_targets,
    all_predictions
)

test_precision = precision_score(
    all_targets,
    all_predictions,
    average="macro",
    zero_division=0
)

test_recall = recall_score(
    all_targets,
    all_predictions,
    average="macro",
    zero_division=0
)

test_f1 = f1_score(
    all_targets,
    all_predictions,
    average="macro",
    zero_division=0
)


print(
    f"\nTest Accuracy       : {test_accuracy:.4f}"
)

print(
    f"Test Macro Precision: {test_precision:.4f}"
)

print(
    f"Test Macro Recall   : {test_recall:.4f}"
)

print(
    f"Test Macro F1       : {test_f1:.4f}"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    all_targets,
    all_predictions,
    target_names=class_names,
    digits=4,
    zero_division=0
)

print("\nClassification Report:")
print(report)

with open(
    os.path.join(
        OUTPUT_DIR,
        "classification_report.txt"
    ),
    "w"
) as f:

    f.write(report)


# ============================================================
# PER-CLASS METRICS JSON
# ============================================================

report_dict = classification_report(
    all_targets,
    all_predictions,
    target_names=class_names,
    output_dict=True,
    zero_division=0
)

with open(
    os.path.join(
        OUTPUT_DIR,
        "per_class_metrics.json"
    ),
    "w"
) as f:

    json.dump(
        report_dict,
        f,
        indent=2
    )


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    all_targets,
    all_predictions
)

plt.figure(
    figsize=(12, 10)
)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.xlabel(
    "Predicted Label"
)

plt.ylabel(
    "True Label"
)

plt.title(
    "CSTF-Net Test Confusion Matrix"
)

plt.xticks(
    rotation=45,
    ha="right"
)

plt.yticks(
    rotation=0
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "confusion_matrix.png"
    ),
    dpi=200
)

plt.close()


# ============================================================
# TEST METRICS JSON
# ============================================================

test_metrics = {
    "architecture": "CSTF-Net",
    "best_epoch": int(best_epoch),
    "best_validation_macro_f1": float(
        best_val_f1
    ),
    "test_accuracy": float(
        test_accuracy
    ),
    "test_macro_precision": float(
        test_precision
    ),
    "test_macro_recall": float(
        test_recall
    ),
    "test_macro_f1": float(
        test_f1
    ),
    "num_test_images": len(test_dataset),
    "num_classes": num_classes,
    "class_names": class_names,
    "image_size": IMAGE_SIZE,
    "batch_size": BATCH_SIZE,
    "learning_rate": LEARNING_RATE,
    "weight_decay": WEIGHT_DECAY,
    "label_smoothing": LABEL_SMOOTHING,
}

with open(
    os.path.join(
        OUTPUT_DIR,
        "test_metrics.json"
    ),
    "w"
) as f:

    json.dump(
        test_metrics,
        f,
        indent=2
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("CSTF-NET FINAL RESULTS")
print("=" * 70)

print(
    f"Best validation Macro-F1 : "
    f"{best_val_f1:.4f}"
)

print(
    f"Test Accuracy            : "
    f"{test_accuracy:.4f}"
)

print(
    f"Test Macro Precision     : "
    f"{test_precision:.4f}"
)

print(
    f"Test Macro Recall        : "
    f"{test_recall:.4f}"
)

print(
    f"Test Macro F1            : "
    f"{test_f1:.4f}"
)

print(
    f"\nOutputs saved to:"
    f"\n{OUTPUT_DIR}"
)

print("\nFiles generated:")

print("  best_model.pth")
print("  training_history.json")
print("  loss_curve.png")
print("  macro_f1_curve.png")
print("  classification_report.txt")
print("  per_class_metrics.json")
print("  confusion_matrix.png")
print("  test_metrics.json")

print("\nDone.")