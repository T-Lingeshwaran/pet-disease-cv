import os
import json
import random
import gc

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import EfficientNet_B0_Weights

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    precision_score,
    classification_report,
    confusion_matrix,
)

# ============================================================
# MCFA-NET V2
# Multi-Color Feature Attention Network
#
# Main changes from v1:
# 1. Separate normalization for RGB / HSV / YCbCr.
# 2. Optional partial backbone freezing for safer transfer learning.
# 3. Learnable post-Transformer attention pooling instead of mean fusion.
# 4. Explicit RGB residual path.
# 5. Reduced color jitter so color-space information is not destroyed.
# 6. Saves branch weights for analysis.
#
# This remains a proposed architecture; superiority is NOT assumed.
# ============================================================

DATA_DIR = "data/eye_face/combined"
OUTPUT_DIR = "outputs/eye_mcfa_v2"

IMAGE_SIZE = 224
BATCH_SIZE = 6                 # RTX 3050 6 GB safe starting point
NUM_EPOCHS = 50
PATIENCE = 7

LR = 1e-4
BACKBONE_LR_MULT = 0.5
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.05

TOKEN_DIM = 128
NUM_HEADS = 4
NUM_TRANSFORMER_LAYERS = 2
DROPOUT = 0.25

# Freeze early EfficientNet blocks initially.
# 0 = train everything.
# 4 = freeze features[0:4], train later blocks + heads.
FREEZE_EARLY_BLOCKS = 4

NUM_WORKERS = 0
SEED = 42
USE_AMP = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# REPRODUCIBILITY
# ============================================================

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


seed_everything(SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# DATASET
# ============================================================

for split in ["train", "valid", "test"]:
    split_path = os.path.join(DATA_DIR, split)
    if not os.path.isdir(split_path):
        raise FileNotFoundError(f"Missing dataset split: {split_path}")


# Color jitter is intentionally mild: MCFA relies on color-space cues.
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(
        brightness=0.10,
        contrast=0.10,
        saturation=0.05,
        hue=0.01,
    ),
    transforms.ToTensor(),
])

eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

train_dataset = datasets.ImageFolder(
    os.path.join(DATA_DIR, "train"),
    transform=train_transform,
)
valid_dataset = datasets.ImageFolder(
    os.path.join(DATA_DIR, "valid"),
    transform=eval_transform,
)
test_dataset = datasets.ImageFolder(
    os.path.join(DATA_DIR, "test"),
    transform=eval_transform,
)

class_names = train_dataset.classes
num_classes = len(class_names)

if valid_dataset.class_to_idx != train_dataset.class_to_idx:
    raise RuntimeError("Validation class mapping differs from training.")
if test_dataset.class_to_idx != train_dataset.class_to_idx:
    raise RuntimeError("Test class mapping differs from training.")

print("=" * 75)
print("MCFA-NET V2")
print("Multi-Color Feature Attention Network v2")
print("=" * 75)
print(f"Device       : {DEVICE}")
print(f"Dataset      : {DATA_DIR}")
print(f"Image size   : {IMAGE_SIZE}")
print(f"Batch size   : {BATCH_SIZE}")
print(f"Epochs       : {NUM_EPOCHS}")
print(f"Frozen blocks: {FREEZE_EARLY_BLOCKS}")
print(f"Classes      : {num_classes}")
print(f"Train        : {len(train_dataset)}")
print(f"Valid        : {len(valid_dataset)}")
print(f"Test         : {len(test_dataset)}")
print("=" * 75)

print("\nClasses:")
for i, name in enumerate(class_names):
    print(f"{i:2d}: {name}")


# ============================================================
# CLASS WEIGHTS
# ============================================================

train_targets = np.asarray(train_dataset.targets)
class_counts = np.bincount(train_targets, minlength=num_classes).astype(np.float32)

class_weights_np = 1.0 / np.sqrt(np.maximum(class_counts, 1.0))
class_weights_np /= class_weights_np.mean()
class_weights = torch.tensor(class_weights_np, dtype=torch.float32, device=DEVICE)

print("\nTraining class distribution:")
for i, count in enumerate(class_counts.astype(int)):
    print(f"{class_names[i]:25s}: {count}")

print("\nClass weights:")
for i, weight in enumerate(class_weights):
    print(f"{class_names[i]:25s}: {weight.item():.4f}")


# ============================================================
# DATALOADERS
# ============================================================

pin_memory = DEVICE.type == "cuda"

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=pin_memory,
)
valid_loader = DataLoader(
    valid_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=pin_memory,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=pin_memory,
)


# ============================================================
# COLOR CONVERSIONS
# ============================================================

def rgb_to_ycbcr(x):
    r = x[:, 0:1]
    g = x[:, 1:2]
    b = x[:, 2:3]

    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 0.5
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 0.5

    return torch.cat([y, cb, cr], dim=1).clamp(0.0, 1.0)


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

    value = max_val

    return torch.stack([hue, saturation, value], dim=1)


# ============================================================
# SEPARATE COLOR-SPACE NORMALIZATION
#
# RGB uses ImageNet statistics because the EfficientNet weights
# were pretrained on RGB images with this preprocessing.
#
# HSV / YCbCr use fixed statistics appropriate to their numeric
# ranges. These are intentionally NOT ImageNet RGB statistics.
# ============================================================

RGB_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
RGB_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

# HSV channels are naturally in [0, 1].
HSV_MEAN = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
HSV_STD = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)

# Y, Cb and Cr are represented in approximately [0, 1].
YCBCR_MEAN = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
YCBCR_STD = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)


def normalize_with_stats(x, mean, std):
    mean = mean.to(device=x.device, dtype=x.dtype)
    std = std.to(device=x.device, dtype=x.dtype)
    return (x - mean) / std


# ============================================================
# BACKBONE HELPERS
# ============================================================

def freeze_early_features(backbone, n_blocks):
    """
    Freeze only the first n entries of EfficientNet.features.
    Later feature blocks and the projection/fusion heads remain
    trainable.
    """
    n_blocks = max(0, min(n_blocks, len(backbone.features)))

    for idx, block in enumerate(backbone.features):
        requires_grad = idx >= n_blocks
        for param in block.parameters():
            param.requires_grad = requires_grad


def make_efficientnet():
    weights = EfficientNet_B0_Weights.DEFAULT
    backbone = models.efficientnet_b0(weights=weights)

    feature_dim = backbone.classifier[-1].in_features
    backbone.classifier = nn.Identity()

    freeze_early_features(backbone, FREEZE_EARLY_BLOCKS)

    return backbone, feature_dim


# ============================================================
# MCFA-NET V2
# ============================================================

class MCFA_Net_V2(nn.Module):
    """
    Multi-Color Feature Attention Network v2.

    RGB / HSV / YCbCr
        -> independent EfficientNet-B0 feature extraction
        -> 128-D color tokens
        -> adaptive branch gating
        -> cross-color Transformer
        -> learned attention pooling
        -> RGB residual preservation
        -> gated fusion
        -> classifier

    Unlike v1, the three color tokens are not simply averaged.
    """

    def __init__(
        self,
        num_classes,
        token_dim=128,
        num_heads=4,
        num_layers=2,
        dropout=0.25,
    ):
        super().__init__()

        self.rgb_backbone, feature_dim = make_efficientnet()
        self.hsv_backbone, _ = make_efficientnet()
        self.ycbcr_backbone, _ = make_efficientnet()

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

        # Produces one adaptive weight per color branch.
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

        # NEW: learned attention pooling instead of mean(attended).
        self.fusion_score = nn.Sequential(
            nn.Linear(token_dim, token_dim // 2),
            nn.GELU(),
            nn.Linear(token_dim // 2, 1),
        )

        # NEW: preserve the strong RGB representation.
        self.rgb_residual = nn.Sequential(
            nn.Linear(token_dim, token_dim),
            nn.LayerNorm(token_dim),
        )

        # NEW: learned gate decides how much transformed color
        # information versus RGB residual information to retain.
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

        # --------------------------------------------------------
        # COLOR SPACES
        # --------------------------------------------------------

        rgb = x
        hsv = rgb_to_hsv(x)
        ycbcr = rgb_to_ycbcr(x)

        # --------------------------------------------------------
        # COLOR-SPECIFIC NORMALIZATION
        # --------------------------------------------------------

        rgb = normalize_with_stats(rgb, RGB_MEAN, RGB_STD)
        hsv = normalize_with_stats(hsv, HSV_MEAN, HSV_STD)
        ycbcr = normalize_with_stats(ycbcr, YCBCR_MEAN, YCBCR_STD)

        # --------------------------------------------------------
        # BACKBONES
        # --------------------------------------------------------

        rgb_features = self.rgb_backbone(rgb)
        hsv_features = self.hsv_backbone(hsv)
        ycbcr_features = self.ycbcr_backbone(ycbcr)

        # --------------------------------------------------------
        # COLOR TOKENS
        # --------------------------------------------------------

        rgb_token = self.rgb_projection(rgb_features)
        hsv_token = self.hsv_projection(hsv_features)
        ycbcr_token = self.ycbcr_projection(ycbcr_features)

        tokens = torch.stack(
            [rgb_token, hsv_token, ycbcr_token],
            dim=1,
        )

        # --------------------------------------------------------
        # ADAPTIVE BRANCH GATING
        # --------------------------------------------------------

        combined = torch.cat(
            [rgb_token, hsv_token, ycbcr_token],
            dim=1,
        )

        branch_logits = self.branch_gate(combined)
        branch_weights = torch.softmax(branch_logits, dim=1)

        gated_tokens = (
            tokens * branch_weights.unsqueeze(-1)
        )

        # --------------------------------------------------------
        # CROSS-COLOR TRANSFORMER
        # --------------------------------------------------------

        attended = self.cross_color_transformer(gated_tokens)

        # --------------------------------------------------------
        # LEARNED ATTENTION POOLING
        # --------------------------------------------------------

        fusion_scores = self.fusion_score(attended)
        fusion_weights = torch.softmax(fusion_scores, dim=1)

        fused_color = (
            attended * fusion_weights
        ).sum(dim=1)

        # --------------------------------------------------------
        # RGB RESIDUAL
        # --------------------------------------------------------

        rgb_residual = self.rgb_residual(rgb_token)

        # --------------------------------------------------------
        # GATED RESIDUAL FUSION
        # --------------------------------------------------------

        fusion_input = torch.cat(
            [fused_color, rgb_residual],
            dim=1,
        )

        residual_gate = self.residual_gate(fusion_input)

        fused = (
            residual_gate * fused_color
            + (1.0 - residual_gate) * rgb_residual
        )

        fused = self.fusion_norm(fused)

        # --------------------------------------------------------
        # CLASSIFIER
        # --------------------------------------------------------

        logits = self.classifier(fused)

        if return_branch_weights:
            return logits, {
                "branch_weights": branch_weights,
                "fusion_weights": fusion_weights.squeeze(-1),
                "residual_gate_mean": residual_gate.mean(dim=1),
            }

        return logits


# ============================================================
# MODEL
# ============================================================

print("\nBuilding MCFA-Net v2...")

model = MCFA_Net_V2(
    num_classes=num_classes,
    token_dim=TOKEN_DIM,
    num_heads=NUM_HEADS,
    num_layers=NUM_TRANSFORMER_LAYERS,
    dropout=DROPOUT,
).to(DEVICE)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(
    p.numel() for p in model.parameters()
    if p.requires_grad
)

print(f"Total parameters    : {total_params / 1e6:.2f} M")
print(f"Trainable parameters: {trainable_params / 1e6:.2f} M")


# ============================================================
# LOSS
# ============================================================

criterion = nn.CrossEntropyLoss(
    weight=class_weights,
    label_smoothing=LABEL_SMOOTHING,
)


# ============================================================
# OPTIMIZER
#
# Backbone receives a smaller LR than newly initialized
# projection/attention/classifier layers.
# ============================================================

backbone_params = []
head_params = []

backbone_modules = [
    model.rgb_backbone,
    model.hsv_backbone,
    model.ycbcr_backbone,
]

backbone_param_ids = set()

for module in backbone_modules:
    for param in module.parameters():
        if param.requires_grad:
            backbone_params.append(param)
            backbone_param_ids.add(id(param))

for param in model.parameters():
    if param.requires_grad and id(param) not in backbone_param_ids:
        head_params.append(param)

optimizer = torch.optim.AdamW(
    [
        {
            "params": backbone_params,
            "lr": LR * BACKBONE_LR_MULT,
        },
        {
            "params": head_params,
            "lr": LR,
        },
    ],
    weight_decay=WEIGHT_DECAY,
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=2,
)


# ============================================================
# AMP
# ============================================================

if USE_AMP and DEVICE.type == "cuda":
    scaler = torch.amp.GradScaler("cuda")
else:
    scaler = None


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "macro_recall": recall_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "macro_f1": f1_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
    }


# ============================================================
# TRAIN
# ============================================================

def train_one_epoch():
    model.train()

    running_loss = 0.0
    all_targets = []
    all_predictions = []

    for images, targets in train_loader:
        images = images.to(DEVICE, non_blocking=True)
        targets = targets.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.amp.autocast(
                device_type="cuda",
                dtype=torch.float16,
            ):
                outputs = model(images)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        else:
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)

        predictions = torch.argmax(outputs, dim=1)

        all_targets.extend(
            targets.detach().cpu().numpy()
        )
        all_predictions.extend(
            predictions.detach().cpu().numpy()
        )

    metrics = calculate_metrics(
        all_targets,
        all_predictions,
    )

    metrics["loss"] = running_loss / len(train_dataset)

    return metrics


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def validate():
    model.eval()

    running_loss = 0.0
    all_targets = []
    all_predictions = []

    for images, targets in valid_loader:
        images = images.to(DEVICE, non_blocking=True)
        targets = targets.to(DEVICE, non_blocking=True)

        if USE_AMP and DEVICE.type == "cuda":
            with torch.amp.autocast(
                device_type="cuda",
                dtype=torch.float16,
            ):
                outputs = model(images)
                loss = criterion(outputs, targets)
        else:
            outputs = model(images)
            loss = criterion(outputs, targets)

        running_loss += loss.item() * images.size(0)

        predictions = torch.argmax(outputs, dim=1)

        all_targets.extend(targets.cpu().numpy())
        all_predictions.extend(
            predictions.cpu().numpy()
        )

    metrics = calculate_metrics(
        all_targets,
        all_predictions,
    )

    metrics["loss"] = running_loss / len(valid_dataset)

    return metrics


# ============================================================
# TRAINING LOOP
# ============================================================

best_val_f1 = -1.0
best_epoch = 0
epochs_without_improvement = 0
history = []

best_model_path = os.path.join(
    OUTPUT_DIR,
    "best_model.pth",
)

print("\n" + "=" * 75)
print("STARTING MCFA-NET V2 TRAINING")
print("=" * 75)

for epoch in range(1, NUM_EPOCHS + 1):

    print(f"\nEpoch {epoch}/{NUM_EPOCHS}")

    train_metrics = train_one_epoch()
    valid_metrics = validate()

    scheduler.step(valid_metrics["macro_f1"])

    current_lrs = [
        group["lr"] for group in optimizer.param_groups
    ]

    history.append({
        "epoch": epoch,
        "backbone_lr": current_lrs[0],
        "head_lr": current_lrs[1],

        "train_loss": train_metrics["loss"],
        "train_accuracy": train_metrics["accuracy"],
        "train_macro_precision": train_metrics["macro_precision"],
        "train_macro_recall": train_metrics["macro_recall"],
        "train_macro_f1": train_metrics["macro_f1"],

        "val_loss": valid_metrics["loss"],
        "val_accuracy": valid_metrics["accuracy"],
        "val_macro_precision": valid_metrics["macro_precision"],
        "val_macro_recall": valid_metrics["macro_recall"],
        "val_macro_f1": valid_metrics["macro_f1"],
    })

    print(
        f"Train | Loss {train_metrics['loss']:.4f} | "
        f"Acc {train_metrics['accuracy']:.4f} | "
        f"F1 {train_metrics['macro_f1']:.4f}"
    )

    print(
        f"Valid | Loss {valid_metrics['loss']:.4f} | "
        f"Acc {valid_metrics['accuracy']:.4f} | "
        f"F1 {valid_metrics['macro_f1']:.4f}"
    )

    print(
        f"LR | Backbone {current_lrs[0]:.7f} | "
        f"Head {current_lrs[1]:.7f}"
    )

    if valid_metrics["macro_f1"] > best_val_f1:
        best_val_f1 = valid_metrics["macro_f1"]
        best_epoch = epoch
        epochs_without_improvement = 0

        torch.save(
            {
                "model_name": "MCFA-Net-v2",
                "architecture": "MCFA-Net-v2",
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),

                "best_val_macro_f1": best_val_f1,

                "class_names": class_names,
                "class_to_idx": train_dataset.class_to_idx,

                "image_size": IMAGE_SIZE,
                "token_dim": TOKEN_DIM,
                "num_heads": NUM_HEADS,
                "num_transformer_layers":
                    NUM_TRANSFORMER_LAYERS,
                "freeze_early_blocks":
                    FREEZE_EARLY_BLOCKS,
            },
            best_model_path,
        )

        print(
            f"*** BEST MODEL SAVED "
            f"(Val Macro-F1 = {best_val_f1:.4f}) ***"
        )

    else:
        epochs_without_improvement += 1

        print(
            f"No improvement "
            f"({epochs_without_improvement}/{PATIENCE})"
        )

    if epochs_without_improvement >= PATIENCE:
        print("\nEarly stopping triggered.")
        break


# ============================================================
# HISTORY
# ============================================================

history_df = pd.DataFrame(history)

history_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "training_history.csv",
    ),
    index=False,
)

with open(
    os.path.join(
        OUTPUT_DIR,
        "training_history.json",
    ),
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        history,
        f,
        indent=2,
    )


# ============================================================
# CURVES
# ============================================================

plt.figure(figsize=(10, 6))
plt.plot(
    history_df["epoch"],
    history_df["train_accuracy"],
    label="Train Accuracy",
)
plt.plot(
    history_df["epoch"],
    history_df["val_accuracy"],
    label="Validation Accuracy",
)
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("MCFA-Net v2 Eye Classification - Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "accuracy_curve.png",
    ),
    dpi=200,
)
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(
    history_df["epoch"],
    history_df["train_macro_f1"],
    label="Train Macro-F1",
)
plt.plot(
    history_df["epoch"],
    history_df["val_macro_f1"],
    label="Validation Macro-F1",
)
plt.xlabel("Epoch")
plt.ylabel("Macro-F1")
plt.title("MCFA-Net v2 Eye Classification - Macro-F1")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "macro_f1_curve.png",
    ),
    dpi=200,
)
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(
    history_df["epoch"],
    history_df["train_loss"],
    label="Train Loss",
)
plt.plot(
    history_df["epoch"],
    history_df["val_loss"],
    label="Validation Loss",
)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("MCFA-Net v2 Eye Classification - Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "loss_curve.png",
    ),
    dpi=200,
)
plt.close()


# ============================================================
# LOAD BEST MODEL
# ============================================================

print("\n" + "=" * 75)
print("LOADING BEST MCFA-NET V2")
print("=" * 75)

checkpoint = torch.load(
    best_model_path,
    map_location=DEVICE,
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)
model.eval()

print(
    f"Best epoch: {checkpoint['epoch']}"
)
print(
    f"Best validation Macro-F1: "
    f"{checkpoint['best_val_macro_f1']:.4f}"
)


# ============================================================
# TEST
# ============================================================

all_targets = []
all_predictions = []
all_probabilities = []

branch_weight_rows = []

print("\nRunning test evaluation...")

with torch.no_grad():
    for images, targets in test_loader:

        images = images.to(
            DEVICE,
            non_blocking=True,
        )

        if USE_AMP and DEVICE.type == "cuda":
            with torch.amp.autocast(
                device_type="cuda",
                dtype=torch.float16,
            ):
                outputs, extras = model(
                    images,
                    return_branch_weights=True,
                )
        else:
            outputs, extras = model(
                images,
                return_branch_weights=True,
            )

        probabilities = torch.softmax(
            outputs,
            dim=1,
        )

        predictions = torch.argmax(
            probabilities,
            dim=1,
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

        bw = extras["branch_weights"].cpu().numpy()
        fw = extras["fusion_weights"].cpu().numpy()

        for i in range(len(images)):
            branch_weight_rows.append({
                "rgb_weight": float(bw[i, 0]),
                "hsv_weight": float(bw[i, 1]),
                "ycbcr_weight": float(bw[i, 2]),
                "rgb_fusion_weight": float(fw[i, 0]),
                "hsv_fusion_weight": float(fw[i, 1]),
                "ycbcr_fusion_weight": float(fw[i, 2]),
            })


y_true = np.asarray(all_targets)
y_pred = np.asarray(all_predictions)
y_prob = np.asarray(all_probabilities)


# ============================================================
# TEST METRICS
# ============================================================

test_metrics = calculate_metrics(
    y_true,
    y_pred,
)

print("\n" + "=" * 75)
print("MCFA-NET V2 TEST RESULTS")
print("=" * 75)

print(
    f"Accuracy        : "
    f"{test_metrics['accuracy']:.4f}"
)
print(
    f"Macro Precision : "
    f"{test_metrics['macro_precision']:.4f}"
)
print(
    f"Macro Recall    : "
    f"{test_metrics['macro_recall']:.4f}"
)
print(
    f"Macro F1        : "
    f"{test_metrics['macro_f1']:.4f}"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report_dict = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    output_dict=True,
    zero_division=0,
)

report_text = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    zero_division=0,
)

print("\nClassification Report:")
print(report_text)

with open(
    os.path.join(
        OUTPUT_DIR,
        "classification_report.txt",
    ),
    "w",
    encoding="utf-8",
) as f:
    f.write(report_text)

with open(
    os.path.join(
        OUTPUT_DIR,
        "classification_report.json",
    ),
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        report_dict,
        f,
        indent=2,
    )


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred,
)

plt.figure(figsize=(12, 10))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=class_names,
    yticklabels=class_names,
    cmap="Blues",
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title(
    "MCFA-Net v2 Eye Classification - Confusion Matrix"
)
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "confusion_matrix.png",
    ),
    dpi=200,
)
plt.close()


# ============================================================
# PER-CLASS METRICS
# ============================================================

per_class_rows = []

for class_name in class_names:
    row = report_dict[class_name]

    per_class_rows.append({
        "class": class_name,
        "precision": row["precision"],
        "recall": row["recall"],
        "f1_score": row["f1-score"],
        "support": row["support"],
    })

per_class_df = pd.DataFrame(
    per_class_rows
)

per_class_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "per_class_metrics.csv",
    ),
    index=False,
)


# ============================================================
# COLOR ATTENTION ANALYSIS
# ============================================================

branch_df = pd.DataFrame(branch_weight_rows)

branch_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "color_attention_weights.csv",
    ),
    index=False,
)

mean_branch_weights = {
    "rgb": float(branch_df["rgb_weight"].mean()),
    "hsv": float(branch_df["hsv_weight"].mean()),
    "ycbcr": float(branch_df["ycbcr_weight"].mean()),
}

mean_fusion_weights = {
    "rgb": float(
        branch_df["rgb_fusion_weight"].mean()
    ),
    "hsv": float(
        branch_df["hsv_fusion_weight"].mean()
    ),
    "ycbcr": float(
        branch_df["ycbcr_fusion_weight"].mean()
    ),
}

print("\nMean learned branch weights:")
for name, value in mean_branch_weights.items():
    print(f"{name:6s}: {value:.4f}")

print("\nMean Transformer fusion weights:")
for name, value in mean_fusion_weights.items():
    print(f"{name:6s}: {value:.4f}")


# ============================================================
# FINAL METRICS
# ============================================================

final_metrics = {
    "model": "MCFA-Net-v2",
    "description": (
        "Multi-Color Feature Attention Network v2"
    ),
    "dataset": "eye_face/combined",

    "best_epoch": int(best_epoch),
    "best_validation_macro_f1": float(
        best_val_f1
    ),

    "test_accuracy": float(
        test_metrics["accuracy"]
    ),
    "test_macro_precision": float(
        test_metrics["macro_precision"]
    ),
    "test_macro_recall": float(
        test_metrics["macro_recall"]
    ),
    "test_macro_f1": float(
        test_metrics["macro_f1"]
    ),

    "num_classes": int(num_classes),
    "image_size": IMAGE_SIZE,
    "batch_size": BATCH_SIZE,
    "epochs_requested": NUM_EPOCHS,

    "token_dim": TOKEN_DIM,
    "num_heads": NUM_HEADS,
    "num_transformer_layers":
        NUM_TRANSFORMER_LAYERS,

    "freeze_early_blocks":
        FREEZE_EARLY_BLOCKS,

    "total_parameters": int(total_params),
    "trainable_parameters":
        int(trainable_params),

    "mean_branch_weights":
        mean_branch_weights,

    "mean_fusion_weights":
        mean_fusion_weights,
}

with open(
    os.path.join(
        OUTPUT_DIR,
        "test_metrics.json",
    ),
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        final_metrics,
        f,
        indent=2,
    )


# ============================================================
# CLASS MAPPING
# ============================================================

with open(
    os.path.join(
        OUTPUT_DIR,
        "class_names.json",
    ),
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        {
            "class_names": class_names,
            "class_to_idx":
                train_dataset.class_to_idx,
        },
        f,
        indent=2,
    )


# ============================================================
# CLEANUP
# ============================================================

del model
gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()


print("\n" + "=" * 75)
print("MCFA-NET V2 TRAINING COMPLETE")
print("=" * 75)
print(f"Best epoch            : {best_epoch}")
print(
    f"Best validation F1    : "
    f"{best_val_f1:.4f}"
)
print(
    f"Test Accuracy         : "
    f"{test_metrics['accuracy']:.4f}"
)
print(
    f"Test Macro-F1         : "
    f"{test_metrics['macro_f1']:.4f}"
)
print(f"\nModel saved to:\n{best_model_path}")
print(f"\nResults saved to:\n{OUTPUT_DIR}")
