"""
CSF-Net-inspired Combined Dog + Cat Eye Disease Classifier

Architecture:
    RGB  -> MobileNetV3-Small ─┐
    HSV  -> MobileNetV3-Small ─┼─> Self-Attention -> Classifier
    YCbCr-> MobileNetV3-Small ─┘

Adapted for:
    9-class combined dog + cat eye dataset

Dataset:
    data/eye_face/combined/
        train/
        valid/
        test/

Outputs:
    outputs/eye_csfnet/

IMPORTANT:
This is an implementation inspired by the 2026 CSF-Net paper,
adapted to your 9-class combined dog+cat dataset.
It is not an exact reproduction of the paper's six-class feline experiment.
"""

from pathlib import Path
import json
import random
import time
import copy

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import (
    mobilenet_v3_small,
    MobileNet_V3_Small_Weights,
)

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

import matplotlib.pyplot as plt
from tqdm import tqdm


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = Path("data/eye_face/combined")
OUTPUT_DIR = Path("outputs/eye_csfnet")

IMAGE_SIZE = 224

BATCH_SIZE = 8
NUM_WORKERS = 2

NUM_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 7

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

LABEL_SMOOTHING = 0.05

SEED = 42

USE_AMP = True

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

IMAGENET_MEAN = torch.tensor(
    [0.485, 0.456, 0.406]
).view(3, 1, 1)

IMAGENET_STD = torch.tensor(
    [0.229, 0.224, 0.225]
).view(3, 1, 1)


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


# ============================================================
# COLOR SPACE CONVERSIONS
# ============================================================

def rgb_to_hsv_tensor(rgb):
    """
    RGB tensor:
        [B, 3, H, W]
        values in [0, 1]

    Returns:
        HSV tensor [B, 3, H, W]
    """

    r = rgb[:, 0]
    g = rgb[:, 1]
    b = rgb[:, 2]

    maxc = torch.max(rgb, dim=1).values
    minc = torch.min(rgb, dim=1).values

    delta = maxc - minc

    # Hue
    h = torch.zeros_like(maxc)

    mask = delta > 1e-6

    # Red is max
    mask_r = mask & (maxc == r)

    h[mask_r] = (
        ((g[mask_r] - b[mask_r]) / delta[mask_r]) % 6
    )

    # Green is max
    mask_g = mask & (maxc == g)

    h[mask_g] = (
        (b[mask_g] - r[mask_g]) / delta[mask_g]
    ) + 2

    # Blue is max
    mask_b = mask & (maxc == b)

    h[mask_b] = (
        (r[mask_b] - g[mask_b]) / delta[mask_b]
    ) + 4

    h = h / 6.0

    # Saturation
    s = torch.zeros_like(maxc)

    nonzero_max = maxc > 1e-6

    s[nonzero_max] = (
        delta[nonzero_max] /
        maxc[nonzero_max]
    )

    # Value
    v = maxc

    return torch.stack([h, s, v], dim=1)


def rgb_to_ycbcr_tensor(rgb):
    """
    Approximate RGB -> YCbCr conversion.
    Input values [0, 1].
    Output scaled approximately to [0, 1].
    """

    r = rgb[:, 0]
    g = rgb[:, 1]
    b = rgb[:, 2]

    y = (
        0.299 * r
        + 0.587 * g
        + 0.114 * b
    )

    cb = (
        -0.168736 * r
        -0.331264 * g
        + 0.500000 * b
        + 0.5
    )

    cr = (
        0.500000 * r
        -0.418688 * g
        -0.081312 * b
        + 0.5
    )

    return torch.stack([y, cb, cr], dim=1).clamp(0, 1)


# ============================================================
# DATASET
# ============================================================

class EyeColorSpaceDataset(torch.utils.data.Dataset):

    def __init__(self, root, train=False):

        self.dataset = datasets.ImageFolder(root)

        self.classes = self.dataset.classes
        self.samples = self.dataset.samples

        self.train = train

        if train:

            self.transform = transforms.Compose([
                transforms.Resize(
                    (IMAGE_SIZE, IMAGE_SIZE)
                ),

                transforms.RandomHorizontalFlip(
                    p=0.5
                ),

                transforms.RandomRotation(
                    degrees=10
                ),

                transforms.ColorJitter(
                    brightness=0.15,
                    contrast=0.15,
                    saturation=0.15,
                    hue=0.03,
                ),

                transforms.ToTensor(),
            ])

        else:

            self.transform = transforms.Compose([
                transforms.Resize(
                    (IMAGE_SIZE, IMAGE_SIZE)
                ),

                transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        path, label = self.samples[index]

        image = Image.open(path).convert("RGB")

        rgb = self.transform(image)

        # Add batch dimension
        rgb_batch = rgb.unsqueeze(0)

        hsv = rgb_to_hsv_tensor(
            rgb_batch
        ).squeeze(0)

        ycbcr = rgb_to_ycbcr_tensor(
            rgb_batch
        ).squeeze(0)

        # RGB uses ImageNet normalization
        rgb = (
            rgb - IMAGENET_MEAN
        ) / IMAGENET_STD

        # HSV and YCbCr remain in approximately
        # [0, 1], following the color-space idea
        # of CSF-Net.

        return (
            rgb,
            hsv,
            ycbcr,
            label
        )


# ============================================================
# MOBILENET FEATURE EXTRACTOR
# ============================================================

class MobileNetFeatureExtractor(nn.Module):

    def __init__(self, pretrained=True):

        super().__init__()

        if pretrained:

            weights = (
                MobileNet_V3_Small_Weights.DEFAULT
            )

            model = mobilenet_v3_small(
                weights=weights
            )

        else:

            model = mobilenet_v3_small(
                weights=None
            )

        # Feature extractor
        self.features = model.features

        self.avgpool = model.avgpool

        # MobileNetV3-Small classifier:
        # Linear(576 -> 1024)
        self.projection = model.classifier[0]

        self.output_dim = 1024

    def forward(self, x):

        x = self.features(x)

        x = self.avgpool(x)

        x = torch.flatten(
            x,
            1
        )

        x = self.projection(x)

        return x


# ============================================================
# CSF-NET
# ============================================================

class CSFNet(nn.Module):

    def __init__(
        self,
        num_classes,
    ):

        super().__init__()

        # RGB branch:
        # ImageNet pretrained
        self.rgb_branch = (
            MobileNetFeatureExtractor(
                pretrained=True
            )
        )

        # HSV branch:
        # Random initialization
        self.hsv_branch = (
            MobileNetFeatureExtractor(
                pretrained=False
            )
        )

        # YCbCr branch:
        # Random initialization
        self.ycbcr_branch = (
            MobileNetFeatureExtractor(
                pretrained=False
            )
        )

        feature_dim = 1024

        # Self-attention over the three
        # color-space feature tokens.
        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=1,
            batch_first=True
        )

        self.norm = nn.LayerNorm(
            feature_dim
        )

        # Final classifier
        self.classifier = nn.Sequential(

            nn.Linear(
                feature_dim,
                512
            ),

            nn.BatchNorm1d(512),

            nn.ReLU(inplace=True),

            nn.Dropout(0.35),

            nn.Linear(
                512,
                num_classes
            )
        )

    def forward(
        self,
        rgb,
        hsv,
        ycbcr
    ):

        rgb_features = (
            self.rgb_branch(rgb)
        )

        hsv_features = (
            self.hsv_branch(hsv)
        )

        ycbcr_features = (
            self.ycbcr_branch(ycbcr)
        )

        # [B, 3, 1024]
        tokens = torch.stack(
            [
                rgb_features,
                hsv_features,
                ycbcr_features
            ],
            dim=1
        )

        # Self-attention
        attended, _ = self.attention(
            tokens,
            tokens,
            tokens
        )

        attended = self.norm(
            attended + tokens
        )

        # Global fusion across
        # the three color spaces
        fused = attended.mean(
            dim=1
        )

        logits = self.classifier(
            fused
        )

        return logits


# ============================================================
# CLASS WEIGHTS
# ============================================================

def compute_class_weights(dataset):

    labels = [
        label
        for _, label in dataset.samples
    ]

    counts = np.bincount(
        labels,
        minlength=len(dataset.classes)
    )

    # sqrt inverse-frequency weighting
    weights = 1.0 / np.sqrt(
        counts.astype(np.float32)
    )

    weights = (
        weights / weights.mean()
    )

    return (
        torch.tensor(
            weights,
            dtype=torch.float32
        ),
        counts
    )


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    scaler,
):

    model.train()

    running_loss = 0.0

    all_predictions = []
    all_labels = []

    progress = tqdm(
        loader,
        desc="Training",
        leave=False
    )

    for (
        rgb,
        hsv,
        ycbcr,
        labels
    ) in progress:

        rgb = rgb.to(
            DEVICE,
            non_blocking=True
        )

        hsv = hsv.to(
            DEVICE,
            non_blocking=True
        )

        ycbcr = ycbcr.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        with torch.amp.autocast(
            device_type="cuda",
            enabled=USE_AMP and DEVICE.type == "cuda"
        ):

            logits = model(
                rgb,
                hsv,
                ycbcr
            )

            loss = criterion(
                logits,
                labels
            )

        scaler.scale(
            loss
        ).backward()

        scaler.step(
            optimizer
        )

        scaler.update()

        running_loss += (
            loss.item()
            * labels.size(0)
        )

        predictions = (
            logits.argmax(dim=1)
        )

        all_predictions.extend(
            predictions.detach()
            .cpu()
            .numpy()
        )

        all_labels.extend(
            labels.detach()
            .cpu()
            .numpy()
        )

        progress.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    epoch_loss = (
        running_loss /
        len(loader.dataset)
    )

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    _, _, macro_f1, _ = (
        precision_recall_fscore_support(
            all_labels,
            all_predictions,
            average="macro",
            zero_division=0
        )
    )

    return (
        epoch_loss,
        accuracy,
        macro_f1
    )


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
):

    model.eval()

    running_loss = 0.0

    all_predictions = []
    all_labels = []

    for (
        rgb,
        hsv,
        ycbcr,
        labels
    ) in tqdm(
        loader,
        desc="Validation",
        leave=False
    ):

        rgb = rgb.to(
            DEVICE,
            non_blocking=True
        )

        hsv = hsv.to(
            DEVICE,
            non_blocking=True
        )

        ycbcr = ycbcr.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )

        with torch.amp.autocast(
            device_type="cuda",
            enabled=USE_AMP and DEVICE.type == "cuda"
        ):

            logits = model(
                rgb,
                hsv,
                ycbcr
            )

            loss = criterion(
                logits,
                labels
            )

        running_loss += (
            loss.item()
            * labels.size(0)
        )

        predictions = (
            logits.argmax(dim=1)
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            labels.cpu().numpy()
        )

    epoch_loss = (
        running_loss /
        len(loader.dataset)
    )

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    precision, recall, macro_f1, _ = (
        precision_recall_fscore_support(
            all_labels,
            all_predictions,
            average="macro",
            zero_division=0
        )
    )

    return {
        "loss": epoch_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "macro_f1": macro_f1,
        "labels": all_labels,
        "predictions": all_predictions,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    seed_everything(SEED)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 70)
    print("CSF-NET COMBINED DOG + CAT EYE DISEASE CLASSIFIER")
    print("=" * 70)

    print(
        f"Device: {DEVICE}"
    )

    if torch.cuda.is_available():

        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    print(
        f"Dataset: {DATA_DIR}"
    )

    print(
        f"Output:  {OUTPUT_DIR}"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # DATASETS
    # --------------------------------------------------------

    print("\nLoading datasets...")

    train_dataset = EyeColorSpaceDataset(
        DATA_DIR / "train",
        train=True
    )

    valid_dataset = EyeColorSpaceDataset(
        DATA_DIR / "valid",
        train=False
    )

    test_dataset = EyeColorSpaceDataset(
        DATA_DIR / "test",
        train=False
    )

    classes = train_dataset.classes

    print("\nClasses:")

    for i, name in enumerate(classes):

        print(
            f"  {i}: {name}"
        )

    print(
        f"\nNumber of classes: "
        f"{len(classes)}"
    )

    print(
        f"Training images:   "
        f"{len(train_dataset)}"
    )

    print(
        f"Validation images: "
        f"{len(valid_dataset)}"
    )

    print(
        f"Test images:       "
        f"{len(test_dataset)}"
    )

    # --------------------------------------------------------
    # CLASS WEIGHTS
    # --------------------------------------------------------

    class_weights, counts = (
        compute_class_weights(
            train_dataset
        )
    )

    print(
        "\nTraining class distribution:"
    )

    for name, count in zip(
        classes,
        counts
    ):

        print(
            f"  {name:25s}: {count}"
        )

    print(
        "\nClass weights:"
    )

    for name, weight in zip(
        classes,
        class_weights
    ):

        print(
            f"  {name:25s}: "
            f"{weight.item():.4f}"
        )

    class_weights = (
        class_weights.to(DEVICE)
    )

    # --------------------------------------------------------
    # DATALOADERS
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(
            NUM_WORKERS > 0
        ),
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(
            NUM_WORKERS > 0
        ),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(
            NUM_WORKERS > 0
        ),
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    print(
        "\nLoading CSF-Net..."
    )

    model = CSFNet(
        num_classes=len(classes)
    )

    model = model.to(DEVICE)

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"Total parameters: "
        f"{total_params:,}"
    )

    print(
        f"Trainable parameters: "
        f"{trainable_params:,}"
    )

    # --------------------------------------------------------
    # LOSS
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=LABEL_SMOOTHING
    )

    # --------------------------------------------------------
    # OPTIMIZER
    # --------------------------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # --------------------------------------------------------
    # SCHEDULER
    # --------------------------------------------------------

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
        min_lr=1e-6
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(
            USE_AMP
            and DEVICE.type == "cuda"
        )
    )

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)

    history = []

    best_val_f1 = -float("inf")
    best_epoch = 0
    patience_counter = 0

    best_model_state = None

    for epoch in range(
        1,
        NUM_EPOCHS + 1
    ):

        print(
            f"\nEpoch "
            f"{epoch}/{NUM_EPOCHS}"
        )

        print("-" * 50)

        start_time = time.time()

        train_loss, train_acc, train_f1 = (
            train_one_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                scaler
            )
        )

        val_results = evaluate(
            model,
            valid_loader,
            criterion
        )

        scheduler.step(
            val_results["macro_f1"]
        )

        current_lr = (
            optimizer.param_groups[0]["lr"]
        )

        epoch_time = (
            time.time() - start_time
        )

        print(
            f"Train loss:     "
            f"{train_loss:.4f}"
        )

        print(
            f"Train accuracy: "
            f"{train_acc:.4f}"
        )

        print(
            f"Train macro-F1: "
            f"{train_f1:.4f}"
        )

        print(
            f"Valid loss:     "
            f"{val_results['loss']:.4f}"
        )

        print(
            f"Valid accuracy: "
            f"{val_results['accuracy']:.4f}"
        )

        print(
            f"Valid macro-F1: "
            f"{val_results['macro_f1']:.4f}"
        )

        print(
            f"Learning rate:  "
            f"{current_lr:.2e}"
        )

        print(
            f"Time:           "
            f"{epoch_time:.1f}s"
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "train_macro_f1": train_f1,
            "valid_loss": val_results["loss"],
            "valid_accuracy": val_results["accuracy"],
            "valid_macro_f1": val_results["macro_f1"],
            "learning_rate": current_lr,
        })

        # ----------------------------------------------------
        # BEST MODEL
        # ----------------------------------------------------

        if (
            val_results["macro_f1"]
            > best_val_f1
        ):

            best_val_f1 = (
                val_results["macro_f1"]
            )

            best_epoch = epoch

            patience_counter = 0

            best_model_state = copy.deepcopy(
                model.state_dict()
            )

            torch.save(
                {
                    "model_state_dict":
                        best_model_state,

                    "classes":
                        classes,

                    "epoch":
                        epoch,

                    "val_macro_f1":
                        best_val_f1,

                    "config": {
                        "image_size":
                            IMAGE_SIZE,

                        "batch_size":
                            BATCH_SIZE,

                        "learning_rate":
                            LEARNING_RATE,

                        "num_classes":
                            len(classes),
                    }
                },
                OUTPUT_DIR /
                "best_csfnet.pth"
            )

            print(
                "\n*** Saved new best model ***"
            )

            print(
                f"Best validation macro-F1: "
                f"{best_val_f1:.4f}"
            )

        else:

            patience_counter += 1

            print(
                f"\nNo improvement "
                f"({patience_counter}/"
                f"{EARLY_STOPPING_PATIENCE})"
            )

        # ----------------------------------------------------
        # EARLY STOPPING
        # ----------------------------------------------------

        if (
            patience_counter
            >= EARLY_STOPPING_PATIENCE
        ):

            print(
                "\nEarly stopping."
            )

            break

    # --------------------------------------------------------
    # SAVE HISTORY
    # --------------------------------------------------------

    history_df = pd.DataFrame(
        history
    )

    history_df.to_csv(
        OUTPUT_DIR /
        "training_history.csv",
        index=False
    )

    with open(
        OUTPUT_DIR /
        "training_history.json",
        "w"
    ) as f:

        json.dump(
            history,
            f,
            indent=2
        )

    # --------------------------------------------------------
    # LOAD BEST MODEL
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("LOADING BEST MODEL")
    print("=" * 70)

    checkpoint = torch.load(
        OUTPUT_DIR /
        "best_csfnet.pth",
        map_location=DEVICE,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    print(
        f"Best validation macro-F1: "
        f"{checkpoint['val_macro_f1']:.4f}"
    )

    print(
        f"Best epoch: "
        f"{checkpoint['epoch']}"
    )

    # --------------------------------------------------------
    # FINAL TEST
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("FINAL TEST EVALUATION")
    print("=" * 70)

    test_results = evaluate(
        model,
        test_loader,
        criterion
    )

    print(
        f"\nTest loss:       "
        f"{test_results['loss']:.4f}"
    )

    print(
        f"Test accuracy:   "
        f"{test_results['accuracy']:.4f}"
    )

    print(
        f"Test macro-F1:   "
        f"{test_results['macro_f1']:.4f}"
    )

    print(
        f"Test macro-recall: "
        f"{test_results['recall']:.4f}"
    )

    # --------------------------------------------------------
    # CLASSIFICATION REPORT
    # --------------------------------------------------------

    report = classification_report(
        test_results["labels"],
        test_results["predictions"],
        target_names=classes,
        digits=4,
        zero_division=0
    )

    print("\nClassification Report:")
    print(report)

    with open(
        OUTPUT_DIR /
        "classification_report.txt",
        "w"
    ) as f:

        f.write(report)

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    cm = confusion_matrix(
        test_results["labels"],
        test_results["predictions"]
    )

    fig, ax = plt.subplots(
        figsize=(13, 11)
    )

    im = ax.imshow(
        cm,
        interpolation="nearest",
        cmap="viridis"
    )

    plt.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(classes)),
        yticks=np.arange(len(classes)),
        xticklabels=classes,
        yticklabels=classes,
        ylabel="True label",
        xlabel="Predicted label",
        title="CSF-Net Test Confusion Matrix"
    )

    plt.setp(
        ax.get_xticklabels(),
        rotation=45,
        ha="right"
    )

    threshold = (
        cm.max() / 2.0
    )

    for i in range(
        cm.shape[0]
    ):

        for j in range(
            cm.shape[1]
        ):

            ax.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
                color=(
                    "white"
                    if cm[i, j] > threshold
                    else "black"
                )
            )

    fig.tight_layout()

    plt.savefig(
        OUTPUT_DIR /
        "confusion_matrix.png",
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    # --------------------------------------------------------
    # TRAINING CURVES
    # --------------------------------------------------------

    epochs = history_df["epoch"]

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        epochs,
        history_df["train_loss"],
        label="Train Loss"
    )

    plt.plot(
        epochs,
        history_df["valid_loss"],
        label="Validation Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("CSF-Net Loss")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.savefig(
        OUTPUT_DIR /
        "loss_curve.png",
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        epochs,
        history_df["train_macro_f1"],
        label="Train Macro-F1"
    )

    plt.plot(
        epochs,
        history_df["valid_macro_f1"],
        label="Validation Macro-F1"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.title("CSF-Net Macro-F1")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.savefig(
        OUTPUT_DIR /
        "macro_f1_curve.png",
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    # --------------------------------------------------------
    # SAVE TEST METRICS
    # --------------------------------------------------------

    metrics = {
        "test_loss":
            test_results["loss"],

        "test_accuracy":
            test_results["accuracy"],

        "test_macro_f1":
            test_results["macro_f1"],

        "test_macro_recall":
            test_results["recall"],

        "best_validation_macro_f1":
            best_val_f1,

        "best_epoch":
            best_epoch,

        "num_classes":
            len(classes),

        "classes":
            classes,

        "test_samples":
            len(test_dataset),

        "architecture":
            "CSF-Net-inspired",
    }

    with open(
        OUTPUT_DIR /
        "test_metrics.json",
        "w"
    ) as f:

        json.dump(
            metrics,
            f,
            indent=2
        )

    # --------------------------------------------------------
    # DATASET INFO
    # --------------------------------------------------------

    dataset_info = {

        "train_images":
            len(train_dataset),

        "validation_images":
            len(valid_dataset),

        "test_images":
            len(test_dataset),

        "classes":
            classes,

        "class_counts":
            {
                name: int(count)
                for name, count in zip(
                    classes,
                    counts
                )
            },

        "image_size":
            IMAGE_SIZE,

        "architecture":
            "CSF-Net-inspired",

        "species":
            "combined dog + cat",
    }

    with open(
        OUTPUT_DIR /
        "dataset_info.json",
        "w"
    ) as f:

        json.dump(
            dataset_info,
            f,
            indent=2
        )

    print("\n")
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print(
        f"\nBest validation macro-F1: "
        f"{best_val_f1:.4f}"
    )

    print(
        f"Final test accuracy: "
        f"{test_results['accuracy']:.4f}"
    )

    print(
        f"Final test macro-F1: "
        f"{test_results['macro_f1']:.4f}"
    )

    print(
        "\nOutput files:"
    )

    print(
        "  best_csfnet.pth"
    )

    print(
        "  training_history.csv"
    )

    print(
        "  classification_report.txt"
    )

    print(
        "  confusion_matrix.png"
    )

    print(
        "  macro_f1_curve.png"
    )

    print(
        "  test_metrics.json"
    )

    print(
        "\nDone."
    )


# ============================================================
# WINDOWS-SAFE ENTRY POINT
# ============================================================

if __name__ == "__main__":

    from multiprocessing import freeze_support

    freeze_support()

    main()