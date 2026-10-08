import os
import json
import random
from pathlib import Path
from multiprocessing import freeze_support

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    classification_report,
    confusion_matrix,
)


# ============================================================
# CONFIG
# ============================================================

DATA_DIR = Path("data/skin_coat/combined_clean")
OUTPUT_DIR = Path("outputs/skin_efficientnet_v2s")

IMAGE_SIZE = 384

BATCH_SIZE = 8
NUM_EPOCHS = 50
PATIENCE = 7

LR = 1e-4
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.05

NUM_WORKERS = 2

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
        torch.cuda.manual_seed_all(seed)


# ============================================================
# DEVICE
# ============================================================

def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    else:
        device = torch.device("cpu")
        print("WARNING: CUDA not available. Using CPU.")

    return device


# ============================================================
# DATA
# ============================================================

def build_dataloaders():

    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

        transforms.RandomHorizontalFlip(p=0.5),

        transforms.RandomRotation(
            degrees=12
        ),

        transforms.ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.15,
            hue=0.03
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    train_dir = DATA_DIR / "train"
    valid_dir = DATA_DIR / "valid"
    test_dir = DATA_DIR / "test"

    train_dataset = datasets.ImageFolder(
        train_dir,
        transform=train_transform
    )

    valid_dataset = datasets.ImageFolder(
        valid_dir,
        transform=eval_transform
    )

    test_dataset = datasets.ImageFolder(
        test_dir,
        transform=eval_transform
    )

    # Make sure all splits use identical class ordering
    assert train_dataset.class_to_idx == valid_dataset.class_to_idx
    assert train_dataset.class_to_idx == test_dataset.class_to_idx

    class_names = train_dataset.classes

    print("\nClasses:")
    for i, name in enumerate(class_names):
        print(f"  {i}: {name}")

    print("\nDataset sizes:")
    print(f"  Train: {len(train_dataset)}")
    print(f"  Valid: {len(valid_dataset)}")
    print(f"  Test : {len(test_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0)
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0)
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0)
    )

    return (
        train_dataset,
        valid_dataset,
        test_dataset,
        train_loader,
        valid_loader,
        test_loader,
        class_names
    )


# ============================================================
# CLASS WEIGHTS
# ============================================================

def calculate_class_weights(dataset, num_classes):

    targets = np.array(dataset.targets)

    counts = np.bincount(
        targets,
        minlength=num_classes
    )

    print("\nTraining class counts:")

    for idx, count in enumerate(counts):
        print(f"  {dataset.classes[idx]:40s}: {count}")

    # Square-root inverse-frequency weighting.
    # Less aggressive than plain inverse-frequency weighting.
    weights = 1.0 / np.sqrt(counts)

    weights = weights / weights.mean()

    return torch.tensor(
        weights,
        dtype=torch.float32
    )


# ============================================================
# MODEL
# ============================================================

def build_model(num_classes):

    print("\nLoading ImageNet-pretrained EfficientNetV2-S...")

    weights = EfficientNet_V2_S_Weights.DEFAULT

    model = efficientnet_v2_s(
        weights=weights
    )

    # Original classifier:
    # Dropout -> Linear(1280, 1000)

    in_features = model.classifier[1].in_features

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(
            in_features,
            num_classes
        )
    )

    return model


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scaler,
    device
):

    model.train()

    running_loss = 0.0

    all_predictions = []
    all_targets = []

    for images, labels in loader:

        images = images.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        with torch.amp.autocast(
            device_type=device.type,
            enabled=USE_AMP and device.type == "cuda"
        ):

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

        if scaler is not None:

            scaler.scale(loss).backward()

            scaler.step(optimizer)

            scaler.update()

        else:

            loss.backward()

            optimizer.step()

        running_loss += (
            loss.item() * images.size(0)
        )

        predictions = outputs.argmax(
            dim=1
        )

        all_predictions.extend(
            predictions.detach().cpu().numpy()
        )

        all_targets.extend(
            labels.detach().cpu().numpy()
        )

    epoch_loss = (
        running_loss / len(loader.dataset)
    )

    epoch_accuracy = accuracy_score(
        all_targets,
        all_predictions
    )

    epoch_f1 = f1_score(
        all_targets,
        all_predictions,
        average="macro",
        zero_division=0
    )

    return (
        epoch_loss,
        epoch_accuracy,
        epoch_f1
    )


# ============================================================
# VALIDATE
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    device
):

    model.eval()

    running_loss = 0.0

    all_predictions = []
    all_targets = []

    for images, labels in loader:

        images = images.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        with torch.amp.autocast(
            device_type=device.type,
            enabled=USE_AMP and device.type == "cuda"
        ):

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

        running_loss += (
            loss.item() * images.size(0)
        )

        predictions = outputs.argmax(
            dim=1
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_targets.extend(
            labels.cpu().numpy()
        )

    epoch_loss = (
        running_loss / len(loader.dataset)
    )

    epoch_accuracy = accuracy_score(
        all_targets,
        all_predictions
    )

    epoch_f1 = f1_score(
        all_targets,
        all_predictions,
        average="macro",
        zero_division=0
    )

    epoch_recall = recall_score(
        all_targets,
        all_predictions,
        average="macro",
        zero_division=0
    )

    return (
        epoch_loss,
        epoch_accuracy,
        epoch_f1,
        epoch_recall,
        all_targets,
        all_predictions
    )


# ============================================================
# PLOTS
# ============================================================

def save_training_curves(history):

    epochs = range(
        1,
        len(history["train_loss"]) + 1
    )

    # Loss
    plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        history["train_loss"],
        label="Train Loss"
    )

    plt.plot(
        epochs,
        history["valid_loss"],
        label="Validation Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Skin Model Loss")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "loss_curve.png",
        dpi=200
    )

    plt.close()

    # Accuracy
    plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        history["train_accuracy"],
        label="Train Accuracy"
    )

    plt.plot(
        epochs,
        history["valid_accuracy"],
        label="Validation Accuracy"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Skin Model Accuracy")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "accuracy_curve.png",
        dpi=200
    )

    plt.close()

    # Macro F1
    plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        history["train_f1"],
        label="Train Macro-F1"
    )

    plt.plot(
        epochs,
        history["valid_f1"],
        label="Validation Macro-F1"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.title("Skin Model Macro-F1")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "macro_f1_curve.png",
        dpi=200
    )

    plt.close()


# ============================================================
# CONFUSION MATRIX
# ============================================================

def save_confusion_matrix(
    targets,
    predictions,
    class_names
):

    cm = confusion_matrix(
        targets,
        predictions
    )

    plt.figure(
        figsize=(12, 10)
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=class_names,
        yticklabels=class_names
    )

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(
        "EfficientNetV2-S Skin Classification Confusion Matrix"
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
        OUTPUT_DIR / "confusion_matrix.png",
        dpi=200
    )

    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():

    set_seed(SEED)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    device = get_device()

    (
        train_dataset,
        valid_dataset,
        test_dataset,
        train_loader,
        valid_loader,
        test_loader,
        class_names
    ) = build_dataloaders()

    num_classes = len(class_names)

    # --------------------------------------------------------
    # CLASS WEIGHTS
    # --------------------------------------------------------

    class_weights = calculate_class_weights(
        train_dataset,
        num_classes
    ).to(device)

    print("\nClass weights:")

    for name, weight in zip(
        class_names,
        class_weights.cpu().numpy()
    ):

        print(
            f"  {name:40s}: {weight:.4f}"
        )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = build_model(
        num_classes
    )

    model = model.to(device)

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

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    # --------------------------------------------------------
    # LR SCHEDULER
    # --------------------------------------------------------

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2
    )

    # --------------------------------------------------------
    # AMP
    # --------------------------------------------------------

    if device.type == "cuda" and USE_AMP:

        scaler = torch.amp.GradScaler(
            "cuda"
        )

    else:

        scaler = None

    # --------------------------------------------------------
    # HISTORY
    # --------------------------------------------------------

    history = {
        "train_loss": [],
        "valid_loss": [],
        "train_accuracy": [],
        "valid_accuracy": [],
        "train_f1": [],
        "valid_f1": [],
        "valid_recall": [],
        "learning_rate": []
    }

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    best_valid_f1 = -1.0

    epochs_without_improvement = 0

    best_model_path = (
        OUTPUT_DIR /
        "best_model.pth"
    )

    print("\n")
    print("=" * 70)
    print("STARTING EFFICIENTNETV2-S TRAINING")
    print("=" * 70)

    print(f"Device       : {device}")
    print(f"Image size   : {IMAGE_SIZE}")
    print(f"Batch size   : {BATCH_SIZE}")
    print(f"Workers      : {NUM_WORKERS}")
    print(f"Epochs       : {NUM_EPOCHS}")
    print(f"Classes      : {num_classes}")
    print("=" * 70)

    for epoch in range(
        1,
        NUM_EPOCHS + 1
    ):

        print(
            f"\nEpoch {epoch}/{NUM_EPOCHS}"
        )

        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        train_loss, train_accuracy, train_f1 = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device
        )

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        (
            valid_loss,
            valid_accuracy,
            valid_f1,
            valid_recall,
            _,
            _
        ) = evaluate(
            model,
            valid_loader,
            criterion,
            device
        )

        # ----------------------------------------------------
        # LR
        # ----------------------------------------------------

        scheduler.step(
            valid_f1
        )

        current_lr = optimizer.param_groups[0]["lr"]

        # ----------------------------------------------------
        # HISTORY
        # ----------------------------------------------------

        history["train_loss"].append(
            train_loss
        )

        history["valid_loss"].append(
            valid_loss
        )

        history["train_accuracy"].append(
            train_accuracy
        )

        history["valid_accuracy"].append(
            valid_accuracy
        )

        history["train_f1"].append(
            train_f1
        )

        history["valid_f1"].append(
            valid_f1
        )

        history["valid_recall"].append(
            valid_recall
        )

        history["learning_rate"].append(
            current_lr
        )

        # ----------------------------------------------------
        # PRINT
        # ----------------------------------------------------

        print(
            f"Train Loss     : {train_loss:.4f}"
        )

        print(
            f"Train Accuracy : {train_accuracy:.4f}"
        )

        print(
            f"Train Macro-F1 : {train_f1:.4f}"
        )

        print(
            f"Valid Loss     : {valid_loss:.4f}"
        )

        print(
            f"Valid Accuracy : {valid_accuracy:.4f}"
        )

        print(
            f"Valid Macro-F1 : {valid_f1:.4f}"
        )

        print(
            f"Valid Recall   : {valid_recall:.4f}"
        )

        print(
            f"Learning Rate  : {current_lr:.7f}"
        )

        # ----------------------------------------------------
        # BEST MODEL
        # ----------------------------------------------------

        if valid_f1 > best_valid_f1:

            best_valid_f1 = valid_f1

            epochs_without_improvement = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_valid_f1": best_valid_f1,
                    "class_names": class_names,
                    "image_size": IMAGE_SIZE
                },
                best_model_path
            )

            print(
                f"\n*** NEW BEST MODEL ***"
            )

            print(
                f"Validation Macro-F1: "
                f"{best_valid_f1:.4f}"
            )

        else:

            epochs_without_improvement += 1

            print(
                f"\nNo improvement "
                f"({epochs_without_improvement}/{PATIENCE})"
            )

        # ----------------------------------------------------
        # EARLY STOPPING
        # ----------------------------------------------------

        if epochs_without_improvement >= PATIENCE:

            print(
                "\nEarly stopping triggered."
            )

            break

    # --------------------------------------------------------
    # SAVE HISTORY
    # --------------------------------------------------------

    with open(
        OUTPUT_DIR / "training_history.json",
        "w"
    ) as f:

        json.dump(
            history,
            f,
            indent=4
        )

    save_training_curves(
        history
    )

    # --------------------------------------------------------
    # LOAD BEST MODEL
    # --------------------------------------------------------

    print(
        "\nLoading best validation model..."
    )

    checkpoint = torch.load(
        best_model_path,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    print(
        f"Best validation Macro-F1: "
        f"{checkpoint['best_valid_f1']:.4f}"
    )

    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("FINAL TEST EVALUATION")
    print("=" * 70)

    (
        test_loss,
        test_accuracy,
        test_f1,
        test_recall,
        test_targets,
        test_predictions
    ) = evaluate(
        model,
        test_loader,
        criterion,
        device
    )

    print(
        f"\nTest Loss      : {test_loss:.4f}"
    )

    print(
        f"Test Accuracy  : {test_accuracy:.4f}"
    )

    print(
        f"Test Macro-F1  : {test_f1:.4f}"
    )

    print(
        f"Test Macro-Recall: {test_recall:.4f}"
    )

    # --------------------------------------------------------
    # CLASSIFICATION REPORT
    # --------------------------------------------------------

    report = classification_report(
        test_targets,
        test_predictions,
        target_names=class_names,
        digits=4,
        zero_division=0
    )

    print("\nClassification Report:")
    print(report)

    with open(
        OUTPUT_DIR / "classification_report.txt",
        "w"
    ) as f:

        f.write(report)

    # --------------------------------------------------------
    # PER-CLASS CSV
    # --------------------------------------------------------

    report_dict = classification_report(
        test_targets,
        test_predictions,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    rows = []

    for class_name in class_names:

        rows.append({
            "class": class_name,
            "precision": report_dict[class_name]["precision"],
            "recall": report_dict[class_name]["recall"],
            "f1": report_dict[class_name]["f1-score"],
            "support": report_dict[class_name]["support"]
        })

    pd.DataFrame(rows).to_csv(
        OUTPUT_DIR / "per_class_metrics.csv",
        index=False
    )

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    save_confusion_matrix(
        test_targets,
        test_predictions,
        class_names
    )

    # --------------------------------------------------------
    # TEST SUMMARY
    # --------------------------------------------------------

    test_summary = {
        "model": "EfficientNetV2-S",
        "num_classes": num_classes,
        "image_size": IMAGE_SIZE,
        "batch_size": BATCH_SIZE,
        "best_validation_macro_f1": float(best_valid_f1),
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "test_macro_f1": float(test_f1),
        "test_macro_recall": float(test_recall),
        "classes": class_names
    }

    with open(
        OUTPUT_DIR / "test_metrics.json",
        "w"
    ) as f:

        json.dump(
            test_summary,
            f,
            indent=4
        )

    print("\n")
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print(
        f"Best model: {best_model_path}"
    )

    print(
        f"Output directory: {OUTPUT_DIR}"
    )


# ============================================================
# WINDOWS SAFE ENTRY POINT
# ============================================================

if __name__ == "__main__":

    freeze_support()

    main()