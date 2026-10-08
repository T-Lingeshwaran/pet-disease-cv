from pathlib import Path
from collections import Counter
from multiprocessing import freeze_support
import json
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torchvision import datasets, transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = Path("data/skin_coat/combined_clean")
OUTPUT_DIR = Path("outputs/skin_efficientnet")

IMAGE_SIZE = 224
BATCH_SIZE = 16

NUM_EPOCHS = 50
PATIENCE = 7

LEARNING_RATE = 1e-4
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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True


# ============================================================
# DATA TRANSFORMS
# ============================================================

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.RandomHorizontalFlip(p=0.5),

    transforms.RandomRotation(
        degrees=12,
        fill=0
    ),

    transforms.ColorJitter(
        brightness=0.20,
        contrast=0.20,
        saturation=0.15,
        hue=0.03,
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
    ),
])


eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
    ),
])


# ============================================================
# DATASET
# ============================================================

def build_datasets():

    train_dir = DATA_DIR / "train"
    valid_dir = DATA_DIR / "valid"
    test_dir = DATA_DIR / "test"

    if not train_dir.exists():
        raise FileNotFoundError(
            f"Training directory not found: {train_dir}"
        )

    if not valid_dir.exists():
        raise FileNotFoundError(
            f"Validation directory not found: {valid_dir}"
        )

    if not test_dir.exists():
        raise FileNotFoundError(
            f"Test directory not found: {test_dir}"
        )

    train_dataset = datasets.ImageFolder(
        train_dir,
        transform=train_transform,
    )

    valid_dataset = datasets.ImageFolder(
        valid_dir,
        transform=eval_transform,
    )

    test_dataset = datasets.ImageFolder(
        test_dir,
        transform=eval_transform,
    )

    # Make sure every split has identical class ordering.
    if train_dataset.class_to_idx != valid_dataset.class_to_idx:
        raise RuntimeError(
            "Train and validation class mappings do not match."
        )

    if train_dataset.class_to_idx != test_dataset.class_to_idx:
        raise RuntimeError(
            "Train and test class mappings do not match."
        )

    print("\nClasses:")
    for idx, name in enumerate(train_dataset.classes):
        print(f"  {idx}: {name}")

    print("\nDataset sizes:")
    print(f"  Train: {len(train_dataset)}")
    print(f"  Valid: {len(valid_dataset)}")
    print(f"  Test : {len(test_dataset)}")

    return train_dataset, valid_dataset, test_dataset


# ============================================================
# CLASS WEIGHTS
# ============================================================

def calculate_class_weights(dataset):

    counts = Counter(
        label for _, label in dataset.samples
    )

    num_classes = len(dataset.classes)

    frequencies = np.array(
        [counts[i] for i in range(num_classes)],
        dtype=np.float32,
    )

    # Square-root inverse frequency weighting.
    #
    # This is intentionally less aggressive than plain
    # inverse-frequency weighting because the dataset is
    # strongly imbalanced.
    weights = 1.0 / np.sqrt(frequencies)

    # Normalize so mean weight = 1.
    weights = weights / weights.mean()

    weights_tensor = torch.tensor(
        weights,
        dtype=torch.float32,
        device=DEVICE,
    )

    print("\nClass weights:")
    for idx, class_name in enumerate(dataset.classes):
        print(
            f"  {class_name:40s} "
            f"count={int(frequencies[idx]):4d} "
            f"weight={weights[idx]:.4f}"
        )

    return weights_tensor


# ============================================================
# DATALOADERS
# ============================================================

def create_dataloaders(
    train_dataset,
    valid_dataset,
    test_dataset,
):

    loader_kwargs = {
        "num_workers": NUM_WORKERS,
        "pin_memory": torch.cuda.is_available(),
    }

    if NUM_WORKERS > 0:
        loader_kwargs["persistent_workers"] = True

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False,
        **loader_kwargs,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        **loader_kwargs,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        **loader_kwargs,
    )

    return train_loader, valid_loader, test_loader


# ============================================================
# MODEL
# ============================================================

def build_model(num_classes):

    print("\nLoading ImageNet-pretrained EfficientNet-B0...")

    weights = EfficientNet_B0_Weights.DEFAULT

    model = efficientnet_b0(
        weights=weights
    )

    # EfficientNet-B0 classifier:
    #
    # Dropout -> Linear
    #
    # Replace final layer for our 11 skin classes.
    in_features = model.classifier[1].in_features

    model.classifier[1] = nn.Linear(
        in_features,
        num_classes,
    )

    model = model.to(DEVICE)

    return model


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(y_true, y_pred):

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    return {
        "accuracy": float(accuracy),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
    }


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scaler,
):

    model.train()

    running_loss = 0.0

    all_targets = []
    all_predictions = []

    for images, targets in loader:

        images = images.to(
            DEVICE,
            non_blocking=True,
        )

        targets = targets.to(
            DEVICE,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        with torch.amp.autocast(
            device_type=DEVICE.type,
            enabled=USE_AMP and DEVICE.type == "cuda",
        ):

            outputs = model(images)

            loss = criterion(
                outputs,
                targets,
            )

        scaler.scale(loss).backward()

        scaler.step(optimizer)

        scaler.update()

        running_loss += (
            loss.item() * images.size(0)
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
        running_loss / len(loader.dataset)
    )

    metrics = calculate_metrics(
        all_targets,
        all_predictions,
    )

    return epoch_loss, metrics


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

    all_targets = []
    all_predictions = []

    for images, targets in loader:

        images = images.to(
            DEVICE,
            non_blocking=True,
        )

        targets = targets.to(
            DEVICE,
            non_blocking=True,
        )

        with torch.amp.autocast(
            device_type=DEVICE.type,
            enabled=USE_AMP and DEVICE.type == "cuda",
        ):

            outputs = model(images)

            loss = criterion(
                outputs,
                targets,
            )

        running_loss += (
            loss.item() * images.size(0)
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
        running_loss / len(loader.dataset)
    )

    metrics = calculate_metrics(
        all_targets,
        all_predictions,
    )

    return (
        epoch_loss,
        metrics,
        all_targets,
        all_predictions,
    )


# ============================================================
# SAVE TRAINING CURVES
# ============================================================

def save_curves(history):

    history_df = pd.DataFrame(history)

    history_df.to_csv(
        OUTPUT_DIR / "training_history.csv",
        index=False,
    )

    # Loss curve
    plt.figure(figsize=(8, 5))

    plt.plot(
        history_df["epoch"],
        history_df["train_loss"],
        label="Train Loss",
    )

    plt.plot(
        history_df["epoch"],
        history_df["valid_loss"],
        label="Validation Loss",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Skin Model Loss")

    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "loss_curve.png",
        dpi=200,
    )

    plt.close()

    # Accuracy
    plt.figure(figsize=(8, 5))

    plt.plot(
        history_df["epoch"],
        history_df["train_accuracy"],
        label="Train Accuracy",
    )

    plt.plot(
        history_df["epoch"],
        history_df["valid_accuracy"],
        label="Validation Accuracy",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Skin Model Accuracy")

    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "accuracy_curve.png",
        dpi=200,
    )

    plt.close()

    # Macro F1
    plt.figure(figsize=(8, 5))

    plt.plot(
        history_df["epoch"],
        history_df["train_macro_f1"],
        label="Train Macro-F1",
    )

    plt.plot(
        history_df["epoch"],
        history_df["valid_macro_f1"],
        label="Validation Macro-F1",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.title("Skin Model Macro-F1")

    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "macro_f1_curve.png",
        dpi=200,
    )

    plt.close()


# ============================================================
# CONFUSION MATRIX
# ============================================================

def save_confusion_matrix(
    y_true,
    y_pred,
    class_names,
):

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
    )

    np.savetxt(
        OUTPUT_DIR / "confusion_matrix.csv",
        cm,
        delimiter=",",
        fmt="%d",
    )

    fig_width = max(
        10,
        len(class_names) * 0.8,
    )

    plt.figure(
        figsize=(fig_width, 9)
    )

    plt.imshow(cm)

    plt.title(
        "Skin Disease Confusion Matrix"
    )

    plt.colorbar()

    plt.xticks(
        range(len(class_names)),
        class_names,
        rotation=45,
        ha="right",
    )

    plt.yticks(
        range(len(class_names)),
        class_names,
    )

    plt.xlabel("Predicted")
    plt.ylabel("True")

    # Write values into cells
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):

            plt.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
            )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "confusion_matrix.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


# ============================================================
# MAIN TRAINING
# ============================================================

def main():

    set_seed(SEED)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print("SKIN DISEASE CLASSIFICATION")
    print("EfficientNet-B0")
    print("=" * 70)

    print(f"\nDevice: {DEVICE}")

    if torch.cuda.is_available():
        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

        print(
            f"CUDA: {torch.version.cuda}"
        )

    print(f"Batch size: {BATCH_SIZE}")
    print(f"Image size: {IMAGE_SIZE}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Workers: {NUM_WORKERS}")
    print(f"AMP: {USE_AMP}")

    # --------------------------------------------------------
    # DATA
    # --------------------------------------------------------

    train_dataset, valid_dataset, test_dataset = (
        build_datasets()
    )

    train_loader, valid_loader, test_loader = (
        create_dataloaders(
            train_dataset,
            valid_dataset,
            test_dataset,
        )
    )

    # --------------------------------------------------------
    # CLASS WEIGHTS
    # --------------------------------------------------------

    class_weights = calculate_class_weights(
        train_dataset
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = build_model(
        num_classes=len(train_dataset.classes)
    )

    # --------------------------------------------------------
    # LOSS
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=LABEL_SMOOTHING,
    )

    # --------------------------------------------------------
    # OPTIMIZER
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # --------------------------------------------------------
    # LR SCHEDULER
    # --------------------------------------------------------

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
        min_lr=1e-7,
    )

    # --------------------------------------------------------
    # AMP
    # --------------------------------------------------------

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=USE_AMP and DEVICE.type == "cuda",
    )

    # --------------------------------------------------------
    # SAVE DATASET INFO
    # --------------------------------------------------------

    dataset_info = {
        "data_dir": str(DATA_DIR),
        "train_size": len(train_dataset),
        "valid_size": len(valid_dataset),
        "test_size": len(test_dataset),
        "num_classes": len(train_dataset.classes),
        "classes": train_dataset.classes,
        "class_to_idx": train_dataset.class_to_idx,
        "image_size": IMAGE_SIZE,
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "epochs": NUM_EPOCHS,
        "patience": PATIENCE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "label_smoothing": LABEL_SMOOTHING,
        "seed": SEED,
    }

    with open(
        OUTPUT_DIR / "dataset_info.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            dataset_info,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # TRAINING LOOP
    # --------------------------------------------------------

    history = []

    best_valid_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0

    print("\n")
    print("=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)

    for epoch in range(
        1,
        NUM_EPOCHS + 1,
    ):

        print(
            f"\nEpoch {epoch}/{NUM_EPOCHS}"
        )

        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        train_loss, train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
        )

        # ----------------------------------------------------
        # VALIDATE
        # ----------------------------------------------------

        valid_loss, valid_metrics, _, _ = evaluate(
            model,
            valid_loader,
            criterion,
        )

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_metrics['accuracy']:.4f} | "
            f"Train Macro-F1: {train_metrics['macro_f1']:.4f}"
        )

        print(
            f"Valid Loss: {valid_loss:.4f} | "
            f"Valid Acc: {valid_metrics['accuracy']:.4f} | "
            f"Valid Macro-F1: {valid_metrics['macro_f1']:.4f}"
        )

        print(
            f"Learning Rate: {current_lr:.7f}"
        )

        # ----------------------------------------------------
        # HISTORY
        # ----------------------------------------------------

        history.append({
            "epoch": epoch,

            "train_loss": train_loss,
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_precision": train_metrics[
                "macro_precision"
            ],
            "train_macro_recall": train_metrics[
                "macro_recall"
            ],
            "train_macro_f1": train_metrics[
                "macro_f1"
            ],

            "valid_loss": valid_loss,
            "valid_accuracy": valid_metrics["accuracy"],
            "valid_macro_precision": valid_metrics[
                "macro_precision"
            ],
            "valid_macro_recall": valid_metrics[
                "macro_recall"
            ],
            "valid_macro_f1": valid_metrics[
                "macro_f1"
            ],

            "learning_rate": current_lr,
        })

        # ----------------------------------------------------
        # SCHEDULER
        # ----------------------------------------------------

        scheduler.step(
            valid_metrics["macro_f1"]
        )

        # ----------------------------------------------------
        # BEST MODEL
        # ----------------------------------------------------

        if valid_metrics["macro_f1"] > best_valid_f1:

            best_valid_f1 = valid_metrics[
                "macro_f1"
            ]

            best_epoch = epoch

            epochs_without_improvement = 0

            checkpoint = {
                "epoch": epoch,

                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "best_valid_macro_f1":
                    best_valid_f1,

                "classes":
                    train_dataset.classes,

                "class_to_idx":
                    train_dataset.class_to_idx,

                "image_size":
                    IMAGE_SIZE,
            }

            torch.save(
                checkpoint,
                OUTPUT_DIR / "best_efficientnet_b0.pth",
            )

            print(
                f"*** NEW BEST MODEL "
                f"(validation Macro-F1: "
                f"{best_valid_f1:.4f}) ***"
            )

        else:

            epochs_without_improvement += 1

            print(
                f"No improvement for "
                f"{epochs_without_improvement} "
                f"epoch(s)."
            )

        # ----------------------------------------------------
        # EARLY STOPPING
        # ----------------------------------------------------

        if epochs_without_improvement >= PATIENCE:

            print(
                f"\nEarly stopping triggered "
                f"after {epoch} epochs."
            )

            break

    # --------------------------------------------------------
    # SAVE HISTORY
    # --------------------------------------------------------

    save_curves(history)

    with open(
        OUTPUT_DIR / "training_history.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            history,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # LOAD BEST MODEL
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("LOADING BEST CHECKPOINT")
    print("=" * 70)

    checkpoint = torch.load(
        OUTPUT_DIR / "best_efficientnet_b0.pth",
        map_location=DEVICE,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    print(
        f"Best epoch: {checkpoint['epoch']}"
    )

    print(
        f"Best validation Macro-F1: "
        f"{checkpoint['best_valid_macro_f1']:.4f}"
    )

    # --------------------------------------------------------
    # FINAL TEST
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("FINAL TEST EVALUATION")
    print("=" * 70)

    test_loss, test_metrics, y_true, y_pred = evaluate(
        model,
        test_loader,
        criterion,
    )

    print(
        f"\nTest Loss: {test_loss:.4f}"
    )

    print(
        f"Test Accuracy: "
        f"{test_metrics['accuracy']:.4f}"
    )

    print(
        f"Test Macro Precision: "
        f"{test_metrics['macro_precision']:.4f}"
    )

    print(
        f"Test Macro Recall: "
        f"{test_metrics['macro_recall']:.4f}"
    )

    print(
        f"Test Macro F1: "
        f"{test_metrics['macro_f1']:.4f}"
    )

    # --------------------------------------------------------
    # CLASSIFICATION REPORT
    # --------------------------------------------------------

    report = classification_report(
        y_true,
        y_pred,
        labels=list(
            range(len(train_dataset.classes))
        ),
        target_names=train_dataset.classes,
        digits=4,
        zero_division=0,
    )

    print("\n")
    print("=" * 70)
    print("CLASSIFICATION REPORT")
    print("=" * 70)

    print(report)

    with open(
        OUTPUT_DIR / "classification_report.txt",
        "w",
        encoding="utf-8",
    ) as f:

        f.write(report)

    # --------------------------------------------------------
    # PER-CLASS METRICS CSV
    # --------------------------------------------------------

    precision, recall, f1, support = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=list(
                range(len(train_dataset.classes))
            ),
            zero_division=0,
        )
    )

    per_class_df = pd.DataFrame({
        "class": train_dataset.classes,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
    })

    per_class_df.to_csv(
        OUTPUT_DIR / "per_class_metrics.csv",
        index=False,
    )

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    save_confusion_matrix(
        y_true,
        y_pred,
        train_dataset.classes,
    )

    # --------------------------------------------------------
    # TEST METRICS JSON
    # --------------------------------------------------------

    test_results = {
        "best_epoch": int(best_epoch),
        "best_validation_macro_f1":
            float(best_valid_f1),

        "test_loss":
            float(test_loss),

        "test_accuracy":
            float(test_metrics["accuracy"]),

        "test_macro_precision":
            float(test_metrics["macro_precision"]),

        "test_macro_recall":
            float(test_metrics["macro_recall"]),

        "test_macro_f1":
            float(test_metrics["macro_f1"]),

        "classes":
            train_dataset.classes,
    }

    with open(
        OUTPUT_DIR / "test_metrics.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            test_results,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # FINISHED
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print(
        f"\nBest epoch: {best_epoch}"
    )

    print(
        f"Best validation Macro-F1: "
        f"{best_valid_f1:.4f}"
    )

    print(
        f"Test Accuracy: "
        f"{test_metrics['accuracy']:.4f}"
    )

    print(
        f"Test Macro-F1: "
        f"{test_metrics['macro_f1']:.4f}"
    )

    print(
        f"\nResults saved to:"
        f"\n{OUTPUT_DIR}"
    )


# ============================================================
# WINDOWS-SAFE ENTRY POINT
# ============================================================

if __name__ == "__main__":

    freeze_support()

    main()