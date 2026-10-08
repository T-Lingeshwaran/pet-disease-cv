from pathlib import Path
import json
import csv

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import EfficientNet_B0_Weights

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
)

from tqdm import tqdm


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = Path("data/eye_face/combined")
OUTPUT_DIR = Path("outputs/eye_combined")

IMAGE_SIZE = 224
BATCH_SIZE = 16
NUM_EPOCHS = 50

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

# Stop training if validation macro-F1 does not improve
EARLY_STOPPING_PATIENCE = 7    
# Number of DataLoader workers.
# 0 is safest on Windows.
NUM_WORKERS = 0

SEED = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================

torch.manual_seed(SEED)
np.random.seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

USE_AMP = DEVICE.type == "cuda"

print("=" * 70)
print("COMBINED DOG + CAT EYE DISEASE CLASSIFIER")
print("=" * 70)
print(f"Device: {DEVICE}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

print(f"Dataset: {DATA_DIR}")
print(f"Output:  {OUTPUT_DIR}")
print("=" * 70)


# ============================================================
# CHECK DATASET
# ============================================================

required_dirs = [
    DATA_DIR / "train",
    DATA_DIR / "valid",
    DATA_DIR / "test",
]

for directory in required_dirs:
    if not directory.exists():
        raise FileNotFoundError(
            f"Required dataset directory does not exist:\n{directory}"
        )


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# IMAGE TRANSFORMS
# ============================================================

weights = EfficientNet_B0_Weights.DEFAULT

# Training augmentation
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.RandomHorizontalFlip(p=0.5),

    transforms.RandomRotation(
        degrees=10
    ),

    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15,
        saturation=0.10,
        hue=0.02,
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# Validation/test:
# NO random augmentation
eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# ============================================================
# DATASETS
# ============================================================

print("\nLoading datasets...")

train_dataset = datasets.ImageFolder(
    DATA_DIR / "train",
    transform=train_transform,
)

valid_dataset = datasets.ImageFolder(
    DATA_DIR / "valid",
    transform=eval_transform,
)

test_dataset = datasets.ImageFolder(
    DATA_DIR / "test",
    transform=eval_transform,
)


# Make sure class ordering is identical
if train_dataset.classes != valid_dataset.classes:
    raise RuntimeError(
        "Train and validation class names/order do not match."
    )

if train_dataset.classes != test_dataset.classes:
    raise RuntimeError(
        "Train and test class names/order do not match."
    )


class_names = train_dataset.classes
num_classes = len(class_names)


print("\nClasses:")
for index, class_name in enumerate(class_names):
    print(f"  {index}: {class_name}")

print(f"\nNumber of classes: {num_classes}")

print(f"Training images:   {len(train_dataset)}")
print(f"Validation images: {len(valid_dataset)}")
print(f"Test images:       {len(test_dataset)}")


# ============================================================
# DATA LOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=(DEVICE.type == "cuda"),
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=(DEVICE.type == "cuda"),
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=(DEVICE.type == "cuda"),
)


# ============================================================
# CLASS COUNTS
# ============================================================

class_counts = np.zeros(num_classes, dtype=np.int64)

for _, label in train_dataset.samples:
    class_counts[label] += 1


print("\nTraining class distribution:")

for index, class_name in enumerate(class_names):
    print(
        f"  {class_name:25s}: "
        f"{class_counts[index]}"
    )


# ============================================================
# CLASS WEIGHTS
# ============================================================
#
# The combined dataset is strongly imbalanced.
#
# We use square-root inverse-frequency weighting.
# This gives minority classes more importance without making
# the smallest classes overwhelmingly dominant.
# ============================================================

class_weights = 1.0 / np.sqrt(class_counts)

# Normalize weights so mean weight = 1
class_weights = class_weights / class_weights.mean()

class_weights_tensor = torch.tensor(
    class_weights,
    dtype=torch.float32,
    device=DEVICE,
)

print("\nClass weights:")

for index, class_name in enumerate(class_names):
    print(
        f"  {class_name:25s}: "
        f"{class_weights[index]:.4f}"
    )


# ============================================================
# MODEL
# ============================================================

print("\nLoading EfficientNet-B0...")

model = models.efficientnet_b0(
    weights=weights
)


# ------------------------------------------------------------
# Replace classifier
# ------------------------------------------------------------

in_features = model.classifier[1].in_features

model.classifier[1] = nn.Linear(
    in_features,
    num_classes,
)


model = model.to(DEVICE)


# ============================================================
# LOSS
# ============================================================

criterion = nn.CrossEntropyLoss(
    weight=class_weights_tensor,
    label_smoothing=0.05,
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)


# ============================================================
# LR SCHEDULER
# ============================================================

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=7,
)


# ============================================================
# AMP SCALER
# ============================================================

scaler = torch.amp.GradScaler(
    "cuda",
    enabled=USE_AMP,
)


# ============================================================
# HELPER: CALCULATE METRICS
# ============================================================

def calculate_metrics(
    y_true,
    y_pred,
    y_prob,
):
    """
    Calculate classification metrics.
    """

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        )
    )

    weighted_precision, weighted_recall, weighted_f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        )
    )

    metrics = {
        "accuracy": float(accuracy),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
    }

    # Multiclass ROC-AUC
    try:
        auc = roc_auc_score(
            y_true,
            y_prob,
            multi_class="ovr",
            average="macro",
        )

        metrics["macro_roc_auc_ovr"] = float(auc)

    except ValueError:
        metrics["macro_roc_auc_ovr"] = None

    return metrics


# ============================================================
# TRAINING FUNCTION
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

    all_labels = []
    all_predictions = []

    progress = tqdm(
        loader,
        desc="Training",
        leave=False,
    )

    for images, labels in progress:

        images = images.to(
            DEVICE,
            non_blocking=True,
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        # Mixed precision
        with torch.amp.autocast(
            device_type=DEVICE.type,
            enabled=USE_AMP,
        ):

            outputs = model(images)

            loss = criterion(
                outputs,
                labels,
            )

        scaler.scale(loss).backward()

        scaler.step(optimizer)

        scaler.update()

        running_loss += loss.item()

        predictions = outputs.argmax(
            dim=1
        )

        all_labels.extend(
            labels.detach()
            .cpu()
            .numpy()
        )

        all_predictions.extend(
            predictions.detach()
            .cpu()
            .numpy()
        )

        progress.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    epoch_loss = (
        running_loss / len(loader)
    )

    train_y_true = np.array(all_labels)
    train_y_pred = np.array(all_predictions)

    precision, recall, f1, _ = precision_recall_fscore_support(
    train_y_true,
    train_y_pred,
    average="macro",
    zero_division=0,
)

    metrics = {
    "accuracy": float(
        accuracy_score(
            train_y_true,
            train_y_pred,
        )
    ),
    "macro_precision": float(precision),
    "macro_recall": float(recall),
    "macro_f1": float(f1),
}

    return epoch_loss, metrics


# ============================================================
# VALIDATION FUNCTION
# ============================================================

def evaluate(
    model,
    loader,
    criterion,
):
    model.eval()

    running_loss = 0.0

    all_labels = []
    all_predictions = []
    all_probabilities = []

    with torch.no_grad():

        progress = tqdm(
            loader,
            desc="Validation",
            leave=False,
        )

        for images, labels in progress:

            images = images.to(
                DEVICE,
                non_blocking=True,
            )

            labels = labels.to(
                DEVICE,
                non_blocking=True,
            )

            with torch.amp.autocast(
                device_type=DEVICE.type,
                enabled=USE_AMP,
            ):

                outputs = model(images)

                loss = criterion(
                    outputs,
                    labels,
                )

            probabilities = torch.softmax(
                outputs,
                dim=1,
            )

            predictions = outputs.argmax(
                dim=1
            )

            running_loss += loss.item()

            all_labels.extend(
                labels.cpu().numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_probabilities.extend(
                probabilities.cpu().numpy()
            )

    y_true = np.array(
        all_labels
    )

    y_pred = np.array(
        all_predictions
    )

    y_prob = np.array(
        all_probabilities
    )

    epoch_loss = (
        running_loss / len(loader)
    )

    metrics = calculate_metrics(
        y_true,
        y_pred,
        y_prob,
    )

    return (
        epoch_loss,
        metrics,
        y_true,
        y_pred,
        y_prob,
    )


# ============================================================
# TRAINING HISTORY
# ============================================================

history = {
    "epoch": [],
    "train_loss": [],
    "train_accuracy": [],
    "train_macro_f1": [],
    "valid_loss": [],
    "valid_accuracy": [],
    "valid_macro_f1": [],
    "valid_macro_roc_auc": [],
    "learning_rate": [],
}


# ============================================================
# BEST MODEL TRACKING
# ============================================================

best_val_f1 = -float("inf")

epochs_without_improvement = 0

best_model_path = (
    OUTPUT_DIR /
    "best_efficientnet_b0.pth"
)


# ============================================================
# TRAINING LOOP
# ============================================================

print("\n")
print("=" * 70)
print("STARTING TRAINING")
print("=" * 70)


for epoch in range(NUM_EPOCHS):

    print(
        f"\nEpoch {epoch + 1}/{NUM_EPOCHS}"
    )

    print("-" * 50)

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    train_loss, train_metrics = (
        train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
        )
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    (
        valid_loss,
        valid_metrics,
        _,
        _,
        _,
    ) = evaluate(
        model,
        valid_loader,
        criterion,
    )

    # --------------------------------------------------------
    # Scheduler
    # --------------------------------------------------------

    scheduler.step(
        valid_metrics["macro_f1"]
    )

    current_lr = optimizer.param_groups[0][
        "lr"
    ]

    # --------------------------------------------------------
    # Store history
    # --------------------------------------------------------

    history["epoch"].append(
        epoch + 1
    )

    history["train_loss"].append(
        train_loss
    )

    history["train_accuracy"].append(
        train_metrics["accuracy"]
    )

    history["train_macro_f1"].append(
        train_metrics["macro_f1"]
    )

    history["valid_loss"].append(
        valid_loss
    )

    history["valid_accuracy"].append(
        valid_metrics["accuracy"]
    )

    history["valid_macro_f1"].append(
        valid_metrics["macro_f1"]
    )

    history["valid_macro_roc_auc"].append(
        valid_metrics["macro_roc_auc_ovr"]
    )

    history["learning_rate"].append(
        current_lr
    )

    # --------------------------------------------------------
    # Print metrics
    # --------------------------------------------------------

    print(
        f"\nTrain loss:     {train_loss:.4f}"
    )

    print(
        f"Train accuracy: {train_metrics['accuracy']:.4f}"
    )

    print(
        f"Train macro-F1: {train_metrics['macro_f1']:.4f}"
    )

    print(
        f"Valid loss:     {valid_loss:.4f}"
    )

    print(
        f"Valid accuracy: {valid_metrics['accuracy']:.4f}"
    )

    print(
        f"Valid macro-F1: {valid_metrics['macro_f1']:.4f}"
    )

    if valid_metrics["macro_roc_auc_ovr"] is not None:

        print(
            f"Valid ROC-AUC:  "
            f"{valid_metrics['macro_roc_auc_ovr']:.4f}"
        )

    print(
        f"Learning rate:  {current_lr:.2e}"
    )

    # --------------------------------------------------------
    # Save best model based on macro-F1
    # --------------------------------------------------------

    if (
        valid_metrics["macro_f1"]
        > best_val_f1
    ):

        best_val_f1 = (
            valid_metrics["macro_f1"]
        )

        epochs_without_improvement = 0

        checkpoint = {
            "model_state_dict":
                model.state_dict(),

            "classes":
                class_names,

            "num_classes":
                num_classes,

            "image_size":
                IMAGE_SIZE,

            "mean":
                [0.485, 0.456, 0.406],

            "std":
                [0.229, 0.224, 0.225],

            "best_val_macro_f1":
                best_val_f1,

            "epoch":
                epoch + 1,
        }

        torch.save(
            checkpoint,
            best_model_path,
        )

        print(
            "\n*** Saved new best model ***"
        )

        print(
            f"Path: {best_model_path}"
        )

    else:

        epochs_without_improvement += 1

        print(
            f"\nNo improvement "
            f"({epochs_without_improvement}/"
            f"{EARLY_STOPPING_PATIENCE})"
        )

    # --------------------------------------------------------
    # Early stopping
    # --------------------------------------------------------

    if (
        epochs_without_improvement
        >= EARLY_STOPPING_PATIENCE
    ):

        print(
            "\nEarly stopping."
        )

        break


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

history_csv = (
    OUTPUT_DIR /
    "training_history.csv"
)

with open(
    history_csv,
    "w",
    newline="",
) as file:

    writer = csv.writer(file)

    writer.writerow(
        history.keys()
    )

    for row in zip(
        *history.values()
    ):

        writer.writerow(row)


history_json = (
    OUTPUT_DIR /
    "training_history.json"
)

with open(
    history_json,
    "w",
) as file:

    json.dump(
        history,
        file,
        indent=4,
    )


# ============================================================
# PLOT TRAINING CURVES
# ============================================================

epochs = history["epoch"]


plt.figure(figsize=(8, 5))

plt.plot(
    epochs,
    history["train_loss"],
    label="Train Loss",
)

plt.plot(
    epochs,
    history["valid_loss"],
    label="Validation Loss",
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("EfficientNet-B0 Training and Validation Loss")
plt.legend()
plt.grid(True)

plt.tight_layout()

loss_plot = (
    OUTPUT_DIR /
    "loss_curve.png"
)

plt.savefig(
    loss_plot,
    dpi=200,
)

plt.close()


# ------------------------------------------------------------
# Accuracy
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    epochs,
    history["train_accuracy"],
    label="Train Accuracy",
)

plt.plot(
    epochs,
    history["valid_accuracy"],
    label="Validation Accuracy",
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("EfficientNet-B0 Accuracy")
plt.legend()
plt.grid(True)

plt.tight_layout()

accuracy_plot = (
    OUTPUT_DIR /
    "accuracy_curve.png"
)

plt.savefig(
    accuracy_plot,
    dpi=200,
)

plt.close()


# ------------------------------------------------------------
# Macro-F1
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    epochs,
    history["train_macro_f1"],
    label="Train Macro-F1",
)

plt.plot(
    epochs,
    history["valid_macro_f1"],
    label="Validation Macro-F1",
)

plt.xlabel("Epoch")
plt.ylabel("Macro-F1")
plt.title("EfficientNet-B0 Macro-F1")
plt.legend()
plt.grid(True)

plt.tight_layout()

f1_plot = (
    OUTPUT_DIR /
    "macro_f1_curve.png"
)

plt.savefig(
    f1_plot,
    dpi=200,
)

plt.close()


# ============================================================
# LOAD BEST MODEL
# ============================================================

print("\n")
print("=" * 70)
print("LOADING BEST MODEL")
print("=" * 70)

checkpoint = torch.load(
    best_model_path,
    map_location=DEVICE,
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print(
    f"Best validation macro-F1: "
    f"{checkpoint['best_val_macro_f1']:.4f}"
)

print(
    f"Best epoch: "
    f"{checkpoint['epoch']}"
)


# ============================================================
# FINAL TEST EVALUATION
# ============================================================

print("\n")
print("=" * 70)
print("FINAL TEST EVALUATION")
print("=" * 70)


(
    test_loss,
    test_metrics,
    test_y_true,
    test_y_pred,
    test_y_prob,
) = evaluate(
    model,
    test_loader,
    criterion,
)


print(
    f"\nTest loss:      {test_loss:.4f}"
)

print(
    f"Test accuracy:  "
    f"{test_metrics['accuracy']:.4f}"
)

print(
    f"Test macro-F1:  "
    f"{test_metrics['macro_f1']:.4f}"
)

print(
    f"Test macro-recall: "
    f"{test_metrics['macro_recall']:.4f}"
)

if test_metrics["macro_roc_auc_ovr"] is not None:

    print(
        f"Test ROC-AUC:   "
        f"{test_metrics['macro_roc_auc_ovr']:.4f}"
    )


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    test_y_true,
    test_y_pred,
    target_names=class_names,
    digits=4,
    zero_division=0,
)


print("\nClassification Report:")
print(report)


report_path = (
    OUTPUT_DIR /
    "classification_report.txt"
)

with open(
    report_path,
    "w",
) as file:

    file.write(
        "Combined Dog + Cat Eye Classifier\n"
    )

    file.write(
        "EfficientNet-B0\n\n"
    )

    file.write(
        report
    )


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    test_y_true,
    test_y_pred,
    labels=np.arange(num_classes),
)


fig, ax = plt.subplots(
    figsize=(11, 10)
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names,
)

display.plot(
    ax=ax,
    xticks_rotation=45,
)

ax.set_title(
    "Combined Dog + Cat Eye Classifier\nTest Confusion Matrix"
)

plt.tight_layout()

confusion_path = (
    OUTPUT_DIR /
    "confusion_matrix.png"
)

plt.savefig(
    confusion_path,
    dpi=200,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# PER-CLASS METRICS
# ============================================================

precision, recall, f1, support = (
    precision_recall_fscore_support(
        test_y_true,
        test_y_pred,
        labels=np.arange(num_classes),
        zero_division=0,
    )
)


per_class_metrics = {}

for index, class_name in enumerate(
    class_names
):

    per_class_metrics[class_name] = {
        "precision": float(
            precision[index]
        ),

        "recall": float(
            recall[index]
        ),

        "f1": float(
            f1[index]
        ),

        "support": int(
            support[index]
        ),
    }


# ============================================================
# CAT/DOG SOURCE-SPECIFIC METRICS
# ============================================================
#
# Combined test images were prefixed:
#
# cat_...
# dog_...
#
# This lets us see how the combined model behaves on each
# source dataset separately.
# ============================================================

test_paths = [
    path
    for path, _ in test_dataset.samples
]


cat_indices = [
    index
    for index, path in enumerate(test_paths)
    if Path(path).name.startswith("cat_")
]


dog_indices = [
    index
    for index, path in enumerate(test_paths)
    if Path(path).name.startswith("dog_")
]


def subset_metrics(
    indices,
    y_true,
    y_pred,
    y_prob,
):

    if len(indices) == 0:
        return None

    subset_true = y_true[
        indices
    ]

    subset_pred = y_pred[
        indices
    ]

    subset_prob = y_prob[
        indices
    ]

    return calculate_metrics(
        subset_true,
        subset_pred,
        subset_prob,
    )


cat_metrics = subset_metrics(
    cat_indices,
    test_y_true,
    test_y_pred,
    test_y_prob,
)


dog_metrics = subset_metrics(
    dog_indices,
    test_y_true,
    test_y_pred,
    test_y_prob,
)


print("\n")
print("=" * 70)
print("SOURCE-SPECIFIC TEST METRICS")
print("=" * 70)


if cat_metrics is not None:

    print(
        f"\nCat test samples: "
        f"{len(cat_indices)}"
    )

    print(
        f"Cat accuracy: "
        f"{cat_metrics['accuracy']:.4f}"
    )

    print(
        f"Cat macro-F1: "
        f"{cat_metrics['macro_f1']:.4f}"
    )


if dog_metrics is not None:

    print(
        f"\nDog test samples: "
        f"{len(dog_indices)}"
    )

    print(
        f"Dog accuracy: "
        f"{dog_metrics['accuracy']:.4f}"
    )

    print(
        f"Dog macro-F1: "
        f"{dog_metrics['macro_f1']:.4f}"
    )


# ============================================================
# SAVE ALL METRICS
# ============================================================

all_metrics = {
    "model": "EfficientNet-B0",
    "dataset": "combined dog + cat",
    "num_classes": num_classes,
    "classes": class_names,

    "train_samples": len(train_dataset),
    "validation_samples": len(valid_dataset),
    "test_samples": len(test_dataset),

    "best_epoch":
        checkpoint["epoch"],

    "best_validation_macro_f1":
        checkpoint["best_val_macro_f1"],

    "test_loss":
        float(test_loss),

    "test_metrics":
        test_metrics,

    "per_class_metrics":
        per_class_metrics,

    "cat_test_samples":
        len(cat_indices),

    "cat_metrics":
        cat_metrics,

    "dog_test_samples":
        len(dog_indices),

    "dog_metrics":
        dog_metrics,
}


metrics_path = (
    OUTPUT_DIR /
    "test_metrics.json"
)

with open(
    metrics_path,
    "w",
) as file:

    json.dump(
        all_metrics,
        file,
        indent=4,
    )


# ============================================================
# SAVE DATASET INFORMATION
# ============================================================

dataset_info = {
    "classes": class_names,

    "class_counts": {
        class_names[i]:
            int(class_counts[i])
        for i in range(num_classes)
    },

    "class_weights": {
        class_names[i]:
            float(class_weights[i])
        for i in range(num_classes)
    },

    "train_samples":
        len(train_dataset),

    "validation_samples":
        len(valid_dataset),

    "test_samples":
        len(test_dataset),
}


dataset_info_path = (
    OUTPUT_DIR /
    "dataset_info.json"
)

with open(
    dataset_info_path,
    "w",
) as file:

    json.dump(
        dataset_info,
        file,
        indent=4,
    )


# ============================================================
# FINAL OUTPUT SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print("\nOutput files:")

print(
    f"Best model:          {best_model_path}"
)

print(
    f"Training history:    {history_csv}"
)

print(
    f"Training JSON:       {history_json}"
)

print(
    f"Classification:      {report_path}"
)

print(
    f"Confusion matrix:    {confusion_path}"
)

print(
    f"Loss curve:          {loss_plot}"
)

print(
    f"Accuracy curve:      {accuracy_plot}"
)

print(
    f"Macro-F1 curve:      {f1_plot}"
)

print(
    f"Test metrics:        {metrics_path}"
)

print(
    f"Dataset information: {dataset_info_path}"
)

print("\nFinal test results:")

print(
    f"  Accuracy: "
    f"{test_metrics['accuracy']:.4f}"
)

print(
    f"  Macro-F1: "
    f"{test_metrics['macro_f1']:.4f}"
)

print(
    f"  Macro Recall: "
    f"{test_metrics['macro_recall']:.4f}"
)

if test_metrics["macro_roc_auc_ovr"] is not None:

    print(
        f"  Macro ROC-AUC: "
        f"{test_metrics['macro_roc_auc_ovr']:.4f}"
    )

print("\nDone.")