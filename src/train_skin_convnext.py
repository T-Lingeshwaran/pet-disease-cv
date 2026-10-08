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

from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import ConvNeXt_Tiny_Weights

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    precision_score,
    classification_report,
    confusion_matrix,
)


# ============================================================
# CONFIG
# ============================================================

DATA_DIR = "data/skin_coat/combined_clean"
OUTPUT_DIR = "outputs/skin_convnext_tiny"

IMAGE_SIZE = 224
BATCH_SIZE = 16

NUM_EPOCHS = 50
PATIENCE = 7

LR = 1e-4
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.05

# Windows-safe
NUM_WORKERS = 0

SEED = 42

USE_AMP = True

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def seed_everything(seed=42):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


seed_everything(SEED)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("CONVNEXT-TINY SKIN CONDITION CLASSIFIER")
print("=" * 70)

print(f"Device      : {DEVICE}")
print(f"Dataset     : {DATA_DIR}")
print(f"Image size  : {IMAGE_SIZE}")
print(f"Batch size  : {BATCH_SIZE}")
print(f"Epochs      : {NUM_EPOCHS}")
print("=" * 70)


# ============================================================
# DATASET CHECK
# ============================================================

for split in ["train", "valid", "test"]:

    split_path = os.path.join(
        DATA_DIR,
        split
    )

    if not os.path.isdir(split_path):

        raise FileNotFoundError(
            f"Missing dataset split: {split_path}"
        )


# ============================================================
# TRANSFORMS
# ============================================================

train_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    transforms.RandomRotation(
        degrees=12
    ),

    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15,
        saturation=0.10,
        hue=0.03
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
        ]
    )
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
        ]
    )
])


# ============================================================
# DATASETS
# ============================================================

train_dataset = datasets.ImageFolder(
    os.path.join(
        DATA_DIR,
        "train"
    ),
    transform=train_transform
)

valid_dataset = datasets.ImageFolder(
    os.path.join(
        DATA_DIR,
        "valid"
    ),
    transform=eval_transform
)

test_dataset = datasets.ImageFolder(
    os.path.join(
        DATA_DIR,
        "test"
    ),
    transform=eval_transform
)


class_names = train_dataset.classes

num_classes = len(
    class_names
)


print("\nClasses:")

for i, name in enumerate(
    class_names
):

    print(
        f"{i:2d}: {name}"
    )


print(
    f"\nNumber of classes: "
    f"{num_classes}"
)


# ============================================================
# VERIFY CLASS MAPPINGS
# ============================================================

if (
    valid_dataset.class_to_idx
    != train_dataset.class_to_idx
):

    raise RuntimeError(
        "Validation class mapping differs from training."
    )


if (
    test_dataset.class_to_idx
    != train_dataset.class_to_idx
):

    raise RuntimeError(
        "Test class mapping differs from training."
    )


# ============================================================
# DATASET SIZES
# ============================================================

print("\nDataset sizes:")

print(
    f"Train: {len(train_dataset)}"
)

print(
    f"Valid: {len(valid_dataset)}"
)

print(
    f"Test : {len(test_dataset)}"
)


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

train_targets = np.array(
    train_dataset.targets
)

class_counts = np.bincount(
    train_targets,
    minlength=num_classes
)


print(
    "\nTraining class distribution:"
)

for idx, count in enumerate(
    class_counts
):

    print(
        f"{class_names[idx]:35s}: {count}"
    )


# ============================================================
# CLASS WEIGHTS
# ============================================================

# Same strategy used in our other models:
# sqrt inverse-frequency weighting.

class_counts_float = (
    class_counts.astype(
        np.float32
    )
)

class_weights = (
    1.0 /
    np.sqrt(
        np.maximum(
            class_counts_float,
            1.0
        )
    )
)

class_weights = (
    class_weights /
    class_weights.mean()
)


class_weights = torch.tensor(
    class_weights,
    dtype=torch.float32,
    device=DEVICE
)


print(
    "\nClass weights:"
)

for i, weight in enumerate(
    class_weights
):

    print(
        f"{class_names[i]:35s}: "
        f"{weight.item():.4f}"
    )


# ============================================================
# DATALOADERS
# ============================================================

pin_memory = (
    DEVICE.type == "cuda"
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=pin_memory
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=pin_memory
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=pin_memory
)


# ============================================================
# MODEL
# ============================================================

print(
    "\nLoading ImageNet-pretrained "
    "ConvNeXt-Tiny..."
)

weights = (
    ConvNeXt_Tiny_Weights.DEFAULT
)

model = models.convnext_tiny(
    weights=weights
)


# Torchvision ConvNeXt classifier:
#
# model.classifier =
# Sequential(
#   LayerNorm2d
#   Flatten
#   Linear(768, 1000)
# )
#
# Replace final classification layer.

in_features = (
    model.classifier[-1].in_features
)


model.classifier[-1] = nn.Linear(
    in_features,
    num_classes
)


model = model.to(
    DEVICE
)


print(
    "ConvNeXt-Tiny loaded."
)

print(
    f"Classifier input features: "
    f"{in_features}"
)

print(
    f"Classifier output classes : "
    f"{num_classes}"
)


# ============================================================
# LOSS
# ============================================================

criterion = nn.CrossEntropyLoss(

    weight=class_weights,

    label_smoothing=LABEL_SMOOTHING
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(

    model.parameters(),

    lr=LR,

    weight_decay=WEIGHT_DECAY
)


# ============================================================
# LR SCHEDULER
# ============================================================

scheduler = (
    torch.optim.lr_scheduler
    .ReduceLROnPlateau(

        optimizer,

        mode="max",

        factor=0.5,

        patience=2
    )
)


# ============================================================
# AMP
# ============================================================

if (
    USE_AMP
    and DEVICE.type == "cuda"
):

    scaler = torch.amp.GradScaler(
        "cuda"
    )

else:

    scaler = None


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    y_true,
    y_pred
):

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    macro_recall = recall_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    macro_precision = precision_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    return {

        "accuracy":
            accuracy,

        "macro_f1":
            macro_f1,

        "macro_recall":
            macro_recall,

        "macro_precision":
            macro_precision
    }


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch():

    model.train()

    running_loss = 0.0

    all_targets = []
    all_predictions = []


    for images, targets in train_loader:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        targets = targets.to(
            DEVICE,
            non_blocking=True
        )


        optimizer.zero_grad(
            set_to_none=True
        )


        if scaler is not None:

            with torch.amp.autocast(

                device_type="cuda",

                dtype=torch.float16
            ):

                outputs = model(
                    images
                )

                loss = criterion(
                    outputs,
                    targets
                )


            scaler.scale(
                loss
            ).backward()


            scaler.step(
                optimizer
            )

            scaler.update()


        else:

            outputs = model(
                images
            )

            loss = criterion(
                outputs,
                targets
            )

            loss.backward()

            optimizer.step()


        running_loss += (
            loss.item()
            * images.size(0)
        )


        predictions = torch.argmax(
            outputs,
            dim=1
        )


        all_targets.extend(
            targets.detach()
            .cpu()
            .numpy()
        )


        all_predictions.extend(
            predictions.detach()
            .cpu()
            .numpy()
        )


    epoch_loss = (
        running_loss /
        len(train_dataset)
    )


    metrics = calculate_metrics(
        all_targets,
        all_predictions
    )

    metrics["loss"] = (
        epoch_loss
    )


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

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        targets = targets.to(
            DEVICE,
            non_blocking=True
        )


        if (
            USE_AMP
            and DEVICE.type == "cuda"
        ):

            with torch.amp.autocast(

                device_type="cuda",

                dtype=torch.float16
            ):

                outputs = model(
                    images
                )

                loss = criterion(
                    outputs,
                    targets
                )

        else:

            outputs = model(
                images
            )

            loss = criterion(
                outputs,
                targets
            )


        running_loss += (
            loss.item()
            * images.size(0)
        )


        predictions = torch.argmax(
            outputs,
            dim=1
        )


        all_targets.extend(
            targets.cpu().numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )


    epoch_loss = (
        running_loss /
        len(valid_dataset)
    )


    metrics = calculate_metrics(
        all_targets,
        all_predictions
    )

    metrics["loss"] = (
        epoch_loss
    )


    return metrics


# ============================================================
# TRAINING
# ============================================================

best_val_f1 = -1.0

best_epoch = 0

epochs_without_improvement = 0

history = []


best_model_path = os.path.join(
    OUTPUT_DIR,
    "best_model.pth"
)


print("\n" + "=" * 70)
print("STARTING TRAINING")
print("=" * 70)


for epoch in range(
    1,
    NUM_EPOCHS + 1
):

    print(
        f"\nEpoch "
        f"{epoch}/{NUM_EPOCHS}"
    )


    train_metrics = (
        train_one_epoch()
    )


    valid_metrics = (
        validate()
    )


    scheduler.step(
        valid_metrics[
            "macro_f1"
        ]
    )


    current_lr = (
        optimizer
        .param_groups[0]["lr"]
    )


    row = {

        "epoch":
            epoch,

        "lr":
            current_lr,

        "train_loss":
            train_metrics["loss"],

        "train_accuracy":
            train_metrics["accuracy"],

        "train_macro_f1":
            train_metrics["macro_f1"],

        "train_macro_recall":
            train_metrics["macro_recall"],

        "train_macro_precision":
            train_metrics["macro_precision"],

        "val_loss":
            valid_metrics["loss"],

        "val_accuracy":
            valid_metrics["accuracy"],

        "val_macro_f1":
            valid_metrics["macro_f1"],

        "val_macro_recall":
            valid_metrics["macro_recall"],

        "val_macro_precision":
            valid_metrics["macro_precision"],
    }


    history.append(
        row
    )


    print(
        f"Train | "
        f"Loss "
        f"{train_metrics['loss']:.4f} | "
        f"Acc "
        f"{train_metrics['accuracy']:.4f} | "
        f"F1 "
        f"{train_metrics['macro_f1']:.4f}"
    )


    print(
        f"Valid | "
        f"Loss "
        f"{valid_metrics['loss']:.4f} | "
        f"Acc "
        f"{valid_metrics['accuracy']:.4f} | "
        f"F1 "
        f"{valid_metrics['macro_f1']:.4f}"
    )


    print(
        f"LR: "
        f"{current_lr:.7f}"
    )


    # ========================================================
    # SAVE BEST MODEL
    # ========================================================

    if (
        valid_metrics["macro_f1"]
        > best_val_f1
    ):

        best_val_f1 = (
            valid_metrics["macro_f1"]
        )

        best_epoch = epoch

        epochs_without_improvement = 0


        torch.save(

            {

                "epoch":
                    epoch,

                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "scheduler_state_dict":
                    scheduler.state_dict(),

                "best_val_macro_f1":
                    best_val_f1,

                "class_names":
                    class_names,

                "class_to_idx":
                    train_dataset.class_to_idx,

                "image_size":
                    IMAGE_SIZE,

                "model_name":
                    "ConvNeXt-Tiny"
            },

            best_model_path
        )


        print(
            f"*** BEST MODEL SAVED "
            f"(Val Macro-F1 = "
            f"{best_val_f1:.4f}) ***"
        )


    else:

        epochs_without_improvement += 1


        print(
            f"No improvement "
            f"({epochs_without_improvement}/"
            f"{PATIENCE})"
        )


    # ========================================================
    # EARLY STOPPING
    # ========================================================

    if (
        epochs_without_improvement
        >= PATIENCE
    ):

        print(
            "\nEarly stopping triggered."
        )

        break


# ============================================================
# SAVE HISTORY
# ============================================================

history_df = pd.DataFrame(
    history
)


history_df.to_csv(

    os.path.join(
        OUTPUT_DIR,
        "training_history.csv"
    ),

    index=False
)


# ============================================================
# CURVES
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.plot(

    history_df["epoch"],

    history_df[
        "train_macro_f1"
    ],

    label="Train Macro-F1"
)

plt.plot(

    history_df["epoch"],

    history_df[
        "val_macro_f1"
    ],

    label="Validation Macro-F1"
)


plt.xlabel("Epoch")
plt.ylabel("Macro-F1")

plt.title(
    "ConvNeXt-Tiny Skin Classification - Macro-F1"
)

plt.legend()

plt.grid(True)

plt.tight_layout()


plt.savefig(

    os.path.join(
        OUTPUT_DIR,
        "macro_f1_curve.png"
    ),

    dpi=200
)

plt.close()


plt.figure(
    figsize=(10, 6)
)

plt.plot(

    history_df["epoch"],

    history_df["train_loss"],

    label="Train Loss"
)

plt.plot(

    history_df["epoch"],

    history_df["val_loss"],

    label="Validation Loss"
)


plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title(
    "ConvNeXt-Tiny Skin Classification - Loss"
)

plt.legend()

plt.grid(True)

plt.tight_layout()


plt.savefig(

    os.path.join(
        OUTPUT_DIR,
        "loss_curve.png"
    ),

    dpi=200
)

plt.close()


# ============================================================
# LOAD BEST MODEL
# ============================================================

print("\n" + "=" * 70)
print("LOADING BEST MODEL")
print("=" * 70)


checkpoint = torch.load(

    best_model_path,

    map_location=DEVICE
)


model.load_state_dict(
    checkpoint[
        "model_state_dict"
    ]
)


print(
    f"Best epoch: "
    f"{checkpoint['epoch']}"
)


print(
    f"Best validation Macro-F1: "
    f"{checkpoint['best_val_macro_f1']:.4f}"
)


# ============================================================
# TEST
# ============================================================

@torch.no_grad()
def evaluate_test():

    model.eval()

    all_targets = []
    all_predictions = []
    all_probabilities = []


    for images, targets in test_loader:

        images = images.to(
            DEVICE,
            non_blocking=True
        )


        if (
            USE_AMP
            and DEVICE.type == "cuda"
        ):

            with torch.amp.autocast(

                device_type="cuda",

                dtype=torch.float16
            ):

                outputs = model(
                    images
                )

        else:

            outputs = model(
                images
            )


        probabilities = torch.softmax(
            outputs,
            dim=1
        )


        predictions = torch.argmax(
            probabilities,
            dim=1
        )


        all_targets.extend(
            targets.numpy()
        )


        all_predictions.extend(
            predictions
            .cpu()
            .numpy()
        )


        all_probabilities.extend(
            probabilities
            .cpu()
            .numpy()
        )


    return (

        np.array(all_targets),

        np.array(all_predictions),

        np.array(all_probabilities)
    )


y_true, y_pred, y_prob = (
    evaluate_test()
)


# ============================================================
# TEST METRICS
# ============================================================

test_metrics = calculate_metrics(
    y_true,
    y_pred
)


print("\n" + "=" * 70)
print("TEST RESULTS")
print("=" * 70)


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

    zero_division=0
)


report_text = classification_report(

    y_true,

    y_pred,

    target_names=class_names,

    zero_division=0
)


print(
    "\nClassification Report:"
)

print(
    report_text
)


with open(

    os.path.join(
        OUTPUT_DIR,
        "classification_report.txt"
    ),

    "w",

    encoding="utf-8"

) as f:

    f.write(
        report_text
    )


with open(

    os.path.join(
        OUTPUT_DIR,
        "classification_report.json"
    ),

    "w",

    encoding="utf-8"

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

    y_true,

    y_pred
)


plt.figure(
    figsize=(13, 10)
)


sns.heatmap(

    cm,

    annot=True,

    fmt="d",

    xticklabels=class_names,

    yticklabels=class_names,

    cmap="Blues"
)


plt.xlabel(
    "Predicted"
)

plt.ylabel(
    "True"
)


plt.title(
    "ConvNeXt-Tiny Skin Classification - "
    "Confusion Matrix"
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
# PER-CLASS METRICS
# ============================================================

per_class_rows = []


for class_name in class_names:

    row = report_dict[
        class_name
    ]


    per_class_rows.append(

        {

            "class":
                class_name,

            "precision":
                row["precision"],

            "recall":
                row["recall"],

            "f1_score":
                row["f1-score"],

            "support":
                row["support"]
        }
    )


per_class_df = pd.DataFrame(
    per_class_rows
)


per_class_df.to_csv(

    os.path.join(
        OUTPUT_DIR,
        "per_class_metrics.csv"
    ),

    index=False
)


# ============================================================
# SAVE FINAL METRICS
# ============================================================

final_metrics = {

    "model":
        "ConvNeXt-Tiny",

    "dataset":
        "skin_coat/combined_clean",

    "best_epoch":
        int(best_epoch),

    "best_validation_macro_f1":
        float(best_val_f1),

    "test_accuracy":
        float(
            test_metrics[
                "accuracy"
            ]
        ),

    "test_macro_precision":
        float(
            test_metrics[
                "macro_precision"
            ]
        ),

    "test_macro_recall":
        float(
            test_metrics[
                "macro_recall"
            ]
        ),

    "test_macro_f1":
        float(
            test_metrics[
                "macro_f1"
            ]
        ),

    "num_classes":
        int(num_classes),

    "image_size":
        IMAGE_SIZE,

    "batch_size":
        BATCH_SIZE,

    "epochs_requested":
        NUM_EPOCHS
}


with open(

    os.path.join(
        OUTPUT_DIR,
        "test_metrics.json"
    ),

    "w",

    encoding="utf-8"

) as f:

    json.dump(

        final_metrics,

        f,

        indent=2
    )


# ============================================================
# CLASS MAPPING
# ============================================================

with open(

    os.path.join(
        OUTPUT_DIR,
        "class_names.json"
    ),

    "w",

    encoding="utf-8"

) as f:

    json.dump(

        {

            "class_names":
                class_names,

            "class_to_idx":
                train_dataset.class_to_idx

        },

        f,

        indent=2
    )


# ============================================================
# CLEANUP
# ============================================================

del model

gc.collect()


if torch.cuda.is_available():

    torch.cuda.empty_cache()


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print(
    f"Best epoch: "
    f"{best_epoch}"
)

print(
    f"Best validation Macro-F1: "
    f"{best_val_f1:.4f}"
)

print(
    f"Test Macro-F1: "
    f"{test_metrics['macro_f1']:.4f}"
)

print(
    f"Test Accuracy: "
    f"{test_metrics['accuracy']:.4f}"
)

print(
    "\nModel saved to:"
)

print(
    best_model_path
)

print(
    "\nResults saved to:"
)

print(
    OUTPUT_DIR
)