import os
import json
import gc

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

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

DATA_DIR = "data/eye_face/combined"
OUTPUT_DIR = "outputs/eye_densenet121"

CHECKPOINT_PATH = os.path.join(
    OUTPUT_DIR,
    "best_model.pth"
)

IMAGE_SIZE = 224
BATCH_SIZE = 16

# Windows-safe
NUM_WORKERS = 0

USE_AMP = True

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("DENSENET-121 EYE MODEL - TEST EVALUATION")
print("=" * 70)

print(f"Device     : {DEVICE}")
print(f"Dataset    : {DATA_DIR}")
print(f"Checkpoint : {CHECKPOINT_PATH}")
print("=" * 70)


# ============================================================
# CHECKPOINT CHECK
# ============================================================

if not os.path.exists(CHECKPOINT_PATH):

    raise FileNotFoundError(
        f"Checkpoint not found:\n"
        f"{CHECKPOINT_PATH}\n\n"
        f"Make sure DenseNet training saved "
        f"best_model.pth before running evaluation."
    )


# ============================================================
# TEST TRANSFORM
# ============================================================

test_transform = transforms.Compose([

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
# TEST DATASET
# ============================================================

test_dataset = datasets.ImageFolder(

    os.path.join(
        DATA_DIR,
        "test"
    ),

    transform=test_transform
)


class_names = test_dataset.classes

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
    f"\nTest images: "
    f"{len(test_dataset)}"
)


# ============================================================
# LOAD CHECKPOINT
# ============================================================

print(
    "\nLoading checkpoint..."
)

checkpoint = torch.load(

    CHECKPOINT_PATH,

    map_location=DEVICE
)


# Prefer the class mapping saved during training
if "class_names" in checkpoint:

    class_names = checkpoint[
        "class_names"
    ]

    num_classes = len(
        class_names
    )


print(
    f"Best training epoch: "
    f"{checkpoint.get('epoch', 'unknown')}"
)

print(
    f"Best validation Macro-F1: "
    f"{checkpoint.get('best_val_macro_f1', 'unknown')}"
)


# ============================================================
# DATALOADER
# ============================================================

test_loader = DataLoader(

    test_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=NUM_WORKERS,

    pin_memory=(
        DEVICE.type == "cuda"
    )
)


# ============================================================
# BUILD DENSENET-121
# ============================================================

print(
    "\nBuilding DenseNet-121..."
)

model = models.densenet121(
    weights=None
)


in_features = (
    model.classifier.in_features
)


model.classifier = nn.Sequential(

    nn.Dropout(
        p=0.30
    ),

    nn.Linear(
        in_features,
        num_classes
    )
)


model.load_state_dict(
    checkpoint[
        "model_state_dict"
    ]
)


model = model.to(
    DEVICE
)


model.eval()


print(
    "DenseNet-121 loaded successfully."
)


# ============================================================
# TEST INFERENCE
# ============================================================

print(
    "\nRunning test evaluation..."
)


all_targets = []
all_predictions = []
all_probabilities = []


with torch.no_grad():

    for batch_idx, (
        images,
        targets
    ) in enumerate(test_loader):

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


        if (
            batch_idx + 1
        ) % 10 == 0:

            print(
                f"Processed "
                f"{min((batch_idx + 1) * BATCH_SIZE, len(test_dataset))}"
                f"/{len(test_dataset)} images"
            )


# ============================================================
# NUMPY
# ============================================================

y_true = np.array(
    all_targets
)

y_pred = np.array(
    all_predictions
)

y_prob = np.array(
    all_probabilities
)


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)

macro_precision = precision_score(
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

macro_f1 = f1_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0
)


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 70)
print("DENSENET-121 TEST RESULTS")
print("=" * 70)

print(
    f"Accuracy        : "
    f"{accuracy:.4f}"
)

print(
    f"Macro Precision : "
    f"{macro_precision:.4f}"
)

print(
    f"Macro Recall    : "
    f"{macro_recall:.4f}"
)

print(
    f"Macro F1        : "
    f"{macro_f1:.4f}"
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
    figsize=(12, 10)
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
    "DenseNet-121 Eye Classification - "
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


    per_class_rows.append({

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
    })


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
# SAVE TEST METRICS
# ============================================================

test_metrics = {

    "model":
        "DenseNet-121",

    "dataset":
        "eye_face/combined",

    "best_epoch":
        int(
            checkpoint.get(
                "epoch",
                -1
            )
        ),

    "best_validation_macro_f1":
        float(
            checkpoint.get(
                "best_val_macro_f1",
                -1
            )
        ),

    "test_accuracy":
        float(
            accuracy
        ),

    "test_macro_precision":
        float(
            macro_precision
        ),

    "test_macro_recall":
        float(
            macro_recall
        ),

    "test_macro_f1":
        float(
            macro_f1
        ),

    "num_classes":
        int(
            num_classes
        ),

    "test_samples":
        int(
            len(test_dataset)
        )
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

        test_metrics,

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
print("EVALUATION COMPLETE")
print("=" * 70)

print(
    f"Best epoch: "
    f"{checkpoint.get('epoch', 'unknown')}"
)

print(
    f"Validation Macro-F1: "
    f"{checkpoint.get('best_val_macro_f1', 'unknown')}"
)

print(
    f"Test Accuracy: "
    f"{accuracy:.4f}"
)

print(
    f"Test Macro-F1: "
    f"{macro_f1:.4f}"
)

print(
    "\nResults saved to:"
)

print(
    OUTPUT_DIR
)