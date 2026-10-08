from pathlib import Path
import json

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

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
MODEL_PATH = Path(
    "outputs/eye_combined/best_efficientnet_b0.pth"
)

OUTPUT_DIR = Path("outputs/eye_combined/evaluation")

IMAGE_SIZE = 224
BATCH_SIZE = 16
NUM_WORKERS = 0


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

USE_AMP = DEVICE.type == "cuda"


print("=" * 70)
print("EFFICIENTNET-B0 EYE MODEL EVALUATION")
print("=" * 70)

print(f"Device: {DEVICE}")

if torch.cuda.is_available():
    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )

print(f"Model: {MODEL_PATH}")
print(f"Test data: {DATA_DIR / 'test'}")

print("=" * 70)


# ============================================================
# CHECK PATHS
# ============================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"\nModel checkpoint not found:\n{MODEL_PATH}\n"
        "\nMake sure your trained .pth file exists."
    )


TEST_DIR = DATA_DIR / "test"

if not TEST_DIR.exists():
    raise FileNotFoundError(
        f"\nTest directory not found:\n{TEST_DIR}"
    )


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# TRANSFORM
# ============================================================

eval_transform = transforms.Compose([
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# ============================================================
# TEST DATASET
# ============================================================

print("\nLoading test dataset...")

test_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=eval_transform,
)

class_names = test_dataset.classes
num_classes = len(class_names)


print("\nClasses:")

for index, class_name in enumerate(
    class_names
):
    print(
        f"  {index}: {class_name}"
    )

print(
    f"\nTotal test images: "
    f"{len(test_dataset)}"
)


test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=(DEVICE.type == "cuda"),
)


# ============================================================
# LOAD CHECKPOINT
# ============================================================

print("\nLoading checkpoint...")

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
)


# ============================================================
# VERIFY CLASSES
# ============================================================

checkpoint_classes = checkpoint.get(
    "classes",
    None
)

if checkpoint_classes is not None:

    print("\nCheckpoint classes:")

    for index, class_name in enumerate(
        checkpoint_classes
    ):
        print(
            f"  {index}: {class_name}"
        )

    if list(checkpoint_classes) != list(
        class_names
    ):

        raise RuntimeError(
            "\nClass mismatch!\n\n"
            f"Checkpoint classes: "
            f"{checkpoint_classes}\n"
            f"Test classes: "
            f"{class_names}"
        )


# ============================================================
# CREATE EFFICIENTNET-B0
# ============================================================

print("\nCreating EfficientNet-B0...")

model = models.efficientnet_b0(
    weights=None
)


in_features = (
    model.classifier[1].in_features
)


model.classifier[1] = nn.Linear(
    in_features,
    num_classes,
)


# ============================================================
# LOAD TRAINED WEIGHTS
# ============================================================

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(DEVICE)

model.eval()


print("Checkpoint loaded successfully.")


if "epoch" in checkpoint:

    print(
        f"Best training epoch: "
        f"{checkpoint['epoch']}"
    )


if "best_val_macro_f1" in checkpoint:

    print(
        f"Best validation macro-F1: "
        f"{checkpoint['best_val_macro_f1']:.4f}"
    )


# ============================================================
# EVALUATION
# ============================================================

print("\n")
print("=" * 70)
print("RUNNING TEST SET")
print("=" * 70)


all_labels = []
all_predictions = []
all_probabilities = []


with torch.no_grad():

    progress = tqdm(
        test_loader,
        desc="Testing"
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

        probabilities = torch.softmax(
            outputs,
            dim=1,
        )

        predictions = outputs.argmax(
            dim=1
        )

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


# ============================================================
# OVERALL METRICS
# ============================================================

accuracy = accuracy_score(
    y_true,
    y_pred,
)


precision, recall, f1, support = (
    precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=np.arange(num_classes),
        zero_division=0,
    )
)


macro_precision = np.mean(
    precision
)

macro_recall = np.mean(
    recall
)

macro_f1 = np.mean(
    f1
)


weighted_precision, weighted_recall, weighted_f1, _ = (
    precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )
)


# ============================================================
# ROC-AUC
# ============================================================

try:

    macro_roc_auc = roc_auc_score(
        y_true,
        y_prob,
        multi_class="ovr",
        average="macro",
    )

except ValueError:

    macro_roc_auc = None


# ============================================================
# PRINT OVERALL RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("FINAL TEST RESULTS")
print("=" * 70)

print(
    f"\nTest samples:       {len(y_true)}"
)

print(
    f"Accuracy:            {accuracy:.4f}"
)

print(
    f"Macro Precision:     {macro_precision:.4f}"
)

print(
    f"Macro Recall:        {macro_recall:.4f}"
)

print(
    f"Macro F1:            {macro_f1:.4f}"
)

print(
    f"Weighted Precision:  {weighted_precision:.4f}"
)

print(
    f"Weighted Recall:     {weighted_recall:.4f}"
)

print(
    f"Weighted F1:         {weighted_f1:.4f}"
)

if macro_roc_auc is not None:

    print(
        f"Macro ROC-AUC:       {macro_roc_auc:.4f}"
    )

else:

    print(
        "Macro ROC-AUC:       Could not calculate"
    )


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    y_true,
    y_pred,
    labels=np.arange(num_classes),
    target_names=class_names,
    digits=4,
    zero_division=0,
)


print("\n")
print("=" * 70)
print("PER-CLASS CLASSIFICATION REPORT")
print("=" * 70)

print(report)


report_path = (
    OUTPUT_DIR /
    "classification_report.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8",
) as file:

    file.write(
        "Combined Dog + Cat Eye Classifier\n"
    )

    file.write(
        "EfficientNet-B0\n\n"
    )

    file.write(report)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=np.arange(num_classes),
)


plt.figure(
    figsize=(11, 10)
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names,
)

display.plot(
    xticks_rotation=45,
    values_format="d",
)

plt.title(
    "EfficientNet-B0\n"
    "Combined Dog + Cat Eye Test Confusion Matrix"
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
# CAT / DOG SPLIT
# ============================================================

test_paths = [
    path
    for path, _ in test_dataset.samples
]


cat_indices = []

dog_indices = []

other_indices = []


for index, path in enumerate(
    test_paths
):

    filename = Path(path).name.lower()

    if filename.startswith("cat_"):

        cat_indices.append(index)

    elif filename.startswith("dog_"):

        dog_indices.append(index)

    else:

        other_indices.append(index)


# ============================================================
# SOURCE METRIC FUNCTION
# ============================================================

def calculate_source_metrics(
    indices,
    source_name,
):

    if len(indices) == 0:

        return None

    source_true = y_true[
        indices
    ]

    source_pred = y_pred[
        indices
    ]

    source_prob = y_prob[
        indices
    ]

    source_accuracy = accuracy_score(
        source_true,
        source_pred,
    )

    source_precision, source_recall, source_f1, _ = (
        precision_recall_fscore_support(
            source_true,
            source_pred,
            labels=np.arange(num_classes),
            average="macro",
            zero_division=0,
        )
    )

    try:

        source_auc = roc_auc_score(
            source_true,
            source_prob,
            multi_class="ovr",
            average="macro",
            labels=np.arange(num_classes),
        )

    except ValueError:

        source_auc = None


    print("\n")
    print(
        f"{source_name.upper()} TEST RESULTS"
    )

    print("-" * 50)

    print(
        f"Samples:       {len(indices)}"
    )

    print(
        f"Accuracy:      {source_accuracy:.4f}"
    )

    print(
        f"Macro Precision: "
        f"{source_precision:.4f}"
    )

    print(
        f"Macro Recall: "
        f"{source_recall:.4f}"
    )

    print(
        f"Macro F1:      {source_f1:.4f}"
    )

    if source_auc is not None:

        print(
            f"Macro ROC-AUC: "
            f"{source_auc:.4f}"
        )


    return {
        "samples": len(indices),
        "accuracy": float(
            source_accuracy
        ),
        "macro_precision": float(
            source_precision
        ),
        "macro_recall": float(
            source_recall
        ),
        "macro_f1": float(
            source_f1
        ),
        "macro_roc_auc_ovr":
            None
            if source_auc is None
            else float(source_auc),
    }


# ============================================================
# CAT RESULTS
# ============================================================

cat_metrics = calculate_source_metrics(
    cat_indices,
    "cat",
)


# ============================================================
# DOG RESULTS
# ============================================================

dog_metrics = calculate_source_metrics(
    dog_indices,
    "dog",
)


# ============================================================
# SAVE METRICS JSON
# ============================================================

results = {
    "model": "EfficientNet-B0",

    "checkpoint": str(
        MODEL_PATH
    ),

    "dataset": "combined dog + cat",

    "test_samples": len(y_true),

    "classes": class_names,

    "overall": {
        "accuracy": float(
            accuracy
        ),

        "macro_precision": float(
            macro_precision
        ),

        "macro_recall": float(
            macro_recall
        ),

        "macro_f1": float(
            macro_f1
        ),

        "weighted_precision": float(
            weighted_precision
        ),

        "weighted_recall": float(
            weighted_recall
        ),

        "weighted_f1": float(
            weighted_f1
        ),

        "macro_roc_auc_ovr":
            None
            if macro_roc_auc is None
            else float(macro_roc_auc),
    },

    "per_class": per_class_metrics,

    "cat": cat_metrics,

    "dog": dog_metrics,

    "cat_test_samples": len(
        cat_indices
    ),

    "dog_test_samples": len(
        dog_indices
    ),

    "unidentified_source_samples":
        len(other_indices),
}


metrics_path = (
    OUTPUT_DIR /
    "test_metrics.json"
)

with open(
    metrics_path,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        results,
        file,
        indent=4,
    )


# ============================================================
# SAVE RAW PREDICTIONS
# ============================================================

predictions_path = (
    OUTPUT_DIR /
    "test_predictions.csv"
)


with open(
    predictions_path,
    "w",
    encoding="utf-8",
) as file:

    file.write(
        "image,true_class,predicted_class,confidence,correct\n"
    )

    for index, path in enumerate(
        test_paths
    ):

        true_index = int(
            y_true[index]
        )

        predicted_index = int(
            y_pred[index]
        )

        confidence = float(
            y_prob[index][predicted_index]
        )

        correct = (
            true_index == predicted_index
        )

        # Escape commas in paths if needed
        image_path = str(
            path
        ).replace(
            '"',
            '""'
        )

        file.write(
            f'"{image_path}",'
            f'"{class_names[true_index]}",'
            f'"{class_names[predicted_index]}",'
            f"{confidence:.6f},"
            f"{correct}\n"
        )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("EVALUATION COMPLETE")
print("=" * 70)

print("\nSaved files:")

print(
    f"Classification report:\n"
    f"  {report_path}"
)

print(
    f"\nConfusion matrix:\n"
    f"  {confusion_path}"
)

print(
    f"\nMetrics:\n"
    f"  {metrics_path}"
)

print(
    f"\nPredictions:\n"
    f"  {predictions_path}"
)

print("\nFinal numbers:")

print(
    f"  Accuracy       = {accuracy:.4f}"
)

print(
    f"  Macro F1       = {macro_f1:.4f}"
)

print(
    f"  Macro Recall   = {macro_recall:.4f}"
)

if macro_roc_auc is not None:

    print(
        f"  Macro ROC-AUC  = {macro_roc_auc:.4f}"
    )

print(
    f"\nCat samples: {len(cat_indices)}"
)

print(
    f"Dog samples: {len(dog_indices)}"
)

print("\nDone.")