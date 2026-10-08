import json
import random
from pathlib import Path
from multiprocessing import freeze_support

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import (
    efficientnet_b0,
    EfficientNet_B0_Weights
)

from PIL import Image

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# CONFIG
# ============================================================

EYE_DIR = Path("data/eye_face/combined")
SKIN_DIR = Path("data/skin_coat/combined_clean")

OUTPUT_DIR = Path("outputs/modality")

IMAGE_SIZE = 224

BATCH_SIZE = 32
NUM_EPOCHS = 25
PATIENCE = 5

LR = 1e-4
WEIGHT_DECAY = 1e-4

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
# IMAGE EXTENSIONS
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


# ============================================================
# COLLECT FILES
# ============================================================

def collect_images(directory):

    files = []

    for path in directory.rglob("*"):

        if (
            path.is_file()
            and path.suffix.lower()
            in IMAGE_EXTENSIONS
        ):
            files.append(path)

    return files


# ============================================================
# MODALITY DATASET
# ============================================================

class ModalityDataset(Dataset):

    def __init__(
        self,
        eye_dir,
        skin_dir,
        split,
        transform=None
    ):

        self.transform = transform

        self.samples = []

        # ----------------------------------------------------
        # Eye = 0
        # ----------------------------------------------------

        eye_split = eye_dir / split

        eye_images = collect_images(
            eye_split
        )

        for path in eye_images:

            self.samples.append(
                (
                    path,
                    0
                )
            )

        # ----------------------------------------------------
        # Skin = 1
        # ----------------------------------------------------

        skin_split = skin_dir / split

        skin_images = collect_images(
            skin_split
        )

        for path in skin_images:

            self.samples.append(
                (
                    path,
                    1
                )
            )

        random.shuffle(
            self.samples
        )

        print(
            f"{split}: "
            f"{len(eye_images)} eye + "
            f"{len(skin_images)} skin = "
            f"{len(self.samples)} images"
        )

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, index):

        path, label = self.samples[index]

        image = Image.open(
            path
        ).convert("RGB")

        if self.transform is not None:

            image = self.transform(
                image
            )

        return image, label


# ============================================================
# TRANSFORMS
# ============================================================

def build_datasets():

    train_transform = transforms.Compose([

        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        ),

        transforms.RandomHorizontalFlip(
            p=0.5
        ),

        transforms.RandomRotation(
            10
        ),

        transforms.ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.15,
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

    train_dataset = ModalityDataset(
        EYE_DIR,
        SKIN_DIR,
        "train",
        train_transform
    )

    valid_dataset = ModalityDataset(
        EYE_DIR,
        SKIN_DIR,
        "valid",
        eval_transform
    )

    test_dataset = ModalityDataset(
        EYE_DIR,
        SKIN_DIR,
        "test",
        eval_transform
    )

    return (
        train_dataset,
        valid_dataset,
        test_dataset
    )


# ============================================================
# CLASS WEIGHTS
# ============================================================

def calculate_class_weights(dataset):

    labels = [
        label
        for _, label in dataset.samples
    ]

    counts = np.bincount(
        labels,
        minlength=2
    )

    print("\nModality counts:")

    print(
        f"  Eye : {counts[0]}"
    )

    print(
        f"  Skin: {counts[1]}"
    )

    weights = 1.0 / np.sqrt(
        counts
    )

    weights = (
        weights / weights.mean()
    )

    return torch.tensor(
        weights,
        dtype=torch.float32
    )


# ============================================================
# MODEL
# ============================================================

def build_model():

    print(
        "\nLoading ImageNet-pretrained "
        "EfficientNet-B0..."
    )

    weights = (
        EfficientNet_B0_Weights.DEFAULT
    )

    model = efficientnet_b0(
        weights=weights
    )

    in_features = (
        model.classifier[1].in_features
    )

    model.classifier = nn.Sequential(

        nn.Dropout(
            p=0.3
        ),

        nn.Linear(
            in_features,
            2
        )
    )

    return model


# ============================================================
# TRAIN
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

    predictions = []
    targets = []

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
            enabled=(
                USE_AMP
                and device.type == "cuda"
            )
        ):

            outputs = model(
                images
            )

            loss = criterion(
                outputs,
                labels
            )

        if scaler is not None:

            scaler.scale(
                loss
            ).backward()

            scaler.step(
                optimizer
            )

            scaler.update()

        else:

            loss.backward()

            optimizer.step()

        running_loss += (
            loss.item()
            * images.size(0)
        )

        preds = outputs.argmax(
            dim=1
        )

        predictions.extend(
            preds.detach()
            .cpu()
            .numpy()
        )

        targets.extend(
            labels.detach()
            .cpu()
            .numpy()
        )

    loss = (
        running_loss
        / len(loader.dataset)
    )

    accuracy = accuracy_score(
        targets,
        predictions
    )

    f1 = f1_score(
        targets,
        predictions,
        average="macro",
        zero_division=0
    )

    return loss, accuracy, f1


# ============================================================
# EVALUATION
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

    predictions = []
    targets = []

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
            enabled=(
                USE_AMP
                and device.type == "cuda"
            )
        ):

            outputs = model(
                images
            )

            loss = criterion(
                outputs,
                labels
            )

        running_loss += (
            loss.item()
            * images.size(0)
        )

        preds = outputs.argmax(
            dim=1
        )

        predictions.extend(
            preds.cpu().numpy()
        )

        targets.extend(
            labels.cpu().numpy()
        )

    loss = (
        running_loss
        / len(loader.dataset)
    )

    accuracy = accuracy_score(
        targets,
        predictions
    )

    f1 = f1_score(
        targets,
        predictions,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        targets,
        predictions,
        average="macro",
        zero_division=0
    )

    return (
        loss,
        accuracy,
        f1,
        recall,
        targets,
        predictions
    )


# ============================================================
# MAIN
# ============================================================

def main():

    set_seed(
        SEED
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # DEVICE
    # --------------------------------------------------------

    if torch.cuda.is_available():

        device = torch.device(
            "cuda"
        )

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    else:

        device = torch.device(
            "cpu"
        )

        print(
            "CUDA unavailable - using CPU"
        )

    # --------------------------------------------------------
    # DATA
    # --------------------------------------------------------

    (
        train_dataset,
        valid_dataset,
        test_dataset
    ) = build_datasets()

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(
            NUM_WORKERS > 0
        )
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(
            NUM_WORKERS > 0
        )
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(
            NUM_WORKERS > 0
        )
    )

    # --------------------------------------------------------
    # CLASS WEIGHTS
    # --------------------------------------------------------

    class_weights = (
        calculate_class_weights(
            train_dataset
        ).to(device)
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = build_model()

    model = model.to(
        device
    )

    # --------------------------------------------------------
    # LOSS
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=0.05
    )

    # --------------------------------------------------------
    # OPTIMIZER
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=2
        )
    )

    # --------------------------------------------------------
    # AMP
    # --------------------------------------------------------

    if (
        device.type == "cuda"
        and USE_AMP
    ):

        scaler = torch.amp.GradScaler(
            "cuda"
        )

    else:

        scaler = None

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    best_f1 = -1

    patience_counter = 0

    history = {
        "train_loss": [],
        "valid_loss": [],
        "train_accuracy": [],
        "valid_accuracy": [],
        "train_f1": [],
        "valid_f1": []
    }

    print("\n")
    print("=" * 70)
    print("STARTING MODALITY CLASSIFIER")
    print("=" * 70)

    for epoch in range(
        1,
        NUM_EPOCHS + 1
    ):

        print(
            f"\nEpoch {epoch}/{NUM_EPOCHS}"
        )

        train_loss, train_acc, train_f1 = (
            train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                scaler,
                device
            )
        )

        (
            valid_loss,
            valid_acc,
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

        scheduler.step(
            valid_f1
        )

        history[
            "train_loss"
        ].append(
            train_loss
        )

        history[
            "valid_loss"
        ].append(
            valid_loss
        )

        history[
            "train_accuracy"
        ].append(
            train_acc
        )

        history[
            "valid_accuracy"
        ].append(
            valid_acc
        )

        history[
            "train_f1"
        ].append(
            train_f1
        )

        history[
            "valid_f1"
        ].append(
            valid_f1
        )

        print(
            f"Train Loss : {train_loss:.4f}"
        )

        print(
            f"Train Acc  : {train_acc:.4f}"
        )

        print(
            f"Train F1   : {train_f1:.4f}"
        )

        print(
            f"Valid Loss : {valid_loss:.4f}"
        )

        print(
            f"Valid Acc  : {valid_acc:.4f}"
        )

        print(
            f"Valid F1   : {valid_f1:.4f}"
        )

        print(
            f"Valid Rec  : {valid_recall:.4f}"
        )

        # ----------------------------------------------------
        # BEST MODEL
        # ----------------------------------------------------

        if valid_f1 > best_f1:

            best_f1 = valid_f1

            patience_counter = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict":
                        model.state_dict(),

                    "optimizer_state_dict":
                        optimizer.state_dict(),

                    "best_valid_f1":
                        best_f1,

                    "class_names": [
                        "eye",
                        "skin"
                    ],

                    "image_size":
                        IMAGE_SIZE
                },

                OUTPUT_DIR
                / "best_model.pth"
            )

            print(
                "*** NEW BEST MODEL SAVED ***"
            )

        else:

            patience_counter += 1

            print(
                f"No improvement "
                f"({patience_counter}/"
                f"{PATIENCE})"
            )

        if (
            patience_counter
            >= PATIENCE
        ):

            print(
                "\nEarly stopping."
            )

            break

    # --------------------------------------------------------
    # SAVE HISTORY
    # --------------------------------------------------------

    with open(
        OUTPUT_DIR
        / "training_history.json",
        "w"
    ) as f:

        json.dump(
            history,
            f,
            indent=4
        )

    # --------------------------------------------------------
    # LOAD BEST
    # --------------------------------------------------------

    checkpoint = torch.load(
        OUTPUT_DIR
        / "best_model.pth",
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------

    (
        test_loss,
        test_acc,
        test_f1,
        test_recall,
        targets,
        predictions
    ) = evaluate(
        model,
        test_loader,
        criterion,
        device
    )

    print("\n")
    print("=" * 70)
    print("FINAL MODALITY TEST")
    print("=" * 70)

    print(
        f"Test Loss   : {test_loss:.4f}"
    )

    print(
        f"Test Accuracy: {test_acc:.4f}"
    )

    print(
        f"Test Macro-F1: {test_f1:.4f}"
    )

    print(
        f"Test Recall  : {test_recall:.4f}"
    )

    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    report = classification_report(
        targets,
        predictions,
        target_names=[
            "eye",
            "skin"
        ],
        digits=4,
        zero_division=0
    )

    print("\n")
    print(report)

    with open(
        OUTPUT_DIR
        / "classification_report.txt",
        "w"
    ) as f:

        f.write(report)

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    cm = confusion_matrix(
        targets,
        predictions
    )

    np.savetxt(
        OUTPUT_DIR
        / "confusion_matrix.csv",
        cm,
        fmt="%d",
        delimiter=","
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    summary = {
        "model":
            "EfficientNet-B0",

        "task":
            "eye_vs_skin",

        "test_accuracy":
            float(test_acc),

        "test_macro_f1":
            float(test_f1),

        "test_macro_recall":
            float(test_recall),

        "best_validation_f1":
            float(best_f1),

        "classes": [
            "eye",
            "skin"
        ]
    }

    with open(
        OUTPUT_DIR
        / "test_metrics.json",
        "w"
    ) as f:

        json.dump(
            summary,
            f,
            indent=4
        )

    print("\n")
    print("=" * 70)
    print("MODALITY TRAINING COMPLETE")
    print("=" * 70)

    print(
        "Saved:"
    )

    print(
        OUTPUT_DIR
        / "best_model.pth"
    )


# ============================================================
# WINDOWS SAFE ENTRY POINT
# ============================================================

if __name__ == "__main__":

    freeze_support()

    main()