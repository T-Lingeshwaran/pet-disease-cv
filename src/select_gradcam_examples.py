"""
Memory-safe Grad-CAM example selector.

Selects:
    5 skin examples
    5 eye examples

Important:
    Models are loaded ONCE.
    Images are processed ONE AT A TIME.

Run from project root:

    python src/select_gradcam_examples.py
"""

from pathlib import Path
import csv
import gc
import random

import numpy as np
import torch

from predict import (
    DEVICE,
    load_image,

    # Model builders
    build_skin_model,
    CSFNet,

    # Checkpoint utilities
    load_checkpoint,
    get_state_dict,
    clean_state_dict,
    get_classes,

    # Image utilities
    prepare_eye_inputs,

    # Grad-CAM
    GradCAM,
    EyeGradCAM,

    # Visualization
    create_gradcam_overlay,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SKIN_DATASET = (
    PROJECT_ROOT
    / "data"
    / "skin_coat"
    / "combined_clean"
    / "test"
)

EYE_DATASET = (
    PROJECT_ROOT
    / "data"
    / "eye_face"
    / "combined"
    / "test"
)

SKIN_CHECKPOINT = (
    PROJECT_ROOT
    / "outputs"
    / "skin_efficientnet_v2s"
    / "best_model.pth"
)

EYE_CHECKPOINT = (
    PROJECT_ROOT
    / "outputs"
    / "eye_csfnet"
    / "best_csfnet.pth"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "gradcam_examples"
)

SKIN_OUTPUT = (
    OUTPUT_DIR
    / "skin"
)

EYE_OUTPUT = (
    OUTPUT_DIR
    / "eye"
)

SKIN_OUTPUT.mkdir(
    parents=True,
    exist_ok=True
)

EYE_OUTPUT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

SEED = 42

NUM_FINAL = 5

# Number of images to evaluate.
#
# We don't need to process all 554 skin images just
# to find five good visual examples.
#
# Increase these if necessary.
MAX_SKIN_CANDIDATES = 120
MAX_EYE_CANDIDATES = 120

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


# ============================================================
# MEMORY SETTINGS
# ============================================================

def cleanup_gpu():
    """
    Aggressive cleanup after each image.
    """

    gc.collect()

    if torch.cuda.is_available():

        torch.cuda.empty_cache()

        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


# ============================================================
# RANDOM / REPRODUCIBLE SAMPLING
# ============================================================

random.seed(SEED)

np.random.seed(
    SEED
)

torch.manual_seed(
    SEED
)


# ============================================================
# FIND IMAGES
# ============================================================

def find_images(dataset_dir):

    images = []

    if not dataset_dir.exists():

        raise FileNotFoundError(
            f"Dataset not found:\n{dataset_dir}"
        )

    for class_dir in sorted(
        dataset_dir.iterdir()
    ):

        if not class_dir.is_dir():
            continue

        for image_path in class_dir.iterdir():

            if (
                image_path.suffix.lower()
                in IMAGE_EXTENSIONS
            ):

                images.append(
                    image_path
                )

    return images


def sample_images(
    images,
    maximum
):

    if len(images) <= maximum:

        return images

    rng = random.Random(
        SEED
    )

    return rng.sample(
        images,
        maximum
    )


# ============================================================
# CAM QUALITY
# ============================================================

def calculate_cam_quality(
    cam
):

    cam = np.asarray(
        cam,
        dtype=np.float32
    )

    cam = np.nan_to_num(
        cam,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    cam = np.clip(
        cam,
        0.0,
        1.0
    )

    peak = float(
        cam.max()
    )

    mean_activation = float(
        cam.mean()
    )

    # --------------------------------------------------------
    # Activated area
    # --------------------------------------------------------

    mask = (
        cam >= 0.50
    )

    coverage = float(
        mask.mean()
    )

    # --------------------------------------------------------
    # Concentration
    # --------------------------------------------------------

    if coverage > 1e-6:

        concentration = (
            mean_activation
            / coverage
        )

    else:

        concentration = 0.0

    concentration = float(
        np.clip(
            concentration,
            0.0,
            1.0
        )
    )

    # --------------------------------------------------------
    # We don't want:
    #
    #   almost no activation
    #
    # OR
    #
    #   entire image activated.
    #
    # Moderate localized activation is preferred.
    # --------------------------------------------------------

    if coverage < 0.005:

        localization = 0.20

    elif coverage <= 0.40:

        localization = 1.00

    elif coverage <= 0.60:

        localization = 0.75

    elif coverage <= 0.80:

        localization = 0.45

    else:

        localization = 0.20

    quality = (
        0.45 * localization
        + 0.30 * concentration
        + 0.25 * peak
    )

    return {
        "coverage": coverage,
        "peak": peak,
        "mean_activation": mean_activation,
        "concentration": concentration,
        "quality": float(quality),
    }


# ============================================================
# LOAD SKIN MODEL ONCE
# ============================================================

def load_skin_model():

    print()
    print(
        "Loading skin model ONCE..."
    )

    checkpoint = load_checkpoint(
        SKIN_CHECKPOINT
    )

    fallback_classes = [
        "Bacterial_dermatosis",
        "demodicosis",
        "Dermatitis",
        "Flea Allergy",
        "Fungal_infections",
        "hotspot",
        "Hypersensitivity",
        "Hypersensitivity_allergic_dermatosis",
        "mange",
        "ringworm",
        "Scabies",
    ]

    classes = get_classes(
        checkpoint,
        fallback_classes
    )

    model = build_skin_model(
        len(classes)
    )

    state_dict = clean_state_dict(
        get_state_dict(checkpoint)
    )

    model.load_state_dict(
        state_dict,
        strict=True
    )

    model = model.to(
        DEVICE
    )

    model.eval()

    print(
        f"Skin classes: {len(classes)}"
    )

    return model, classes


# ============================================================
# LOAD EYE MODEL ONCE
# ============================================================

def load_eye_model():

    print()
    print(
        "Loading eye model ONCE..."
    )

    checkpoint = load_checkpoint(
        EYE_CHECKPOINT
    )

    fallback_classes = [
        "Blepharitis",
        "Cataract",
        "Cherry_Eye",
        "Conjunctivitis",
        "Corneal_Sequestrum",
        "Corneal_Ulcer",
        "Glaucoma",
        "Health",
        "Non_ulcerative",
    ]

    classes = get_classes(
        checkpoint,
        fallback_classes
    )

    model = CSFNet(
        len(classes)
    )

    state_dict = clean_state_dict(
        get_state_dict(checkpoint)
    )

    model.load_state_dict(
        state_dict,
        strict=True
    )

    model = model.to(
        DEVICE
    )

    model.eval()

    print(
        f"Eye classes: {len(classes)}"
    )

    return model, classes


# ============================================================
# EVALUATE ONE SKIN IMAGE
# ============================================================

def evaluate_skin_image(
    model,
    classes,
    image_path
):

    cam_generator = None

    try:

        image = load_image(
            image_path
        )

        # ----------------------------------------------------
        # Prepare tensor
        # ----------------------------------------------------

        tensor = (
            __import__(
                "predict"
            ).resize_tensor(
                image,
                384
            )
        )

        tensor = (
            __import__(
                "predict"
            ).normalize_imagenet(
                tensor
            )
        )

        tensor = tensor.to(
            DEVICE
        )

        # ----------------------------------------------------
        # Create Grad-CAM BEFORE forward
        # ----------------------------------------------------

        target_layer = (
            model.features[-1]
        )

        cam_generator = GradCAM(
            model,
            target_layer
        )

        # ----------------------------------------------------
        # Forward
        # ----------------------------------------------------

        logits = model(
            tensor
        )

        probabilities = torch.softmax(
            logits,
            dim=1
        )[0]

        confidence, index = (
            torch.max(
                probabilities,
                dim=0
            )
        )

        predicted_index = int(
            index.item()
        )

        predicted_class = (
            classes[predicted_index]
        )

        confidence = float(
            confidence.item()
        )

        ground_truth = (
            image_path.parent.name
        )

        correct = (
            predicted_class.strip().lower()
            ==
            ground_truth.strip().lower()
        )

        # ----------------------------------------------------
        # Only generate CAM for promising predictions.
        #
        # This saves GPU memory/time.
        # ----------------------------------------------------

        if not correct or confidence < 0.60:

            cam_generator.remove_hooks()

            return None

        cam = cam_generator.generate(
            logits,
            predicted_index,
            (
                384,
                384
            )
        )

        cam_quality = (
            calculate_cam_quality(
                cam
            )
        )

        selection_score = (
            0.55 * confidence
            + 0.45 * cam_quality["quality"]
        )

        result = {
            "path": image_path,
            "ground_truth": ground_truth,
            "prediction": predicted_class,
            "confidence": confidence,
            "correct": correct,
            "cam": cam.copy(),
            "coverage": cam_quality["coverage"],
            "cam_quality": cam_quality["quality"],
            "selection_score": selection_score,
        }

        cam_generator.remove_hooks()

        return result

    except torch.cuda.OutOfMemoryError:

        print(
            "    CUDA OOM -> skipping image"
        )

        if cam_generator is not None:
            cam_generator.remove_hooks()

        cleanup_gpu()

        return None

    except Exception as exc:

        print(
            f"    ERROR -> {exc}"
        )

        if cam_generator is not None:
            cam_generator.remove_hooks()

        return None

    finally:

        cleanup_gpu()


# ============================================================
# EVALUATE ONE EYE IMAGE
# ============================================================

def evaluate_eye_image(
    model,
    classes,
    image_path
):

    cam_generator = None

    try:

        image = load_image(
            image_path
        )

        (
            rgb,
            hsv,
            ycbcr
        ) = prepare_eye_inputs(
            image
        )

        # ----------------------------------------------------
        # IMPORTANT:
        # Hooks BEFORE forward.
        # ----------------------------------------------------

        cam_generator = EyeGradCAM(
            model
        )

        logits = model(
            rgb,
            hsv,
            ycbcr
        )

        probabilities = torch.softmax(
            logits,
            dim=1
        )[0]

        confidence, index = (
            torch.max(
                probabilities,
                dim=0
            )
        )

        predicted_index = int(
            index.item()
        )

        predicted_class = (
            classes[predicted_index]
        )

        confidence = float(
            confidence.item()
        )

        ground_truth = (
            image_path.parent.name
        )

        correct = (
            predicted_class.strip().lower()
            ==
            ground_truth.strip().lower()
        )

        # ----------------------------------------------------
        # Skip poor candidates before CAM.
        # ----------------------------------------------------

        if not correct or confidence < 0.60:

            cam_generator.remove_hooks()

            return None

        cam = cam_generator.generate(
            logits,
            predicted_index,
            (
                224,
                224
            )
        )

        cam_quality = (
            calculate_cam_quality(
                cam
            )
        )

        selection_score = (
            0.55 * confidence
            + 0.45 * cam_quality["quality"]
        )

        result = {
            "path": image_path,
            "ground_truth": ground_truth,
            "prediction": predicted_class,
            "confidence": confidence,
            "correct": correct,
            "cam": cam.copy(),
            "coverage": cam_quality["coverage"],
            "cam_quality": cam_quality["quality"],
            "selection_score": selection_score,
        }

        cam_generator.remove_hooks()

        return result

    except torch.cuda.OutOfMemoryError:

        print(
            "    CUDA OOM -> skipping image"
        )

        if cam_generator is not None:
            cam_generator.remove_hooks()

        cleanup_gpu()

        return None

    except Exception as exc:

        print(
            f"    ERROR -> {exc}"
        )

        if cam_generator is not None:
            cam_generator.remove_hooks()

        return None

    finally:

        cleanup_gpu()


# ============================================================
# SELECT 5 WITH CLASS DIVERSITY
# ============================================================

def select_best(
    results,
    number=5
):

    results = [
        r for r in results
        if r is not None
        and r["correct"]
    ]

    results.sort(
        key=lambda x: x["selection_score"],
        reverse=True
    )

    selected = []

    used_classes = set()

    # --------------------------------------------------------
    # First: different classes.
    # --------------------------------------------------------

    for result in results:

        class_name = (
            result["ground_truth"]
        )

        if class_name in used_classes:
            continue

        selected.append(
            result
        )

        used_classes.add(
            class_name
        )

        if len(selected) >= number:
            return selected

    # --------------------------------------------------------
    # If fewer than 5 classes are available,
    # fill with strongest remaining examples.
    # --------------------------------------------------------

    selected_paths = {
        r["path"]
        for r in selected
    }

    for result in results:

        if result["path"] in selected_paths:
            continue

        selected.append(
            result
        )

        if len(selected) >= number:
            break

    return selected


# ============================================================
# SAVE RESULTS
# ============================================================

def save_examples(
    selected,
    output_dir,
    modality
):

    rows = []

    for rank, result in enumerate(
        selected,
        start=1
    ):

        image_path = (
            result["path"]
        )

        image = load_image(
            image_path
        )

        safe_class = (
            result["ground_truth"]
            .replace(" ", "_")
            .replace("/", "_")
            .replace("\\", "_")
        )

        stem = (
            image_path.stem
        )

        # ----------------------------------------------------
        # Original
        # ----------------------------------------------------

        original_path = (
            output_dir
            / (
                f"{rank:02d}_"
                f"{safe_class}_"
                f"{stem}_original"
                f"{image_path.suffix}"
            )
        )

        image.save(
            original_path
        )

        # ----------------------------------------------------
        # Side-by-side Grad-CAM
        # ----------------------------------------------------

        gradcam_path = (
            output_dir
            / (
                f"{rank:02d}_"
                f"{safe_class}_"
                f"{stem}_gradcam.png"
            )
        )

        create_gradcam_overlay(
            image,
            result["cam"],
            gradcam_path,
            (
                f"{result['prediction']} "
                f"| "
                f"{result['confidence'] * 100:.1f}%"
            )
        )

        rows.append({
            "modality": modality,
            "rank": rank,
            "ground_truth": result["ground_truth"],
            "prediction": result["prediction"],
            "confidence": round(
                result["confidence"],
                4
            ),
            "cam_quality": round(
                result["cam_quality"],
                4
            ),
            "cam_coverage": round(
                result["coverage"],
                4
            ),
            "selection_score": round(
                result["selection_score"],
                4
            ),
            "original": str(
                original_path
            ),
            "gradcam": str(
                gradcam_path
            ),
        })

    return rows


# ============================================================
# CSV
# ============================================================

def save_csv(
    rows
):

    csv_path = (
        OUTPUT_DIR
        / "selected_examples.csv"
    )

    if not rows:
        return

    fieldnames = list(
        rows[0].keys()
    )

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(
            rows
        )

    print()
    print(
        f"CSV saved:"
    )

    print(
        csv_path
    )


# ============================================================
# PRINT
# ============================================================

def print_results(
    results,
    modality
):

    print()
    print("=" * 80)

    print(
        f"BEST {modality.upper()} EXAMPLES"
    )

    print("=" * 80)

    for i, result in enumerate(
        results,
        start=1
    ):

        print()

        print(
            f"{i}. {result['path'].name}"
        )

        print(
            f"   Class      : "
            f"{result['ground_truth']}"
        )

        print(
            f"   Prediction : "
            f"{result['prediction']}"
        )

        print(
            f"   Confidence : "
            f"{result['confidence'] * 100:.2f}%"
        )

        print(
            f"   CAM quality: "
            f"{result['cam_quality']:.3f}"
        )

        print(
            f"   CAM area   : "
            f"{result['coverage'] * 100:.2f}%"
        )

        print(
            f"   Score      : "
            f"{result['selection_score']:.3f}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 80)
    print(
        "       MEMORY-SAFE GRAD-CAM SELECTOR"
    )
    print("=" * 80)

    print()
    print(
        f"Device: {DEVICE}"
    )

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

        print(
            "GPU memory:",
            round(
                torch.cuda.get_device_properties(0).total_memory
                / (1024 ** 3),
                2
            ),
            "GB"
        )

    # ========================================================
    # LOAD MODELS ONCE
    # ========================================================

    skin_model, skin_classes = (
        load_skin_model()
    )

    eye_model, eye_classes = (
        load_eye_model()
    )

    # ========================================================
    # SKIN
    # ========================================================

    print()
    print(
        "=" * 80
    )

    print(
        "SCANNING SKIN"
    )

    print(
        "=" * 80
    )

    skin_images = find_images(
        SKIN_DATASET
    )

    print(
        f"Total skin images: "
        f"{len(skin_images)}"
    )

    skin_images = sample_images(
        skin_images,
        MAX_SKIN_CANDIDATES
    )

    print(
        f"Testing candidates: "
        f"{len(skin_images)}"
    )

    skin_results = []

    for i, image_path in enumerate(
        skin_images,
        start=1
    ):

        print(
            f"[SKIN {i}/{len(skin_images)}] "
            f"{image_path.name}"
        )

        result = evaluate_skin_image(
            skin_model,
            skin_classes,
            image_path
        )

        if result is not None:

            skin_results.append(
                result
            )

            print(
                f"    ✓ "
                f"{result['prediction']} "
                f"{result['confidence'] * 100:.1f}% "
                f"| CAM "
                f"{result['cam_quality']:.2f}"
            )

    selected_skin = select_best(
        skin_results,
        NUM_FINAL
    )

    print_results(
        selected_skin,
        "skin"
    )

    skin_rows = save_examples(
        selected_skin,
        SKIN_OUTPUT,
        "skin"
    )

    # ========================================================
    # CLEAN SKIN MODEL
    # ========================================================

    del skin_model

    cleanup_gpu()

    # ========================================================
    # EYE
    # ========================================================

    print()
    print(
        "=" * 80
    )

    print(
        "SCANNING EYE"
    )

    print(
        "=" * 80
    )

    eye_images = find_images(
        EYE_DATASET
    )

    print(
        f"Total eye images: "
        f"{len(eye_images)}"
    )

    eye_images = sample_images(
        eye_images,
        MAX_EYE_CANDIDATES
    )

    print(
        f"Testing candidates: "
        f"{len(eye_images)}"
    )

    eye_results = []

    for i, image_path in enumerate(
        eye_images,
        start=1
    ):

        print(
            f"[EYE {i}/{len(eye_images)}] "
            f"{image_path.name}"
        )

        result = evaluate_eye_image(
            eye_model,
            eye_classes,
            image_path
        )

        if result is not None:

            eye_results.append(
                result
            )

            print(
                f"    ✓ "
                f"{result['prediction']} "
                f"{result['confidence'] * 100:.1f}% "
                f"| CAM "
                f"{result['cam_quality']:.2f}"
            )

    selected_eye = select_best(
        eye_results,
        NUM_FINAL
    )

    print_results(
        selected_eye,
        "eye"
    )

    eye_rows = save_examples(
        selected_eye,
        EYE_OUTPUT,
        "eye"
    )

    # ========================================================
    # CLEAN
    # ========================================================

    del eye_model

    cleanup_gpu()

    # ========================================================
    # CSV
    # ========================================================

    all_rows = (
        skin_rows
        + eye_rows
    )

    save_csv(
        all_rows
    )

    # ========================================================
    # FINAL
    # ========================================================

    print()
    print("=" * 80)

    print(
        "DONE — YOUR GPU SURVIVED 😎"
    )

    print("=" * 80)

    print()

    print(
        f"Selected skin: "
        f"{len(selected_skin)}/5"
    )

    print(
        f"Selected eye : "
        f"{len(selected_eye)}/5"
    )

    print()

    print(
        "Skin output:"
    )

    print(
        SKIN_OUTPUT
    )

    print()

    print(
        "Eye output:"
    )

    print(
        EYE_OUTPUT
    )

    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()