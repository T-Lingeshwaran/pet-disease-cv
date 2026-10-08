
"""
Select strong Grad-CAM examples for the proposed pet-health models.

IMPORTANT:
This script is a CANDIDATE RANKER, not a localization ground-truth metric.
It ranks images using a fixed, pre-declared heuristic:
    - correct prediction (optional filter)
    - prediction confidence
    - Grad-CAM concentration
    - Grad-CAM spatial compactness
    - Grad-CAM/LayerCAM agreement
    - low CAM entropy

It is intended to find clean qualitative examples for inspection.
For a scientific report, inspect the selected images manually and also
retain a small random/stratified sample so the report is not based only
on cherry-picked examples.

It imports the architecture and CAM implementations from the current
compare_explainability_V2.py in the same src/ directory.
"""

from __future__ import annotations

import argparse
import gc
import heapq
import json
import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont

import compare_explainability_V2 as exp


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Default: find examples with strong model evidence and good
# cross-method consistency, not simply the highest confidence.
WEIGHTS = {
    "confidence": 0.20,
    "concentration": 0.30,
    "compactness": 0.20,
    "agreement": 0.20,
    "entropy_focus": 0.10,
}


def parse_args():
    p = argparse.ArgumentParser(
        description="Rank test images by Grad-CAM qualitative quality."
    )
    p.add_argument(
        "--task",
        choices=["eye", "skin"],
        required=True,
    )
    p.add_argument(
        "--model",
        choices=["proposed", "baseline"],
        default="proposed",
    )
    p.add_argument(
        "--test-dir",
        type=str,
        default=None,
        help="Override the test directory.",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of examples to keep per true class.",
    )
    p.add_argument(
        "--min-confidence",
        type=float,
        default=0.70,
    )
    p.add_argument(
        "--only-correct",
        action="store_true",
        help="Only rank correctly classified test images.",
    )
    p.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Optional limit for debugging. 0 = all images.",
    )
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory.",
    )
    return p.parse_args()


def image_paths(test_dir: Path):
    rows = []
    for class_dir in sorted(test_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        true_class = class_dir.name
        for path in sorted(class_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
                rows.append((path, true_class))
    return rows


def safe_entropy_focus(cam: np.ndarray) -> float:
    """
    1 = highly concentrated CAM
    0 = highly diffuse CAM
    """
    x = np.clip(cam.astype(np.float64), 0.0, None)
    s = x.sum()
    if s <= 1e-12:
        return 0.0

    p = x / s
    entropy = -(p * np.log(p + 1e-12)).sum()
    max_entropy = math.log(p.size)
    normalized = entropy / max_entropy
    return float(np.clip(1.0 - normalized, 0.0, 1.0))


def top_mask(cam: np.ndarray, percentile: float = 70.0):
    threshold = np.percentile(cam, percentile)
    return cam >= threshold


def concentration_score(cam: np.ndarray, top_fraction: float = 0.20):
    """
    Measures how much CAM mass lies in its strongest top_fraction pixels.
    Higher means more concentrated.
    """
    x = np.clip(cam.astype(np.float64), 0.0, None)
    total = x.sum()
    if total <= 1e-12:
        return 0.0

    k = max(1, int(round(x.size * top_fraction)))
    flat = np.sort(x.reshape(-1))
    mass = flat[-k:].sum() / total

    # A uniform map gives approximately top_fraction.
    # Rescale so uniform ~= 0 and very concentrated ~= 1.
    score = (mass - top_fraction) / max(1e-9, 1.0 - top_fraction)
    return float(np.clip(score, 0.0, 1.0))


def compactness_score(cam: np.ndarray, percentile: float = 70.0):
    """
    Fraction of activated pixels contained in the largest connected
    component. This rewards one/few coherent regions rather than noise.
    """
    mask = top_mask(cam, percentile).astype(np.uint8)

    # Small morphological cleanup only for scoring.
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    total = int(mask.sum())
    if total == 0:
        return 0.0

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8,
    )
    if n_labels <= 1:
        return 0.0

    largest = int(stats[1:, cv2.CC_STAT_AREA].max())
    return float(np.clip(largest / total, 0.0, 1.0))


def cam_agreement(grad_cam: np.ndarray, layer_cam: np.ndarray):
    """
    IoU between the strongest 30% of Grad-CAM and LayerCAM.
    """
    a = grad_cam >= np.percentile(grad_cam, 70.0)
    b = layer_cam >= np.percentile(layer_cam, 70.0)

    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0

    inter = np.logical_and(a, b).sum()
    return float(inter / union)


def score_cam(
    confidence: float,
    grad_cam: np.ndarray,
    layer_cam: np.ndarray,
):
    concentration = concentration_score(grad_cam)
    compactness = compactness_score(grad_cam)
    agreement = cam_agreement(grad_cam, layer_cam)
    entropy_focus = safe_entropy_focus(grad_cam)

    score = (
        WEIGHTS["confidence"] * confidence
        + WEIGHTS["concentration"] * concentration
        + WEIGHTS["compactness"] * compactness
        + WEIGHTS["agreement"] * agreement
        + WEIGHTS["entropy_focus"] * entropy_focus
    )

    return float(score), {
        "confidence": float(confidence),
        "concentration": float(concentration),
        "compactness": float(compactness),
        "gradcam_layercam_agreement": float(agreement),
        "entropy_focus": float(entropy_focus),
        "selection_score": float(score),
    }


def model_setup(task: str, model_name: str):
    if task == "eye":
        if model_name == "proposed":
            model, classes = exp.load_model(
                exp.EYE_PROPOSED_CHECKPOINT,
                exp.MCFA_Net_V2(len(exp.EYE_CLASSES)),
            )
            return model, classes or exp.EYE_CLASSES, "MCFA-Net V2"

        model, classes = exp.load_model(
            exp.EYE_BASELINE_CHECKPOINT,
            exp.build_eye_efficientnet(len(exp.EYE_CLASSES)),
        )
        return model, classes or exp.EYE_CLASSES, "EfficientNet-B0"

    if model_name == "proposed":
        model, classes = exp.load_model(
            exp.SKIN_PROPOSED_CHECKPOINT,
            exp.CSTFNet(len(exp.SKIN_CLASSES)),
        )
        return model, classes or exp.SKIN_CLASSES, "CSTF-Net"

    model, classes = exp.load_model(
        exp.SKIN_BASELINE_CHECKPOINT,
        exp.build_skin_efficientnet_v2s(len(exp.SKIN_CLASSES)),
    )
    return model, classes or exp.SKIN_CLASSES, "EfficientNetV2-S"


def make_input(task: str, model_name: str, image: Image.Image):
    if task == "eye":
        size = exp.EYE_IMAGE_SIZE
    elif task == "skin" and model_name == "baseline":
        # EfficientNetV2-S was evaluated with its 384px input pipeline.
        size = exp.SKIN_BASELINE_IMAGE_SIZE
    elif task == "skin" and model_name == "proposed":
        size = exp.SKIN_PROPOSED_IMAGE_SIZE
    else:
        size = 224

    tensor = exp.preprocess_rgb(image, size).to(exp.DEVICE)
    return tensor


def _cam_from_hook(hook):
    """Build a Grad-CAM/LayerCAM map from the hook's captured tensors."""
    if hasattr(hook, "generate"):
        return hook.generate
    raise RuntimeError("CAM hook does not expose generate().")


def _forward_single_branch(model, tensor, pred_idx, hook):
    """
    Low-memory single forward/backward pass.

    Only ONE CAM hook is installed at a time. This is intentionally slower
    than keeping hooks on all branches simultaneously, but it is much safer
    on 6 GB GPUs.
    """
    model.zero_grad(set_to_none=True)

    logits = model(tensor)
    probabilities = torch.softmax(logits, dim=1)[0]
    confidence = float(probabilities[pred_idx].item())

    model.zero_grad(set_to_none=True)
    cam = hook.generate(logits, pred_idx)

    # Detach everything before returning so the autograd graph can be freed.
    cam = np.asarray(cam, dtype=np.float32).copy()

    del logits, probabilities
    model.zero_grad(set_to_none=True)

    return confidence, cam


def _predict_only(model, tensor):
    """One inference pass used to determine the predicted class."""
    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]
        pred_idx = int(torch.argmax(probabilities).item())
        confidence = float(probabilities[pred_idx].item())

    del logits, probabilities
    return pred_idx, confidence


def _branch_cam_low_memory(model, tensor, pred_idx, layer, hook_cls):
    """
    Run exactly one forward/backward for one target layer.

    This prevents the previous implementation from retaining six large
    activation/gradient hook buffers simultaneously.
    """
    hook = hook_cls(layer)
    try:
        model.zero_grad(set_to_none=True)

        logits = model(tensor)
        hook_confidence = float(
            torch.softmax(logits, dim=1)[0, pred_idx].item()
        )

        model.zero_grad(set_to_none=True)
        cam = hook.generate(logits, pred_idx)
        cam = np.asarray(cam, dtype=np.float32).copy()

        del logits
        model.zero_grad(set_to_none=True)
        return cam
    finally:
        hook.remove()


def run_one(model, classes, task, model_name, image):
    """
    Low-VRAM inference + Grad-CAM + LayerCAM.

    IMPORTANT:
    The old selector installed Grad-CAM and LayerCAM hooks on all branches
    simultaneously. MCFA/CSTF have multiple feature branches, so that could
    retain many large CUDA activation/gradient tensors on a 6 GB GPU.

    This version deliberately trades speed for memory:
      1. predict once
      2. compute one branch CAM at a time
      3. immediately detach/delete its graph
      4. call empty_cache between branches
    """
    tensor = make_input(task, model_name, image)

    if (
        (task == "eye" and model_name == "baseline")
        or (task == "skin" and model_name == "baseline")
    ):
        tensor = exp.normalize_imagenet(tensor)

    # ---------------------------------------------------------
    # First prediction pass — no autograd graph retained.
    # ---------------------------------------------------------
    pred_idx, confidence = _predict_only(model, tensor)

    # ---------------------------------------------------------
    # Proposed eye: three EfficientNet branches.
    # Compute Grad-CAM and LayerCAM one branch at a time.
    # ---------------------------------------------------------
    if task == "eye" and model_name == "proposed":
        layers = [
            model.rgb_backbone.features[-1],
            model.hsv_backbone.features[-1],
            model.ycbcr_backbone.features[-1],
        ]

        # We need the branch weights from the model. Run one normal
        # forward with gradients disabled, so this does not consume
        # a persistent autograd graph.
        with torch.no_grad():
            _, info = model(tensor, return_branch_weights=True)
            weights = (
                info["branch_weights"][0]
                .detach()
                .float()
                .cpu()
                .numpy()
            )
        del info
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        grad_maps = []
        layer_maps = []

        for layer in layers:
            grad_maps.append(
                _branch_cam_low_memory(
                    model, tensor, pred_idx, layer, exp.GradCAMHook
                )
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        for layer in layers:
            layer_maps.append(
                _branch_cam_low_memory(
                    model, tensor, pred_idx, layer, exp.LayerCAMHook
                )
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        grad_cam = exp.normalize_cam(
            weights[0] * grad_maps[0]
            + weights[1] * grad_maps[1]
            + weights[2] * grad_maps[2]
        )
        layer_cam = exp.normalize_cam(
            weights[0] * layer_maps[0]
            + weights[1] * layer_maps[1]
            + weights[2] * layer_maps[2]
        )

        del grad_maps, layer_maps, weights, tensor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return pred_idx, confidence, grad_cam, layer_cam

    # ---------------------------------------------------------
    # Proposed skin: RGB / Color / Texture branches.
    # ---------------------------------------------------------
    if task == "skin" and model_name == "proposed":
        layers = [
            model.rgb_branch.features[-1],
            model.color_branch.features[-1],
            model.texture_branch.features[-1],
        ]

        # Obtain the learned branch weights without retaining a graph.
        with torch.no_grad():
            outputs = model(tensor, return_features=True)
            weights = (
                outputs["branch_weights"][0]
                .detach()
                .float()
                .cpu()
                .numpy()
            )
        del outputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        grad_maps = []
        layer_maps = []

        for layer in layers:
            grad_maps.append(
                _branch_cam_low_memory(
                    model, tensor, pred_idx, layer, exp.GradCAMHook
                )
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        for layer in layers:
            layer_maps.append(
                _branch_cam_low_memory(
                    model, tensor, pred_idx, layer, exp.LayerCAMHook
                )
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        grad_cam = exp.normalize_cam(
            weights[0] * grad_maps[0]
            + weights[1] * grad_maps[1]
            + weights[2] * grad_maps[2]
        )
        layer_cam = exp.normalize_cam(
            weights[0] * layer_maps[0]
            + weights[1] * layer_maps[1]
            + weights[2] * layer_maps[2]
        )

        del grad_maps, layer_maps, weights, tensor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return pred_idx, confidence, grad_cam, layer_cam

    # ---------------------------------------------------------
    # Single-branch baseline.
    # ---------------------------------------------------------
    target_layer = model.features[-1]

    grad_cam = _branch_cam_low_memory(
        model, tensor, pred_idx, target_layer, exp.GradCAMHook
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    layer_cam = _branch_cam_low_memory(
        model, tensor, pred_idx, target_layer, exp.LayerCAMHook
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    del tensor
    return pred_idx, confidence, grad_cam, layer_cam


def save_candidate(candidate, output_dir: Path, rank: int):
    cls = candidate["true_class"]
    safe_cls = "".join(
        c if c.isalnum() or c in "-_" else "_"
        for c in cls
    )

    class_dir = output_dir / safe_cls
    class_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(candidate["image_path"]).stem
    base = class_dir / f"{rank:02d}_{stem}"

    original_path = base.with_suffix(".jpg")
    grad_path = base.with_name(base.name + "_gradcam.png")
    layer_path = base.with_name(base.name + "_layercam.png")

    image = Image.open(candidate["image_path"]).convert("RGB")
    image.save(original_path, quality=95)

    exp.save_overlay(
        image,
        candidate["grad_cam"],
        grad_path,
        (
            f"{candidate['model']} Grad-CAM | "
            f"true={candidate['true_class']} | "
            f"pred={candidate['predicted_class']} "
            f"{candidate['confidence']:.1%} | "
            f"score={candidate['selection_score']:.3f}"
        ),
    )

    exp.save_overlay(
        image,
        candidate["layer_cam"],
        layer_path,
        (
            f"{candidate['model']} LayerCAM | "
            f"true={candidate['true_class']} | "
            f"pred={candidate['predicted_class']} "
            f"{candidate['confidence']:.1%}"
        ),
    )

    return {
        "original": str(original_path),
        "gradcam": str(grad_path),
        "layercam": str(layer_path),
    }


def make_montage(candidates, output_path: Path):
    if not candidates:
        return

    panels = []

    for c in candidates:
        image = Image.open(c["image_path"]).convert("RGB")
        # Use the same visualization function's output logic directly.
        grad = exp.resize_cam(c["grad_cam"], image.width, image.height)
        grad = exp.normalize_cam(grad)

        heat = (grad * 255.0).astype(np.uint8)
        heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
        heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)

        overlay = (
            0.55 * np.asarray(image, dtype=np.float32)
            + 0.45 * heat.astype(np.float32)
        )
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)

        panel = Image.fromarray(overlay)
        panel = panel.resize((320, 240))

        draw = ImageDraw.Draw(panel)
        text = (
            f"{c['true_class']} -> {c['predicted_class']}\n"
            f"conf={c['confidence']:.1%}  "
            f"score={c['selection_score']:.3f}"
        )
        draw.rectangle((0, 0, 320, 48), fill=(0, 0, 0))
        draw.text((5, 5), text, fill=(255, 255, 255))
        panels.append(panel)

    cols = min(3, len(panels))
    rows = math.ceil(len(panels) / cols)

    canvas = Image.new("RGB", (cols * 320, rows * 240), "black")

    for i, panel in enumerate(panels):
        x = (i % cols) * 320
        y = (i // cols) * 240
        canvas.paste(panel, (x, y))

    canvas.save(output_path)


def main():
    args = parse_args()

    if args.test_dir:
        test_dir = Path(args.test_dir)
    elif args.task == "eye":
        test_dir = exp.PROJECT_ROOT / "data" / "eye_face" / "combined" / "test"
    else:
        test_dir = (
            exp.PROJECT_ROOT
            / "data"
            / "skin_coat"
            / "combined_clean"
            / "test"
        )

    if not test_dir.exists():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = (
            exp.PROJECT_ROOT
            / "outputs"
            / "explainability"
            / "selected"
            / args.task
            / args.model
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    paths = image_paths(test_dir)

    if args.max_images > 0:
        paths = paths[: args.max_images]

    print("=" * 78)
    print("EXCELLENT GRAD-CAM CANDIDATE SELECTOR")
    print("=" * 78)
    print(f"Task           : {args.task}")
    print(f"Model          : {args.model}")
    print(f"Test directory : {test_dir}")
    print(f"Images         : {len(paths)}")
    print(f"Top per class  : {args.top_k}")
    print(f"Min confidence: {args.min_confidence:.1%}")
    print(f"Only correct   : {args.only_correct}")
    print(f"Output         : {output_dir}")
    print("CAM mode       : low-VRAM sequential branches")
    print("=" * 78)

    model, classes, model_label = model_setup(args.task, args.model)

    # Small per-class heaps. Each heap keeps the highest scores.
    heaps = {}

    processed = 0
    kept_candidates = 0
    skipped = 0

    for image_path, true_class in paths:
        try:
            image = exp.load_image(image_path)

            pred_idx, confidence, grad_cam, layer_cam = run_one(
                model,
                classes,
                args.task,
                args.model,
                image,
            )

            predicted_class = classes[pred_idx]

            if confidence < args.min_confidence:
                skipped += 1
                continue

            correct = predicted_class == true_class

            if args.only_correct and not correct:
                skipped += 1
                continue

            selection_score, components = score_cam(
                confidence,
                grad_cam,
                layer_cam,
            )

            candidate = {
                "image_path": str(image_path),
                "true_class": true_class,
                "predicted_class": predicted_class,
                "correct": bool(correct),
                "model": model_label,
                "confidence": confidence,
                "selection_score": selection_score,
                "grad_cam": grad_cam.astype(np.float32),
                "layer_cam": layer_cam.astype(np.float32),
                **components,
            }

            heap = heaps.setdefault(true_class, [])

            # Heap entries are (score, unique counter, candidate).
            # The counter avoids comparing dictionaries on score ties.
            counter = processed

            if len(heap) < args.top_k:
                heapq.heappush(heap, (selection_score, counter, candidate))
                kept_candidates += 1
            elif selection_score > heap[0][0]:
                heapq.heapreplace(heap, (selection_score, counter, candidate))

            processed += 1

            # Conservative cleanup for 6 GB GPUs.
            del image, grad_cam, layer_cam
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if processed % 25 == 0:
                print(f"Processed {processed}/{len(paths)}")

        except Exception as e:
            print(f"[WARN] Failed: {image_path}\n       {e}")
            skipped += 1

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    selected = []

    for true_class, heap in heaps.items():
        ranked = sorted(heap, key=lambda x: x[0], reverse=True)

        for rank, (_, _, candidate) in enumerate(ranked, start=1):
            files = save_candidate(candidate, output_dir, rank)
            candidate = {
                k: v
                for k, v in candidate.items()
                if k not in {"grad_cam", "layer_cam"}
            }
            candidate["saved_files"] = files
            selected.append(candidate)

    selected.sort(
        key=lambda x: x["selection_score"],
        reverse=True,
    )

    # CSV-friendly results.
    csv_rows = []
    for c in selected:
        row = dict(c)
        row["saved_files"] = json.dumps(row["saved_files"])
        csv_rows.append(row)

    df = pd.DataFrame(csv_rows)
    csv_path = output_dir / "selected_candidates.csv"
    df.to_csv(csv_path, index=False)

    json_path = output_dir / "selected_candidates.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            selected,
            f,
            indent=2,
        )

    montage_path = output_dir / "top_candidates_montage.png"
    make_montage(
        [
            {
                **c,
                "grad_cam": np.asarray(
                    Image.open(c["saved_files"]["gradcam"])
                )
                if False else None,
            }
            for c in []
        ],
        montage_path,
    )

    # Build montage directly from selected JSON records is unnecessary;
    # create a compact montage from the saved Grad-CAM images instead.
    montage_sources = []
    for c in selected[: min(12, len(selected))]:
        montage_sources.append(c["saved_files"]["gradcam"])

    if montage_sources:
        thumbs = []
        for path in montage_sources:
            im = Image.open(path).convert("RGB")
            im.thumbnail((360, 270))
            canvas = Image.new("RGB", (360, 300), "black")
            canvas.paste(
                im,
                ((360 - im.width) // 2, 0),
            )
            thumbs.append(canvas)

        cols = min(3, len(thumbs))
        rows = math.ceil(len(thumbs) / cols)
        montage = Image.new(
            "RGB",
            (cols * 360, rows * 300),
            "black",
        )
        for i, im in enumerate(thumbs):
            montage.paste(
                im,
                ((i % cols) * 360, (i // cols) * 300),
            )
        montage.save(montage_path)

    print("\n" + "=" * 78)
    print("SELECTION COMPLETE")
    print("=" * 78)
    print(f"Processed eligible : {processed}")
    print(f"Skipped             : {skipped}")
    print(f"Selected             : {len(selected)}")
    print(f"CSV                 : {csv_path}")
    print(f"JSON                : {json_path}")
    print(f"Montage             : {montage_path}")
    print("\nTop candidates:")
    for i, c in enumerate(selected[:10], start=1):
        print(
            f"{i:2d}. {c['true_class']:30s} "
            f"{c['confidence']:.1%} "
            f"score={c['selection_score']:.3f} "
            f"{c['image_path']}"
        )


if __name__ == "__main__":
    main()
