"""
visualize.py

Renders the numeric gait report as a chart a non-technical viewer (or a
vet) can read in a few seconds. The skeleton-overlay video itself is
already produced by DeepLabCut during pose_extraction.py (the
"*_labeled.mp4" file) — this module just adds the summary chart and a
plain-language report card.
"""

import json
import matplotlib.pyplot as plt

from gait_metrics import GaitReport
from classify import ClassificationResult


LEG_ORDER = ["front_left", "front_right", "back_left", "back_right"]
LEG_LABELS = {
    "front_left": "Front-Left",
    "front_right": "Front-Right",
    "back_left": "Back-Left",
    "back_right": "Back-Right",
}
FLAG_COLORS = {"normal": "#4CAF50", "mild": "#FFC107", "significant": "#E53935", "inconclusive": "#9E9E9E"}


def plot_gait_report(report: GaitReport, result: ClassificationResult, out_png: str) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # Left panel: per-leg stance ratio
    legs = [l for l in LEG_ORDER if l in report.stance_ratio]
    values = [report.stance_ratio[l] for l in legs]
    colors = [
        FLAG_COLORS["significant"] if l == result.suspected_leg and result.flag != "normal"
        else "#607D8B"
        for l in legs
    ]
    axes[0].bar([LEG_LABELS[l] for l in legs], values, color=colors)
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Stance ratio (fraction of cycle planted)")
    axes[0].set_title("Per-leg stance ratio")
    axes[0].tick_params(axis="x", rotation=20)

    # Right panel: stride symmetry pairs
    pairs = list(report.stride_symmetry.keys())
    sym_values = [report.stride_symmetry[p] for p in pairs]
    axes[1].bar(pairs, sym_values, color="#3F51B5")
    axes[1].axhline(0.85, color="orange", linestyle="--", linewidth=1, label="mild threshold")
    axes[1].axhline(0.65, color="red", linestyle="--", linewidth=1, label="significant threshold")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("Symmetry score (1.0 = perfectly symmetric)")
    axes[1].set_title("Stride symmetry")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].legend(fontsize=8)

    fig.suptitle(
        f"Gait screening result: {result.flag.upper()}  "
        f"(confidence {result.confidence:.0%}, suspected leg: {LEG_LABELS.get(result.suspected_leg, result.suspected_leg)})",
        fontsize=12,
        color=FLAG_COLORS.get(result.flag, "black"),
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


def write_json_report(report: GaitReport, result: ClassificationResult, out_json: str) -> str:
    payload = {
        "flag": result.flag,
        "confidence": result.confidence,
        "suspected_leg": result.suspected_leg,
        "reasons": result.reasons,
        "metrics": {
            "stance_ratio": report.stance_ratio,
            "stride_length_px": report.stride_length,
            "stride_symmetry": report.stride_symmetry,
            "head_bob_amplitude_px": report.head_bob_amplitude_px,
            "head_bob_normalized": report.head_bob_normalized,
            "frames_used": report.frames_used,
        },
        "warnings": report.warnings,
        "disclaimer": (
            "This is a gait-screening tool, not a diagnosis. It does not "
            "detect fractures or internal injury. If flagged mild or "
            "significant, consult a veterinarian."
        ),
    }
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)
    return out_json