"""
gait_metrics.py

Converts a tidy keypoint CSV (frame, bodypart, x, y, likelihood) into the
numeric gait signals a vet-style lameness read relies on:

  1. Per-leg stance ratio  -> how much of the gait cycle each paw spends
     planted vs. swinging. A sore leg is typically unloaded faster
     (shorter stance) than the healthy legs.
  2. Stride length symmetry -> compares left/right and front/hind stride
     lengths. Healthy quadrupeds are close to symmetric; a limping animal
     shortens the stride on the sore side.
  3. Head-bob amplitude -> vertical head movement over the gait cycle.
     A well-established equine/canine sign: the head rises when weight
     hits a sore FRONT leg and drops when weight hits a sore HIND leg
     (Feuser et al.; AAEP lameness scale uses the same principle).

SuperAnimal-Quadruped bodypart names (verify against your own run's
printed list with `list_available_bodyparts`, since exact names can
vary slightly by DeepLabCut version):
    nose, upper_jaw, lower_jaw, right_eye, left_eye,
    right_earbase, right_earend, left_earbase, left_earend,
    neck_base, neck_end, throat_base, throat_end,
    back_base, back_middle, back_end,
    front_left_thai, front_left_knee, front_left_paw,
    front_right_thai, front_right_knee, front_right_paw,
    back_left_thai, back_left_knee, back_left_paw,
    back_right_thai, back_right_knee, back_right_paw,
    tail_base, tail_end, belly_bottom,
    body_middle_right, body_middle_left
"""

from dataclasses import dataclass, field
import numpy as np
import pandas as pd


# Logical leg name -> SuperAnimal-Quadruped bodypart used as that paw's marker.
DEFAULT_LEG_MAP = {
    "front_left": "front_left_paw",
    "front_right": "front_right_paw",
    "back_left": "back_left_paw",
    "back_right": "back_right_paw",
}
HEAD_POINT = "nose"
LIKELIHOOD_THRESHOLD = 0.4


@dataclass
class GaitReport:
    stance_ratio: dict = field(default_factory=dict)       # leg -> 0..1
    stride_length: dict = field(default_factory=dict)      # leg -> px (mean)
    stride_symmetry: dict = field(default_factory=dict)    # pair -> 0..1 (1 = perfectly symmetric)
    head_bob_amplitude_px: float = 0.0
    head_bob_normalized: float = 0.0                       # amplitude / body length, comparable across videos
    frames_used: int = 0
    warnings: list = field(default_factory=list)


def list_available_bodyparts(tidy_csv: str) -> list:
    """Print/return the bodypart names actually present in a run's output,
    so you can fix DEFAULT_LEG_MAP / HEAD_POINT if your DLC version's
    naming differs."""
    df = pd.read_csv(tidy_csv)
    parts = sorted(df["bodypart"].unique().tolist())
    return parts


def _pivot(tidy_df: pd.DataFrame, bodypart: str) -> pd.DataFrame:
    sub = tidy_df[tidy_df["bodypart"] == bodypart].sort_values("frame")
    sub = sub.set_index("frame")[["x", "y", "likelihood"]]
    return sub


def _leg_is_planted(y_series: pd.Series, threshold_frac: float = 0.15) -> np.ndarray:
    """
    Heuristic stance detector: a paw is 'planted' when its vertical
    position is near the local maximum (lowest point on screen = paw down,
    since image y increases downward) for a sustained stretch, i.e. its
    vertical velocity is near zero at a low point in the cycle.
    Returns a boolean array, one entry per frame.
    """
    y = y_series.values.astype(float)
    if len(y) < 5:
        return np.zeros_like(y, dtype=bool)

    vel = np.gradient(y)
    y_range = np.nanmax(y) - np.nanmin(y)
    if y_range == 0 or np.isnan(y_range):
        return np.zeros_like(y, dtype=bool)

    near_bottom = y > (np.nanmax(y) - threshold_frac * y_range)
    low_velocity = np.abs(vel) < (0.05 * y_range)
    return near_bottom & low_velocity


def _stride_lengths_px(x_series: pd.Series, planted_mask: np.ndarray) -> list:
    """Distance the paw's x-position advances between consecutive
    plant events — a proxy for stride length in pixels (relative measure,
    not calibrated to real-world units unless you know camera scale)."""
    x = x_series.values.astype(float)
    plant_frames = np.where(planted_mask)[0]
    if len(plant_frames) < 2:
        return []

    # Collapse consecutive planted frames into single "plant events"
    events = []
    run_start = plant_frames[0]
    prev = plant_frames[0]
    for f in plant_frames[1:]:
        if f - prev > 1:
            events.append((run_start + prev) // 2)
            run_start = f
        prev = f
    events.append((run_start + prev) // 2)

    strides = [abs(x[events[i + 1]] - x[events[i]]) for i in range(len(events) - 1)]
    return strides


def compute_gait_report(tidy_csv: str, leg_map: dict = None) -> GaitReport:
    """
    Main entry point. Reads a tidy keypoint CSV and returns a GaitReport
    with per-leg stance ratio, stride length/symmetry, and head-bob signal.
    """
    leg_map = leg_map or DEFAULT_LEG_MAP
    tidy_df = pd.read_csv(tidy_csv)
    report = GaitReport()

    leg_series = {}
    for leg, bodypart in leg_map.items():
        pdf = _pivot(tidy_df, bodypart)
        if pdf.empty:
            report.warnings.append(
                f"Bodypart '{bodypart}' not found for leg '{leg}' — check "
                "list_available_bodyparts() and update leg_map."
            )
            continue
        pdf = pdf[pdf["likelihood"] >= LIKELIHOOD_THRESHOLD]
        if pdf.empty:
            report.warnings.append(
                f"All '{bodypart}' detections were below the confidence "
                f"threshold ({LIKELIHOOD_THRESHOLD}) — video quality/angle issue?"
            )
            continue
        leg_series[leg] = pdf

    if not leg_series:
        report.warnings.append("No usable leg keypoints — cannot compute gait metrics.")
        return report

    report.frames_used = max(len(v) for v in leg_series.values())

    stride_by_leg = {}
    for leg, pdf in leg_series.items():
        planted = _leg_is_planted(pdf["y"])
        report.stance_ratio[leg] = float(np.mean(planted)) if len(planted) else 0.0
        strides = _stride_lengths_px(pdf["x"], planted)
        stride_by_leg[leg] = strides
        report.stride_length[leg] = float(np.mean(strides)) if strides else 0.0

    # Symmetry: compare left vs right on each axle, and front vs hind
    def sym(a, b):
        if a == 0 and b == 0:
            return 1.0
        return 1.0 - (abs(a - b) / max(a, b, 1e-6))

    sl = report.stride_length
    if "front_left" in sl and "front_right" in sl:
        report.stride_symmetry["front_left_vs_right"] = sym(sl["front_left"], sl["front_right"])
    if "back_left" in sl and "back_right" in sl:
        report.stride_symmetry["back_left_vs_right"] = sym(sl["back_left"], sl["back_right"])
    if "front_left" in sl and "back_left" in sl:
        report.stride_symmetry["left_front_vs_back"] = sym(sl["front_left"], sl["back_left"])
    if "front_right" in sl and "back_right" in sl:
        report.stride_symmetry["right_front_vs_back"] = sym(sl["front_right"], sl["back_right"])

    # Head-bob: vertical head movement, normalized by body length so the
    # score is comparable across videos shot at different distances/zooms.
    head_pdf = _pivot(tidy_df, HEAD_POINT)
    head_pdf = head_pdf[head_pdf["likelihood"] >= LIKELIHOOD_THRESHOLD]
    if not head_pdf.empty and len(head_pdf) > 5:
        head_y = head_pdf["y"].values.astype(float)
        amplitude = float(np.nanpercentile(head_y, 95) - np.nanpercentile(head_y, 5))
        report.head_bob_amplitude_px = amplitude

        # Body length proxy: nose-to-tail_base distance, median across frames.
        # This is a real physical span (unlike a single point's variance),
        # so it gives a stable, meaningful denominator for normalization.
        tail_pdf = _pivot(tidy_df, "tail_base")
        body_len_proxy = None
        if not tail_pdf.empty:
            joined = head_pdf[["x", "y"]].join(
                tail_pdf[["x", "y"]], lsuffix="_head", rsuffix="_tail", how="inner"
            )
            if not joined.empty:
                dists = np.hypot(
                    joined["x_head"] - joined["x_tail"], joined["y_head"] - joined["y_tail"]
                )
                body_len_proxy = float(np.nanmedian(dists))

        if not body_len_proxy or body_len_proxy < 1e-6:
            # Fallback: full vertical range of the back line as a rough scale.
            back_pdf = _pivot(tidy_df, "back_base")
            if not back_pdf.empty:
                body_len_proxy = float(np.nanmax(back_pdf["y"]) - np.nanmin(back_pdf["y"])) or amplitude
            else:
                body_len_proxy = amplitude if amplitude else 1.0
            report.warnings.append(
                "No usable nose-to-tail_base pair for body-length normalization; "
                "used a fallback scale. Head-bob score is less reliable — verify "
                "'tail_base' is being detected via list_available_bodyparts()."
            )

        report.head_bob_normalized = amplitude / max(body_len_proxy, 1e-6)
    else:
        report.warnings.append("Head keypoint too sparse for head-bob calculation.")

    return report
