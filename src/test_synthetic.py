"""
test_synthetic.py

Sanity-checks gait_metrics.py, classify.py, and visualize.py using
synthetic keypoint trajectories (no video, no DeepLabCut required) —
one simulating a healthy symmetric gait, one simulating a front-left limp.
Run this after cloning to confirm your environment/math is wired up
correctly before spending time on real pose-estimation inference.
"""

import numpy as np
import pandas as pd

from gait_metrics import compute_gait_report, DEFAULT_LEG_MAP, HEAD_POINT
from classify import rule_based_flag
from visualize import plot_gait_report, write_json_report


def make_synthetic_csv(path: str, limp_leg: str = None, n_frames: int = 150, fps: int = 30):
    """
    Simulate a simple trot: each leg's paw follows a sinusoidal vertical
    bob (down when planted, up when swinging) with a phase offset per leg
    (diagonal-pair trot gait), and moves forward in x while planted.
    If limp_leg is set, that leg's stance duration shortens and its
    stride amplitude shrinks, and the head gets an extra bob synced to
    that leg's plant phase.
    """
    t = np.arange(n_frames)
    rows = []

    phase_offset = {
        "front_left": 0.0,
        "back_right": 0.0,      # diagonal pair with front_left
        "front_right": np.pi,
        "back_left": np.pi,     # diagonal pair with front_right
    }

    base_x_speed = 4.0
    y_baseline = 300
    y_amplitude = 20

    for leg, bodypart in DEFAULT_LEG_MAP.items():
        phase = phase_offset[leg]
        amp = y_amplitude
        duty = 0.5  # fraction of cycle planted, healthy default

        if limp_leg and leg == limp_leg:
            amp *= 0.5       # shorter stride amplitude on sore leg
            duty = 0.3        # unloads faster -> shorter stance ratio

        cycle_len = 30
        for f in t:
            cyc_pos = ((f + phase / (2 * np.pi) * cycle_len) % cycle_len) / cycle_len
            planted = cyc_pos < duty
            y = y_baseline + (amp if planted else amp * (1 - (cyc_pos - duty) / (1 - duty)))
            x = base_x_speed * f + (0 if planted else amp * 2 * (cyc_pos - duty))
            rows.append({"frame": f, "bodypart": bodypart, "x": x, "y": y, "likelihood": 0.9})

    # Head: baseline bob, extra bob synced to limp leg's stance phase if limping
    head_y_base = 100
    for f in t:
        y = head_y_base + 5 * np.sin(2 * np.pi * f / 30)
        if limp_leg:
            leg_phase = phase_offset[limp_leg]
            extra = 15 * np.abs(np.sin(2 * np.pi * f / 30 + leg_phase))
            y += extra
        rows.append({"frame": f, "bodypart": HEAD_POINT, "x": base_x_speed * f, "y": y, "likelihood": 0.9})

    # back_base: near-static reference point (kept for fallback path)
    # tail_base: placed ~120px behind the head to give a realistic
    # nose-to-tail body length for head-bob normalization
    body_length_px = 120
    for f in t:
        rows.append({"frame": f, "bodypart": "back_base", "x": base_x_speed * f, "y": 250, "likelihood": 0.9})
        rows.append({
            "frame": f, "bodypart": "tail_base",
            "x": base_x_speed * f - body_length_px, "y": 250, "likelihood": 0.9,
        })

    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return path


if __name__ == "__main__":
    healthy_csv = "/tmp/healthy_keypoints.csv"
    limp_csv = "/tmp/limping_keypoints.csv"

    make_synthetic_csv(healthy_csv, limp_leg=None)
    make_synthetic_csv(limp_csv, limp_leg="front_left")

    print("=" * 60)
    print("HEALTHY (synthetic)")
    print("=" * 60)
    report_h = compute_gait_report(healthy_csv)
    result_h = rule_based_flag(report_h)
    print("Stance ratio:", report_h.stance_ratio)
    print("Stride symmetry:", report_h.stride_symmetry)
    print("Head-bob normalized:", report_h.head_bob_normalized)
    print("=> Flag:", result_h.flag, "| Confidence:", result_h.confidence, "| Suspected leg:", result_h.suspected_leg)
    for r in result_h.reasons:
        print("   -", r)

    print()
    print("=" * 60)
    print("LIMPING front_left (synthetic)")
    print("=" * 60)
    report_l = compute_gait_report(limp_csv)
    result_l = rule_based_flag(report_l)
    print("Stance ratio:", report_l.stance_ratio)
    print("Stride symmetry:", report_l.stride_symmetry)
    print("Head-bob normalized:", report_l.head_bob_normalized)
    print("=> Flag:", result_l.flag, "| Confidence:", result_l.confidence, "| Suspected leg:", result_l.suspected_leg)
    for r in result_l.reasons:
        print("   -", r)

    plot_gait_report(report_h, result_h, "/tmp/healthy_report.png")
    plot_gait_report(report_l, result_l, "/tmp/limping_report.png")
    write_json_report(report_h, result_h, "/tmp/healthy_report.json")
    write_json_report(report_l, result_l, "/tmp/limping_report.json")
    print("\nCharts + JSON written to /tmp/ for visual sanity check.")