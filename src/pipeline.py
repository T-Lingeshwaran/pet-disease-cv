"""
pipeline.py

End-to-end CLI: give it your pet's video, get back a skeleton-overlay
video, a gait score chart, and a JSON report with a normal/mild/significant
flag.

Usage:
    python pipeline.py --video data/gait/videos/limping/buddy_limping.mp4
    python pipeline.py --video data/gait/videos/limping/buddy_limping.mp4 --no-video-adapt
"""

import argparse
from pathlib import Path

from pose_extraction import extract_keypoints, keypoints_to_tidy_csv
from gait_metrics import compute_gait_report, list_available_bodyparts
from classify import rule_based_flag
from visualize import plot_gait_report, write_json_report


def run(video_path: str, out_dir: str = "outputs", video_adapt: bool = True):
    video_path = Path(video_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = video_path.stem

    print(f"[1/4] Running SuperAnimal-Quadruped pose estimation on {video_path.name} ...")
    keypoints_df, labeled_video = extract_keypoints(str(video_path), video_adapt=video_adapt)

    tidy_csv = str(out_dir / f"{stem}_keypoints.csv")
    keypoints_to_tidy_csv(keypoints_df, tidy_csv)
    print(f"      Keypoints saved to {tidy_csv}")
    if labeled_video:
        print(f"      Skeleton overlay video: {labeled_video}")

    print("[2/4] Computing gait metrics (stance ratio, stride symmetry, head-bob) ...")
    report = compute_gait_report(tidy_csv)
    if report.warnings:
        print("      Warnings:")
        for w in report.warnings:
            print(f"        - {w}")
        print(f"      Available bodyparts in this run: {list_available_bodyparts(tidy_csv)}")

    print("[3/4] Classifying gait (rule-based v1) ...")
    result = rule_based_flag(report)
    print(f"      Flag: {result.flag}  |  Confidence: {result.confidence}  |  Suspected leg: {result.suspected_leg}")
    for reason in result.reasons:
        print(f"        - {reason}")

    print("[4/4] Writing report chart + JSON ...")
    png_path = str(out_dir / f"{stem}_gait_report.png")
    json_path = str(out_dir / f"{stem}_report.json")
    plot_gait_report(report, result, png_path)
    write_json_report(report, result, json_path)

    print("\nDone. Outputs:")
    print(f"  - {png_path}")
    print(f"  - {json_path}")
    if labeled_video:
        print(f"  - {labeled_video}")

    return report, result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pet gait anomaly screening pipeline")
    parser.add_argument("--video", required=True, help="Path to a dog/cat video (mp4/avi/mov)")
    parser.add_argument("--out", default="outputs", help="Output directory")
    parser.add_argument(
        "--no-video-adapt",
        action="store_true",
        help="Skip DeepLabCut's per-video self-adaptation step (faster, slightly less stable keypoints)",
    )
    args = parser.parse_args()
    run(args.video, out_dir=args.out, video_adapt=not args.no_video_adapt)
