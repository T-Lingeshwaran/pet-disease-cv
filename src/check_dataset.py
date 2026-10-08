"""
check_dataset.py

Run after every collection session:

    python src/check_dataset.py            # from the repo root
    python src/check_dataset.py --data path/to/data

For each module it checks that
  1. every filename in the label CSV exists on disk,
  2. every media file on disk has a row in the CSV,
  3. label columns only contain allowed values (see data/DATA_GUIDE.md),
  4. every pet_id appears in pets.csv,
  5. (gait only) the subfolder agrees with the label.

Exit code is 1 if any problem is found, so it can also run as a pre-commit check.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

MEDIA_EXT = {".mp4", ".mov", ".avi", ".mkv", ".jpg", ".jpeg", ".png", ".webp", ".dcm", ".bmp"}

# module -> (label csv, media folder, allowed values per column)
MODULES = {
    "gait": (
        "gait/gait_labels.csv",
        "gait/videos",
        {
            "gait_type": {"walk", "trot"},
            "camera_angle": {"side", "front", "rear", "other"},
            "label": {"healthy", "mild", "significant", "uncertain"},
            "suspected_leg": {"none", "front_left", "front_right", "back_left", "back_right"},
            "label_confirmed_by": {"owner", "vet"},
            "source": {"own", "external"},
        },
    ),
    "posture_activity": (
        "posture_activity/posture_labels.csv",
        "posture_activity/videos",
        {
            "activity": {"walking", "standing", "lying", "sitting", "standing_up", "climbing_stairs"},
            "behavior_flag": {"normal", "difficulty_rising", "weight_shifting", "guarding"},
            "label_confirmed_by": {"owner", "vet"},
            "source": {"own", "external"},
        },
    ),
    "body_condition": (
        "body_condition/body_condition_labels.csv",
        "body_condition/images",
        {
            "view": {"side", "top"},
            "bcs_9pt": {str(i) for i in range(1, 10)},
            "scored_by": {"owner", "vet"},
            "source": {"own", "external"},
        },
    ),
    "eye_face": (
        "eye_face/eye_labels.csv",
        "eye_face/images",
        {
            "eye": {"left", "right", "both"},
            "finding": {"normal", "redness", "discharge", "foreign_matter", "cloudiness", "other"},
            "label_confirmed_by": {"owner", "vet"},
            "source": {"own", "external"},
        },
    ),
    "skin_coat": (
        "skin_coat/skin_labels.csv",
        "skin_coat/images",
        {
            "finding": {"normal", "hair_loss", "redness", "lesion", "other"},
            "label_confirmed_by": {"owner", "vet"},
            "source": {"own", "external"},
        },
    ),
    "injury_triage": (
        "injury_triage/injury_labels.csv",
        "injury_triage",
        {
            "media_type": {"video", "photo", "xray"},
            "injury_signs": {"none", "non_weight_bearing", "swelling", "abnormal_limb_angle", "guarding"},
            "fracture_confirmed": {"yes", "no", "unknown"},
            "label_confirmed_by": {"owner", "vet"},
            "source": {"own", "external"},
        },
    ),
}

SYMPTOM_VALUES = {
    "appetite": {"normal", "reduced", "increased"},
    "water_intake": {"normal", "reduced", "increased"},
    "energy": {"normal", "low", "high"},
    "vocalization": {"none", "whining", "yelping"},
    "avoids_stairs": {"yes", "no"},
    "limping_noticed": {"yes", "no"},
}


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def _check_values(df, allowed, problems, tag):
    for col, ok in allowed.items():
        if col not in df.columns:
            problems.append(f"[{tag}] missing column '{col}'")
            continue
        for i, v in df[col].items():
            v = v.strip()
            if v == "":
                problems.append(f"[{tag}] row {i + 2}: '{col}' is empty")
            elif v not in ok:
                problems.append(f"[{tag}] row {i + 2}: '{col}' = '{v}' (allowed: {sorted(ok)})")


def check(data_dir: Path) -> int:
    problems, summary = [], []

    pets_csv = data_dir / "pets.csv"
    pet_ids = set()
    if pets_csv.exists():
        pet_ids = set(_read(pets_csv)["pet_id"])
    else:
        problems.append("pets.csv not found")

    for name, (csv_rel, media_rel, allowed) in MODULES.items():
        csv_path, media_dir = data_dir / csv_rel, data_dir / media_rel
        if not csv_path.exists():
            problems.append(f"[{name}] label file missing: {csv_rel}")
            continue
        df = _read(csv_path)

        on_disk = {}
        for p in media_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in MEDIA_EXT:
                if p.name in on_disk:
                    problems.append(f"[{name}] duplicate filename on disk: {p.name}")
                on_disk[p.name] = p

        logged = list(df["filename"]) if "filename" in df.columns else []
        for f in logged:
            if f not in on_disk:
                problems.append(f"[{name}] in CSV but not on disk: {f}")
        for f in on_disk:
            if f not in set(logged):
                problems.append(f"[{name}] on disk but not in CSV: {f}")
        if len(logged) != len(set(logged)):
            problems.append(f"[{name}] duplicate rows for the same filename")

        _check_values(df, allowed, problems, name)

        if pet_ids and "pet_id" in df.columns:
            for i, pid in df["pet_id"].items():
                if pid not in pet_ids:
                    problems.append(f"[{name}] row {i + 2}: pet_id '{pid}' not in pets.csv")

        if name == "gait":
            for _, row in df.iterrows():
                p = on_disk.get(row["filename"])
                if p is None:
                    continue
                folder, label = p.parent.name, row["label"]
                ok = (
                    (folder == "healthy" and label == "healthy")
                    or (folder == "limping" and label in {"mild", "significant"})
                    or (folder == "uncertain" and label == "uncertain")
                )
                if not ok:
                    problems.append(
                        f"[gait] {row['filename']}: sits in '{folder}/' but labelled '{label}'"
                    )

        summary.append(f"{name:18s} {len(logged):4d} logged | {len(on_disk):4d} on disk")

    sym_csv = data_dir / "symptoms/symptom_log.csv"
    if sym_csv.exists():
        sdf = _read(sym_csv)
        _check_values(sdf, SYMPTOM_VALUES, problems, "symptoms")
        if pet_ids:
            for i, pid in sdf["pet_id"].items():
                if pid not in pet_ids:
                    problems.append(f"[symptoms] row {i + 2}: pet_id '{pid}' not in pets.csv")
        summary.append(f"{'symptoms':18s} {len(sdf):4d} rows")
    else:
        problems.append("[symptoms] symptom_log.csv missing")

    print("Dataset summary")
    print("-" * 40)
    print("\n".join(summary))
    print("-" * 40)
    if problems:
        print(f"{len(problems)} problem(s) found:")
        for p in problems:
            print("  -", p)
        return 1
    print("No problems found.")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Validate the pet-health dataset folders and label CSVs")
    ap.add_argument("--data", default="data", help="path to the data folder (default: data)")
    args = ap.parse_args()
    sys.exit(check(Path(args.data)))
