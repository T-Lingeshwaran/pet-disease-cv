from pathlib import Path
import json
import csv
import re

# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
RESULTS_DIR = OUTPUT_DIR / "model_results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# HELPERS
# ============================================================

def clean_number(value):
    """Convert a value to float if possible."""
    if value is None:
        return None

    if isinstance(value, (int, float)):
        return float(value)

    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def find_json_files(folder):
    if not folder.exists():
        return []

    return list(folder.rglob("*.json"))


def find_metric_files(folder):
    """
    Find files whose names look like metric/result files.
    """
    if not folder.exists():
        return []

    files = []

    for path in folder.rglob("*"):
        if not path.is_file():
            continue

        name = path.name.lower()

        if (
            "metric" in name
            or "result" in name
            or "report" in name
            or "test" in name
        ):
            files.append(path)

    return files


def read_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


# ============================================================
# METRIC EXTRACTION
# ============================================================

def extract_metrics(data):
    """
    Try to extract common classification metrics from
    different JSON structures.
    """

    if not isinstance(data, dict):
        return {}

    metrics = {}

    # Direct metric names
    possible_names = {
        "accuracy": [
            "accuracy",
            "test_accuracy",
            "val_accuracy",
            "acc",
        ],

        "macro_f1": [
            "macro_f1",
            "test_macro_f1",
            "val_macro_f1",
            "f1_macro",
        ],

        "macro_precision": [
            "macro_precision",
            "test_macro_precision",
            "val_macro_precision",
            "precision_macro",
        ],

        "macro_recall": [
            "macro_recall",
            "test_macro_recall",
            "val_macro_recall",
            "recall_macro",
        ],

        "loss": [
            "test_loss",
            "val_loss",
            "loss",
        ],
    }

    for output_name, candidates in possible_names.items():

        for key in candidates:

            if key in data:

                value = clean_number(data[key])

                if value is not None:
                    metrics[output_name] = value
                    break

    # --------------------------------------------------------
    # Sometimes metrics are nested under "test"
    # --------------------------------------------------------

    for section_name in ["test", "test_metrics", "metrics"]:

        section = data.get(section_name)

        if not isinstance(section, dict):
            continue

        nested = extract_metrics(section)

        for key, value in nested.items():

            if key not in metrics:
                metrics[key] = value

    return metrics


# ============================================================
# KNOWN MODEL LOCATIONS
# ============================================================

MODELS = [
    {
        "name": "Eye EfficientNet-B0",
        "task": "Eye condition classification",
        "folder": OUTPUT_DIR / "eye_efficientnet",
    },

    {
        "name": "Eye CSF-Net",
        "task": "Eye condition classification",
        "folder": OUTPUT_DIR / "eye_csfnet",
    },

    {
        "name": "Skin EfficientNet-B0",
        "task": "Skin condition classification",
        "folder": OUTPUT_DIR / "skin_efficientnet",
    },

    {
        "name": "Skin EfficientNetV2-S",
        "task": "Skin condition classification",
        "folder": OUTPUT_DIR / "skin_efficientnet_v2s",
    },

    {
        "name": "Modality Router",
        "task": "Eye vs Skin classification",
        "folder": OUTPUT_DIR / "modality",
    },
]


# ============================================================
# COLLECT RESULTS
# ============================================================

all_results = []

print("=" * 70)
print("PET HEALTH CV — MODEL RESULTS")
print("=" * 70)

for model in MODELS:

    name = model["name"]
    task = model["task"]
    folder = model["folder"]

    print()
    print("-" * 70)
    print(f"MODEL: {name}")
    print(f"TASK : {task}")
    print(f"PATH : {folder}")

    if not folder.exists():

        print("STATUS: Output folder not found")
        print("        Skipping.")

        continue

    json_files = find_json_files(folder)

    if not json_files:

        print("STATUS: No JSON result files found")

        # Still record the model
        all_results.append({
            "model": name,
            "task": task,
            "accuracy": "",
            "macro_f1": "",
            "macro_precision": "",
            "macro_recall": "",
            "loss": "",
            "source_file": "",
        })

        continue

    found = False

    for json_file in json_files:

        data = read_json(json_file)

        if data is None:
            continue

        metrics = extract_metrics(data)

        if not metrics:
            continue

        # Prefer test metrics if available.
        result = {
            "model": name,
            "task": task,
            "accuracy": metrics.get("accuracy", ""),
            "macro_f1": metrics.get("macro_f1", ""),
            "macro_precision": metrics.get(
                "macro_precision", ""
            ),
            "macro_recall": metrics.get(
                "macro_recall", ""
            ),
            "loss": metrics.get("loss", ""),
            "source_file": str(
                json_file.relative_to(PROJECT_ROOT)
            ),
        }

        all_results.append(result)

        print(f"RESULT FILE: {json_file.name}")

        for key in [
            "accuracy",
            "macro_f1",
            "macro_precision",
            "macro_recall",
            "loss",
        ]:

            value = result[key]

            if value != "":
                print(f"  {key:20s}: {value}")

        found = True

        # Stop after first useful result file.
        break

    if not found:
        print("STATUS: JSON files found, but no standard metrics detected.")


# ============================================================
# SAVE CSV
# ============================================================

csv_path = RESULTS_DIR / "all_model_results.csv"

fieldnames = [
    "model",
    "task",
    "accuracy",
    "macro_f1",
    "macro_precision",
    "macro_recall",
    "loss",
    "source_file",
]

with open(csv_path, "w", newline="", encoding="utf-8") as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(all_results)


# ============================================================
# SAVE TEXT SUMMARY
# ============================================================

summary_path = RESULTS_DIR / "summary.txt"

with open(summary_path, "w", encoding="utf-8") as f:

    f.write("PET HEALTH COMPUTER VISION PROJECT\n")
    f.write("MODEL PERFORMANCE SUMMARY\n")
    f.write("=" * 70 + "\n\n")

    for result in all_results:

        f.write(f"Model : {result['model']}\n")
        f.write(f"Task  : {result['task']}\n")

        if result["accuracy"] != "":
            f.write(
                f"Accuracy       : "
                f"{float(result['accuracy']):.4f}\n"
            )

        if result["macro_f1"] != "":
            f.write(
                f"Macro F1       : "
                f"{float(result['macro_f1']):.4f}\n"
            )

        if result["macro_precision"] != "":
            f.write(
                f"Macro Precision: "
                f"{float(result['macro_precision']):.4f}\n"
            )

        if result["macro_recall"] != "":
            f.write(
                f"Macro Recall   : "
                f"{float(result['macro_recall']):.4f}\n"
            )

        if result["loss"] != "":
            f.write(
                f"Loss           : "
                f"{float(result['loss']):.4f}\n"
            )

        f.write(
            f"Source         : "
            f"{result['source_file']}\n"
        )

        f.write("-" * 70 + "\n")


# ============================================================
# PRINT FINAL TABLE
# ============================================================

print()
print()
print("=" * 100)
print("FINAL MODEL COMPARISON")
print("=" * 100)

header = (
    f"{'Model':30s}"
    f"{'Accuracy':>12s}"
    f"{'Macro F1':>12s}"
    f"{'Precision':>12s}"
    f"{'Recall':>12s}"
)

print(header)
print("-" * 100)

for result in all_results:

    def fmt(value):
        if value == "" or value is None:
            return "N/A"
        return f"{float(value):.4f}"

    print(
        f"{result['model'][:30]:30s}"
        f"{fmt(result['accuracy']):>12s}"
        f"{fmt(result['macro_f1']):>12s}"
        f"{fmt(result['macro_precision']):>12s}"
        f"{fmt(result['macro_recall']):>12s}"
    )

print("-" * 100)

print()
print(f"CSV saved to:")
print(csv_path)

print()
print(f"Summary saved to:")
print(summary_path)

print()
print("=" * 70)
print("DONE")
print("=" * 70)