import os
import json
import matplotlib.pyplot as plt


# ============================================================
# MODEL OUTPUT DIRECTORIES
# ============================================================

MODEL_DIRS = [
    "outputs/eye_combined",
    "outputs/eye_densenet121",
    "outputs/eye_csfnet",
    "outputs/eye_mcfa",

    "outputs/skin_efficientnet",
    "outputs/skin_efficientnet_v2s",
    "outputs/skin_convnext_tiny",
    "outputs/skin_cstf",

    "outputs/modality",
]


# ============================================================
# FIND ACCURACY VALUES
# ============================================================

def extract_accuracy(history):

    train_accuracy = None
    val_accuracy = None

    # --------------------------------------------------------
    # FORMAT 1
    # Dictionary-based history
    # --------------------------------------------------------

    if isinstance(history, dict):

        train_accuracy = history.get(
            "train_accuracy"
        )

        val_accuracy = history.get(
            "val_accuracy"
        )

        # Older naming
        if train_accuracy is None:
            train_accuracy = history.get(
                "train_acc"
            )

        if val_accuracy is None:
            val_accuracy = history.get(
                "val_acc"
            )

    # --------------------------------------------------------
    # FORMAT 2
    # List-based history
    # --------------------------------------------------------

    elif isinstance(history, list):

        if len(history) == 0:
            return None, None

        # Print the first entry so we can understand
        # the structure if necessary.
        first = history[0]

        if isinstance(first, dict):

            # Possible naming conventions
            train_keys = [
                "train_accuracy",
                "train_acc",
                "accuracy",
            ]

            val_keys = [
                "val_accuracy",
                "val_acc",
                "validation_accuracy",
                "validation_acc",
            ]

            for key in train_keys:

                if key in first:
                    train_accuracy = [
                        item.get(key)
                        for item in history
                    ]
                    break

            for key in val_keys:

                if key in first:
                    val_accuracy = [
                        item.get(key)
                        for item in history
                    ]
                    break

    return train_accuracy, val_accuracy


# ============================================================
# GENERATE ACCURACY CURVE
# ============================================================

def generate_accuracy_curve(output_dir):

    model_name = os.path.basename(output_dir)

    curve_path = os.path.join(
        output_dir,
        "accuracy_curve.png"
    )

    json_path = os.path.join(
        output_dir,
        "training_history.json"
    )

    csv_path = os.path.join(
        output_dir,
        "training_history.csv"
    )

    print("\n" + "-" * 60)
    print(f"Model: {model_name}")

    # --------------------------------------------------------
    # Existing curve
    # --------------------------------------------------------

    if os.path.exists(curve_path):

        print("✓ accuracy_curve.png already exists")
        print("  Skipping.")

        return "exists"

    # --------------------------------------------------------
    # Try JSON first
    # --------------------------------------------------------

    history = None

    if os.path.exists(json_path):

        try:

            with open(
                json_path,
                "r"
            ) as f:

                history = json.load(f)

            print(
                "✓ Found training_history.json"
            )

        except Exception as e:

            print(
                f"⚠ Could not read JSON: {e}"
            )

    # --------------------------------------------------------
    # If JSON unavailable, try CSV
    # --------------------------------------------------------

    if history is None and os.path.exists(csv_path):

        try:

            import csv

            with open(
                csv_path,
                "r",
                newline=""
            ) as f:

                reader = csv.DictReader(f)

                history = list(reader)

            print(
                "✓ Found training_history.csv"
            )

        except Exception as e:

            print(
                f"⚠ Could not read CSV: {e}"
            )

    # --------------------------------------------------------
    # No history
    # --------------------------------------------------------

    if history is None:

        print(
            "⚠ No training history found."
        )

        print(
            "  Cannot generate accuracy curve."
        )

        return "missing_history"

    # --------------------------------------------------------
    # Extract accuracy
    # --------------------------------------------------------

    train_accuracy = []
    val_accuracy = []

    # ========================================================
    # JSON DICTIONARY
    # ========================================================

    if isinstance(history, dict):

        train_accuracy = (
            history.get("train_accuracy")
            or history.get("train_acc")
            or []
        )

        val_accuracy = (
            history.get("val_accuracy")
            or history.get("val_acc")
            or []
        )

    # ========================================================
    # JSON LIST / CSV ROWS
    # ========================================================

    elif isinstance(history, list):

        if len(history) > 0:

            first = history[0]

            if isinstance(first, dict):

                # Possible column names
                train_keys = [
                    "train_accuracy",
                    "train_acc",
                    "Train Accuracy",
                    "train_accuracy_score",
                ]

                val_keys = [
                    "val_accuracy",
                    "val_acc",
                    "validation_accuracy",
                    "validation_acc",
                    "Val Accuracy",
                ]

                train_key = None
                val_key = None

                for key in train_keys:

                    if key in first:

                        train_key = key
                        break

                for key in val_keys:

                    if key in first:

                        val_key = key
                        break

                if train_key is not None:

                    train_accuracy = [
                        row.get(train_key)
                        for row in history
                    ]

                if val_key is not None:

                    val_accuracy = [
                        row.get(val_key)
                        for row in history
                    ]

    # --------------------------------------------------------
    # Convert CSV strings to floats
    # --------------------------------------------------------

    try:

        train_accuracy = [
            float(x)
            for x in train_accuracy
            if x not in (None, "")
        ]

        val_accuracy = [
            float(x)
            for x in val_accuracy
            if x not in (None, "")
        ]

    except Exception as e:

        print(
            f"⚠ Could not convert accuracy values: {e}"
        )

        return "error"

    # --------------------------------------------------------
    # Check
    # --------------------------------------------------------

    if len(train_accuracy) == 0:

        print(
            "⚠ Training accuracy not found."
        )

        return "missing_accuracy"

    if len(val_accuracy) == 0:

        print(
            "⚠ Validation accuracy not found."
        )

        return "missing_accuracy"

    # --------------------------------------------------------
    # Match lengths
    # --------------------------------------------------------

    num_epochs = min(
        len(train_accuracy),
        len(val_accuracy)
    )

    train_accuracy = train_accuracy[
        :num_epochs
    ]

    val_accuracy = val_accuracy[
        :num_epochs
    ]

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    epochs = range(
        1,
        num_epochs + 1
    )

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        epochs,
        train_accuracy,
        label="Train Accuracy"
    )

    plt.plot(
        epochs,
        val_accuracy,
        label="Validation Accuracy"
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Accuracy"
    )

    plt.title(
        f"{model_name} Training and Validation Accuracy"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        curve_path,
        dpi=200
    )

    plt.close()

    print(
        "✓ Accuracy curve generated:"
    )

    print(
        f"  {curve_path}"
    )

    return "generated"

    # --------------------------------------------------------
    # Missing history
    # --------------------------------------------------------

    if not os.path.exists(history_path):

        print(
            "⚠ training_history.json not found"
        )

        print(
            "  Cannot generate accuracy curve."
        )

        return "missing_history"

    # --------------------------------------------------------
    # Load JSON
    # --------------------------------------------------------

    try:

        with open(
            history_path,
            "r"
        ) as f:

            history = json.load(f)

    except Exception as e:

        print(
            f"⚠ Could not read training history: {e}"
        )

        return "error"

    # --------------------------------------------------------
    # Extract accuracy
    # --------------------------------------------------------

    train_accuracy, val_accuracy = (
        extract_accuracy(history)
    )

    # --------------------------------------------------------
    # Check accuracy
    # --------------------------------------------------------

    if train_accuracy is None:

        print(
            "⚠ Training accuracy not found."
        )

        print(
            "  Cannot generate curve."
        )

        return "missing_accuracy"

    if val_accuracy is None:

        print(
            "⚠ Validation accuracy not found."
        )

        print(
            "  Cannot generate curve."
        )

        return "missing_accuracy"

    # --------------------------------------------------------
    # Remove None values
    # --------------------------------------------------------

    valid_train = [
        value
        for value in train_accuracy
        if value is not None
    ]

    valid_val = [
        value
        for value in val_accuracy
        if value is not None
    ]

    if len(valid_train) == 0:

        print(
            "⚠ Training accuracy contains no usable values."
        )

        return "missing_accuracy"

    if len(valid_val) == 0:

        print(
            "⚠ Validation accuracy contains no usable values."
        )

        return "missing_accuracy"

    # --------------------------------------------------------
    # Match lengths
    # --------------------------------------------------------

    num_epochs = min(
        len(train_accuracy),
        len(val_accuracy)
    )

    train_accuracy = train_accuracy[
        :num_epochs
    ]

    val_accuracy = val_accuracy[
        :num_epochs
    ]

    # --------------------------------------------------------
    # Epoch numbers
    # --------------------------------------------------------

    epochs = range(
        1,
        num_epochs + 1
    )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        epochs,
        train_accuracy,
        label="Train Accuracy"
    )

    plt.plot(
        epochs,
        val_accuracy,
        label="Validation Accuracy"
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Accuracy"
    )

    plt.title(
        f"{model_name} Training and Validation Accuracy"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        curve_path,
        dpi=200
    )

    plt.close()

    print(
        "✓ Accuracy curve generated:"
    )

    print(
        f"  {curve_path}"
    )

    return "generated"


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("MISSING ACCURACY CURVE GENERATOR")
    print("=" * 60)

    generated = 0
    skipped = 0
    unable = 0

    for output_dir in MODEL_DIRS:

        # ----------------------------------------------------
        # Missing model directory
        # ----------------------------------------------------

        if not os.path.isdir(output_dir):

            print("\n" + "-" * 60)

            print(
                f"Model directory not found: {output_dir}"
            )

            print(
                "Skipping."
            )

            skipped += 1

            continue

        # ----------------------------------------------------
        # Process model
        # ----------------------------------------------------

        try:

            result = generate_accuracy_curve(
                output_dir
            )

        except Exception as e:

            print(
                f"⚠ Unexpected error: {e}"
            )

            print(
                "  Skipping this model and continuing."
            )

            unable += 1

            continue

        # ----------------------------------------------------
        # Count results
        # ----------------------------------------------------

        if result == "generated":

            generated += 1

        elif result == "exists":

            skipped += 1

        else:

            unable += 1

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print(
        f"Generated : {generated}"
    )

    print(
        f"Skipped   : {skipped}"
    )

    print(
        f"Unable    : {unable}"
    )

    print("=" * 60)


if __name__ == "__main__":

    main()