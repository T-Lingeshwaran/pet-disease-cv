from pathlib import Path
from PIL import Image
import shutil

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(".")

SOURCE_DIR = (
    PROJECT_ROOT
    / "data"
    / "eye_face"
    / "dog"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "eye_face"
    / "dog_classification"
)


# ============================================================
# DOG CLASS MAPPING
# ============================================================

CLASS_NAMES = {
    0: "cataract",
    1: "cherry_eye",
    2: "glaucoma",
}


# ============================================================
# SETTINGS
# ============================================================

# Small padding around the YOLO bounding box.
# This keeps a little surrounding eye context.
PADDING = 0.15

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
}


# ============================================================
# CLEAN OUTPUT DIRECTORY
# ============================================================

if OUTPUT_DIR.exists():

    print(
        f"Output directory already exists:\n"
        f"{OUTPUT_DIR}"
    )

    answer = input(
        "\nDelete it and recreate it? [y/N]: "
    ).strip().lower()

    if answer != "y":

        print(
            "Cancelled. Existing data was not modified."
        )

        raise SystemExit

    shutil.rmtree(OUTPUT_DIR)


# ============================================================
# CREATE OUTPUT STRUCTURE
# ============================================================

for split in ["train", "valid", "test"]:

    for class_name in CLASS_NAMES.values():

        (
            OUTPUT_DIR
            / split
            / class_name
        ).mkdir(
            parents=True,
            exist_ok=True
        )


# ============================================================
# PROCESS DATASET
# ============================================================

total_crops = 0

for split in ["train", "valid", "test"]:

    image_dir = (
        SOURCE_DIR
        / split
        / "images"
    )

    label_dir = (
        SOURCE_DIR
        / split
        / "labels"
    )

    if not image_dir.exists():

        print(
            f"WARNING: missing image directory:\n"
            f"{image_dir}"
        )

        continue

    if not label_dir.exists():

        print(
            f"WARNING: missing label directory:\n"
            f"{label_dir}"
        )

        continue


    image_files = [
        file
        for file in image_dir.iterdir()
        if file.is_file()
        and file.suffix.lower()
        in IMAGE_EXTENSIONS
    ]


    print(
        f"\nProcessing {split}: "
        f"{len(image_files)} images"
    )


    for image_path in image_files:

        label_path = (
            label_dir
            / f"{image_path.stem}.txt"
        )

        if not label_path.exists():

            print(
                f"WARNING: no label for "
                f"{image_path.name}"
            )

            continue


        # ----------------------------------------------------
        # Open image
        # ----------------------------------------------------

        try:

            image = Image.open(
                image_path
            ).convert("RGB")

        except Exception as error:

            print(
                f"Could not open "
                f"{image_path.name}: {error}"
            )

            continue


        image_width, image_height = (
            image.size
        )


        # ----------------------------------------------------
        # Read YOLO annotations
        # ----------------------------------------------------

        with open(
            label_path,
            "r",
            encoding="utf-8"
        ) as file:

            lines = [
                line.strip()
                for line in file
                if line.strip()
            ]


        crop_number = 0


        for line in lines:

            parts = line.split()

            if len(parts) != 5:

                print(
                    f"WARNING: malformed label "
                    f"in {label_path.name}: {line}"
                )

                continue


            try:

                class_id = int(
                    parts[0]
                )

                x_center = float(
                    parts[1]
                )

                y_center = float(
                    parts[2]
                )

                box_width = float(
                    parts[3]
                )

                box_height = float(
                    parts[4]
                )

            except ValueError:

                print(
                    f"WARNING: invalid label "
                    f"in {label_path.name}: {line}"
                )

                continue


            if class_id not in CLASS_NAMES:

                print(
                    f"WARNING: unknown class "
                    f"{class_id} in "
                    f"{label_path.name}"
                )

                continue


            class_name = CLASS_NAMES[
                class_id
            ]


            # ------------------------------------------------
            # YOLO normalized coordinates
            # → pixel coordinates
            # ------------------------------------------------

            x_center_px = (
                x_center
                * image_width
            )

            y_center_px = (
                y_center
                * image_height
            )

            box_width_px = (
                box_width
                * image_width
            )

            box_height_px = (
                box_height
                * image_height
            )


            # ------------------------------------------------
            # Bounding box
            # ------------------------------------------------

            x1 = (
                x_center_px
                - box_width_px / 2
            )

            y1 = (
                y_center_px
                - box_height_px / 2
            )

            x2 = (
                x_center_px
                + box_width_px / 2
            )

            y2 = (
                y_center_px
                + box_height_px / 2
            )


            # ------------------------------------------------
            # Add padding
            # ------------------------------------------------

            padding_x = (
                box_width_px
                * PADDING
            )

            padding_y = (
                box_height_px
                * PADDING
            )

            x1 -= padding_x
            y1 -= padding_y

            x2 += padding_x
            y2 += padding_y


            # ------------------------------------------------
            # Clamp to image
            # ------------------------------------------------

            x1 = max(
                0,
                int(x1)
            )

            y1 = max(
                0,
                int(y1)
            )

            x2 = min(
                image_width,
                int(x2)
            )

            y2 = min(
                image_height,
                int(y2)
            )


            # ------------------------------------------------
            # Validate crop
            # ------------------------------------------------

            if x2 <= x1 or y2 <= y1:

                print(
                    f"WARNING: invalid crop "
                    f"for {image_path.name}"
                )

                continue


            # ------------------------------------------------
            # Crop
            # ------------------------------------------------

            crop = image.crop(
                (
                    x1,
                    y1,
                    x2,
                    y2
                )
            )


            # ------------------------------------------------
            # Save
            # ------------------------------------------------

            output_class_dir = (
                OUTPUT_DIR
                / split
                / class_name
            )


            output_filename = (
                f"{image_path.stem}"
                f"_eye{crop_number}.jpg"
            )


            output_path = (
                output_class_dir
                / output_filename
            )


            crop.save(
                output_path,
                quality=95
            )


            crop_number += 1
            total_crops += 1


print("\n" + "=" * 60)

print(
    "DOG YOLO → CLASSIFICATION CONVERSION COMPLETE"
)

print("=" * 60)

print(
    f"Total eye crops created: {total_crops}"
)

print(
    f"Output directory:\n{OUTPUT_DIR}"
)

print(
    "\nOriginal dog dataset was NOT modified."
)

print(
    "\nClasses:"
)

for class_id, class_name in CLASS_NAMES.items():

    print(
        f"  {class_id} → {class_name}"
    )