from pathlib import Path
import shutil

# ============================================================
# PATHS
# ============================================================

BASE = Path("data/eye_face")

CAT_DIR = BASE / "cat"
DOG_DIR = BASE / "dog_classification"

COMBINED_DIR = BASE / "combined"


# ============================================================
# CLASS MAPPING
# ============================================================

# Source folder -> final unified class name

CAT_CLASSES = {
    "Blepharitis": "Blepharitis",
    "Conjunctivitis": "Conjunctivitis",
    "Corneal Sequestrum": "Corneal_Sequestrum",
    "Corneal Ulcer": "Corneal_Ulcer",
    "Health": "Health",
    "Non-ulcerative": "Non_ulcerative",
}

DOG_CLASSES = {
    "cataract": "Cataract",
    "cherry_eye": "Cherry_Eye",
    "glaucoma": "Glaucoma",
}


# ============================================================
# SETTINGS
# ============================================================

SPLITS = [
    "train",
    "valid",
    "test",
]

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
}


# ============================================================
# SAFETY CHECK
# ============================================================

if COMBINED_DIR.exists():

    existing_files = [
        file
        for file in COMBINED_DIR.rglob("*")
        if file.is_file()
    ]

    if existing_files:

        print(
            f"Combined dataset already contains "
            f"{len(existing_files)} files."
        )

        answer = input(
            "\nDelete and rebuild combined dataset? [y/N]: "
        ).strip().lower()

        if answer != "y":

            print(
                "Cancelled. Nothing was changed."
            )

            raise SystemExit

        shutil.rmtree(COMBINED_DIR)


# ============================================================
# CREATE DIRECTORIES
# ============================================================

all_classes = list(
    CAT_CLASSES.values()
) + list(
    DOG_CLASSES.values()
)

for split in SPLITS:

    for class_name in all_classes:

        (
            COMBINED_DIR
            / split
            / class_name
        ).mkdir(
            parents=True,
            exist_ok=True
        )


# ============================================================
# COPY FUNCTION
# ============================================================

def copy_images(
    source_root,
    class_mapping,
    source_name
):

    copied = 0

    print(
        f"\nProcessing {source_name}..."
    )

    for split in SPLITS:

        for source_class, target_class in (
            class_mapping.items()
        ):

            source_dir = (
                source_root
                / split
                / source_class
            )

            target_dir = (
                COMBINED_DIR
                / split
                / target_class
            )

            if not source_dir.exists():

                print(
                    f"WARNING: missing:\n"
                    f"{source_dir}"
                )

                continue


            images = [
                file
                for file in source_dir.iterdir()
                if file.is_file()
                and file.suffix.lower()
                in IMAGE_EXTENSIONS
            ]


            for image in images:

                # Add source prefix so filenames
                # can never collide.

                new_name = (
                    f"{source_name}_"
                    f"{image.name}"
                )

                destination = (
                    target_dir
                    / new_name
                )

                # Extra safety check

                if destination.exists():

                    raise RuntimeError(
                        f"Filename collision:\n"
                        f"{destination}"
                    )

                shutil.copy2(
                    image,
                    destination
                )

                copied += 1


    print(
        f"{source_name}: copied {copied} images"
    )

    return copied


# ============================================================
# COPY CAT DATA
# ============================================================

cat_count = copy_images(
    CAT_DIR,
    CAT_CLASSES,
    "cat"
)


# ============================================================
# COPY DOG DATA
# ============================================================

dog_count = copy_images(
    DOG_DIR,
    DOG_CLASSES,
    "dog"
)


# ============================================================
# FINAL REPORT
# ============================================================

print("\n" + "=" * 65)

print(
    "COMBINED EYE DATASET CREATED"
)

print("=" * 65)

print(
    f"Cat images copied: {cat_count}"
)

print(
    f"Dog images copied: {dog_count}"
)

print(
    f"Total images:      {cat_count + dog_count}"
)

print(
    f"\nLocation:\n{COMBINED_DIR}"
)


# ============================================================
# COUNT FINAL DATASET
# ============================================================

print("\nFinal class distribution:\n")

grand_total = 0

for split in SPLITS:

    print(
        f"\n--- {split.upper()} ---"
    )

    split_total = 0

    for class_name in all_classes:

        class_dir = (
            COMBINED_DIR
            / split
            / class_name
        )

        count = len([
            file
            for file in class_dir.iterdir()
            if file.is_file()
            and file.suffix.lower()
            in IMAGE_EXTENSIONS
        ])

        print(
            f"{class_name:25s} {count:5d}"
        )

        split_total += count

    print(
        f"{'TOTAL':25s} {split_total:5d}"
    )

    grand_total += split_total


print(
    f"\n{'GRAND TOTAL':25s} {grand_total:5d}"
)

print(
    "\nOriginal cat/ and dog/ datasets "
    "were NOT modified."
)

print(
    "\nReady for EfficientNet-B0 training."
)