from pathlib import Path
import random
import shutil

# ============================================================
# CHANGE THIS PATH
# ============================================================

DATASET_DIR = Path(
    "data/eye_face/cat"
)

TRAIN_DIR = DATASET_DIR / "train"
VALID_DIR = DATASET_DIR / "valid"
TEST_DIR = DATASET_DIR / "test"

TEST_RATIO = 0.50
RANDOM_SEED = 42

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


# ============================================================
# CHECK FOLDERS
# ============================================================

if not TRAIN_DIR.exists():
    raise FileNotFoundError(
        f"Could not find:\n{TRAIN_DIR}"
    )

if not VALID_DIR.exists():
    raise FileNotFoundError(
        f"Could not find:\n{VALID_DIR}"
    )

print("\nCat dataset found.")
print("Train: ", TRAIN_DIR)
print("Valid: ", VALID_DIR)
print()


# ============================================================
# FIND CLASSES
# ============================================================

classes = sorted([
    folder
    for folder in VALID_DIR.iterdir()
    if folder.is_dir()
])

if not classes:
    raise RuntimeError(
        "No class folders were found inside valid/"
    )

print("Classes found:")

for class_folder in classes:
    print("  ", class_folder.name)


# ============================================================
# CREATE TEST DIRECTORY
# ============================================================

if TEST_DIR.exists():

    existing_files = list(
        TEST_DIR.rglob("*")
    )

    existing_files = [
        x for x in existing_files
        if x.is_file()
    ]

    if existing_files:

        raise RuntimeError(
            "\nThe test folder already contains files.\n"
            "Delete the existing test folder before "
            "running this script again."
        )

TEST_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# RANDOM SEED
# ============================================================

random.seed(RANDOM_SEED)


# ============================================================
# SPLIT EACH CLASS
# ============================================================

total_valid = 0
total_test = 0

print("\nSplitting valid/ into valid/ + test/...\n")


for class_folder in classes:

    images = [
        file
        for file in class_folder.iterdir()
        if file.is_file()
        and file.suffix.lower()
        in IMAGE_EXTENSIONS
    ]

    random.shuffle(images)

    total = len(images)

    if total < 2:

        print(
            f"WARNING: {class_folder.name} "
            f"has only {total} image(s)."
        )

        continue


    # --------------------------------------------------------
    # 50% goes to test
    # --------------------------------------------------------

    test_count = round(
        total * TEST_RATIO
    )

    # Keep at least one image in valid
    test_count = max(
        1,
        min(
            test_count,
            total - 1
        )
    )


    test_images = images[:test_count]

    remaining_valid = (
        images[test_count:]
    )


    # --------------------------------------------------------
    # Create matching test class folder
    # --------------------------------------------------------

    test_class_folder = (
        TEST_DIR / class_folder.name
    )

    test_class_folder.mkdir(
        parents=True,
        exist_ok=True
    )


    # --------------------------------------------------------
    # MOVE images from valid → test
    # --------------------------------------------------------

    for image in test_images:

        destination = (
            test_class_folder
            / image.name
        )

        shutil.move(
            str(image),
            str(destination)
        )


    # --------------------------------------------------------
    # Print statistics
    # --------------------------------------------------------

    print(
        f"{class_folder.name}:"
    )

    print(
        f"    Original valid: {total}"
    )

    print(
        f"    Remaining valid: {len(remaining_valid)}"
    )

    print(
        f"    Test: {len(test_images)}"
    )

    print()


    total_valid += total
    total_test += len(test_images)


# ============================================================
# SUMMARY
# ============================================================

remaining_valid = (
    total_valid - total_test
)

print("=" * 55)
print("CAT DATASET SPLIT COMPLETE")
print("=" * 55)

print(
    f"Original valid images : {total_valid}"
)

print(
    f"New valid images      : {remaining_valid}"
)

print(
    f"New test images       : {total_test}"
)

print()

print("Final structure:")

print("""
cat/
│
├── train/
│   ├── class_1/
│   ├── class_2/
│   └── ...
│
├── valid/
│   ├── class_1/
│   ├── class_2/
│   └── ...
│
└── test/
    ├── class_1/
    ├── class_2/
    └── ...
""")

print(
    f"Random seed: {RANDOM_SEED}"
)

print(
    "\nTrain was NOT modified."
)

print(
    "Half of the original valid images "
    "were moved into test."
)