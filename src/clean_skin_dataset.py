from pathlib import Path
import shutil

SOURCE = Path("data/skin_coat/combined")
DEST = Path("data/skin_coat/combined_clean")

# Classes we want to keep
KEEP_CLASSES = [
    "Bacterial_dermatosis",
    "demodicosis",
    "Dermatitis",
    "Flea Allergy",
    "Fungal_infections",
    "hotspot",
    "Hypersensitivity",
    "Hypersensitivity_allergic_dermatosis",
    "mange",
    "ringworm",
    "Scabies",
]

SPLITS = ["train", "valid", "test"]


def main():
    print("=" * 60)
    print("Cleaning skin disease dataset")
    print("=" * 60)

    if not SOURCE.exists():
        raise FileNotFoundError(f"Source dataset not found: {SOURCE}")

    if DEST.exists():
        print(f"\nRemoving previous clean dataset: {DEST}")
        shutil.rmtree(DEST)

    total = 0

    for split in SPLITS:
        source_split = SOURCE / split
        dest_split = DEST / split

        dest_split.mkdir(parents=True, exist_ok=True)

        print(f"\n[{split.upper()}]")

        for class_name in KEEP_CLASSES:
            # Test originally contains "Ringworm" with capital R.
            source_class_name = class_name

            if split == "test" and class_name == "ringworm":
                source_class_name = "Ringworm"

            source_class = source_split / source_class_name
            dest_class = dest_split / class_name

            if not source_class.exists():
                print(f"WARNING: missing {source_class}")
                continue

            dest_class.mkdir(parents=True, exist_ok=True)

            count = 0

            for file in source_class.iterdir():
                if file.is_file():
                    shutil.copy2(file, dest_class / file.name)
                    count += 1

            print(f"{class_name:40s} {count:5d}")
            total += count

    print("\n" + "=" * 60)
    print(f"Total clean images: {total}")
    print(f"Output: {DEST}")
    print("=" * 60)

    print("\nRemoved from training:")
    print("  - Unlabeled")
    print("\nNormalized:")
    print("  - Ringworm / ringworm -> ringworm")


if __name__ == "__main__":
    main()