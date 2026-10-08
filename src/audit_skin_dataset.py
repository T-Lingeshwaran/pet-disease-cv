from pathlib import Path
from collections import Counter

DATA_DIR = Path("data/skin_coat/combined_clean")
SPLITS = ["train", "valid", "test"]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def count_images(folder):
    return sum(
        1
        for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    )


def main():
    print("=" * 70)
    print("SKIN DATASET AUDIT")
    print("=" * 70)

    all_counts = {}

    for split in SPLITS:
        split_dir = DATA_DIR / split
        counts = {}

        for class_dir in sorted(split_dir.iterdir()):
            if class_dir.is_dir():
                counts[class_dir.name] = count_images(class_dir)

        all_counts[split] = counts

        print(f"\n{split.upper()}")
        print("-" * 70)

        for name, count in counts.items():
            print(f"{name:40s} {count:6d}")

        print(f"{'TOTAL':40s} {sum(counts.values()):6d}")

    # Combined totals
    print("\n" + "=" * 70)
    print("COMBINED TOTALS")
    print("=" * 70)

    combined = Counter()

    for split in SPLITS:
        combined.update(all_counts[split])

    for name, count in combined.most_common():
        print(f"{name:40s} {count:6d}")

    print("-" * 70)
    print(f"{'TOTAL':40s} {sum(combined.values()):6d}")

    largest = max(combined.values())
    smallest = min(combined.values())

    print(f"\nLargest class : {largest}")
    print(f"Smallest class: {smallest}")
    print(f"Imbalance ratio: {largest / smallest:.2f}x")

    print("\nClass order:")
    print(sorted(combined.keys()))


if __name__ == "__main__":
    main()