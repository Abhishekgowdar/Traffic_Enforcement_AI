"""
Prepare the `archive` detection dataset for YOLO training.

Reads the YOLO-format images/labels from the source folder, keeps only images
that have a matching non-empty label, splits into train/val, and copies them
into data/datasets/plates_det/ with a data.yaml.
"""

import argparse
import random
import shutil
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def find_label(labels_dir: Path, stem: str) -> Path | None:
    candidate = labels_dir / f"{stem}.txt"
    return candidate if candidate.exists() else None


def main():
    parser = argparse.ArgumentParser(description="Prepare YOLO detection dataset")
    parser.add_argument("--src", default=r"C:\Users\BWM743\Downloads\archive",
                        help="Source folder containing images/ and labels/")
    parser.add_argument("--out", default="data/datasets/plates_det",
                        help="Output dataset folder")
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    src = Path(args.src)
    src_images = src / "images"
    src_labels = src / "labels"
    out = Path(args.out)

    if not src_images.exists() or not src_labels.exists():
        raise SystemExit(f"Expected {src_images} and {src_labels} to exist")

    # Match images to labels
    pairs = []
    for img in src_images.iterdir():
        if img.suffix.lower() not in IMG_EXTS:
            continue
        label = find_label(src_labels, img.stem)
        if label and label.stat().st_size > 0:
            pairs.append((img, label))

    print(f"Found {len(pairs)} image/label pairs")
    if not pairs:
        raise SystemExit("No valid image/label pairs found")

    random.seed(args.seed)
    random.shuffle(pairs)

    n_val = int(len(pairs) * args.val_split)
    splits = {"val": pairs[:n_val], "train": pairs[n_val:]}

    # Reset output tree
    if out.exists():
        shutil.rmtree(out)
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    for split, items in splits.items():
        for img, label in items:
            shutil.copy2(img, out / "images" / split / img.name)
            shutil.copy2(label, out / "labels" / split / f"{img.stem}.txt")
        print(f"  {split}: {len(items)} images")

    # data.yaml (absolute path so YOLO finds it from anywhere)
    data_yaml = out / "data.yaml"
    data_yaml.write_text(
        f"path: {out.resolve().as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"nc: 1\n"
        f"names: ['license_plate']\n",
        encoding="utf-8",
    )
    print(f"Wrote {data_yaml}")
    print("Done.")


if __name__ == "__main__":
    main()
