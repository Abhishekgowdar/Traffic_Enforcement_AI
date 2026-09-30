"""
Benchmark the OCR pipeline against the labeled `number_plate` dataset.

Reads number_plate.xlsx (columns: Image, GroundTruthNumber), runs each cropped
plate through the OCR pipeline, and reports exact-match and character-level
accuracy. Use --limit to sample a subset for a quick estimate.

Example:
    python scripts/benchmark_ocr.py --limit 150
    python scripts/benchmark_ocr.py --limit 0   # full dataset (slow)
"""

import argparse
import sys
from pathlib import Path

import cv2
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.ocr import MultiOCREngine


def normalize(text: str) -> str:
    return "".join(ch for ch in str(text).upper() if ch.isalnum())


def levenshtein(a: str, b: str) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def main():
    parser = argparse.ArgumentParser(description="Benchmark OCR on labeled plates")
    parser.add_argument("--src", default=r"C:\Users\BWM743\Downloads\number_plate")
    parser.add_argument("--limit", type=int, default=150,
                        help="Number of images to test (0 = all)")
    parser.add_argument("--show-errors", type=int, default=15,
                        help="How many mismatches to print")
    args = parser.parse_args()

    src = Path(args.src)
    xlsx = src / "number_plate.xlsx"
    images_dir = src / "images"

    df = pd.read_excel(xlsx)
    if args.limit and args.limit > 0:
        df = df.head(args.limit)

    print(f"Benchmarking OCR on {len(df)} plates...\n")
    ocr = MultiOCREngine()

    exact = 0
    total_chars = 0
    total_edits = 0
    errors = []

    for _, row in df.iterrows():
        img_path = images_dir / str(row["Image"])
        truth = normalize(row["GroundTruthNumber"])
        image = cv2.imread(str(img_path))
        if image is None:
            continue

        result = ocr.read_text(image, preprocess=False, postprocess=True, plate_format="indian")
        pred = normalize(result.get("text", ""))

        if pred == truth:
            exact += 1
        else:
            errors.append((row["Image"], truth, pred))

        total_chars += len(truth)
        total_edits += levenshtein(pred, truth)

    n = len(df)
    char_acc = 100 * (1 - total_edits / total_chars) if total_chars else 0
    print("=" * 48)
    print(f"Exact-match accuracy : {exact}/{n} = {100 * exact / n:.1f}%")
    print(f"Character accuracy   : {char_acc:.1f}%")
    print("=" * 48)

    if errors and args.show_errors:
        print(f"\nSample mismatches (truth -> predicted):")
        for name, truth, pred in errors[:args.show_errors]:
            print(f"  {name}: {truth} -> {pred or '(empty)'}")


if __name__ == "__main__":
    main()
