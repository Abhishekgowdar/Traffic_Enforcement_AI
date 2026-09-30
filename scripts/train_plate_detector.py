"""
Train a YOLOv8 license-plate detector on the prepared `plates_det` dataset.

Examples
--------
Quick smoke test (tiny subset, 1 epoch) to verify the pipeline:
    python scripts/train_plate_detector.py --quick

Full CPU training (run it and leave it; this is slow on CPU):
    python scripts/train_plate_detector.py --epochs 40 --imgsz 512 --batch 16

The best weights are copied to models/detection/plate_yolov8.pt on completion.
"""

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 plate detector")
    parser.add_argument("--data", default=str(ROOT / "data/datasets/plates_det/data.yaml"))
    parser.add_argument("--model", default="yolov8n.pt", help="Base model to fine-tune")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="cpu", help="'cpu' or '0' for GPU")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--fraction", type=float, default=1.0,
                        help="Fraction of the training set to use")
    parser.add_argument("--quick", action="store_true",
                        help="Fast smoke test: 1 epoch, small imgsz, 5%% of data")
    args = parser.parse_args()

    if args.quick:
        args.epochs = 1
        args.imgsz = 320
        args.fraction = 0.05
        args.workers = 0

    from ultralytics import YOLO

    print(f"Loading base model: {args.model}")
    model = YOLO(args.model)

    print(f"Training on {args.data}  (epochs={args.epochs}, imgsz={args.imgsz}, "
          f"batch={args.batch}, device={args.device}, fraction={args.fraction})")

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        fraction=args.fraction,
        project=str(ROOT / "models/detection/runs"),
        name="plate_yolov8n",
        exist_ok=True,
        patience=15,
        verbose=True,
    )

    # Locate and publish the best weights
    save_dir = Path(results.save_dir) if hasattr(results, "save_dir") else \
        ROOT / "models/detection/runs/plate_yolov8n"
    best = save_dir / "weights" / "best.pt"
    if best.exists():
        dest = ROOT / "models/detection/plate_yolov8.pt"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best, dest)
        print(f"\n✓ Best weights copied to: {dest}")
    else:
        print(f"\n⚠ Could not find best.pt under {save_dir}")

    print("Training complete.")


if __name__ == "__main__":
    main()
