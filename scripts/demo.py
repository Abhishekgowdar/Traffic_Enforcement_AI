"""
Demo Script - Vehicle Detection

Quick test of the traffic enforcement system.
"""

import sys
from pathlib import Path
import argparse
import cv2
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.detection import VehicleDetector
from src.utils.image_utils import draw_bbox, save_image
from rich.console import Console
from rich.table import Table

console = Console()
logging.basicConfig(level=logging.INFO)


def demo_detection(image_path: str, output_path: str = None, show: bool = False):
    """
    Run vehicle detection demo.

    Args:
        image_path: Path to input image
        output_path: Path to save output (optional)
        show: Display result window (default: False, use --show to enable)
    """
    console.print("\n[bold cyan]🚦 Traffic Enforcement AI - Demo[/bold cyan]\n")

    # Load image
    console.print(f"📂 Loading image: {image_path}")
    image = cv2.imread(image_path)

    if image is None:
        console.print(f"[red]✗ Error: Could not load image {image_path}[/red]")
        return

    console.print(f"✓ Image loaded: {image.shape[1]}x{image.shape[0]}\n")

    # Initialize detector
    console.print("🤖 Initializing detector...")
    detector = VehicleDetector()

    # Get detector info
    info = detector.get_info()
    console.print(f"✓ Using {info['model_type']} on {info['device']}\n")

    # Detect vehicles
    console.print("🔍 Detecting vehicles...")
    detections = detector.detect_vehicles(image)

    console.print(f"✓ Found {len(detections)} vehicle(s)\n")

    # Create results table
    if detections:
        table = Table(title="Detection Results")
        table.add_column("ID", style="cyan")
        table.add_column("Class", style="green")
        table.add_column("Confidence", style="yellow")
        table.add_column("BBox", style="magenta")

        for i, det in enumerate(detections, 1):
            bbox = det['bbox']
            table.add_row(
                str(i),
                det['class_name'],
                f"{det['confidence']:.2%}",
                f"({int(bbox[0])}, {int(bbox[1])}, {int(bbox[2])}, {int(bbox[3])})"
            )

        console.print(table)

        # Draw boxes on image
        result_image = image.copy()
        for det in detections:
            label = f"{det['class_name']} {det['confidence']:.2f}"
            result_image = draw_bbox(
                result_image,
                det['bbox'],
                label=label,
                color=(0, 255, 0)
            )

        # Save result
        if output_path:
            save_image(result_image, output_path)
            console.print(f"\n✓ Result saved to: {output_path}")
        else:
            output_path = "data/outputs/demo_result.jpg"
            Path("data/outputs").mkdir(parents=True, exist_ok=True)
            save_image(result_image, output_path)
            console.print(f"\n✓ Result saved to: {output_path}")

        # Show result
        if show:
            console.print("\n[dim]Press any key to close the window...[/dim]")
            cv2.imshow("Detection Result", result_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    else:
        console.print("[yellow]No vehicles detected in the image.[/yellow]")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Vehicle Detection Demo")
    parser.add_argument(
        "--image",
        type=str,
        help="Path to input image",
        default=None
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to save output image",
        default=None
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display result window (requires OpenCV GUI support)"
    )

    args = parser.parse_args()

    # Check if image path provided
    if args.image is None:
        console.print("[yellow]No image provided. Usage:[/yellow]")
        console.print("  python scripts/demo.py --image path/to/image.jpg")
        console.print("\n[dim]Example:[/dim]")
        console.print("  python scripts/demo.py --image data/samples/car.jpg")

        # Check if sample image exists
        sample_path = "data/samples/test_image.jpg"
        if Path(sample_path).exists():
            console.print(f"\n[green]Found sample image. Using: {sample_path}[/green]")
            demo_detection(sample_path, args.output, args.show)
        else:
            console.print("\n[red]No sample images found. Please provide an image path.[/red]")
            sys.exit(1)
    else:
        demo_detection(args.image, args.output, args.show)


if __name__ == "__main__":
    main()
