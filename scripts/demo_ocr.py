"""
Demo Script - OCR (License Plate Reading)

Test the OCR system for reading license plates.
"""

import sys
from pathlib import Path
import argparse
import cv2
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ocr import MultiOCREngine
from rich.console import Console
from rich.table import Table

console = Console()
logging.basicConfig(level=logging.INFO)


def demo_ocr(image_path: str):
    """
    Run OCR demo on license plate image.

    Args:
        image_path: Path to license plate image
    """
    console.print("\n[bold cyan]🔤 Traffic Enforcement AI - OCR Demo[/bold cyan]\n")

    # Load image
    console.print(f"📂 Loading image: {image_path}")
    image = cv2.imread(image_path)

    if image is None:
        console.print(f"[red]✗ Error: Could not load image {image_path}[/red]")
        return

    console.print(f"✓ Image loaded: {image.shape[1]}x{image.shape[0]}\n")

    # Show image preview
    console.print("[dim]Image preview saved to: data/outputs/ocr_input.jpg[/dim]")
    Path("data/outputs").mkdir(parents=True, exist_ok=True)
    cv2.imwrite("data/outputs/ocr_input.jpg", image)

    # Initialize OCR engine
    console.print("🤖 Initializing OCR engine...")
    try:
        ocr_engine = MultiOCREngine()
        console.print("✓ OCR engine ready\n")
    except Exception as e:
        console.print(f"[red]✗ Error initializing OCR: {e}[/red]")
        return

    # Run OCR
    console.print("🔍 Reading text from image...")
    try:
        result = ocr_engine.read_text(image)

        if result and 'text' in result:
            console.print(f"\n[bold green]✓ Text detected: {result['text']}[/bold green]")
            console.print(f"[dim]Confidence: {result.get('confidence', 0)*100:.2f}%[/dim]")

            # Show if corrections were made
            if result.get('corrected', False):
                console.print(f"\n[yellow]Post-processing corrections applied:[/yellow]")
                console.print(f"  Raw OCR: {result.get('raw_text', 'N/A')}")
                console.print(f"  Corrected: {result['text']}")
                console.print(f"  [dim]Confidence adjusted: {result.get('raw_confidence', 0)*100:.2f}% → {result.get('confidence', 0)*100:.2f}%[/dim]")

                # Show Indian plate details if available
                if 'details' in result:
                    details = result['details']
                    console.print(f"\n[cyan]Indian License Plate Details:[/cyan]")
                    console.print(f"  State Code: {details.get('state_code', 'N/A')} {'✓' if details.get('state_valid') else '✗ (invalid)'}")
                    console.print(f"  Formatted: {details.get('formatted', 'N/A')}")
                    console.print(f"  Format Valid: {'✓ Yes' if details.get('format_valid') else '✗ No'}")

                    if details.get('corrections'):
                        console.print(f"\n  [dim]Corrections made:[/dim]")
                        for correction in details['corrections'][:5]:  # Show first 5
                            console.print(f"    • {correction}")

            # Show detailed results if available
            if 'engine_results' in result:
                console.print("\n[bold]Individual Engine Results:[/bold]")
                table = Table(show_header=True, header_style="bold magenta")
                table.add_column("Engine", style="cyan")
                table.add_column("Text", style="white")
                table.add_column("Confidence", justify="right", style="green")

                for engine_name, engine_result in result['engine_results'].items():
                    if engine_result:
                        text = engine_result.get('text', 'N/A')
                        conf = engine_result.get('confidence', 0) * 100
                        table.add_row(engine_name, text, f"{conf:.2f}%")

                console.print(table)

            # Draw result on image
            if 'bbox' in result:
                bbox = result['bbox']
                cv2.rectangle(image,
                            (int(bbox[0]), int(bbox[1])),
                            (int(bbox[2]), int(bbox[3])),
                            (0, 255, 0), 3)

                # Add text label
                text_label = f"{result['text']} ({result.get('confidence', 0)*100:.1f}%)"
                cv2.putText(image, text_label,
                          (int(bbox[0]), int(bbox[1])-10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Save result
            output_path = "data/outputs/ocr_result.jpg"
            cv2.imwrite(output_path, image)
            console.print(f"\n✓ Result saved to: {output_path}")

        else:
            console.print("\n[yellow]No text detected in the image.[/yellow]")

    except Exception as e:
        console.print(f"\n[red]✗ Error during OCR: {e}[/red]")
        import traceback
        console.print(f"[dim]{traceback.format_exc()}[/dim]")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="OCR Demo - License Plate Reading")
    parser.add_argument(
        "--image",
        type=str,
        help="Path to license plate image",
        default=None
    )

    args = parser.parse_args()

    if args.image is None:
        console.print("[yellow]No image provided.[/yellow]")
        console.print("\n[bold]Creating a test license plate...[/bold]")

        # Create test plate
        import subprocess
        result = subprocess.run(
            ["python", "scripts/create_test_plate.py"],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            test_image = "data/samples/test_plate.jpg"
            console.print(f"[green]✓ Test plate created: {test_image}[/green]\n")
            demo_ocr(test_image)
        else:
            console.print("[red]Failed to create test plate[/red]")
            console.print("\n[yellow]Usage:[/yellow]")
            console.print("  python scripts/demo_ocr.py --image path/to/plate.jpg")
    else:
        demo_ocr(args.image)


if __name__ == "__main__":
    main()
