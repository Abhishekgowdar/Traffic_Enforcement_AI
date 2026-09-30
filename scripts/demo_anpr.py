"""
ANPR Demo Script

Demonstrates the complete ANPR pipeline on vehicle images.
"""

import sys
from pathlib import Path
import argparse
import cv2
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.anpr_pipeline import ANPRPipeline
from rich.console import Console
from rich.panel import Panel

console = Console()
logging.basicConfig(level=logging.INFO)


def demo_anpr(image_path: str, show_plate: bool = True, save_result: bool = True):
    """
    Run ANPR demo on a single image.

    Args:
        image_path: Path to vehicle image
        show_plate: Display cropped plate image
        save_result: Save annotated result
    """
    console.print("\n[bold cyan]🚗 ANPR Pipeline Demo[/bold cyan]\n")

    # Load image
    console.print(f"📂 Loading image: {image_path}")
    image = cv2.imread(image_path)

    if image is None:
        console.print(f"[red]✗ Error: Could not load image {image_path}[/red]")
        return

    height, width = image.shape[:2]
    console.print(f"✓ Image loaded: {width}x{height}\n")

    # Initialize ANPR pipeline
    console.print("🤖 Initializing ANPR Pipeline...")
    console.print("   • License Plate Detector")
    console.print("   • OCR Engine (EasyOCR + Indian Plate Corrections)")

    pipeline = ANPRPipeline()
    console.print("✓ Pipeline ready\n")

    # Process image
    console.print("🔍 Processing image...")
    console.print("   [dim]Step 1: Detecting license plate...[/dim]")
    console.print("   [dim]Step 2: Cropping plate region...[/dim]")
    console.print("   [dim]Step 3: Running OCR...[/dim]")
    console.print("   [dim]Step 4: Applying corrections...[/dim]\n")

    result = pipeline.process_image(image, return_details=True)

    # Display results
    if result['status'] == 'success':
        # Success
        plate_number = result['plate_number']
        confidence = result['confidence'] * 100

        console.print(Panel(
            f"[bold green]{plate_number}[/bold green]",
            title="✓ License Plate Detected",
            border_style="green"
        ))

        console.print(f"\n[bold]Details:[/bold]")
        console.print(f"  Confidence: {confidence:.2f}%")
        console.print(f"  Detection Confidence: {result.get('detection_confidence', 0)*100:.2f}%")

        if result.get('corrected'):
            console.print(f"  Raw OCR: {result.get('raw_ocr_text', 'N/A')}")
            console.print(f"  [yellow]✓ Auto-corrected to Indian format[/yellow]")

        if 'ocr_details' in result and 'details' in result['ocr_details']:
            details = result['ocr_details']['details']
            console.print(f"\n[cyan]Indian Plate Validation:[/cyan]")
            console.print(f"  State Code: {details.get('state_code', 'N/A')} {'✓' if details.get('state_valid') else '✗'}")
            console.print(f"  Format: {details.get('formatted', 'N/A')}")
            console.print(f"  Valid: {'✓ Yes' if details.get('format_valid') else '✗ No'}")

            if details.get('corrections'):
                console.print(f"\n  [dim]Corrections applied:[/dim]")
                for correction in details['corrections'][:3]:
                    console.print(f"    • {correction}")

    elif result['status'] == 'no_plate_detected':
        console.print("[yellow]⚠ No license plate detected in image[/yellow]")
        console.print("[dim]  Possible reasons:[/dim]")
        console.print("[dim]  • Plate too small or far away[/dim]")
        console.print("[dim]  • Plate obscured or not visible[/dim]")
        console.print("[dim]  • Image quality too low[/dim]")

    elif result['status'] == 'low_confidence':
        console.print(f"[yellow]⚠ Plate detected but low OCR confidence[/yellow]")
        console.print(f"  Detected: {result.get('plate_number', 'N/A')}")
        console.print(f"  Confidence: {result.get('confidence', 0)*100:.2f}%")

    else:
        console.print(f"[red]✗ Error: {result.get('error', 'Unknown error')}[/red]")

    # Save visualized result
    if save_result and result.get('plate_detected'):
        output_path = "data/outputs/anpr_result.jpg"
        Path("data/outputs").mkdir(parents=True, exist_ok=True)

        annotated = pipeline.visualize_result(image, result, save_path=output_path)
        console.print(f"\n✓ Annotated image saved to: {output_path}")

        # Save cropped plate
        if 'plate_image' in result:
            plate_output = "data/outputs/anpr_plate_crop.jpg"
            cv2.imwrite(plate_output, result['plate_image'])
            console.print(f"✓ Cropped plate saved to: {plate_output}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="ANPR Pipeline Demo")
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to vehicle image"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save results"
    )

    args = parser.parse_args()

    demo_anpr(args.image, save_result=not args.no_save)


if __name__ == "__main__":
    main()
