"""
Batch ANPR Testing Script

Test the complete ANPR pipeline on multiple images.
"""

import sys
from pathlib import Path
import cv2
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.anpr_pipeline import ANPRPipeline
from rich.console import Console
from rich.table import Table
from rich.progress import track

console = Console()
logging.basicConfig(level=logging.WARNING)  # Reduce noise


def batch_test_anpr(folder_path: str):
    """
    Test ANPR pipeline on all images in a folder.

    Args:
        folder_path: Path to folder containing vehicle images
    """
    console.print(f"\n[bold cyan]🚗 Batch ANPR Testing[/bold cyan]\n")

    # Get all image files
    folder = Path(folder_path)
    image_files = list(folder.glob("*.jpg")) + list(folder.glob("*.jpeg")) + list(folder.glob("*.png"))

    if not image_files:
        console.print(f"[red]No images found in {folder_path}[/red]")
        return

    console.print(f"Found {len(image_files)} images in {folder_path}\n")

    # Initialize ANPR pipeline
    console.print("🤖 Initializing ANPR Pipeline...")
    console.print("   • License Plate Detector (YOLOv8 + Fallback)")
    console.print("   • OCR Engine (EasyOCR)")
    console.print("   • Indian Plate Corrections")

    pipeline = ANPRPipeline()
    console.print("✓ Pipeline ready\n")

    # Results storage
    results = []

    # Process each image
    for image_file in track(image_files, description="Processing images..."):
        try:
            # Load image
            image = cv2.imread(str(image_file))

            if image is None:
                results.append({
                    'filename': image_file.name,
                    'status': 'Error',
                    'plate_number': 'Failed to load',
                    'confidence': 0.0,
                    'plate_detected': False
                })
                continue

            # Process with ANPR pipeline
            result = pipeline.process_image(image, return_details=False)

            results.append({
                'filename': image_file.name,
                'status': result['status'],
                'plate_number': result.get('plate_number', 'N/A'),
                'confidence': result.get('confidence', 0.0),
                'plate_detected': result.get('plate_detected', False),
                'detection_conf': result.get('detection_confidence', 0.0),
                'corrected': result.get('corrected', False)
            })

        except Exception as e:
            results.append({
                'filename': image_file.name,
                'status': 'error',
                'plate_number': str(e)[:20],
                'confidence': 0.0,
                'plate_detected': False
            })

    # Display results
    console.print("\n[bold green]📊 Results:[/bold green]\n")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("#", style="dim", width=4)
    table.add_column("Filename", style="cyan", width=20)
    table.add_column("Status", style="white", width=15)
    table.add_column("License Plate", style="white", width=18)
    table.add_column("OCR Conf", justify="right", style="green", width=10)
    table.add_column("Det Conf", justify="right", style="blue", width=10)

    for i, result in enumerate(results, 1):
        # Status styling
        status = result['status']
        if status == 'success':
            status_display = "[green]✓ Success[/green]"
        elif status == 'no_plate_detected':
            status_display = "[yellow]⚠ No Plate[/yellow]"
        elif status == 'low_confidence':
            status_display = "[yellow]⚠ Low Conf[/yellow]"
        else:
            status_display = "[red]✗ Error[/red]"

        # Confidence styling
        ocr_conf = result.get('confidence', 0) * 100
        if ocr_conf >= 70:
            ocr_style = "green"
        elif ocr_conf >= 50:
            ocr_style = "yellow"
        else:
            ocr_style = "red"

        det_conf = result.get('detection_conf', 0) * 100
        det_style = "blue" if det_conf > 0 else "dim"

        plate_text = result['plate_number']
        if result.get('corrected'):
            plate_text += " *"  # Mark corrected plates

        table.add_row(
            str(i),
            result['filename'],
            status_display,
            plate_text,
            f"[{ocr_style}]{ocr_conf:.1f}%[/{ocr_style}]",
            f"[{det_style}]{det_conf:.1f}%[/{det_style}]"
        )

    console.print(table)
    console.print("\n[dim]* = Auto-corrected to Indian format[/dim]")

    # Statistics
    total = len(results)
    successful = sum(1 for r in results if r['status'] == 'success')
    plate_detected = sum(1 for r in results if r.get('plate_detected', False))
    high_conf = sum(1 for r in results if r.get('confidence', 0) >= 0.7)
    corrected = sum(1 for r in results if r.get('corrected', False))
    no_plate = sum(1 for r in results if r['status'] == 'no_plate_detected')
    low_conf = sum(1 for r in results if r['status'] == 'low_confidence')

    console.print(f"\n[bold]Summary:[/bold]")
    console.print(f"  Total images: {total}")
    console.print(f"  Plates detected: {plate_detected} ({plate_detected/total*100:.1f}%)")
    console.print(f"  Successful reads: {successful} ({successful/total*100:.1f}%)")
    console.print(f"  High confidence (≥70%): {high_conf} ({high_conf/total*100:.1f}%)")
    console.print(f"  Auto-corrected: {corrected}")
    console.print(f"  No plate detected: {no_plate}")
    console.print(f"  Low confidence: {low_conf}")

    # Save results to file
    output_file = "data/outputs/batch_anpr_results.txt"
    Path("data/outputs").mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("BATCH ANPR RESULTS\n")
        f.write("=" * 80 + "\n\n")

        for i, result in enumerate(results, 1):
            f.write(f"{i}. {result['filename']}\n")
            f.write(f"   Status: {result['status']}\n")
            f.write(f"   Plate Detected: {result.get('plate_detected', False)}\n")

            if result.get('plate_detected'):
                f.write(f"   Detection Confidence: {result.get('detection_conf', 0)*100:.2f}%\n")

            if result.get('plate_number'):
                f.write(f"   License Plate: {result['plate_number']}\n")
                f.write(f"   OCR Confidence: {result.get('confidence', 0)*100:.2f}%\n")

                if result.get('corrected'):
                    f.write(f"   Auto-corrected: Yes\n")

            f.write("\n")

        f.write(f"\nSummary:\n")
        f.write(f"Total: {total}\n")
        f.write(f"Plates detected: {plate_detected}\n")
        f.write(f"Successful: {successful}\n")
        f.write(f"High confidence: {high_conf}\n")
        f.write(f"Auto-corrected: {corrected}\n")

    console.print(f"\n✓ Detailed results saved to: {output_file}")


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description="Batch ANPR Testing")
    parser.add_argument(
        "--folder",
        type=str,
        default="data/datasets/indian_plates",
        help="Path to folder containing vehicle images"
    )

    args = parser.parse_args()

    batch_test_anpr(args.folder)


if __name__ == "__main__":
    main()
