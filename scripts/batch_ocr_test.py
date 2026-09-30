"""
Batch OCR Testing Script
Test OCR on multiple images in a folder
"""

import sys
from pathlib import Path
import cv2
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ocr import MultiOCREngine
from rich.console import Console
from rich.table import Table
from rich.progress import track

console = Console()
logging.basicConfig(level=logging.WARNING)  # Reduce logging noise


def batch_test_ocr(folder_path: str):
    """
    Test OCR on all images in a folder.

    Args:
        folder_path: Path to folder containing images
    """
    console.print(f"\n[bold cyan]🔤 Batch OCR Testing[/bold cyan]\n")

    # Get all image files
    folder = Path(folder_path)
    image_files = list(folder.glob("*.jpg")) + list(folder.glob("*.jpeg")) + list(folder.glob("*.png"))

    if not image_files:
        console.print(f"[red]No images found in {folder_path}[/red]")
        return

    console.print(f"Found {len(image_files)} images in {folder_path}\n")

    # Initialize OCR engine
    console.print("🤖 Initializing OCR engine...")
    ocr_engine = MultiOCREngine()
    console.print("✓ OCR ready\n")

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
                    'text': 'Failed to load',
                    'confidence': 0.0
                })
                continue

            # Run OCR
            result = ocr_engine.read_text(image, preprocess=True, postprocess=True, plate_format='indian')

            results.append({
                'filename': image_file.name,
                'status': '✓',
                'text': result.get('text', 'N/A'),
                'raw_text': result.get('raw_text', 'N/A'),
                'confidence': result.get('confidence', 0.0),
                'corrected': result.get('corrected', False)
            })

        except Exception as e:
            results.append({
                'filename': image_file.name,
                'status': 'Error',
                'text': str(e)[:30],
                'confidence': 0.0
            })

    # Display results
    console.print("\n[bold green]📊 Results:[/bold green]\n")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("#", style="dim", width=4)
    table.add_column("Filename", style="cyan", width=25)
    table.add_column("Detected Text", style="white", width=20)
    table.add_column("Confidence", justify="right", style="green", width=12)
    table.add_column("Corrected", justify="center", width=10)

    for i, result in enumerate(results, 1):
        confidence_str = f"{result.get('confidence', 0)*100:.1f}%"
        corrected_str = "Yes" if result.get('corrected', False) else "No"

        # Color code confidence
        if result.get('confidence', 0) >= 0.7:
            conf_style = "green"
        elif result.get('confidence', 0) >= 0.5:
            conf_style = "yellow"
        else:
            conf_style = "red"

        table.add_row(
            str(i),
            result['filename'],
            result['text'],
            f"[{conf_style}]{confidence_str}[/{conf_style}]",
            corrected_str
        )

    console.print(table)

    # Statistics
    total = len(results)
    successful = sum(1 for r in results if r['status'] == '✓')
    high_conf = sum(1 for r in results if r.get('confidence', 0) >= 0.7)
    corrected = sum(1 for r in results if r.get('corrected', False))

    console.print(f"\n[bold]Summary:[/bold]")
    console.print(f"  Total images: {total}")
    console.print(f"  Successful: {successful} ({successful/total*100:.1f}%)")
    console.print(f"  High confidence (≥70%): {high_conf} ({high_conf/total*100:.1f}%)")
    console.print(f"  Auto-corrected: {corrected}")

    # Save results to file
    output_file = "data/outputs/batch_ocr_results.txt"
    Path("data/outputs").mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("BATCH OCR RESULTS\n")
        f.write("=" * 80 + "\n\n")

        for i, result in enumerate(results, 1):
            f.write(f"{i}. {result['filename']}\n")
            f.write(f"   Detected: {result['text']}\n")

            if result.get('raw_text') and result.get('corrected'):
                f.write(f"   Raw OCR: {result['raw_text']}\n")

            f.write(f"   Confidence: {result.get('confidence', 0)*100:.2f}%\n")
            f.write(f"   Status: {result['status']}\n")
            f.write("\n")

        f.write(f"\nSummary:\n")
        f.write(f"Total: {total}\n")
        f.write(f"Successful: {successful}\n")
        f.write(f"High confidence: {high_conf}\n")
        f.write(f"Auto-corrected: {corrected}\n")

    console.print(f"\n✓ Results saved to: {output_file}")


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description="Batch OCR Testing")
    parser.add_argument(
        "--folder",
        type=str,
        default="data/datasets/indian_plates",
        help="Path to folder containing images"
    )

    args = parser.parse_args()

    batch_test_ocr(args.folder)


if __name__ == "__main__":
    main()
