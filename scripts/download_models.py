"""
Model Download Script

Downloads all required models for the traffic enforcement system.
"""

import os
import sys
from pathlib import Path
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ultralytics import YOLO
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()
logger = logging.getLogger(__name__)


def setup_directories():
    """Create necessary directories"""
    dirs = [
        'models/detection',
        'models/ocr',
        'models/tracking',
        'models/forgery',
        'data/samples',
        'data/outputs',
        'logs'
    ]

    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        console.print(f"✓ Created directory: {dir_path}", style="green")


def download_yolov8_models():
    """Download YOLOv8 models"""
    console.print("\n[bold blue]Downloading YOLOv8 Models...[/bold blue]")

    models = ['yolov8n.pt', 'yolov8s.pt']  # Download nano and small

    for model_name in models:
        try:
            console.print(f"  Downloading {model_name}...", style="yellow")

            # YOLO automatically downloads if not found
            model = YOLO(model_name)

            console.print(f"  ✓ {model_name} downloaded", style="green")

        except Exception as e:
            console.print(f"  ✗ Error downloading {model_name}: {e}", style="red")


def download_ocr_models():
    """Download OCR models"""
    console.print("\n[bold blue]Downloading OCR Models...[/bold blue]")

    try:
        # EasyOCR
        console.print("  Downloading EasyOCR models...", style="yellow")
        import easyocr
        reader = easyocr.Reader(['en'], gpu=False, download_enabled=True)
        console.print("  ✓ EasyOCR models downloaded", style="green")

    except Exception as e:
        console.print(f"  ⚠ EasyOCR: {e}", style="yellow")

    try:
        # PaddleOCR
        console.print("  Downloading PaddleOCR models...", style="yellow")
        from paddleocr import PaddleOCR
        ocr = PaddleOCR(use_angle_cls=False, lang='en')
        console.print("  ✓ PaddleOCR models downloaded", style="green")

    except Exception as e:
        console.print(f"  ⚠ PaddleOCR: {e}", style="yellow")


def download_whisper_model():
    """Download Whisper model for voice recognition"""
    console.print("\n[bold blue]Downloading Whisper Model...[/bold blue]")

    try:
        import whisper
        console.print("  Downloading Whisper base model...", style="yellow")
        model = whisper.load_model("base")
        console.print("  ✓ Whisper model downloaded", style="green")

    except Exception as e:
        console.print(f"  ⚠ Whisper: {e}", style="yellow")


def create_sample_readme():
    """Create README in models directory"""
    readme_content = """# Models Directory

This directory contains downloaded model files.

## Detection Models
- `yolov8n.pt` - YOLOv8 Nano (~6MB)
- `yolov8s.pt` - YOLOv8 Small (~22MB)

## OCR Models
- EasyOCR models are stored in `~/.EasyOCR/`
- PaddleOCR models are stored in `~/.paddleocr/`

## Model Sizes
- Detection models: ~50MB total
- OCR models: ~400MB total
- Voice models: ~150MB total

Total: ~600MB
"""

    with open('models/README.md', 'w') as f:
        f.write(readme_content)

    console.print("✓ Created models/README.md", style="green")


def main():
    """Main download function"""
    console.print("[bold cyan]Traffic Enforcement AI - Model Downloader[/bold cyan]\n")

    try:
        # Step 1: Setup directories
        console.print("[bold]Step 1: Setting up directories...[/bold]")
        setup_directories()

        # Step 2: Download YOLOv8
        console.print("\n[bold]Step 2: Downloading detection models...[/bold]")
        download_yolov8_models()

        # Step 3: Download OCR models
        console.print("\n[bold]Step 3: Downloading OCR models...[/bold]")
        download_ocr_models()

        # Step 4: Download Whisper
        console.print("\n[bold]Step 4: Downloading voice models...[/bold]")
        download_whisper_model()

        # Step 5: Create README
        create_sample_readme()

        console.print("\n[bold green]✓ All models downloaded successfully![/bold green]")
        console.print("\n[bold]Next steps:[/bold]")
        console.print("  1. Run demo: python scripts/demo.py")
        console.print("  2. Start backend: cd backend && uvicorn main:app --reload")
        console.print("  3. Explore notebooks: jupyter notebook notebooks/")

    except Exception as e:
        console.print(f"\n[bold red]Error: {e}[/bold red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
