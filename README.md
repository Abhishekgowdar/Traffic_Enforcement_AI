# 🚦 Trustworthy Explainable AI Framework for Smart Traffic Enforcement

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> An AI-powered traffic enforcement system using multimodal evidence fusion, human-in-the-loop verification, and voice-based decision support.

## 🌟 Features

- **🚗 Vehicle & License Plate Detection** - YOLOv8-based detection
- **🔤 Multi-OCR Fusion** - EasyOCR + PaddleOCR with confidence weighting
- **📹 Vehicle Tracking** - ByteTrack for multi-object tracking
- **⚠️ Violation Detection** - Red-light, speed, lane violations
- **🔍 Forgery Detection** - Deep learning-based plate manipulation detection
- **🤝 Evidence Fusion** - Multimodal evidence aggregation
- **🎯 Trust Assessment** - Confidence-based decision scoring
- **💡 Explainability (XAI)** - Grad-CAM + SHAP explanations
- **👤 Human-in-the-Loop** - Review interface for uncertain cases
- **🎤 Voice Assistant** - Natural language query support

## 🚀 Quick Start

### Installation

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download models
python scripts/download_models.py
```

### Run Demo

```bash
python scripts/demo.py --image data/samples/test_image.jpg
```

### Start Web Interface

```bash
# Backend
cd backend
uvicorn main:app --reload

# Frontend (new terminal)
cd frontend
npm install
npm run dev
```

## 📁 Project Structure

```
Traffic_Enforcement_AI/
├── configs/              # Configuration files
├── data/                 # Datasets and outputs
├── models/               # Trained models
├── src/                  # Source code
├── backend/              # FastAPI backend
├── frontend/             # React frontend
├── notebooks/            # Jupyter notebooks
├── tests/                # Unit tests
└── scripts/              # Helper scripts
```

## ⚙️ Configuration

Edit `configs/model_config.yaml` to change models:

```yaml
detection:
  active_model: "yolov8s"  # Your 32GB RAM can handle this

ocr:
  active_engines: ["easyocr", "paddleocr"]

hardware:
  device: "cpu"
  num_threads: 8
```

## 📖 Usage

### Python API

```python
from src.detection.detector import VehicleDetector
from src.ocr.ocr_engine import MultiOCREngine

detector = VehicleDetector()
ocr = MultiOCREngine()

image = cv2.imread("car.jpg")
vehicles = detector.detect(image)
plates = ocr.recognize(vehicles[0]['plate_crop'])
```

## 🧪 Testing

```bash
pytest tests/
```

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- Ultralytics YOLOv8
- EasyOCR, PaddleOCR
- ByteTrack, SHAP

---

**⭐ Star this repo if you find it useful!**
