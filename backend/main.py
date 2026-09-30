"""
Traffic Enforcement AI - Backend API

FastAPI backend for the traffic enforcement system.
"""

import sys
import os
import base64
import threading
import time
import webbrowser
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import cv2
import numpy as np
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.detection import VehicleDetector
from src.ocr import MultiOCREngine
from src.anpr_pipeline import ANPRPipeline

# Frontend assets directory
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Traffic Enforcement AI API",
    description="API for intelligent traffic enforcement",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def no_cache_frontend(request, call_next):
    """Prevent the browser from caching the dashboard assets during development."""
    response = await call_next(request)
    path = request.url.path
    if path == "/" or path.startswith("/static"):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response

# Initialize models (lazy loading)
detector = None
ocr_engine = None
anpr_pipeline = None


def get_detector():
    """Get or initialize detector"""
    global detector
    if detector is None:
        logger.info("Initializing vehicle detector...")
        detector = VehicleDetector()
    return detector


def get_ocr():
    """Get or initialize OCR"""
    global ocr_engine
    if ocr_engine is None:
        logger.info("Initializing OCR engine...")
        ocr_engine = MultiOCREngine()
    return ocr_engine


def get_anpr():
    """Get or initialize the full ANPR pipeline"""
    global anpr_pipeline
    if anpr_pipeline is None:
        logger.info("Initializing ANPR pipeline...")
        anpr_pipeline = ANPRPipeline()
    return anpr_pipeline


def _encode_image(image: np.ndarray) -> str:
    """Encode a BGR image as a base64 data URI for the browser"""
    success, buffer = cv2.imencode(".jpg", image)
    if not success:
        return ""
    encoded = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def _read_upload(contents: bytes) -> np.ndarray:
    """Decode uploaded bytes into an OpenCV image"""
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image file")
    return image


@app.get("/")
async def root():
    """Serve the dashboard if built, else return API info"""
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {
        "message": "Traffic Enforcement AI API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/api")
async def api_info():
    """API metadata endpoint"""
    return {
        "message": "Traffic Enforcement AI API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


@app.post("/api/v1/detect")
async def detect_vehicles(file: UploadFile = File(...)):
    """
    Detect vehicles in uploaded image.

    Args:
        file: Image file

    Returns:
        Detections with bounding boxes and confidences
    """
    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        # Get detector
        detector = get_detector()

        # Detect vehicles
        detections = detector.detect_vehicles(image)

        return {
            "success": True,
            "count": len(detections),
            "detections": detections
        }

    except Exception as e:
        logger.error(f"Error during detection: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/ocr")
async def recognize_plate(file: UploadFile = File(...)):
    """
    Recognize license plate text from image.

    Args:
        file: License plate image

    Returns:
        Recognized text and confidence
    """
    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        # Get OCR
        ocr = get_ocr()

        # Recognize
        result = ocr.recognize(image)

        return {
            "success": True,
            "text": result['fused_text'],
            "confidence": result['confidence'],
            "engines": result['individual_results']
        }

    except Exception as e:
        logger.error(f"Error during OCR: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/full-pipeline")
async def full_pipeline(file: UploadFile = File(...)):
    """
    Run full detection + OCR pipeline.

    Args:
        file: Traffic scene image

    Returns:
        Vehicles with license plates and OCR results
    """
    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        # Get models
        detector = get_detector()
        ocr = get_ocr()

        # Detect vehicles
        vehicles = detector.detect_vehicles(image)

        # For each vehicle, try to detect and recognize plate
        # (Simplified - in production, use proper plate detection)
        results = []

        for vehicle in vehicles:
            vehicle_data = {
                'vehicle': vehicle,
                'plate_text': None,
                'plate_confidence': 0.0
            }

            # Extract vehicle region
            bbox = vehicle['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            vehicle_crop = image[y1:y2, x1:x2]

            # Try OCR on vehicle region
            try:
                ocr_result = ocr.recognize(vehicle_crop)
                if ocr_result['confidence'] > 0.5:
                    vehicle_data['plate_text'] = ocr_result['fused_text']
                    vehicle_data['plate_confidence'] = ocr_result['confidence']
            except:
                pass

            results.append(vehicle_data)

        return {
            "success": True,
            "count": len(results),
            "results": results
        }

    except Exception as e:
        logger.error(f"Error during full pipeline: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/anpr")
async def anpr(file: UploadFile = File(...)):
    """
    Run the full ANPR pipeline: detect plate, crop, OCR, and correct.

    Returns the recognised plate number, confidence, and an annotated image.
    """
    try:
        contents = await file.read()
        image = _read_upload(contents)

        pipeline = get_anpr()
        result = pipeline.process_image(image, return_details=True)
        annotated = pipeline.visualize_result(image, result)

        response = {
            "success": result.get("status") == "success",
            "status": result.get("status"),
            "plate_number": result.get("plate_number"),
            "confidence": float(result.get("confidence") or 0.0),
            "detection_confidence": float(result.get("detection_confidence") or 0.0),
            "plate_detected": bool(result.get("plate_detected", False)),
            "raw_ocr_text": result.get("raw_ocr_text"),
            "corrected": bool(result.get("corrected", False)),
            "annotated_image": _encode_image(annotated),
        }

        plate_crop = result.get("plate_image")
        if plate_crop is not None and getattr(plate_crop, "size", 0) > 0:
            response["plate_crop"] = _encode_image(plate_crop)

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during ANPR: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/detect-visual")
async def detect_visual(file: UploadFile = File(...)):
    """
    Detect vehicles and return an annotated image with bounding boxes.
    """
    try:
        contents = await file.read()
        image = _read_upload(contents)

        detector = get_detector()
        detections = detector.detect_vehicles(image)

        annotated = image.copy()
        for det in detections:
            x1, y1, x2, y2 = map(int, det["bbox"])
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 200, 0), 2)
            label = f"{det['class_name']} {det['confidence'] * 100:.0f}%"
            cv2.putText(annotated, label, (x1, max(y1 - 8, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)

        return {
            "success": True,
            "count": len(detections),
            "detections": detections,
            "annotated_image": _encode_image(annotated),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during visual detection: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/models/info")
async def get_models_info():
    """Get information about loaded models"""
    detector = get_detector()
    return {
        "detector": detector.get_info(),
        "ocr": {
            "engines": ["easyocr", "paddleocr"],
            "fusion_method": "confidence_weighted"
        }
    }


# Serve static frontend assets (css/js) if the dashboard exists
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


def _open_dashboard_in_browser():
    """Open the dashboard in Chrome (falls back to the default browser)."""
    base = os.environ.get("TEA_DASHBOARD_URL", "http://127.0.0.1:8000")
    # Append a timestamp so the browser always loads a fresh page (never a stale cache)
    sep = "&" if "?" in base else "?"
    url = f"{base}{sep}t={int(time.time())}"
    chrome_paths = [
        os.path.join(os.environ.get("ProgramFiles", ""), r"Google\Chrome\Application\chrome.exe"),
        os.path.join(os.environ.get("ProgramFiles(x86)", ""), r"Google\Chrome\Application\chrome.exe"),
        os.path.join(os.environ.get("LocalAppData", ""), r"Google\Chrome\Application\chrome.exe"),
    ]
    for path in chrome_paths:
        if path and os.path.exists(path):
            try:
                webbrowser.register("chrome", None, webbrowser.BackgroundBrowser(path))
                webbrowser.get("chrome").open(url)
                return
            except Exception:
                break
    try:
        webbrowser.open(url)
    except Exception as e:
        logger.warning(f"Could not auto-open browser: {e}")


@app.on_event("startup")
async def _auto_open_browser():
    """Launch the dashboard in the browser once the server is ready."""
    if os.environ.get("TEA_NO_BROWSER") == "1":
        return
    # Small delay so the server is accepting connections before the page loads
    threading.Timer(1.5, _open_dashboard_in_browser).start()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
