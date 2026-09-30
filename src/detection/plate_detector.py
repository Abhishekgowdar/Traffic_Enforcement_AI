"""
License Plate Detector

Detects license plates in vehicle images using YOLOv8.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class LicensePlateDetector:
    """
    Detects license plates in images using YOLOv8.
    """

    def __init__(self, model_path: str = None, confidence_threshold: float = 0.25):
        """
        Initialize license plate detector.

        Args:
            model_path: Path to YOLOv8 model (if None, uses pretrained)
            confidence_threshold: Minimum confidence for detections
        """
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.model_loaded = False
        self.is_plate_model = False

        # If no explicit model given, prefer a trained plate detector when available
        if model_path is None:
            trained = Path(__file__).resolve().parents[2] / "models" / "detection" / "plate_yolov8.pt"
            if trained.exists():
                model_path = str(trained)

        # Try to load YOLO
        try:
            from ultralytics import YOLO

            if model_path and Path(model_path).exists():
                logger.info(f"Loading custom plate detection model: {model_path}")
                self.model = YOLO(model_path)
                self.is_plate_model = True
            else:
                # Use YOLOv8 pretrained on COCO (can detect some objects)
                # We'll look for specific patterns that might be plates
                logger.info("Loading YOLOv8 model for plate detection...")
                self.model = YOLO('yolov8n.pt')  # Smallest, fastest model

            self.model_loaded = True
            logger.info("License plate detector initialized")

        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            self.model_loaded = False

    def detect_plates(self, image: np.ndarray) -> List[Dict]:
        """
        Detect license plates in image.

        Args:
            image: Input image (full vehicle image)

        Returns:
            List of detected plates with bounding boxes
        """
        if not self.model_loaded:
            logger.error("Model not loaded, using fallback plate detection")
            return self._fallback_detection(image)

        try:
            # Run YOLOv8 detection
            results = self.model(image, conf=self.confidence_threshold, verbose=False)

            plates = []

            for result in results:
                boxes = result.boxes

                for box in boxes:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0])

                    # Calculate dimensions
                    width = x2 - x1
                    height = y2 - y1
                    aspect_ratio = width / height if height > 0 else 0

                    # A plate-trained model already outputs plate boxes directly;
                    # only apply the aspect-ratio heuristic to the generic COCO model.
                    if self.is_plate_model or (1.5 < aspect_ratio < 6.0):
                        plates.append({
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': confidence,
                            'aspect_ratio': aspect_ratio,
                            'width': int(width),
                            'height': int(height)
                        })

            if not plates:
                logger.warning("No plates detected by YOLO, using fallback")
                return self._fallback_detection(image)

            # Sort by confidence
            plates.sort(key=lambda x: x['confidence'], reverse=True)

            logger.info(f"Detected {len(plates)} potential license plates")
            return plates

        except Exception as e:
            logger.error(f"Error during detection: {e}")
            return self._fallback_detection(image)

    def _fallback_detection(self, image: np.ndarray) -> List[Dict]:
        """
        Fallback detection using traditional CV methods.

        Looks for rectangular regions with plate-like characteristics.
        """
        logger.info("Using fallback plate detection (traditional CV)")

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply edge detection
        edges = cv2.Canny(gray, 100, 200)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        plates = []
        img_height, img_width = image.shape[:2]

        for contour in contours:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)

            # Calculate aspect ratio
            aspect_ratio = w / h if h > 0 else 0

            # Filter by size and aspect ratio
            # Plates are typically 2-5 times wider than tall
            # and occupy 2-20% of image width
            min_width = img_width * 0.05
            max_width = img_width * 0.4
            min_height = img_height * 0.02
            max_height = img_height * 0.2

            if (1.5 < aspect_ratio < 6.0 and
                min_width < w < max_width and
                min_height < h < max_height):

                plates.append({
                    'bbox': [x, y, x + w, y + h],
                    'confidence': 0.5,  # Lower confidence for fallback
                    'aspect_ratio': aspect_ratio,
                    'width': w,
                    'height': h
                })

        # Sort by area (larger plates more likely to be real)
        plates.sort(key=lambda x: x['width'] * x['height'], reverse=True)

        # Keep top 5 candidates
        plates = plates[:5]

        logger.info(f"Fallback detected {len(plates)} potential plates")
        return plates

    def crop_plate(self, image: np.ndarray, bbox: List[int], padding: int = 5) -> np.ndarray:
        """
        Crop license plate from image.

        Args:
            image: Full image
            bbox: Bounding box [x1, y1, x2, y2]
            padding: Pixels to add around plate

        Returns:
            Cropped plate image
        """
        x1, y1, x2, y2 = bbox

        # Add padding
        height, width = image.shape[:2]
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(width, x2 + padding)
        y2 = min(height, y2 + padding)

        # Crop
        plate_image = image[y1:y2, x1:x2]

        logger.info(f"Cropped plate: {plate_image.shape[1]}x{plate_image.shape[0]}")

        return plate_image

    def detect_and_crop(self, image: np.ndarray) -> List[Tuple[np.ndarray, Dict]]:
        """
        Detect plates and return cropped images.

        Args:
            image: Full vehicle image

        Returns:
            List of (cropped_plate_image, detection_info) tuples
        """
        plates = self.detect_plates(image)

        results = []
        for plate in plates:
            cropped = self.crop_plate(image, plate['bbox'])
            results.append((cropped, plate))

        return results


def detect_license_plate(image: np.ndarray, confidence_threshold: float = 0.25) -> Optional[np.ndarray]:
    """
    Convenience function to detect and crop the best license plate.

    Args:
        image: Full vehicle image
        confidence_threshold: Minimum confidence

    Returns:
        Cropped plate image or None
    """
    detector = LicensePlateDetector(confidence_threshold=confidence_threshold)
    results = detector.detect_and_crop(image)

    if results:
        return results[0][0]  # Return best plate

    return None
