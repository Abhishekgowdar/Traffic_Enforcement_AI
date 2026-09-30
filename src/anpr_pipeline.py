"""
ANPR (Automatic Number Plate Recognition) Pipeline

Complete end-to-end pipeline:
1. Detect license plate in vehicle image
2. Crop plate region
3. Run OCR on plate
4. Return recognized plate number
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path

from .detection.plate_detector import LicensePlateDetector
from .ocr import MultiOCREngine

logger = logging.getLogger(__name__)


class ANPRPipeline:
    """
    Complete ANPR pipeline for Indian license plates.
    """

    def __init__(self,
                 plate_detection_confidence: float = 0.25,
                 ocr_confidence_threshold: float = 0.3):
        """
        Initialize ANPR pipeline.

        Args:
            plate_detection_confidence: Minimum confidence for plate detection
            ocr_confidence_threshold: Minimum confidence for OCR results
        """
        logger.info("Initializing ANPR Pipeline...")

        # Initialize plate detector
        self.plate_detector = LicensePlateDetector(
            confidence_threshold=plate_detection_confidence
        )

        # Initialize OCR engine
        self.ocr_engine = MultiOCREngine()

        self.ocr_confidence_threshold = ocr_confidence_threshold

        logger.info("ANPR Pipeline initialized successfully")

    def process_image(self, image: np.ndarray, return_details: bool = True) -> Dict:
        """
        Process a single vehicle image and extract license plate number.

        Args:
            image: Vehicle image (BGR format)
            return_details: Include detailed information in result

        Returns:
            Dictionary with:
                - plate_number: Recognized plate number
                - confidence: OCR confidence
                - plate_detected: Whether plate was detected
                - plate_bbox: Bounding box of detected plate
                - plate_image: Cropped plate image (if return_details=True)
                - ocr_details: Detailed OCR results
        """
        result = {
            'plate_number': None,
            'confidence': 0.0,
            'plate_detected': False,
            'status': 'processing'
        }

        try:
            # Step 1: Detect license plate
            logger.info("Step 1: Detecting license plate...")
            plate_detections = self.plate_detector.detect_and_crop(image)

            if not plate_detections:
                logger.warning("No license plate detected in image")
                result['status'] = 'no_plate_detected'
                return result

            # Get best plate detection
            plate_image, plate_info = plate_detections[0]
            result['plate_detected'] = True
            result['plate_bbox'] = plate_info['bbox']
            result['detection_confidence'] = plate_info['confidence']

            logger.info(f"Plate detected with confidence: {plate_info['confidence']:.2f}")

            # Step 2: Run OCR on plate
            logger.info("Step 2: Running OCR on detected plate...")
            ocr_result = self.ocr_engine.read_text(
                plate_image,
                preprocess=False,
                postprocess=True,
                plate_format='indian'
            )

            # Extract results
            plate_number = ocr_result.get('text', '')
            confidence = ocr_result.get('confidence', 0.0)

            # Fallback: pre-cropped plate images can defeat plate detection, leaving the
            # crop OCR empty/weak. Retry OCR on the full image and keep it if it's better.
            if not plate_number.strip() or confidence < self.ocr_confidence_threshold:
                logger.info("Crop OCR weak; retrying OCR on the full image...")
                whole = self.ocr_engine.read_text(
                    image, preprocess=False, postprocess=True, plate_format='indian'
                )
                whole_text = whole.get('text', '')
                whole_conf = whole.get('confidence', 0.0)
                if whole_text.strip() and whole_conf > confidence:
                    ocr_result = whole
                    plate_number = whole_text
                    confidence = whole_conf
                    plate_image = image
                    result['plate_bbox'] = [0, 0, image.shape[1], image.shape[0]]

            logger.info(f"OCR Result: '{plate_number}' (confidence: {confidence:.2f})")

            # Check if confidence meets threshold
            if confidence < self.ocr_confidence_threshold:
                logger.warning(f"Low OCR confidence: {confidence:.2f}")
                result['status'] = 'low_confidence'
            else:
                result['status'] = 'success'

            # Populate result
            result['plate_number'] = plate_number
            result['confidence'] = confidence
            result['raw_ocr_text'] = ocr_result.get('raw_text', '')
            result['corrected'] = ocr_result.get('corrected', False)

            if return_details:
                result['plate_image'] = plate_image
                result['ocr_details'] = ocr_result
                result['all_plate_detections'] = len(plate_detections)

            return result

        except Exception as e:
            logger.error(f"Error processing image: {e}")
            result['status'] = 'error'
            result['error'] = str(e)
            return result

    def process_multiple(self, images: List[np.ndarray]) -> List[Dict]:
        """
        Process multiple images.

        Args:
            images: List of vehicle images

        Returns:
            List of results
        """
        results = []

        for i, image in enumerate(images):
            logger.info(f"Processing image {i+1}/{len(images)}")
            result = self.process_image(image)
            results.append(result)

        return results

    def process_image_file(self, image_path: str) -> Dict:
        """
        Process an image file.

        Args:
            image_path: Path to image file

        Returns:
            Result dictionary
        """
        # Load image
        image = cv2.imread(image_path)

        if image is None:
            return {
                'plate_number': None,
                'confidence': 0.0,
                'status': 'error',
                'error': 'Failed to load image'
            }

        result = self.process_image(image)
        result['image_path'] = image_path
        result['image_filename'] = Path(image_path).name

        return result

    def visualize_result(self, image: np.ndarray, result: Dict,
                         save_path: Optional[str] = None) -> np.ndarray:
        """
        Draw bounding box and text on image.

        Args:
            image: Original image
            result: Result from process_image()
            save_path: Optional path to save visualized image

        Returns:
            Annotated image
        """
        annotated = image.copy()

        if result.get('plate_detected') and 'plate_bbox' in result:
            bbox = result['plate_bbox']
            x1, y1, x2, y2 = bbox

            # Draw bounding box
            color = (0, 255, 0) if result['status'] == 'success' else (0, 165, 255)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)

            # Draw label
            if result.get('plate_number'):
                label = f"{result['plate_number']} ({result['confidence']*100:.1f}%)"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)

                # Background for text
                cv2.rectangle(annotated,
                            (x1, y1 - label_size[1] - 10),
                            (x1 + label_size[0], y1),
                            color, -1)

                # Text
                cv2.putText(annotated, label,
                          (x1, y1 - 5),
                          cv2.FONT_HERSHEY_SIMPLEX,
                          0.8, (0, 0, 0), 2)

        if save_path:
            cv2.imwrite(save_path, annotated)
            logger.info(f"Visualized result saved to: {save_path}")

        return annotated


def recognize_plate(image: np.ndarray) -> Optional[str]:
    """
    Convenience function to recognize license plate from image.

    Args:
        image: Vehicle image

    Returns:
        Plate number or None
    """
    pipeline = ANPRPipeline()
    result = pipeline.process_image(image, return_details=False)

    if result['status'] == 'success':
        return result['plate_number']

    return None
