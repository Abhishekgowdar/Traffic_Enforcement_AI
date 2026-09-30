"""
YOLOv8 Detector Implementation

Vehicle and license plate detection using YOLOv8.
"""

from typing import List, Dict, Any
import numpy as np
import logging
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    raise ImportError("ultralytics not installed. Run: pip install ultralytics")

from .base_detector import BaseDetector

logger = logging.getLogger(__name__)


class YOLOv8Detector(BaseDetector):
    """YOLOv8-based detector for vehicles and license plates"""

    # COCO class names
    COCO_CLASSES = {
        0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle',
        4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck'
    }

    def __init__(self, model_config: Dict[str, Any], device: str = "cpu"):
        """
        Initialize YOLOv8 detector.

        Args:
            model_config: Model configuration
            device: Device to run on
        """
        super().__init__(model_config, device)
        self.weight_file = model_config.get('weight_file', 'yolov8s.pt')
        self.variant = model_config.get('variant', 'small')
        self.load_model()

    def load_model(self):
        """Load YOLOv8 model"""
        try:
            weight_path = Path(self.weight_file)

            # If weight file doesn't exist, download it
            if not weight_path.exists():
                logger.info(f"Downloading YOLOv8 {self.variant} model...")
                # Use model name to auto-download from Ultralytics
                model_name = f"yolov8{self.variant[0]}.pt"  # e.g., yolov8s.pt
                self.model = YOLO(model_name)
            else:
                logger.info(f"Loading model from {weight_path}")
                self.model = YOLO(str(weight_path))

            # Move to device
            self.model.to(self.device)

            logger.info(f"YOLOv8 model loaded successfully")

        except Exception as e:
            logger.error(f"Error loading YOLOv8 model: {e}")
            raise

    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect objects in image using YOLOv8.

        Args:
            image: Input image (BGR format)

        Returns:
            List of detections
        """
        if self.model is None:
            logger.error("Model not loaded")
            return []

        try:
            # Run inference
            results = self.model(
                image,
                conf=self.confidence_threshold,
                imgsz=self.input_size,
                verbose=False
            )

            detections = []

            # Parse results
            for result in results:
                boxes = result.boxes

                for i in range(len(boxes)):
                    # Get box coordinates
                    box = boxes.xyxy[i].cpu().numpy()
                    x1, y1, x2, y2 = box

                    # Get confidence
                    confidence = float(boxes.conf[i].cpu().numpy())

                    # Get class
                    class_id = int(boxes.cls[i].cpu().numpy())
                    class_name = self.COCO_CLASSES.get(class_id, f"class_{class_id}")

                    detection = {
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'confidence': confidence,
                        'class_id': class_id,
                        'class_name': class_name
                    }

                    detections.append(detection)

            logger.debug(f"Detected {len(detections)} objects")
            return detections

        except Exception as e:
            logger.error(f"Error during detection: {e}")
            return []

    def detect_with_tracking_id(self, image: np.ndarray, track: bool = True) -> List[Dict[str, Any]]:
        """
        Detect objects with tracking IDs.

        Args:
            image: Input image
            track: Enable tracking

        Returns:
            Detections with tracking IDs
        """
        if not track:
            return self.detect(image)

        try:
            # Run tracking
            results = self.model.track(
                image,
                conf=self.confidence_threshold,
                imgsz=self.input_size,
                persist=True,
                verbose=False
            )

            detections = []

            for result in results:
                boxes = result.boxes

                for i in range(len(boxes)):
                    box = boxes.xyxy[i].cpu().numpy()
                    x1, y1, x2, y2 = box

                    confidence = float(boxes.conf[i].cpu().numpy())
                    class_id = int(boxes.cls[i].cpu().numpy())
                    class_name = self.COCO_CLASSES.get(class_id, f"class_{class_id}")

                    # Get tracking ID if available
                    track_id = None
                    if boxes.id is not None:
                        track_id = int(boxes.id[i].cpu().numpy())

                    detection = {
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'confidence': confidence,
                        'class_id': class_id,
                        'class_name': class_name,
                        'track_id': track_id
                    }

                    detections.append(detection)

            return detections

        except Exception as e:
            logger.error(f"Error during tracking: {e}")
            return self.detect(image)


# Convenience function
def create_yolov8_detector(variant: str = "s", device: str = "cpu") -> YOLOv8Detector:
    """
    Create a YOLOv8 detector with default configuration.

    Args:
        variant: Model variant (n, s, m, l, x)
        device: Device to run on

    Returns:
        YOLOv8Detector instance
    """
    config = {
        'variant': variant,
        'weight_file': f'yolov8{variant}.pt',
        'input_size': 640,
        'confidence_threshold': 0.5
    }

    return YOLOv8Detector(config, device)
