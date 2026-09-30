"""
Base Detector Class

Abstract base class for all detection models.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
import numpy as np
import logging

logger = logging.getLogger(__name__)


class BaseDetector(ABC):
    """
    Abstract base class for vehicle and license plate detection.

    All detector implementations must inherit from this class.
    """

    def __init__(self, model_config: Dict[str, Any], device: str = "cpu"):
        """
        Initialize detector.

        Args:
            model_config: Model configuration dictionary
            device: Device to run on ('cpu' or 'cuda')
        """
        self.model_config = model_config
        self.device = device
        self.model = None
        self.confidence_threshold = model_config.get('confidence_threshold', 0.5)
        self.input_size = model_config.get('input_size', 640)

        logger.info(f"Initializing {self.__class__.__name__}")
        logger.info(f"Device: {device}")
        logger.info(f"Confidence threshold: {self.confidence_threshold}")

    @abstractmethod
    def load_model(self):
        """Load the detection model"""
        pass

    @abstractmethod
    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect objects in image.

        Args:
            image: Input image (BGR format)

        Returns:
            List of detections, each containing:
                - bbox: [x1, y1, x2, y2]
                - confidence: float
                - class_id: int
                - class_name: str
        """
        pass

    def detect_vehicles(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect vehicles in image.

        Args:
            image: Input image

        Returns:
            List of vehicle detections
        """
        all_detections = self.detect(image)

        # Filter for vehicle classes (car, truck, bus, motorcycle)
        vehicle_classes = ['car', 'truck', 'bus', 'motorcycle']
        vehicles = [
            det for det in all_detections
            if det['class_name'] in vehicle_classes
        ]

        logger.info(f"Detected {len(vehicles)} vehicles")
        return vehicles

    def detect_license_plates(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect license plates in image.

        Args:
            image: Input image

        Returns:
            List of license plate detections
        """
        all_detections = self.detect(image)

        # Filter for license plate class
        plates = [
            det for det in all_detections
            if det['class_name'] == 'license_plate'
        ]

        logger.info(f"Detected {len(plates)} license plates")
        return plates

    def _filter_by_confidence(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter detections by confidence threshold"""
        return [
            det for det in detections
            if det['confidence'] >= self.confidence_threshold
        ]

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        return {
            'model_type': self.__class__.__name__,
            'device': self.device,
            'confidence_threshold': self.confidence_threshold,
            'input_size': self.input_size,
        }
