"""
Detector Factory

Automatically creates the right detector based on configuration.
Change models by editing config file - no code changes needed!
"""

from typing import Optional
import logging
from pathlib import Path

from ..utils.config_loader import config
from .base_detector import BaseDetector
from .yolov8_detector import YOLOv8Detector

logger = logging.getLogger(__name__)


class DetectorFactory:
    """
    Factory class to create detector instances.

    Automatically selects the right detector based on config.
    """

    @staticmethod
    def create_detector(model_name: Optional[str] = None,
                       device: Optional[str] = None) -> BaseDetector:
        """
        Create a detector instance.

        Args:
            model_name: Model name (if None, uses active model from config)
            device: Device to run on (if None, uses hardware config)

        Returns:
            Detector instance

        Raises:
            ValueError: If model type is not supported
        """
        # Load configuration
        if model_name is None:
            model_config = config.get_active_detection_model()
            model_name = config.model_config['detection']['active_model']
        else:
            model_config = config.model_config['detection']['models'][model_name]

        if device is None:
            device = config.get_hardware_config()['device']

        # Get model type
        model_type = model_config.get('type', 'YOLOv8')

        logger.info(f"Creating detector: {model_name} ({model_type}) on {device}")

        # Create appropriate detector
        if model_type == "YOLOv8":
            return YOLOv8Detector(model_config, device)

        elif model_type == "YOLOv5":
            # Import only if needed
            try:
                from .yolov5_detector import YOLOv5Detector
                return YOLOv5Detector(model_config, device)
            except ImportError:
                logger.error("YOLOv5 detector not implemented yet")
                logger.info("Falling back to YOLOv8")
                return YOLOv8Detector(model_config, device)

        elif model_type == "MobileNetSSD":
            try:
                from .mobilenet_detector import MobileNetDetector
                return MobileNetDetector(model_config, device)
            except ImportError:
                logger.error("MobileNet detector not implemented yet")
                logger.info("Falling back to YOLOv8")
                # Fall back to YOLOv8
                fallback_config = config.model_config['detection']['models']['yolov8n']
                return YOLOv8Detector(fallback_config, device)

        else:
            raise ValueError(f"Unsupported model type: {model_type}")


class VehicleDetector:
    """
    High-level vehicle detector interface.

    This is the main class you should use in your application.
    """

    def __init__(self, model_name: Optional[str] = None,
                 device: Optional[str] = None):
        """
        Initialize vehicle detector.

        Args:
            model_name: Model to use (None = use config)
            device: Device to run on (None = use config)
        """
        self.detector = DetectorFactory.create_detector(model_name, device)
        logger.info("VehicleDetector initialized")

    def detect(self, image):
        """
        Detect vehicles in image.

        Args:
            image: Input image (BGR format)

        Returns:
            List of detections
        """
        return self.detector.detect(image)

    def detect_vehicles(self, image):
        """
        Detect vehicles in image (filtered).

        Args:
            image: Input image

        Returns:
            List of vehicle detections
        """
        return self.detector.detect_vehicles(image)

    def detect_license_plates(self, image):
        """
        Detect license plates in image.

        Args:
            image: Input image

        Returns:
            List of plate detections
        """
        return self.detector.detect_license_plates(image)

    def get_info(self):
        """Get detector information"""
        return self.detector.get_model_info()


# Convenience function for quick usage
def detect_vehicles(image, model_name: Optional[str] = None):
    """
    Quick vehicle detection function.

    Args:
        image: Input image
        model_name: Model to use (None = use config)

    Returns:
        List of vehicle detections

    Example:
        >>> import cv2
        >>> from src.detection import detect_vehicles
        >>> image = cv2.imread("car.jpg")
        >>> vehicles = detect_vehicles(image)
    """
    detector = VehicleDetector(model_name)
    return detector.detect_vehicles(image)
