"""Vehicle and License Plate Detection Module"""

from .detector import VehicleDetector, DetectorFactory, detect_vehicles
from .base_detector import BaseDetector
from .yolov8_detector import YOLOv8Detector
from .plate_detector import LicensePlateDetector, detect_license_plate

__all__ = [
    'VehicleDetector',
    'DetectorFactory',
    'detect_vehicles',
    'BaseDetector',
    'YOLOv8Detector',
    'LicensePlateDetector',
    'detect_license_plate',
]
