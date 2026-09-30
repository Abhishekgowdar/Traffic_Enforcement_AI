"""Traffic Enforcement AI - Source Code"""

__version__ = "1.0.0"
__author__ = "Traffic Enforcement AI Team"

from .detection import VehicleDetector, detect_vehicles
from .utils import config

__all__ = [
    'VehicleDetector',
    'detect_vehicles',
    'config',
]
