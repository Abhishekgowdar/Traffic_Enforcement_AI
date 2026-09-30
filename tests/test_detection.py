"""
Unit Tests for Detection Module
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.detection import VehicleDetector


def test_detector_initialization():
    """Test that detector initializes correctly"""
    detector = VehicleDetector()
    assert detector is not None

    info = detector.get_info()
    assert 'model_type' in info
    assert 'device' in info


def test_detection_on_dummy_image():
    """Test detection on a dummy image"""
    detector = VehicleDetector()

    # Create dummy image (640x640 random)
    dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

    # Should not crash
    results = detector.detect(dummy_image)

    # Results should be a list
    assert isinstance(results, list)


def test_vehicle_detection_filtering():
    """Test that vehicle detection filters correctly"""
    detector = VehicleDetector()

    # Create dummy image
    dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

    # Get vehicle detections
    vehicles = detector.detect_vehicles(dummy_image)

    # Should return list
    assert isinstance(vehicles, list)

    # Each detection should have required keys
    for det in vehicles:
        assert 'bbox' in det
        assert 'confidence' in det
        assert 'class_name' in det
        assert det['class_name'] in ['car', 'truck', 'bus', 'motorcycle']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
