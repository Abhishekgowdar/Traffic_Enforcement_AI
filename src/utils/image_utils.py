"""
Image Processing Utilities

Common image processing functions used across modules.
"""

import cv2
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def load_image(image_path: str) -> Optional[np.ndarray]:
    """
    Load image from file path.

    Args:
        image_path: Path to image file

    Returns:
        Image as numpy array (BGR format) or None if failed
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Failed to load image: {image_path}")
            return None
        return image
    except Exception as e:
        logger.error(f"Error loading image {image_path}: {e}")
        return None


def resize_image(image: np.ndarray, size: Tuple[int, int],
                 keep_aspect_ratio: bool = True) -> np.ndarray:
    """
    Resize image to target size.

    Args:
        image: Input image
        size: Target (width, height)
        keep_aspect_ratio: Maintain aspect ratio

    Returns:
        Resized image
    """
    if keep_aspect_ratio:
        h, w = image.shape[:2]
        target_w, target_h = size

        # Calculate scaling factor
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)

        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        return resized
    else:
        return cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)


def crop_image(image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    """
    Crop image using bounding box.

    Args:
        image: Input image
        bbox: (x1, y1, x2, y2) coordinates

    Returns:
        Cropped image
    """
    x1, y1, x2, y2 = bbox
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    # Clamp coordinates to image bounds
    h, w = image.shape[:2]
    x1 = max(0, min(x1, w))
    y1 = max(0, min(y1, h))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))

    return image[y1:y2, x1:x2]


def enhance_contrast(image: np.ndarray) -> np.ndarray:
    """
    Enhance image contrast using CLAHE.

    Args:
        image: Input image

    Returns:
        Contrast-enhanced image
    """
    if len(image.shape) == 3:
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)

        # Merge back
        enhanced = cv2.merge([l, a, b])
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    else:
        # Grayscale image
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(image)


def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert image to grayscale"""
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def denoise_image(image: np.ndarray) -> np.ndarray:
    """
    Apply denoising to image.

    Args:
        image: Input image

    Returns:
        Denoised image
    """
    if len(image.shape) == 3:
        return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    else:
        return cv2.fastNlMeansDenoising(image, None, 10, 7, 21)


def draw_bbox(image: np.ndarray, bbox: Tuple[int, int, int, int],
              label: str = "", color: Tuple[int, int, int] = (0, 255, 0),
              thickness: int = 2) -> np.ndarray:
    """
    Draw bounding box on image.

    Args:
        image: Input image
        bbox: (x1, y1, x2, y2)
        label: Text label
        color: Box color (BGR)
        thickness: Line thickness

    Returns:
        Image with bounding box
    """
    image = image.copy()
    x1, y1, x2, y2 = map(int, bbox)

    # Draw rectangle
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

    # Draw label if provided
    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 2

        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(
            label, font, font_scale, font_thickness
        )

        # Draw background rectangle
        cv2.rectangle(
            image,
            (x1, y1 - text_height - 10),
            (x1 + text_width, y1),
            color,
            -1
        )

        # Draw text
        cv2.putText(
            image, label, (x1, y1 - 5),
            font, font_scale, (255, 255, 255), font_thickness
        )

    return image


def save_image(image: np.ndarray, output_path: str) -> bool:
    """
    Save image to file.

    Args:
        image: Image to save
        output_path: Output file path

    Returns:
        True if successful, False otherwise
    """
    try:
        cv2.imwrite(output_path, image)
        logger.info(f"Image saved to {output_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving image to {output_path}: {e}")
        return False
