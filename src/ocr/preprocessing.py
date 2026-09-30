"""
OCR Preprocessing Module

Image preprocessing to improve license plate OCR accuracy.
"""

import cv2
import numpy as np
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


class OCRPreprocessor:
    """Preprocesses images for better OCR accuracy"""

    @staticmethod
    def preprocess(image: np.ndarray, debug: bool = False) -> np.ndarray:
        """
        Preprocess image for OCR.

        Args:
            image: Input image
            debug: Save intermediate steps for debugging

        Returns:
            Preprocessed image
        """
        original = image.copy()

        # Step 1: Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Step 2: Resize if too small (OCR works better on larger images)
        height, width = gray.shape
        if height < 100:
            scale = 100 / height
            new_width = int(width * scale)
            new_height = int(height * scale)
            gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            logger.info(f"Resized image from {width}x{height} to {new_width}x{new_height}")

        # Step 3: Denoise
        denoised = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)

        # Step 4: Increase contrast (CLAHE - Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        contrast = clahe.apply(denoised)

        # Step 5: Sharpen
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        sharpened = cv2.filter2D(contrast, -1, kernel)

        # Step 6: Threshold to binary (improves text detection)
        # Try adaptive threshold
        binary = cv2.adaptiveThreshold(
            sharpened,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2
        )

        # Step 7: Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        if debug:
            cv2.imwrite("debug_1_gray.jpg", gray)
            cv2.imwrite("debug_2_denoised.jpg", denoised)
            cv2.imwrite("debug_3_contrast.jpg", contrast)
            cv2.imwrite("debug_4_sharpened.jpg", sharpened)
            cv2.imwrite("debug_5_binary.jpg", binary)
            cv2.imwrite("debug_6_morph.jpg", morph)
            logger.info("Debug images saved")

        return morph

    @staticmethod
    def enhance_for_ocr(image: np.ndarray) -> np.ndarray:
        """
        Simple enhancement for OCR (lighter preprocessing).

        Args:
            image: Input image

        Returns:
            Enhanced image
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Resize if too small
        height, width = gray.shape
        if height < 64:
            scale = 64 / height
            new_width = int(width * scale)
            new_height = 64
            gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

        # Increase contrast
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Light sharpening
        kernel = np.array([[0, -1, 0],
                          [-1, 5, -1],
                          [0, -1, 0]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)

        return sharpened

    @staticmethod
    def deskew(image: np.ndarray) -> np.ndarray:
        """
        Deskew image (correct rotation).

        Args:
            image: Input image

        Returns:
            Deskewed image
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Detect edges
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        # Detect lines using Hough transform
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)

        if lines is not None and len(lines) > 0:
            # Calculate average angle
            angles = []
            for rho, theta in lines[:, 0]:
                angle = np.degrees(theta) - 90
                angles.append(angle)

            median_angle = np.median(angles)

            # Rotate image if angle is significant
            if abs(median_angle) > 0.5:
                h, w = gray.shape
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
                rotated = cv2.warpAffine(image, M, (w, h),
                                        flags=cv2.INTER_CUBIC,
                                        borderMode=cv2.BORDER_REPLICATE)
                logger.info(f"Deskewed image by {median_angle:.2f} degrees")
                return rotated

        return image


def preprocess_plate(image: np.ndarray, method: str = "enhance") -> np.ndarray:
    """
    Convenience function to preprocess license plate image.

    Args:
        image: Input image
        method: Preprocessing method ("enhance" or "full")

    Returns:
        Preprocessed image
    """
    preprocessor = OCRPreprocessor()

    if method == "enhance":
        return preprocessor.enhance_for_ocr(image)
    elif method == "full":
        return preprocessor.preprocess(image)
    else:
        return image
