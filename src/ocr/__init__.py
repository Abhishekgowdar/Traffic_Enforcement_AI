"""License Plate OCR Module"""

from .ocr_engine import MultiOCREngine, recognize_plate
from .preprocessing import OCRPreprocessor, preprocess_plate
from .postprocessing import OCRPostProcessor, correct_plate_text
from .indian_plate_ocr import IndianPlateOCR, correct_indian_plate

__all__ = [
    'MultiOCREngine',
    'recognize_plate',
    'OCRPreprocessor',
    'preprocess_plate',
    'OCRPostProcessor',
    'correct_plate_text',
    'IndianPlateOCR',
    'correct_indian_plate'
]
