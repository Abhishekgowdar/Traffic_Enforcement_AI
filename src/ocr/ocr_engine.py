"""
Multi-OCR Engine with Fusion

Combines multiple OCR engines for better accuracy.
"""

import numpy as np
from typing import List, Dict, Any
import logging
from .preprocessing import preprocess_plate
from .postprocessing import correct_plate_text
from .indian_plate_ocr import correct_indian_plate

logger = logging.getLogger(__name__)


class MultiOCREngine:
    """
    Multi-engine OCR with confidence-based fusion.

    Combines results from EasyOCR, PaddleOCR, and optionally TrOCR.
    """

    def __init__(self, engines: List[str] = None):
        """
        Initialize multi-OCR engine.

        Args:
            engines: List of engine names to use
                     Default: ["easyocr"] (PaddleOCR not available on Python 3.14)
        """
        self.engines = engines or ["easyocr"]
        self.ocr_engines = {}

        logger.info(f"Initializing Multi-OCR with engines: {self.engines}")

        # Initialize engines
        self._init_engines()

    def _init_engines(self):
        """Initialize OCR engines"""
        for engine_name in self.engines:
            if engine_name == "easyocr":
                self._init_easyocr()
            elif engine_name == "paddleocr":
                self._init_paddleocr()
            elif engine_name == "trocr":
                self._init_trocr()

    def _init_easyocr(self):
        """Initialize EasyOCR"""
        try:
            import easyocr
            self.ocr_engines['easyocr'] = easyocr.Reader(['en'], gpu=False)
            logger.info("✓ EasyOCR initialized")
        except Exception as e:
            logger.warning(f"Could not initialize EasyOCR: {e}")

    def _init_paddleocr(self):
        """Initialize PaddleOCR"""
        try:
            from paddleocr import PaddleOCR
            self.ocr_engines['paddleocr'] = PaddleOCR(
                use_angle_cls=False,
                lang='en',
                show_log=False
            )
            logger.info("✓ PaddleOCR initialized")
        except Exception as e:
            logger.warning(f"Could not initialize PaddleOCR: {e}")

    def _init_trocr(self):
        """Initialize TrOCR (placeholder)"""
        logger.warning("TrOCR not implemented yet")

    def read_text(self, image: np.ndarray, preprocess: bool = True, postprocess: bool = True,
                  plate_format: str = 'indian') -> Dict[str, Any]:
        """
        Read text from image (alias for recognize).

        Args:
            image: Input image
            preprocess: Apply preprocessing for better accuracy
            postprocess: Apply post-processing to fix common OCR errors
            plate_format: Plate format ('indian' or 'auto')

        Returns:
            Dictionary with:
                - text: Final fused text (corrected if postprocess=True)
                - confidence: Overall confidence
                - engine_results: Results from each engine
                - raw_text: Original OCR text before correction (if postprocessed)
        """
        # Apply preprocessing if enabled
        if preprocess:
            logger.info("Applying OCR preprocessing...")
            processed_image = preprocess_plate(image, method="enhance")
        else:
            processed_image = image

        result = self.recognize(processed_image)

        raw_text = result.get('fused_text', '')
        raw_confidence = result.get('confidence', 0.0)

        # Apply post-processing if enabled
        if postprocess and raw_text:
            if plate_format == 'indian':
                logger.info("Applying Indian license plate corrections...")
                corrected_text, adjusted_confidence, details = correct_indian_plate(
                    raw_text,
                    raw_confidence
                )

                return {
                    'text': corrected_text,
                    'raw_text': raw_text,
                    'confidence': adjusted_confidence,
                    'raw_confidence': raw_confidence,
                    'engine_results': result.get('individual_results', {}),
                    'best_engine': result.get('best_engine', None),
                    'corrected': (raw_text != corrected_text),
                    'details': details
                }
            else:
                logger.info("Applying generic OCR post-processing...")
                corrected_text, adjusted_confidence = correct_plate_text(
                    raw_text,
                    raw_confidence,
                    plate_format='auto'
                )

                return {
                    'text': corrected_text,
                    'raw_text': raw_text,
                    'confidence': adjusted_confidence,
                    'raw_confidence': raw_confidence,
                    'engine_results': result.get('individual_results', {}),
                    'best_engine': result.get('best_engine', None),
                    'corrected': (raw_text != corrected_text)
                }
        else:
            # No post-processing
            return {
                'text': raw_text,
                'confidence': raw_confidence,
                'engine_results': result.get('individual_results', {}),
                'best_engine': result.get('best_engine', None),
                'corrected': False
            }

    def recognize(self, plate_image: np.ndarray) -> Dict[str, Any]:
        """
        Recognize license plate text using multiple OCR engines.

        Args:
            plate_image: Cropped license plate image

        Returns:
            Dictionary with:
                - fused_text: Final fused text
                - confidence: Overall confidence
                - individual_results: Results from each engine
        """
        if not self.ocr_engines:
            logger.error("No OCR engines initialized")
            return {
                'fused_text': '',
                'confidence': 0.0,
                'individual_results': {}
            }

        individual_results = {}

        # Run each OCR engine
        for engine_name, engine in self.ocr_engines.items():
            try:
                result = self._run_engine(engine_name, engine, plate_image)
                individual_results[engine_name] = result
            except Exception as e:
                logger.error(f"Error running {engine_name}: {e}")

        # Fuse results
        fused_result = self._fuse_results(individual_results)

        return fused_result

    def _run_engine(self, engine_name: str, engine, image: np.ndarray) -> Dict[str, Any]:
        """Run a single OCR engine"""
        if engine_name == "easyocr":
            results = engine.readtext(image)
            if results:
                text = results[0][1]  # Get text
                conf = results[0][2]  # Get confidence
                return {'text': text, 'confidence': conf}

        elif engine_name == "paddleocr":
            results = engine.ocr(image, cls=False)
            if results and results[0]:
                text = results[0][0][1][0]  # Get text
                conf = results[0][0][1][1]  # Get confidence
                return {'text': text, 'confidence': conf}

        return {'text': '', 'confidence': 0.0}

    def _fuse_results(self, individual_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Fuse results from multiple OCR engines.

        Uses confidence-weighted voting.
        """
        if not individual_results:
            return {
                'fused_text': '',
                'confidence': 0.0,
                'individual_results': {}
            }

        # Simple approach: take highest confidence result
        best_result = max(
            individual_results.items(),
            key=lambda x: x[1].get('confidence', 0)
        )

        fused_text = best_result[1]['text']
        fused_confidence = best_result[1]['confidence']

        logger.info(f"OCR Result: '{fused_text}' (confidence: {fused_confidence:.2f})")

        return {
            'fused_text': fused_text,
            'confidence': fused_confidence,
            'individual_results': individual_results,
            'best_engine': best_result[0]
        }


# Convenience function
def recognize_plate(plate_image: np.ndarray) -> str:
    """
    Quick license plate recognition.

    Args:
        plate_image: Cropped plate image

    Returns:
        Recognized text
    """
    ocr = MultiOCREngine()
    result = ocr.recognize(plate_image)
    return result['fused_text']
