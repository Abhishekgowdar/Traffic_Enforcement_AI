"""
Indian License Plate OCR
Specialized for Indian vehicle registration plates
"""

import re
import numpy as np
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class IndianPlateOCR:
    """
    Specialized OCR post-processor for Indian license plates.

    Indian Plate Format: XX##XX#### or XX##X####
    - XX: State code (2 letters)
    - ##: District code (2 digits)
    - XX/X: Series code (1-2 letters)
    - ####: Registration number (4 digits)

    Examples: KL01CA2555, DL7CQ1234, MH02AB1234
    """

    # Common letter-to-digit OCR errors
    LETTER_TO_DIGIT = {
        'O': '0', 'o': '0',
        'I': '1', 'i': '1', 'l': '1', 'L': '1',
        'Z': '2', 'z': '2',
        'E': '3', 'e': '3',
        'A': '4', 'a': '4',
        'S': '5', 's': '5',
        'G': '6', 'g': '6',
        'T': '7', 't': '7',
        'B': '8', 'b': '8',
        'g': '9',  # lowercase g often looks like 9
        'Q': '0', 'q': '0',
        'D': '0',
    }

    # Common digit-to-letter OCR errors
    DIGIT_TO_LETTER = {
        '0': 'O',
        '1': 'I',
        '2': 'Z',
        '3': 'E',
        '4': 'A',
        '5': 'S',
        '6': 'G',
        '7': 'T',
        '8': 'B',
        '9': 'g',
    }

    # Valid Indian state codes (RTO codes)
    STATE_CODES = [
        'AN', 'AP', 'AR', 'AS', 'BR', 'CH', 'CG', 'DD', 'DL', 'DN', 'GA',
        'GJ', 'HP', 'HR', 'JH', 'JK', 'KA', 'KL', 'LA', 'LD', 'MH', 'ML',
        'MN', 'MP', 'MZ', 'NL', 'OD', 'OR', 'PB', 'PY', 'RJ', 'SK', 'TN',
        'TR', 'TS', 'UK', 'UP', 'WB'
    ]

    # Letters that are commonly confused with each other by OCR (shape-based)
    VISUAL_CONFUSION = {
        'H': ['M', 'N'], 'M': ['H', 'N', 'W'], 'N': ['M', 'H'],
        'O': ['D', 'Q', 'C', 'G'], 'D': ['O', 'P'], 'Q': ['O', 'G'],
        'B': ['R', 'P', 'E'], 'R': ['B', 'P', 'K', 'A'], 'P': ['R', 'B', 'F'],
        'C': ['G', 'O'], 'G': ['C', 'O', 'Q', 'S'],
        'K': ['R', 'X'], 'X': ['K', 'Y'], 'Y': ['V', 'X'],
        'U': ['V', 'W'], 'V': ['U', 'W', 'Y'], 'W': ['V', 'U', 'M'],
        'T': ['I', 'L', 'J'], 'I': ['T', 'L', 'J'], 'L': ['I', 'T'], 'J': ['I', 'T'],
        'E': ['F', 'B'], 'F': ['E', 'P'], 'A': ['R', 'H'], 'S': ['G'], 'Z': ['2'],
    }

    @classmethod
    def correct_state_code(cls, code: str) -> Tuple[str, bool]:
        """
        Snap an invalid 2-letter state code to the nearest valid RTO code
        using visual-confusion substitutions (e.g. 'HH' -> 'MH').

        Returns (code, changed).
        """
        code = code.upper()
        if code in cls.STATE_CODES:
            return code, False

        best, best_score = None, (-1, -1)
        for valid in cls.STATE_CODES:
            if len(valid) != len(code):
                continue
            exact = sum(1 for a, b in zip(code, valid) if a == b)
            confusable = sum(
                1 for a, b in zip(code, valid)
                if a != b and b in cls.VISUAL_CONFUSION.get(a, [])
            )
            # Every position must either match exactly or be a known confusion
            if exact + confusable == len(code) and confusable >= 1:
                score = (exact, confusable)
                if score > best_score:
                    best_score, best = score, valid

        if best is not None:
            return best, True
        return code, False

    @classmethod
    def correct_plate(cls, text: str) -> Tuple[str, float, Dict]:
        """
        Correct Indian license plate OCR errors.

        Args:
            text: Raw OCR text

        Returns:
            Tuple of (corrected_text, confidence_multiplier, details)
        """
        original_text = text
        text = text.upper().replace(' ', '').replace('-', '').replace('.', '')

        logger.info(f"Processing Indian plate: '{original_text}' → '{text}'")

        # Check length (should be 9 or 10)
        if len(text) < 8 or len(text) > 11:
            logger.warning(f"Unusual plate length: {len(text)}")
            # Try to fix by removing extra characters
            text = re.sub(r'[^A-Z0-9]', '', text)

        corrections_made = []
        confidence_adjustments = []

        # Process character by character based on position
        result = []

        for i, char in enumerate(text):
            if i < 2:
                # Position 0-1: State code (MUST be letters)
                if char.isdigit():
                    # Convert digit to letter
                    if char in cls.DIGIT_TO_LETTER:
                        corrected = cls.DIGIT_TO_LETTER[char]
                        result.append(corrected)
                        corrections_made.append(f"Pos {i}: {char}→{corrected} (digit to letter)")
                        confidence_adjustments.append(0.95)
                    else:
                        result.append(char)
                        confidence_adjustments.append(0.8)
                else:
                    result.append(char)
                    confidence_adjustments.append(1.0)

            elif i < 4:
                # Position 2-3: District code (MUST be digits)
                if char.isalpha():
                    # Convert letter to digit
                    if char in cls.LETTER_TO_DIGIT:
                        corrected = cls.LETTER_TO_DIGIT[char]
                        result.append(corrected)
                        corrections_made.append(f"Pos {i}: {char}→{corrected} (letter to digit)")
                        confidence_adjustments.append(0.95)
                    else:
                        # Unknown conversion, try common patterns
                        if char == 'O':
                            result.append('0')
                            corrections_made.append(f"Pos {i}: {char}→0")
                            confidence_adjustments.append(0.9)
                        elif char in ['I', 'L']:
                            result.append('1')
                            corrections_made.append(f"Pos {i}: {char}→1")
                            confidence_adjustments.append(0.9)
                        else:
                            result.append(char)
                            confidence_adjustments.append(0.7)
                else:
                    result.append(char)
                    confidence_adjustments.append(1.0)

            elif i < 6 or (len(text) == 10 and i < 6):
                # Position 4-5 (or 4-6 if 10 chars): Series code (MUST be letters)
                if char.isdigit():
                    # Convert digit to letter
                    if char in cls.DIGIT_TO_LETTER:
                        corrected = cls.DIGIT_TO_LETTER[char]
                        result.append(corrected)
                        corrections_made.append(f"Pos {i}: {char}→{corrected} (digit to letter)")
                        confidence_adjustments.append(0.95)
                    else:
                        result.append(char)
                        confidence_adjustments.append(0.8)
                else:
                    result.append(char)
                    confidence_adjustments.append(1.0)

            else:
                # Last 4 positions: Registration number (MUST be digits)
                if char.isalpha():
                    # Convert letter to digit
                    if char in cls.LETTER_TO_DIGIT:
                        corrected = cls.LETTER_TO_DIGIT[char]
                        result.append(corrected)
                        corrections_made.append(f"Pos {i}: {char}→{corrected} (letter to digit)")
                        confidence_adjustments.append(0.95)
                    else:
                        result.append(char)
                        confidence_adjustments.append(0.7)
                else:
                    result.append(char)
                    confidence_adjustments.append(1.0)

        corrected_text = ''.join(result)

        # Validate state code, snapping obvious misreads to the nearest valid code
        state_code = corrected_text[:2]
        state_valid = state_code in cls.STATE_CODES
        if not state_valid:
            fixed_code, changed = cls.correct_state_code(state_code)
            if changed:
                logger.info(f"State code corrected: {state_code} \u2192 {fixed_code}")
                corrections_made.append(f"State: {state_code}\u2192{fixed_code}")
                corrected_text = fixed_code + corrected_text[2:]
                state_code = fixed_code
                state_valid = True
                confidence_adjustments.append(0.9)
            else:
                logger.warning(f"Invalid state code: {state_code}")
                confidence_adjustments.append(0.8)

        # Calculate confidence multiplier
        if confidence_adjustments:
            avg_confidence = sum(confidence_adjustments) / len(confidence_adjustments)
        else:
            avg_confidence = 1.0

        # Format the plate nicely: XX ## XX ####
        if len(corrected_text) == 10:
            formatted = f"{corrected_text[:2]} {corrected_text[2:4]} {corrected_text[4:6]} {corrected_text[6:]}"
        elif len(corrected_text) == 9:
            formatted = f"{corrected_text[:2]} {corrected_text[2:4]} {corrected_text[4]} {corrected_text[5:]}"
        else:
            formatted = corrected_text

        details = {
            'original': original_text,
            'cleaned': text,
            'corrected': corrected_text,
            'formatted': formatted,
            'corrections': corrections_made,
            'state_code': state_code,
            'state_valid': state_valid,
            'length': len(corrected_text),
            'format_valid': cls.validate_format(corrected_text)
        }

        logger.info(f"Corrected: '{text}' → '{corrected_text}' (formatted: '{formatted}')")
        if corrections_made:
            logger.info(f"Corrections: {corrections_made}")

        return corrected_text, avg_confidence, details

    @classmethod
    def validate_format(cls, text: str) -> bool:
        """
        Validate if text matches Indian license plate format.

        Args:
            text: Plate text

        Returns:
            True if valid format
        """
        # Pattern: 2 letters + 2 digits + 1-2 letters + 4 digits
        pattern_10 = r'^[A-Z]{2}\d{2}[A-Z]{2}\d{4}$'
        pattern_9 = r'^[A-Z]{2}\d{2}[A-Z]{1}\d{4}$'

        return bool(re.match(pattern_10, text) or re.match(pattern_9, text))

    @classmethod
    def format_plate(cls, text: str) -> str:
        """
        Format plate text nicely.

        Args:
            text: Plate text (without spaces)

        Returns:
            Formatted plate text
        """
        text = text.replace(' ', '').replace('-', '')

        if len(text) == 10:
            # Format: XX ## XX ####
            return f"{text[:2]} {text[2:4]} {text[4:6]} {text[6:]}"
        elif len(text) == 9:
            # Format: XX ## X ####
            return f"{text[:2]} {text[2:4]} {text[4]} {text[5:]}"
        else:
            return text


def correct_indian_plate(text: str, confidence: float) -> Tuple[str, float, Dict]:
    """
    Convenience function to correct Indian license plate.

    Args:
        text: Raw OCR text
        confidence: OCR confidence

    Returns:
        Tuple of (corrected_text, adjusted_confidence, details)
    """
    corrected, conf_multiplier, details = IndianPlateOCR.correct_plate(text)

    adjusted_confidence = confidence * conf_multiplier

    return corrected, adjusted_confidence, details
