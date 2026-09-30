"""
OCR Post-Processing Module

Cleans and corrects OCR results for license plates.
"""

import re
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


class OCRPostProcessor:
    """Post-processes OCR results to fix common errors"""

    # Common OCR character confusions
    CHAR_CORRECTIONS = {
        # Letters that look like numbers
        'O': '0',  # Letter O → Number 0
        'o': '0',
        'I': '1',  # Letter I → Number 1
        'l': '1',  # Lowercase L → Number 1
        'Z': '2',
        'S': '5',
        'B': '8',
        'g': '9',  # lowercase g → 9
        'G': '0',  # uppercase G → 0 (looks like 0 in some fonts)
        'D': '0',  # D can look like 0
        'Q': '0',  # Q can look like 0

        # Numbers that look like letters (in specific contexts)
        # These are applied more carefully based on position
    }

    # Patterns for different license plate formats
    PLATE_PATTERNS = {
        'indian': r'^[A-Z]{2}\d{2}[A-Z]{1,2}\d{4}$',  # KL01CA2555
        'us': r'^[A-Z0-9]{2,3}\s?[A-Z0-9]{3,4}$',     # ABC 1234
        'european': r'^[A-Z]{2}[\s-]?[A-Z]{2}[\s-]?\d{3,4}$',  # PG-MN-112
        'generic': r'^[A-Z0-9\s-]{5,12}$',
    }

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Basic cleaning of OCR output.

        Args:
            text: Raw OCR text

        Returns:
            Cleaned text
        """
        # Remove extra whitespace
        text = ' '.join(text.split())

        # Remove common OCR artifacts
        text = text.replace('|', '1')
        text = text.replace('!', '1')
        text = text.replace(',', '')
        text = text.replace('.', '')

        # Convert to uppercase
        text = text.upper()

        return text

    @staticmethod
    def correct_common_errors(text: str, plate_format: str = 'auto') -> str:
        """
        Correct common OCR errors based on license plate format.

        Args:
            text: OCR text
            plate_format: Expected format ('indian', 'us', 'european', 'auto')

        Returns:
            Corrected text
        """
        text = OCRPostProcessor.clean_text(text)

        # Auto-detect format if needed
        if plate_format == 'auto':
            plate_format = OCRPostProcessor.detect_format(text)

        logger.info(f"Detected plate format: {plate_format}")

        if plate_format == 'indian':
            # Indian format: KL01CA2555 (State Code + District + Series + Number)
            # Pattern: 2 letters + 2 digits + 1-2 letters + 4 digits
            corrected = OCRPostProcessor._correct_indian_plate(text)

        elif plate_format == 'european':
            # European format: PG-MN-112 or PG MN 112
            corrected = OCRPostProcessor._correct_european_plate(text)

        elif plate_format == 'us':
            # US format: ABC 1234
            corrected = OCRPostProcessor._correct_us_plate(text)

        else:
            # Generic corrections
            corrected = OCRPostProcessor._apply_generic_corrections(text)

        logger.info(f"Original: '{text}' → Corrected: '{corrected}'")
        return corrected

    @staticmethod
    def detect_format(text: str) -> str:
        """
        Auto-detect license plate format.

        Args:
            text: OCR text

        Returns:
            Format name ('indian', 'us', 'european', 'generic')
        """
        original_text = text
        text_clean = text.replace(' ', '').replace('-', '').replace('e', '').upper()

        # Indian: Starts with 2 letters, length 9-10
        # Pattern: KL01CA2555 (2 letters + digits/letters mix)
        if len(text_clean) >= 9 and len(text_clean) <= 10:
            # Check if starts with 2 letters
            if text_clean[:2].isalpha():
                # Likely Indian format
                return 'indian'

        # European: Contains dashes, spaces, or lowercase 'e' (common OCR error for separator)
        if '-' in original_text or ' ' in original_text or 'e' in original_text.lower():
            if len(text_clean) >= 6 and len(text_clean) <= 9:
                return 'european'

        # US: typically shorter
        if len(text_clean) <= 8:
            return 'us'

        return 'indian' if len(text_clean) >= 9 else 'generic'

    @staticmethod
    def _correct_indian_plate(text: str) -> str:
        """
        Correct Indian license plate format.

        Expected: KL01CA2555
        Pattern: [A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}
        """
        # Remove spaces and dashes
        text = text.replace(' ', '').replace('-', '').upper()

        # Length should be 10 (2+2+2+4) or 9 (2+2+1+4)
        if len(text) < 9 or len(text) > 10:
            logger.warning(f"Indian plate length unexpected: {len(text)}")

        result = []

        for i, char in enumerate(text):
            if i < 2:
                # First 2 characters: letters (state code)
                if char.isdigit():
                    # If digit appears, might be OCR error
                    result.append(char)  # Keep as-is for now
                else:
                    result.append(char.upper())

            elif i < 4:
                # Next 2 characters: MUST BE digits (district code)
                if char.isalpha():
                    # Convert common letter-to-digit confusions
                    if char in OCRPostProcessor.CHAR_CORRECTIONS:
                        corrected = OCRPostProcessor.CHAR_CORRECTIONS[char]
                        logger.info(f"Corrected '{char}' → '{corrected}' at position {i+1}")
                        result.append(corrected)
                    else:
                        # Unknown letter in digit position - try common patterns
                        if char == 'O':
                            result.append('0')
                        elif char == 'I' or char == 'L':
                            result.append('1')
                        elif char == 'G':
                            result.append('0')  # G often looks like 0 or 9
                        else:
                            result.append(char)
                else:
                    result.append(char)

            elif i < 6 or (len(text) == 10 and i < 6):
                # Next 1-2 characters: letters (series)
                if char.isdigit():
                    # Might be OCR error, but keep for now
                    result.append(char)
                else:
                    result.append(char.upper())

            else:
                # Last 4 characters: digits
                if char.isalpha():
                    # Convert letter to digit
                    if char in OCRPostProcessor.CHAR_CORRECTIONS:
                        corrected = OCRPostProcessor.CHAR_CORRECTIONS[char]
                        logger.info(f"Corrected '{char}' → '{corrected}' at position {i+1}")
                        result.append(corrected)
                    else:
                        result.append(char)
                else:
                    result.append(char)

        return ''.join(result)

    @staticmethod
    def _correct_european_plate(text: str) -> str:
        """
        Correct European license plate format.

        Expected: PG-MN-112 or PG MN 112
        """
        original_has_dash = '-' in text
        original_has_space = ' ' in text

        # Convert common OCR errors for separators
        # 'e' is often misread as separator
        text = text.replace('e', ' ').replace('E', ' ')

        # Also convert multiple spaces to single space
        text = ' '.join(text.split())

        # Split by separators
        if '-' in text:
            parts = text.split('-')
            separator = '-'
        elif ' ' in text:
            parts = text.split()
            separator = ' '
        else:
            # No clear separator, try to split intelligently
            # Typical European format: 2 letters, 2 letters, 3-4 digits
            # Example: PGMN112 → PG MN 112
            parts = []
            if len(text) >= 6:
                parts.append(text[:2])   # First 2 chars

                # Find where digits start
                digit_start = -1
                for i, char in enumerate(text[2:], start=2):
                    if char.isdigit():
                        digit_start = i
                        break

                if digit_start > 2:
                    parts.append(text[2:digit_start])  # Letters in middle
                    parts.append(text[digit_start:])   # Digits at end
                else:
                    # Couldn't split intelligently, just use what we have
                    parts = [text]
            else:
                parts = [text]

            separator = ' '

        # Clean each part
        cleaned_parts = []
        for part in parts:
            part = part.strip().upper()
            if part:
                cleaned_parts.append(part)

        # Rejoin with appropriate separator
        result = separator.join(cleaned_parts)

        logger.info(f"European plate split: {cleaned_parts} → '{result}'")

        return result

    @staticmethod
    def _correct_us_plate(text: str) -> str:
        """
        Correct US license plate format.

        Expected: ABC 1234
        """
        text = text.upper()

        # Apply generic corrections
        for old, new in OCRPostProcessor.CHAR_CORRECTIONS.items():
            text = text.replace(old, new)

        return text

    @staticmethod
    def _apply_generic_corrections(text: str) -> str:
        """
        Apply generic character corrections.
        """
        result = []

        for char in text:
            if char.upper() in OCRPostProcessor.CHAR_CORRECTIONS:
                result.append(OCRPostProcessor.CHAR_CORRECTIONS[char.upper()])
            else:
                result.append(char.upper())

        return ''.join(result)

    @staticmethod
    def calculate_confidence_adjustment(original: str, corrected: str) -> float:
        """
        Calculate confidence adjustment based on corrections made.

        Args:
            original: Original OCR text
            corrected: Corrected text

        Returns:
            Confidence multiplier (0.9-1.0)
        """
        if original == corrected:
            return 1.0

        # Count differences
        differences = sum(1 for a, b in zip(original, corrected) if a != b)

        # Small penalty for corrections
        penalty = min(0.1, differences * 0.02)

        return 1.0 - penalty


def correct_plate_text(text: str, confidence: float, plate_format: str = 'auto') -> Tuple[str, float]:
    """
    Convenience function to correct OCR text.

    Args:
        text: OCR text
        confidence: Original confidence
        plate_format: Expected format

    Returns:
        Tuple of (corrected_text, adjusted_confidence)
    """
    processor = OCRPostProcessor()

    corrected = processor.correct_common_errors(text, plate_format)

    # Adjust confidence if corrections were made
    conf_adjustment = processor.calculate_confidence_adjustment(text, corrected)
    adjusted_confidence = confidence * conf_adjustment

    return corrected, adjusted_confidence
