"""
Create a synthetic license plate for testing OCR
"""

import cv2
import numpy as np
from pathlib import Path

def create_test_plate():
    """Create a simple license plate image"""
    # Create white background
    img = np.ones((200, 600, 3), dtype=np.uint8) * 255

    # Add border
    cv2.rectangle(img, (10, 10), (590, 190), (0, 0, 0), 3)

    # Add text (license plate number)
    text = "ABC1234"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 3
    thickness = 8

    # Get text size to center it
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
    x = (600 - text_width) // 2
    y = (200 + text_height) // 2

    # Draw text
    cv2.putText(img, text, (x, y), font, font_scale, (0, 0, 0), thickness)

    # Save
    output_path = "data/samples/test_plate.jpg"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, img)

    print(f"[OK] Created test license plate: {output_path}")
    print(f"  Text: {text}")
    return output_path

if __name__ == "__main__":
    create_test_plate()
