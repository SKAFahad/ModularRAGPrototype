"""
parse_image.py

Using docTR for OCR on images, returning a dict with:
  "text":   [recognized lines of text]
  "tables": []
  "images": []

We fix the 'Line' object has no attribute 'value' error by merging word.value for each line.
"""

import os
from typing import Dict, List

import cv2
import numpy as np

from doctr.io import DocumentFile
from doctr.models import ocr_predictor

def parse_image(file_path: str) -> Dict[str, List]:
    """
    The main function for aggregator usage:
      - Load the image as docTR DocumentFile
      - Use docTR's OCR model to detect lines
      - Each line has a list of Word objects. We join word.value to get line text
      - Return {"text": [...], "tables": [], "images": []}

    Since docTR doesn't do table/figure detection, those stay empty.
    """
    if not os.path.isfile(file_path):
        raise ValueError(f"[parse_image] File not found: {file_path}")

    # Load single-page docTR Document
    doc = DocumentFile.from_images(file_path)

    # Initialize docTR's OCR model
    model = ocr_predictor(pretrained=True)

    # Inference
    result = model(doc)

    recognized_text_lines = []

    # docTR result is a Document object
    for page in result.pages:
        # For each block in the page, we gather lines
        for block in page.blocks:
            for line in block.lines:
                # line.words is a list of Word objects, each with word.value
                line_text = " ".join(word.value for word in line.words)
                if line_text.strip():
                    recognized_text_lines.append(line_text)

    # Return the aggregator-friendly dict
    return {
        "text": recognized_text_lines,  # list of recognized line strings
        "tables": [],
        "images": []
    }

if __name__ == "__main__":
    # Quick test usage
    test_image_path = "example.jpg"
    print(f"[parse_image] Testing docTR parse_image with: {test_image_path}")
    parse_result = parse_image(test_image_path)
    print("[parse_image] Returned keys:", parse_result.keys())
    print(" -> text lines:", len(parse_result["text"]))
