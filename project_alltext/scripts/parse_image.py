#!/usr/bin/env python3
"""
parse_image.py
--------------
Parses image files by applying Optical Character Recognition (OCR) using pytesseract.
This converts image content to text.
Each chunk contains:
  - chunk_id: Unique identifier.
  - modality: "text" (since we extract textual content via OCR).
  - content: The text extracted from the image.
  - metadata: File name and an indicator of original modality.
"""

import os
import pytesseract  # Make sure Tesseract is installed and pytesseract is configured.
from PIL import Image

def process_image_file(file_path: str):
    """
    Uses OCR to convert an image to text and returns a text chunk.

    :param file_path: Path to the image file.
    :return: List with one dictionary representing the text chunk.
    """
    chunks = []

    try:
        # Open the image using PIL.
        img = Image.open(file_path)
        # Convert the image to string using pytesseract.
        extracted_text = pytesseract.image_to_string(img)
        
        # Create a unique chunk_id based on the file name.
        chunk_id = f"{os.path.basename(file_path)}_img_0"
        # Append the chunk with modality "text". Optionally, store the original modality.
        chunks.append({
            "chunk_id": chunk_id,
            "modality": "text",
            "content": extracted_text.strip(),
            "metadata": {
                "file_name": os.path.basename(file_path),
                "original_modality": "image"
            }
        })

    except Exception as e:
        print(f"Error processing image file {file_path}: {e}")

    return chunks
