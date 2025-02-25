"""
chunk_image.py

This module converts an image into multiple text chunks by:
  1) Pre-processing the image (grayscale, blur, threshold).
  2) Running OCR with pytesseract to get extracted_text.
  3) Splitting that text into multiple paragraphs or wrapped lines
     using logic from "chunk_text".
  4) Returning a list of chunk dicts, each with:
     {
       "chunk_id": ...,
       "modality": "image",
       "content": ...,
       "metadata": {...},
       "textual_modality": "ocr_extracted"
     }

Requires:
    pip install pytesseract opencv-python nltk
Also ensure Tesseract is installed on your system.

Example usage:
    from chunk_image import chunk_image_file
    chunks = chunk_image_file("my_chart.png")
    # 'chunks' is a list of chunk dicts for each paragraph from the OCR text.
"""

import os
import cv2
import pytesseract

# We'll reuse logic from chunk_text for paragraph splitting and wrapping
import textwrap
import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt', quiet=True)

def preprocess_image(image_path: str):
    """
    Loads the image and applies pre-processing to enhance OCR:
      - Convert to grayscale
      - Median blur
      - Otsu threshold => binary
    Returns the thresholded image (numpy array).
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"[chunk_image] Could not read image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 3)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return thresh

def reflow_paragraphs(text: str, width: int = 80) -> list:
    """
    Splits the text into paragraphs (double newlines),
    tokenizes sentences, then re-wraps lines using textwrap.
    Returns a list of paragraph strings.
    """
    paragraphs = text.split("\n\n")
    formatted_pars = []

    for para in paragraphs:
        clean_para = " ".join(para.split())
        if not clean_para:
            continue

        # Tokenize into sentences for better spacing
        sentences = sent_tokenize(clean_para)
        combined = " ".join(sentences)

        # Wrap lines to 'width'
        wrapped = textwrap.fill(combined, width=width)
        formatted_pars.append(wrapped)

    return formatted_pars

def chunk_image_file(image_path: str, width: int = 80) -> list:
    """
    1) Pre-process the image
    2) OCR with pytesseract => 'extracted_text'
    3) Reflow paragraphs using reflow_paragraphs
    4) For each paragraph, build a chunk dictionary with:
       chunk_id, modality="image", content=paragraph_text,
       metadata={ file_name, paragraph_index }, textual_modality="ocr_extracted"

    Returns a list of chunk dictionaries.
    """
    # Step 1: Preprocess the image
    processed_img = preprocess_image(image_path)

    # Step 2: OCR
    extracted_text = pytesseract.image_to_string(processed_img)

    # Step 3: Split into paragraphs
    paragraphs = reflow_paragraphs(extracted_text, width=width)

    # Step 4: Create chunk dicts
    chunk_dicts = []
    base_name = os.path.basename(image_path)

    for i, para_text in enumerate(paragraphs):
        chunk_id = f"{base_name}_img_0_par_{i}"

        chunk = {
            "chunk_id": chunk_id,
            "modality": "image",
            "content": para_text,
            "metadata": {
                "file_name": base_name,
                "paragraph_index": i
            },
            "textual_modality": "ocr_extracted"
        }
        chunk_dicts.append(chunk)

    return chunk_dicts

if __name__ == "__main__":
    # Simple test usage
    test_image = "chart_image.png"
    chunks = chunk_image_file(test_image)
    print(f"[chunk_image] Created {len(chunks)} chunks from {test_image}.")
    for c in chunks:
        print(c)
