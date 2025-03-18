"""
parse_image.py

This module uses the docTR library to perform Optical Character Recognition (OCR)
on image files (JPEG, PNG, TIFF, etc.). It returns recognized text as well as any
basic metadata in a structured format. The focus is on extracting textual content
that can later be passed through the rest of the RAG pipeline.

Guiding Principles:
1. Offline and open-source: Use docTR locally for end-to-end OCR without external APIs.
2. Modular design: parse_image.py should only handle reading an image file and extracting text.
3. Detailed documentation: Provide clear, thorough comments for easy onboarding.
4. Maintain consistent return structure with other parse_* scripts.

Recommended Return Structure:
{
  "text": <recognized text from the image, combined as a single string>,
  "tables": [],   # No table extraction in this module
  "images": [],   # Typically empty, or references if needed
  "metadata": {
     "file_name": <the input image filename>,
     "page_count": <1 if single image, or more if multi-page TIFF>,
     ...
  }
}

Note:
- docTR is an open-source deep-learning OCR tool. It performs text detection and
  text recognition. If your images contain multiple columns or handwriting, docTR
  generally handles them better than basic Tesseract.
- Make sure to install docTR and its dependencies properly:
    pip install python-doctr[torch]
  or  pip install python-doctr[tensorflow]
  depending on the backend you prefer (PyTorch or TensorFlow).
- For multi-page images (like multi-page TIFFs), docTR can read them as separate pages
  and return text for each page. We'll demonstrate how to handle that.

Usage:
    from parse_image import parse_image

    result = parse_image("path/to/image.png")
    # result is a dict with "text", "tables", "images", and "metadata"
"""

import os
import sys

# docTR imports
try:
    from doctr.io import DocumentFile
    from doctr.models import ocr_predictor
except ImportError:
    raise ImportError(
        "docTR is not installed. Please install docTR to use parse_image.py.\n"
        "Example:\n"
        "  pip install 'python-doctr[torch]'     # for PyTorch backend\n"
        "or\n"
        "  pip install 'python-doctr[tensorflow]' # for TensorFlow backend\n"
    )


def parse_image(file_path: str) -> dict:
    """
    Reads an image file from disk and performs OCR via docTR. Returns a dictionary
    containing recognized text, empty placeholders for tables/images, and metadata.

    :param file_path: The path to the image file (e.g., .png, .jpg, .tiff)
    :type file_path: str

    :return: Dictionary with keys:
      - "text": single string with recognized text from all pages/blocks/lines.
      - "tables": an empty list (no table extraction here).
      - "images": an empty list or references if you wish to store them.
      - "metadata": Additional info, including file_name, page_count, etc.
    :rtype: dict

    The structure is intentionally similar to parse_docx, parse_pdf, etc., so it
    can seamlessly fit into a unified pipeline for further chunking, embedding, etc.

    Key Steps:
      1) Validate that the file exists.
      2) Use docTR's DocumentFile to open the image as a docTR 'Document' object.
         If the image is multi-page (like a multi-page TIFF), docTR will treat
         each page separately.
      3) Load the OCR predictor model (if not already). We use pretrained=True for
         a standard docTR text recognition model.
      4) Perform OCR on the Document, resulting in a high-level structure of pages,
         blocks, lines, and words.
      5) Concatenate recognized words line by line, block by block, page by page,
         producing a final single string of recognized text.
      6) Return a dictionary with the recognized text, an empty "tables" list,
         an empty "images" list, and a metadata sub-dict.

    Requirements:
      - docTR: pip install 'python-doctr[torch]' or 'python-doctr[tensorflow]'
      - A local environment with Torch or TensorFlow depending on your docTR choice.
    """
    # 1) Validate the file existence:
    if not os.path.isfile(file_path):
        raise FileNotFoundError(
            f"parse_image: The file '{file_path}' does not exist or is not accessible."
        )

    # 2) Load the image with docTR.
    #    DocumentFile.from_images supports reading multi-page TIFF, returning multiple pages if present.
    try:
        doc = DocumentFile.from_images(file_path)
    except Exception as e:
        raise RuntimeError(f"parse_image: Failed to read image '{file_path}': {e}")

    # 3) Create an OCR predictor. pretrained=True loads a default model (for printed text).
    #    If you want specialized or multilingual models, docTR supports them as well.
    try:
        ocr_model = ocr_predictor(pretrained=True)
    except Exception as e:
        raise RuntimeError(
            "parse_image: Failed to load docTR OCR model. "
            f"Check your installation. Details: {e}"
        )

    # 4) Perform inference: returns a high-level result structure with pages, blocks, lines, words
    try:
        result = ocr_model(doc)
    except Exception as e:
        raise RuntimeError(
            f"parse_image: docTR OCR failed on '{file_path}'. Possible model/device error: {e}"
        )

    # 5) Build a single recognized text string from all pages/blocks/lines.
    recognized_text_lines = []
    page_count = 0
    for page in result.pages:
        page_count += 1  # increment page count
        # page.blocks is a list of text blocks
        for block in page.blocks:
            # block.lines is a list of line objects
            for line in block.lines:
                # each line has a list of Word objects with .value for recognized text
                line_text = " ".join(word.value for word in line.words)
                if line_text.strip():
                    recognized_text_lines.append(line_text.strip())

    # Join all recognized lines with newlines. Adjust to spaces if desired.
    final_text = "\n".join(recognized_text_lines)

    # We'll keep an empty "tables" key, as docTR does not do table detection by default.
    # We'll also keep "images" empty or you could store references to original file.
    tables_data = []
    images_data = []

    # Build metadata
    metadata = {
        "file_name": os.path.basename(file_path),
        "page_count": page_count
    }

    # Return a consistent structure
    parse_result = {
        "text": final_text,
        "tables": tables_data,
        "images": images_data,
        "metadata": metadata
    }

    return parse_result


if __name__ == "__main__":
    """
    If run directly: python parse_image.py /path/to/imagefile.png
    Will print out recognized text plus metadata, letting you quickly validate OCR.
    """
    if len(sys.argv) < 2:
        print("Usage: python parse_image.py <image_file>")
        sys.exit(1)

    image_path = sys.argv[1]

    try:
        output = parse_image(image_path)
        print(f"Successfully parsed '{image_path}'.")
        print("--- Extracted OCR Text (first 500 chars) ---")
        print(output["text"][:500], "..." if len(output["text"]) > 500 else "")
        print("--------------------------------------------\n")

        # We know "tables" is always empty in this docTR usage unless further logic is added
        print(f"Tables: {output['tables']}")
        print(f"Images: {output['images']}")
        print("Metadata:", output["metadata"])

    except Exception as e:
        print(f"Error while parsing image: {e}")
