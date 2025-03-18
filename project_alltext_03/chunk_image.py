"""
chunk_image.py

This module is responsible for transforming the OCR-extracted text from images
into smaller, more manageable "chunks." These chunks can then be used in downstream
stages such as embedding and retrieval. The idea is to split the text in a way
that preserves context while avoiding overly large segments that might exceed
LLM context windows.

Guiding Principles (per our discussion):
1. Keep the module focused on chunking OCR-based text from images.
2. Provide detailed, maintainable comments so future developers can easily follow the logic.
3. Return a list of dictionaries, each representing a chunk. This consistent structure
   allows the chunked output to integrate seamlessly with the rest of the RAG pipeline.
4. Include metadata, such as the file name or chunk ID, for traceability.

Recommended Return Format (list of chunk dicts):
[
  {
    "chunk_id": <unique ID for the chunk>,
    "modality": "image",       # from an OCR-extracted image
    "content": <the textual content of this chunk>,
    "metadata": {
        "file_name": <image file name or ID>,
        "paragraph_index": <index of the chunk in this file>
        ...
    },
    "textual_modality": "ocr_extracted"
  },
  ...
]

Usage Example:
    from chunk_image import chunk_image_text

    # Suppose 'ocr_text' is the entire OCR result from parse_image
    # and 'img_file_name' is the name of the image file
    chunk_list = chunk_image_text(ocr_text, img_file_name)
    # chunk_list is a list of chunk dicts for each chunk of text
"""

import os
import textwrap
import nltk
from nltk.tokenize import sent_tokenize

# Ensure NLTK's 'punkt' data is downloaded, or handle gracefully
nltk.download('punkt', quiet=True)


def chunk_image_text(
    ocr_text: str,
    file_name: str,
    max_width: int = 80
) -> list:
    """
    Splits the OCR-extracted text from an image into multiple smaller chunks.
    Each chunk is a dictionary with fields:
      - chunk_id: e.g. <file_name>_img_<chunk_index>
      - modality: always "image"
      - content: the chunk's textual content
      - metadata: includes the file name and a chunk/paragraph index
      - textual_modality: "ocr_extracted"

    :param ocr_text: The entire string of text extracted from an image (via OCR).
    :type ocr_text: str

    :param file_name: The name of the source image file. Used to label chunk_id and metadata.
    :type file_name: str

    :param max_width: The approximate character width for wrapping each chunk. Defaults to 80.
    :type max_width: int

    :return: A list of chunk dictionaries, each representing a segment of text.
    :rtype: list

    Steps:
      1) Clean & split the text into paragraphs (by double newlines or some heuristic).
      2) For each paragraph, tokenize into sentences to maintain logical boundaries.
      3) Wrap lines to 'max_width' using textwrap.fill or build chunk sizes
         that are manageable for subsequent LLM context usage.
      4) Create a chunk dict for each wrapped paragraph, storing chunk_id, content,
         and appropriate metadata.

    Example output:
    [
      {
        "chunk_id": "my_image.png_img_0_par_0",
        "modality": "image",
        "content": "This is the first chunk of text from the OCR result...",
        "metadata": {
          "file_name": "my_image.png",
          "paragraph_index": 0
        },
        "textual_modality": "ocr_extracted"
      },
      ...
    ]
    """

    if not ocr_text.strip():
        # If there's no text, return an empty list
        return []

    # 1) Split the text by double newlines into paragraphs
    raw_paragraphs = ocr_text.split("\n\n")
    cleaned_paragraphs = []

    # Remove extra whitespace, skip empty paragraphs
    for para in raw_paragraphs:
        para_str = " ".join(para.split())
        if para_str:
            cleaned_paragraphs.append(para_str)

    # 2) For each paragraph, we tokenize sentences to ensure we keep meaningful boundaries.
    #    Then rejoin them, and wrap the text to 'max_width'.
    chunk_list = []
    paragraph_counter = 0
    for paragraph_text in cleaned_paragraphs:
        # Sentence tokenize for clarity
        sentences = sent_tokenize(paragraph_text)
        combined_para = " ".join(sentences)

        # 3) Wrap lines to max_width
        #    textwrap.fill(...) returns a single string with embedded newlines.
        #    We'll treat that entire block as a single chunk in this example. 
        #    Alternatively, you could split further if you want smaller chunks.
        wrapped_text = textwrap.fill(combined_para, width=max_width)

        # 4) Build the chunk dictionary
        chunk_id = f"{file_name}_img_0_par_{paragraph_counter}"
        chunk_dict = {
            "chunk_id": chunk_id,
            "modality": "image",
            "content": wrapped_text,
            "metadata": {
                "file_name": file_name,
                "paragraph_index": paragraph_counter
            },
            "textual_modality": "ocr_extracted"
        }
        chunk_list.append(chunk_dict)
        paragraph_counter += 1

    return chunk_list


if __name__ == "__main__":
    """
    Standalone test:
      python chunk_image.py "Some OCR text..." "image_file_name"

    This is just for debugging. Real usage is typically within the pipeline, after parse_image.
    """
    import sys

    if len(sys.argv) < 3:
        print("Usage: python chunk_image.py <ocr_text> <image_file_name>")
        sys.exit(1)

    ocr_text_arg = sys.argv[1]
    file_name_arg = sys.argv[2]

    results = chunk_image_text(ocr_text_arg, file_name_arg)
    print(f"[chunk_image] Created {len(results)} chunk(s).")
    for chunk in results:
        print("-- Chunk ID:", chunk["chunk_id"])
        print("   content:", chunk["content"][:80], "..." if len(chunk["content"]) > 80 else "")
        print("   metadata:", chunk["metadata"])
