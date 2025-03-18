"""
parse_text.py

This module handles ingestion of plain text files (.txt). It reads the entire file
content into memory, generating a dictionary with the standard keys we use in our RAG
pipeline:

{
  "text":   <the entire text as a single string>,
  "tables": [],
  "images": [],
  "metadata": {
      "file_name": <txt file name>,
      "file_size_bytes": <optional, size of the file in bytes>,
      ...
  }
}

Guiding Principles (as discussed):
1. Offline, open-source approach: simply read local text files.
2. Modular design: parse_text.py only deals with plain text ingestion.
3. Detailed & maintainable: clear, thorough comments.
4. Consistent return structure with other parse_* scripts
   (pdf, docx, spreadsheet, image) so that the RAG pipeline can treat them uniformly.

Usage:
    from parse_text import parse_text_file

    result = parse_text_file("notes.txt")
    # result is a dict with "text", "tables" (empty), "images" (empty), "metadata"
"""

import os
import sys


def parse_text_file(file_path: str) -> dict:
    """
    Reads a plain text (.txt) file from disk and returns a standardized dictionary
    for the RAG pipeline.

    :param file_path: The path to a plain text file on disk (.txt)
    :type file_path: str

    :return: Dictionary with keys:
      "text":    <string content of the file>,
      "tables":  []  (no table parsing here),
      "images":  []  (no images in plain text),
      "metadata": additional file info (filename, etc.)
    :rtype: dict

    Steps:
      1) Validate file existence (FileNotFoundError if absent).
      2) Open in read-only mode with UTF-8 encoding, read entire content.
      3) Build the return dictionary:
         - "text" holds the file's entire text.
         - "tables" and "images" are empty lists by design for .txt.
         - "metadata" includes filename, size, or any additional info.

    Requirements:
      - No special library needed; standard Python I/O.
    """

    # 1) Check if the file exists
    if not os.path.isfile(file_path):
        raise FileNotFoundError(
            f"parse_text_file: The file '{file_path}' does not exist or is inaccessible."
        )

    # 2) Read the file content
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            file_content = f.read()
    except Exception as e:
        raise RuntimeError(f"parse_text_file: Failed to read '{file_path}': {e}")

    # Optionally gather file size or other metadata
    file_size = os.path.getsize(file_path)
    file_name = os.path.basename(file_path)

    # 3) Construct the dictionary with standard keys
    parse_result = {
        "text": file_content,
        "tables": [],   # plain text has no table concept
        "images": [],   # plain text has no embedded images
        "metadata": {
            "file_name": file_name,
            "file_size_bytes": file_size
        }
    }

    return parse_result


if __name__ == "__main__":
    """
    Command-line testing:
       python parse_text.py somefile.txt

    Will print a snippet of the text, plus metadata.
    """
    if len(sys.argv) < 2:
        print("Usage: python parse_text.py <text_file>")
        sys.exit(1)

    txt_file = sys.argv[1]
    try:
        result = parse_text_file(txt_file)
        print(f"Successfully parsed '{txt_file}'.\n")

        # Show snippet of text
        excerpt = result["text"][:500]
        trailing = "..." if len(result["text"]) > 500 else ""
        print("--- Extracted Text (first 500 chars) ---")
        print(excerpt + trailing)
        print("----------------------------------------")

        # Show metadata
        print("\nMetadata:", result["metadata"])

    except Exception as e:
        print(f"Error parsing text file '{txt_file}': {e}")
