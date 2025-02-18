#!/usr/bin/env python3
"""
parse_txt.py
------------
Parses plain text (.txt) files by reading them line by line.
Each line is stored as a text chunk.
Each chunk contains:
  - chunk_id: Unique identifier.
  - modality: "text".
  - content: The line of text.
  - metadata: File name and line number.
"""

import os

def process_text_file(file_path: str):
    """
    Reads a .txt file and converts each line into a text chunk.

    :param file_path: Path to the text file.
    :return: List of dictionaries representing text chunks.
    """
    chunks = []

    try:
        # Open the text file with UTF-8 encoding.
        with open(file_path, "r", encoding="utf-8") as file:
            lines = file.readlines()

        # Process each line to create a chunk.
        for i, line in enumerate(lines):
            line = line.strip()
            if line:
                chunk_id = f"{os.path.basename(file_path)}_line_{i}"
                chunks.append({
                    "chunk_id": chunk_id,
                    "modality": "text",
                    "content": line,
                    "metadata": {
                        "file_name": os.path.basename(file_path),
                        "line_number": i
                    }
                })
    except Exception as e:
        print(f"Error processing text file {file_path}: {e}")

    return chunks
