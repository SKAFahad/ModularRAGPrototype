# parse_txt.py

import os
from typing import List, Dict

def process_text_file(file_path: str) -> List[Dict]:
    """
    Reads a .txt file line by line, creates 'text' chunks.
    """
    chunks = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            chunk_id = f"{os.path.basename(file_path)}_line_{i}"
            chunks.append({
                "chunk_id": chunk_id,
                "modality": "text",
                "content": line.strip(),
                "metadata": {
                    "file_name": os.path.basename(file_path),
                    "line_number": i
                }
            })
    except Exception as e:
        print(f"Error processing text file {file_path}: {e}")
    return chunks
