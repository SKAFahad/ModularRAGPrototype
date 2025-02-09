# parse_image.py

import os
from typing import List, Dict

def process_image_file(file_path: str) -> List[Dict]:
    """
    For now, we simply store a reference to the image file.
    If needed, we can do OCR or chart analysis later.
    """
    chunks = []
    try:
        chunk_id = f"{os.path.basename(file_path)}_img_0"
        chunks.append({
            "chunk_id": chunk_id,
            "modality": "image",
            "content": file_path,  # path reference
            "metadata": {
                "file_name": os.path.basename(file_path)
            }
        })
    except Exception as e:
        print(f"Error processing image file {file_path}: {e}")
    return chunks
