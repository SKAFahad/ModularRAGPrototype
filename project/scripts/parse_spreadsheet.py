# parse_spreadsheet.py

import pandas as pd
import os
from typing import List, Dict

def process_spreadsheet(file_path: str) -> List[Dict]:
    """
    Reads an Excel file (XLSX/XLS) and outputs each row as a 'table' chunk.
    """
    chunks = []
    try:
        df = pd.read_excel(file_path)

        for i, row in df.iterrows():
            chunk_id = f"{os.path.basename(file_path)}_row_{i}"
            # Build a string representation of each row
            row_text = ", ".join([f"{col}: {row[col]}" for col in df.columns])

            chunks.append({
                "chunk_id": chunk_id,
                "modality": "table",
                "content": row_text,
                "metadata": {
                    "file_name": os.path.basename(file_path),
                    "row_index": i
                }
            })
    except Exception as e:
        print(f"Error processing spreadsheet {file_path}: {e}")
    return chunks
