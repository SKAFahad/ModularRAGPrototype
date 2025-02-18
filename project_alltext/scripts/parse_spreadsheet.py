#!/usr/bin/env python3
"""
parse_spreadsheet.py
--------------------
Parses spreadsheet files to extract textual data.
Each row is converted into a comma-separated string representing column values.
A separate chunk is created for each row.
Each chunk contains:
  - chunk_id: Unique identifier.
  - modality: "text".
  - content: Text string of the row.
  - metadata: File name and row index.
"""

import os
import pandas as pd

def process_spreadsheet(file_path: str):
    """
    Reads a spreadsheet file and converts each row into a text chunk.

    :param file_path: Path to the spreadsheet file.
    :return: List of dictionaries representing text chunks.
    """
    chunks = []

    try:
        # Read the spreadsheet into a pandas DataFrame.
        df = pd.read_excel(file_path)

        # Iterate through each row in the DataFrame.
        for i, row in df.iterrows():
            # Create a text representation for the row (e.g., "Column1: value1, Column2: value2").
            row_text = ", ".join([f"{col}: {row[col]}" for col in df.columns])
            chunk_id = f"{os.path.basename(file_path)}_row_{i}"
            chunks.append({
                "chunk_id": chunk_id,
                "modality": "text",
                "content": row_text,
                "metadata": {
                    "file_name": os.path.basename(file_path),
                    "row_index": i
                }
            })

    except Exception as e:
        print(f"Error processing spreadsheet {file_path}: {e}")

    return chunks
