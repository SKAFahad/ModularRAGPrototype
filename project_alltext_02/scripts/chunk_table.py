"""
chunk_table.py

This module reads a CSV file (via Pandas) and transforms each row into
a chunk dictionary with fields:
  - chunk_id
  - modality="table"
  - content (string describing the row)
  - metadata { file_name, row_index, ... }
  - textual_modality="row_data"

Dependencies:
  pip install pandas

Usage:
    from chunk_table import chunk_table_csv
    chunks = chunk_table_csv("mydata.csv")
    # 'chunks' is a list of row-based chunk dicts
"""

import os
import pandas as pd

def chunk_table_csv(csv_file: str) -> list:
    """
    Reads the CSV into a Pandas DataFrame, then for each row,
    creates a chunk dictionary representing that row's data.

    Example chunk dict:
    {
      "chunk_id": "mydata.csv_row_0",
      "modality": "table",
      "content": "ColumnA: valA, ColumnB: valB",
      "metadata": {
        "file_name": "mydata.csv",
        "row_index": 0
      },
      "textual_modality": "row_data"
    }

    Returns a list of such chunk dicts.
    """
    if not os.path.isfile(csv_file):
        raise FileNotFoundError(f"[chunk_table] Could not find CSV: {csv_file}")

    df = pd.read_csv(csv_file)
    base_name = os.path.basename(csv_file)

    chunk_list = []

    # Iterate over rows with df.iterrows() -> (index, Series)
    for i, row_series in df.iterrows():
        # Convert the row's columns/values to a single string
        # e.g. "ColA: valA, ColB: valB"
        row_strings = []
        for col_name in df.columns:
            val = str(row_series[col_name])
            row_strings.append(f"{col_name}: {val}")
        row_content = ", ".join(row_strings)

        chunk_id = f"{base_name}_row_{i}"

        chunk_dict = {
            "chunk_id": chunk_id,
            "modality": "table",
            "content": row_content,
            "metadata": {
                "file_name": base_name,
                "row_index": i
            },
            "textual_modality": "row_data"
        }
        chunk_list.append(chunk_dict)

    return chunk_list

if __name__ == "__main__":
    # Example standalone usage
    test_csv = "data.csv"
    chunks = chunk_table_csv(test_csv)
    print(f"[chunk_table] Created {len(chunks)} chunks from {test_csv}.")
    # Print first few
    for c in chunks[:3]:
        print(c)
