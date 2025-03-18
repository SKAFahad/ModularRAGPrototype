"""
chunk_table.py

This module transforms table data (e.g., rows from spreadsheets or CSV files)
into smaller text chunks suitable for embedding in a RAG pipeline. The typical use
case is when you have parsed table data via parse_spreadsheet or parse_pdf (with
Camelot/Tabula), and now you want to convert each row (or group of rows) into
a text "chunk."

Guiding Principles:
1. Keep the functionality focused on chunking table data.
2. Provide thorough comments for maintainability and clarity.
3. Return a list of dictionaries, each representing a chunk with consistent
   keys for easier downstream usage (embedding, storage, retrieval, etc.).
4. Align with the rest of our pipeline's chunk structure, including metadata.

Recommended Return Structure (list of chunk dicts):
[
  {
    "chunk_id": <unique ID for the chunk>,
    "modality": "table",
    "content": <text representation of the table row(s)>,
    "metadata": {
       "file_name": <source file name or table name>,
       "row_index": <index of the row or group of rows>,
       ...
    },
    "textual_modality": "row_data"
  },
  ...
]

Usage Example:
    from chunk_table import chunk_table_rows

    # Suppose we have table_data as a 2D list from parse_spreadsheet or parse_pdf
    # e.g., [ [col1, col2, col3], [val1, val2, val3], ... ]
    # and we want to chunk each row into a chunk of text
    chunk_list = chunk_table_rows(table_data, file_name="data.xlsx")
    # chunk_list will be a list of dicts, one per row
"""

import os


def chunk_table_rows(
    table_data: list,
    file_name: str,
    start_index: int = 0
) -> list:
    """
    Converts a 2D list (representing table rows) into chunk dictionaries. Each row becomes
    a chunk with a textual representation of its columns. The function uses a simple mapping:
    row_data -> "Column0: val0, Column1: val1, ..." or a direct joined approach.

    :param table_data: A 2D list of rows, e.g. [ [colA, colB, ...], [valA, valB, ...], ... ]
    :type table_data: list

    :param file_name: The name of the source file or table (for metadata, chunk_id).
    :type file_name: str

    :param start_index: Used if you have multiple tables in one file; you can pass an offset
                        so row_index doesn't clash. Typically 0 if there's just one table.
    :type start_index: int

    :return: A list of chunk dictionaries. Each chunk is a row from the table.
             [
               {
                 "chunk_id": "<file_name>_row_0",
                 "modality": "table",
                 "content": "Col0: val0, Col1: val1, ...",
                 "metadata": {
                     "file_name": <file_name>,
                     "row_index": 0
                 },
                 "textual_modality": "row_data"
               },
               ...
             ]
    :rtype: list

    Steps:
      1) Iterate over each row in 'table_data' (a 2D list).
      2) Convert the row to a textual form (e.g., "Column0: val0, Column1: val1, ...").
         For advanced usage, you might store actual column headers, but that typically
         requires a separate step. By default, we just name columns "Column0", "Column1", etc.
      3) Build a chunk dict with the standard fields (chunk_id, modality="table", content, etc.).
      4) Append the chunk to a results list.
      5) Return the list.

    Potential Enhancements:
      - If the first row is a header row, you might want to use actual column names in the
        textual representation. This is optional and depends on your pipeline approach.
      - If the table is large, you might group rows (e.g., 10 rows per chunk).
      - If columns are numeric or large text, you might skip some or transform them.

    Example:
      table_data = [
         ["Name", "Age", "Role"],
         ["Alice", "30", "Engineer"],
         ["Bob", "25", "Designer"]
      ]
      chunk_table_rows(table_data, "employees.csv") -> 
      [
         {
           "chunk_id": "employees.csv_row_0",
           "modality": "table",
           "content": "Column0: Name, Column1: Age, Column2: Role",
           "metadata": { "file_name": "employees.csv", "row_index": 0 },
           "textual_modality": "row_data"
         },
         {
           "chunk_id": "employees.csv_row_1",
           "modality": "table",
           "content": "Column0: Alice, Column1: 30, Column2: Engineer",
           ...
         },
         ...
      ]
    """

    chunks = []

    # Iterate through each row, building a chunk
    for i, row in enumerate(table_data):
        row_index = i + start_index
        # We'll name columns as "Column0", "Column1", etc.
        # Then join them with commas. Alternatively, just do ",".join(row).
        row_strings = []
        for col_index, cell_value in enumerate(row):
            # convert cell_value to string, strip if needed
            val_str = str(cell_value).strip()
            row_strings.append(f"Column{col_index}: {val_str}")

        # combine row data into one string
        row_content = ", ".join(row_strings)

        chunk_id = f"{file_name}_row_{row_index}"
        chunk_dict = {
            "chunk_id": chunk_id,
            "modality": "table",
            "content": row_content,
            "metadata": {
                "file_name": file_name,
                "row_index": row_index
            },
            "textual_modality": "row_data"
        }
        chunks.append(chunk_dict)

    return chunks


if __name__ == "__main__":
    """
    Example usage / standalone test:
      python chunk_table.py
    We'll just provide a small sample table_data inline for demonstration.
    """

    # Sample table
    sample_data = [
        ["Name", "Age", "Role"],
        ["Alice", "30", "Engineer"],
        ["Bob", "25", "Designer"],
    ]
    file_name = "employees.csv"

    results = chunk_table_rows(sample_data, file_name)
    print(f"[chunk_table] Created {len(results)} chunks from {file_name}.\n")
    for r in results:
        print("Chunk ID:", r["chunk_id"])
        print("Modality:", r["modality"])
        print("Content:", r["content"])
        print("Metadata:", r["metadata"])
        print("Textual Modality:", r["textual_modality"])
        print("------------------------------------------")
