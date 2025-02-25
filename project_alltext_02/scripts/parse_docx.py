"""
parse_docx.py

Fixes the TypeError by flattening nested lists in docx2python's result.body.
We do NOT handle images (no extract_image=True) because docx2python's version
no longer supports that param. So we return "images": [].

Usage:
  from parse_docx import parse_docx
  parse_result = parse_docx("some_doc.docx")
  # parse_result -> { "text": [...], "tables": [...], "images": [] }

If you run this script directly, it saves text/tables but not images.
"""

import os
import pandas as pd
from docx2python import docx2python


def flatten_runs(paragraph_runs):
    """
    docx2python often returns nested lists in paragraph_runs (something like [[run1, run2], [run3]]).
    We flatten them into a single list of strings.
    """
    flattened = []
    for item in paragraph_runs:
        if isinstance(item, list):
            # If 'item' is itself a list, extend our flattened list
            flattened.extend(item)
        else:
            # Otherwise, it's a string (a run)
            flattened.append(item)
    return flattened


def extract_docx_content(docx_path):
    """
    Loads a DOCX file with docx2python, extracts text & tables (no images).
    Returns:
      text_data: List of paragraphs (each a string)
      tables_data: List of nested-list tables
    """
    # docx2python in this version may not accept extract_image=True
    result = docx2python(docx_path)

    # 1) TEXT
    text_data = []
    # result.body is typically [section][paragraph][runs... possibly nested]
    for section in result.body:
        for paragraph_runs in section:
            # 'paragraph_runs' might be a list of lists
            flattened = flatten_runs(paragraph_runs)  # ensures we have a flat list of strings
            paragraph_text = " ".join(flattened).strip()
            if paragraph_text:
                text_data.append(paragraph_text)

    # 2) TABLES
    # docx2python populates .body_tables as a list of tables (nested list of rows/cells)
    tables_data = getattr(result, "body_tables", [])

    return text_data, tables_data


def parse_docx(file_path: str) -> dict:
    """
    The function aggregator calls:
    Returns {"text": [...], "tables": [...], "images": []}.
    """
    text_data, tables_data = extract_docx_content(file_path)

    # aggregator expects an images list, but we have none -> empty
    return {
        "text": text_data,
        "tables": tables_data,
        "images": []
    }


# Optional standalone usage
def save_text(text_data, output_dir):
    """
    Saves a list of paragraphs to one text file in 'output_dir'.
    """
    os.makedirs(output_dir, exist_ok=True)
    text_file = os.path.join(output_dir, "document_text.txt")
    with open(text_file, "w", encoding="utf-8") as f:
        for para in text_data:
            f.write(para + "\n\n")
    print(f"[parse_docx] Wrote paragraphs to: {text_file}")


def save_tables(tables_data, output_dir):
    """
    Saves each table as a CSV in 'output_dir'.
    """
    os.makedirs(output_dir, exist_ok=True)
    if not tables_data:
        print("[parse_docx] No tables found.")
        return
    for idx, table_nested in enumerate(tables_data, start=1):
        df = pd.DataFrame(table_nested)
        csv_path = os.path.join(output_dir, f"table_{idx}.csv")
        df.to_csv(csv_path, index=False, header=False)
        print(f"[parse_docx] Table {idx} saved to: {csv_path}")


def merge_and_save_docx(docx_path, output_dir="docx_output"):
    """
    If you run parse_docx.py directly, this function extracts
    text & tables, then saves them. 'images' is unused & empty.
    """
    text_data, tables_data = extract_docx_content(docx_path)
    save_text(text_data, output_dir)
    save_tables(tables_data, output_dir)
    return {
        "text": text_data,
        "tables": tables_data,
        "images": []
    }


if __name__ == "__main__":
    docx_file = "your_document.docx"
    result = merge_and_save_docx(docx_file)
    print("[parse_docx] Done. Returned dict has:", result.keys())
