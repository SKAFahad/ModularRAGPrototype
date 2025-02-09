"""
Data_Ingestion_and_Chunking.py

This is the main Python script for data ingestion & chunking in our modular RAG prototype.
It scans 'project/data/' (or any folder passed in) for files, identifies them by extension,
and delegates to specialized parsing functions. The result is a unified list of chunk
dictionaries that can be used in subsequent RAG pipeline steps.
"""

import os
import json
from typing import List, Dict

# Import the specialized parsing functions from other Python files
from parse_spreadsheet import process_spreadsheet
from parse_txt import process_text_file
from parse_image import process_image_file
from parse_docx import process_docx_file
from parse_pdf import process_pdf_file

def ingest_and_chunk(data_folder: str) -> List[Dict]:
    """
    Scans the specified data folder for XLSX, TXT, PNG/JPG, DOCX, PDF files
    and invokes the appropriate parse_* function for each.

    :param data_folder: The folder containing data files (e.g. 'project/data/')
    :return: A list of chunk dictionaries with the format:
        {
          "chunk_id": str,
          "modality": str,      # "text", "table", or "image"
          "content": str,       # the chunk text or file reference
          "metadata": {...}     # file_name, page_number, row_index, etc.
        }
    """
    all_chunks = []

    # 1. List all files in the data folder
    for file_name in os.listdir(data_folder):
        full_path = os.path.join(data_folder, file_name)

        # Skip directories (or process recursively if desired)
        if os.path.isdir(full_path):
            continue

        # Identify file extension in lowercase
        _, ext = os.path.splitext(file_name)
        ext = ext.lower()

        # 2. Dispatch to the correct parser
        if ext in [".xlsx", ".xls"]:
            # Spreadsheet
            new_chunks = process_spreadsheet(full_path)
            all_chunks.extend(new_chunks)

        elif ext == ".txt":
            # Plain text
            new_chunks = process_text_file(full_path)
            all_chunks.extend(new_chunks)

        elif ext in [".png", ".jpg", ".jpeg", ".gif"]:
            # Image file
            new_chunks = process_image_file(full_path)
            all_chunks.extend(new_chunks)

        elif ext == ".docx":
            # Word document
            new_chunks = process_docx_file(full_path)
            all_chunks.extend(new_chunks)

        elif ext == ".pdf":
            # PDF
            new_chunks = process_pdf_file(full_path)
            all_chunks.extend(new_chunks)

        else:
            print(f"Skipping unsupported file type: {file_name}")

    return all_chunks


if __name__ == "__main__":
    # Example usage:
    data_folder_path = "project/data"
    chunks = ingest_and_chunk(data_folder_path)
    print(f"Extracted {len(chunks)} chunks from folder '{data_folder_path}'.")

    # Optional: Print the first few chunks to verify content
    for chunk in chunks[:5]:
        print(chunk)

    # --- NEW: Write the chunks to a JSON file so Embedding_Generation.py can load them ---
    output_path = "project/chunked_data.json"
    with open(output_path, "w") as f:
        json.dump(chunks, f, indent=2)

    print(f"Saved chunks to '{output_path}'.")
