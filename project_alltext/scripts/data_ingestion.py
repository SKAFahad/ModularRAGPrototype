#!/usr/bin/env python3
"""
data_ingestion.py
-----------------
This script scans the data folder for files and delegates parsing to specialized functions.
All files (PDF, DOCX, spreadsheets, images, and plain text) are processed so that the output
is a list of text chunks. Each chunk includes:
  - chunk_id: A unique identifier for the chunk.
  - modality: Set to "text" (since we convert all content to text).
  - content: The extracted text.
  - metadata: Additional information such as file name and other relevant details.
The final output is saved as 'chunked_data.json'.
"""

import os
import json

# Import our specialized parser functions from our separate modules.
from parse_pdf import process_pdf_file
from parse_docx import process_docx_file
from parse_spreadsheet import process_spreadsheet
from parse_image import process_image_file
from parse_txt import process_text_file

def ingest_and_chunk(data_folder: str):
    """
    Scans the specified folder for supported file types and aggregates the text chunks.

    :param data_folder: The directory containing your raw data files.
    :return: A list of dictionaries (chunks), each containing chunk_id, modality, content, and metadata.
    """
    all_chunks = []

    # Loop through every file in the provided folder.
    for file_name in os.listdir(data_folder):
        full_path = os.path.join(data_folder, file_name)

        # Skip directories (if you want to process subdirectories recursively, add logic here).
        if os.path.isdir(full_path):
            continue

        # Get the file extension in lowercase for comparison.
        _, ext = os.path.splitext(file_name.lower())

        # Dispatch the file to the correct parser based on its extension.
        if ext == ".pdf":
            new_chunks = process_pdf_file(full_path)
        elif ext == ".docx":
            new_chunks = process_docx_file(full_path)
        elif ext in [".xlsx", ".xls"]:
            new_chunks = process_spreadsheet(full_path)
        elif ext in [".png", ".jpg", ".jpeg"]:
            new_chunks = process_image_file(full_path)
        elif ext == ".txt":
            new_chunks = process_text_file(full_path)
        else:
            print(f"Skipping unsupported file type: {file_name}")
            new_chunks = []

        # Append the chunks from this file to our overall list.
        all_chunks.extend(new_chunks)

    return all_chunks

if __name__ == "__main__":
    # Define the folder that contains all your input files.
    data_folder_path = "project_alltext/data"
    
    # Get all chunks by parsing the files.
    chunks = ingest_and_chunk(data_folder_path)
    print(f"Extracted {len(chunks)} text chunks from '{data_folder_path}'.")

    # Save the aggregated chunks to a JSON file.
    output_file = "project_alltext/chunked_data.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)
    print(f"Saved chunked data to '{output_file}'.")
