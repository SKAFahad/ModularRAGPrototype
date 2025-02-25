#!/usr/bin/env python3
"""
data_ingestion.py
-----------------
This script scans the data folder for files and delegates parsing to specialized functions.
All supported files (PDF, DOCX, spreadsheets, images, and plain text) are processed so that
the output is a list of text chunks. Each chunk includes:
  - chunk_id: A unique identifier for the chunk.
  - modality: Set to "text" (since all content is converted to text).
  - content: The extracted text from the file.
  - metadata: Additional information such as file name and other relevant details.
The final output is saved as 'chunked_data.json' in the project_alltext folder.
"""

import os
import json

# Import specialized parser functions from separate modules.
from parse_pdf import process_pdf_file
from parse_docx import process_docx_file
from parse_spreadsheet import process_spreadsheet
from parse_image import process_image_file
from parse_txt import process_text_file

def ingest_and_chunk(data_folder: str):
    """
    Scans the specified folder for supported file types and aggregates the text chunks.
    
    Args:
        data_folder (str): The absolute path to the folder containing your raw data files.
    
    Returns:
        list: A list of dictionaries (chunks), each containing chunk_id, modality, content, and metadata.
    """
    all_chunks = []

    # Loop through every file in the provided folder.
    for file_name in os.listdir(data_folder):
        # Construct the full path to the file.
        full_path = os.path.join(data_folder, file_name)

        # Skip directories (if you want to process subdirectories, add logic here).
        if os.path.isdir(full_path):
            continue

        # Extract the file extension in lowercase.
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

        # Append the new chunks from this file to our overall list.
        all_chunks.extend(new_chunks)

    return all_chunks

if __name__ == "__main__":
    # Determine the directory where this script is located.
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Since this script is in the 'scripts' folder, the 'data' folder is one level up.
    data_folder_path = os.path.join(current_dir, "..", "data")
    print(f"Using data folder: {data_folder_path}")

    # Attempt to ingest and chunk the data.
    try:
        chunks = ingest_and_chunk(data_folder_path)
        print(f"Extracted {len(chunks)} text chunks from '{data_folder_path}'.")
    except Exception as e:
        print(f"Error during data ingestion: {e}")
        exit(1)

    # Define the output file path in the project_alltext folder (one level up from 'scripts').
    output_file = os.path.join(current_dir, "..", "chunked_data.json")
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2)
        print(f"Saved chunked data to '{output_file}'.")
    except Exception as e:
        print(f"Error saving JSON file: {e}")
