#!/usr/bin/env python3
"""
parse_pdf.py
------------
Parses PDF files to extract textual content. It uses pdfplumber to read text from each page.
Additionally, if tables are detected, it converts them into a string representation.
The output is a list of chunks, where each chunk is a dictionary containing:
  - chunk_id: Unique ID (constructed from file name, page, and paragraph index).
  - modality: "text" (since we convert everything to text).
  - content: Extracted text.
  - metadata: Additional info like file name and page number.
"""

import os
import pdfplumber
import camelot  # For table extraction (if needed)

def process_pdf_file(file_path: str):
    """
    Extracts text from a PDF file and returns a list of text chunks.

    :param file_path: Path to the PDF file.
    :return: List of dictionaries (chunks) with extracted text and metadata.
    """
    chunks = []

    try:
        # Open the PDF using pdfplumber.
        with pdfplumber.open(file_path) as pdf:
            # Iterate over each page in the PDF.
            for page_num, page in enumerate(pdf.pages):
                # Extract raw text from the page.
                page_text = page.extract_text()
                if page_text:
                    # Split text into paragraphs based on double newlines.
                    paragraphs = page_text.split('\n\n')
                    for p_i, paragraph in enumerate(paragraphs):
                        paragraph = paragraph.strip()
                        if paragraph:
                            # Create a unique chunk ID using file name, page number, and paragraph index.
                            chunk_id = f"{os.path.basename(file_path)}_p{page_num}_par{p_i}"
                            chunks.append({
                                "chunk_id": chunk_id,
                                "modality": "text",
                                "content": paragraph,
                                "metadata": {
                                    "file_name": os.path.basename(file_path),
                                    "page_number": page_num
                                }
                            })

                # Optionally, process tables using Camelot (if you wish to include table data as text).
                # Note: This example converts table rows to comma-separated strings.
                try:
                    tables = camelot.read_pdf(file_path, pages=str(page_num + 1))
                    for t_i, table in enumerate(tables):
                        df = table.df
                        # Process each row in the table.
                        for r_i, row in df.iterrows():
                            row_text = ", ".join(row.tolist())
                            chunk_id = f"{os.path.basename(file_path)}_p{page_num}_table{t_i}_row{r_i}"
                            chunks.append({
                                "chunk_id": chunk_id,
                                "modality": "text",
                                "content": row_text,
                                "metadata": {
                                    "file_name": os.path.basename(file_path),
                                    "page_number": page_num,
                                    "table_index": t_i,
                                    "row_index": r_i
                                }
                            })
                except Exception as e:
                    print(f"Table extraction skipped for page {page_num} in {file_path}: {e}")

    except Exception as e:
        print(f"Error processing PDF file {file_path}: {e}")

    return chunks
