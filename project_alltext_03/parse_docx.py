"""
parse_docx.py

This module extracts textual content (and optionally table data) from Microsoft Word
(.docx) files using the docx2python library. It returns the extracted text in a
consistent format that can later be stored or chunked as needed.

Guiding Principles:
1. Keep it open-source and offline: no external API calls, everything runs locally.
2. Build an easily maintainable, well-commented codebase for future onboarding.
3. Make the function parse_docx(file_path) modular—only responsible for extraction.
   Subsequent steps (chunking, embedding, etc.) belong to separate modules.
4. Store results in a structured dictionary, e.g.:
    {
      "text": <string of combined text content>,
      "tables": <list of 2D table data if relevant>,
      "images": <list of placeholders or references if docx2python extracted images>,
      "metadata": {
         "file_name": ...
         "num_sections": ...
         ...
      }
    }

Note:
- docx2python primarily focuses on text extraction from docx.
- If images are embedded, docx2python versions before 2.0 used to handle
  extract_image=True, but newer versions have dropped that feature. We'll
  just return an empty "images" list by default or mention that docx2python
  does not fully extract images.
- For advanced image extraction, you might have to use python-docx or a direct
  unzip approach.

Usage:
    from parse_docx import parse_docx

    result = parse_docx("path/to/mydocument.docx")
    # result is a dictionary with keys: "text", "tables", "images", "metadata"
"""

import os
import sys
from docx2python import docx2python


def parse_docx(file_path: str) -> dict:
    """
    Parse a .docx file to extract all textual content (and table data),
    returning a structured dictionary.

    :param file_path: Path to the .docx file on disk
    :type file_path: str

    :return: A dictionary with keys:
      - "text": A single string containing the combined text from the docx.
      - "tables": A list of table data (each table is a nested list of rows/cells).
      - "images": Currently an empty list, as docx2python no longer supports direct image extraction.
      - "metadata": Additional info about the docx, e.g. file name, etc.
    :rtype: dict

    Example structure returned:
    {
      "text": "All paragraphs, headings, footers, etc. as a single string",
      "tables": [
          [
              ["Row1-Col1", "Row1-Col2"],
              ["Row2-Col1", "Row2-Col2"]
          ],
          ...
      ],
      "images": [],
      "metadata": {
          "file_name": "mydocument.docx",
          "num_sections": 1,
          ...
      }
    }

    Detailed Explanation:
    1. docx2python(file_path) returns a DocxContent object. This object
       contains multiple attributes including 'body', 'footnotes', etc.
    2. The 'body' attribute is typically a list of sections, each containing
       lists for paragraphs and runs.
    3. The docx2python object also provides .body_tables if tables exist.
    4. We combine all paragraphs into one big text block. For advanced usage,
       you might prefer returning each paragraph separately (or implementing
       chunking separately). Here, we join everything for simplicity.
    5. We also gather any table data from .body_tables, storing them in "tables".
       Each table is typically a nested list of rows/cells from docx2python.
    6. docx2python older versions allowed for 'extract_image=True' param to gather
       images, but now it’s dropped. We set "images" to an empty list by default.

    Requirements:
      pip install docx2python
    """

    # Validate that the file exists
    if not os.path.isfile(file_path):
        raise FileNotFoundError(
            f"parse_docx: The file '{file_path}' does not exist or is not accessible."
        )

    # Attempt to parse the docx file
    # docx2python returns an object with attributes:
    # - body (list of sections, each a nested list for paragraphs/runs)
    # - footnotes, endnotes, etc.
    # - body_tables (list of tables, each a nested list of rows/cells)
    try:
        doc_result = docx2python(file_path)
    except Exception as e:
        # If docx2python fails (corrupt .docx, etc.), raise an error for better debugging
        raise RuntimeError(f"parse_docx: Failed to parse '{file_path}': {e}")

    # docx2python splits the doc into sections > paragraphs > runs (possibly nested lists).
    # We'll flatten these paragraphs into a single string. For large documents, you
    # might prefer returning a list of paragraphs. Here we do one big text for simplicity.
    extracted_text = extract_all_text(doc_result.body)

    # docx2python can also capture footnotes and endnotes if present.
    # We might optionally extract them to keep more context. We'll just ignore them
    # for brevity, but you can do the same approach:
    # footnote_text = extract_all_text(doc_result.footnotes)

    # docx2python also populates 'body_tables' if there are tables in the doc
    # body_tables is typically a list of tables, each table is a list of rows,
    # each row is a list of cells. Each cell is a list of paragraphs/runs or sub-lists.
    # We'll keep them in raw form or flatten them as needed.
    doc_tables = doc_result.body_tables  # This might be an empty list if no tables.

    # Convert doc_tables into a more uniform python list-of-lists
    # doc_result.body_tables is already a nested structure:
    # [ table_1, table_2, ...]
    # table_1 => [ row_1, row_2, ...]
    # row_1 => [ cell_1, cell_2, ...]
    # Each cell might be further subdivided. We'll flatten cell content for each table.
    flattened_tables = []
    for table_idx, table in enumerate(doc_tables, start=1):
        # table is a list of rows
        clean_table = []
        for row in table:
            # row is a list of cells
            # each cell might be a list of paragraphs/runs
            # we'll flatten them into a single string
            clean_row = []
            for cell in row:
                # Flatten the nested runs within the cell
                cell_text = flatten_runs(cell)
                cell_text_str = " ".join(cell_text).strip()
                clean_row.append(cell_text_str)
            clean_table.append(clean_row)
        flattened_tables.append(clean_table)

    # docx2python no longer extracts images. We'll return an empty list for images.
    images_list = []

    # Build a metadata dict for extra info
    metadata = {
        "file_name": os.path.basename(file_path),
        # docx2python organizes doc in 'body', 'footnotes', etc. We'll count body sections
        "num_sections": len(doc_result.body) if doc_result.body else 0,
        # You might add more fields if needed (like doc_result.document_info if available)
    }

    # Finally, build the main result dictionary
    parse_result = {
        "text": extracted_text,
        "tables": flattened_tables,
        "images": images_list,
        "metadata": metadata
    }

    return parse_result


def extract_all_text(body_data):
    """
    Recursively traverse docx2python's 'body' structure to flatten all paragraphs/runs
    into a single large text string. Each 'section' is a list, each 'paragraph' is a list
    of runs, which might be further nested.

    :param body_data: docx2python body data (list of sections)
    :return: A single string containing all text from the docx 'body'.
    """
    paragraphs = []

    # docx2python organizes 'body_data' as:
    # [
    #   [   # section 1
    #       [   # paragraph 1
    #           [run1, run2]   # runs
    #       ],
    #       [   # paragraph 2
    #           [run1, run2]
    #       ]
    #   ],
    #   [   # section 2
    #       ...
    #   ],
    #   ...
    # ]
    for section in body_data:
        # 'section' is a list of paragraphs
        for paragraph_runs in section:
            # flatten this paragraph (which may contain multiple runs or nested lists)
            # into a single array of strings
            run_texts = flatten_runs(paragraph_runs)
            # join them with a space or newline
            paragraph_text = " ".join(run_texts).strip()
            if paragraph_text:
                paragraphs.append(paragraph_text)

    # Join paragraphs with double newline for readability
    combined_text = "\n\n".join(paragraphs)
    return combined_text


def flatten_runs(runs):
    """
    docx2python paragraphs often have a nested list structure: 
    runs -> lists of strings or further lists. 
    This helper flattens them into a single list of strings.

    :param runs: A nested list or an item that might be string or list 
    :return: A list of strings that are all the textual runs.
    """
    flattened = []
    if isinstance(runs, list):
        for item in runs:
            if isinstance(item, list):
                # Recursively flatten
                flattened.extend(flatten_runs(item))
            else:
                # It's a string
                flattened.append(item)
    else:
        # If runs is itself a string
        flattened.append(runs)
    return flattened


if __name__ == "__main__":
    # Simple test usage (manual run):
    # Provide a .docx file path as an argument:
    #   python parse_docx.py path/to/test.docx

    if len(sys.argv) < 2:
        print("Usage: python parse_docx.py <docx_file>")
        sys.exit(1)

    docx_file_path = sys.argv[1]

    try:
        result = parse_docx(docx_file_path)
        print(f"Successfully parsed '{docx_file_path}'.")
        print("--- Extracted Text (first 500 chars) ---")
        print(result["text"][:500], "..." if len(result["text"]) > 500 else "")
        print("----------------------------------------\n")

        if result["tables"]:
            print(f"Found {len(result['tables'])} table(s).")
            # Print first table
            print("First table data:")
            for row in result["tables"][0][:5]:  # up to 5 rows 
                print(row)
        else:
            print("No tables found.")

        # images is always empty in this approach
        print(f"Images extracted: {len(result['images'])}")
        print("Metadata:", result["metadata"])
    except Exception as e:
        print(f"Error while parsing DOCX: {e}")
