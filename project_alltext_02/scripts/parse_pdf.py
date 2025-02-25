"""
parse_pdf.py

This parser uses four main tools to handle different aspects of PDF extraction:

1) PyMuPDF (fitz) for:
   - Raw text extraction page by page
   - Collecting embedded images (stored as Pixmap objects)
   - Retrieving PDF metadata

2) Unstructured (partition_pdf) for:
   - Higher-level, semantic segmentation of the PDF into blocks
     (e.g., paragraphs, headings, etc.)

3) Camelot for:
   - Extracting tables by analyzing the PDF's line structures
   - Typically more accurate for PDFs that have visible lines

4) Tabula-py for:
   - A second approach to table extraction via a Java-based PDF parsing library
   - Helps capture tables that Camelot might miss, or vice versa

After gathering raw text, segmented text, images, and tables, we merge them into
a consistent dictionary with keys:
    {
      "text":         (string) raw text concatenated across pages,
      "segmented_text": (list)  unstructured's content blocks,
      "tables":       (list)  each table as a Pandas DataFrame,
      "images":       (list)  pixmap objects or image references,
      "metadata":     (dict)  PDF-level metadata
    }

Dependencies:
    pip install pymupdf
    pip install unstructured
    pip install camelot-py[cv]
    pip install tabula-py
    pip install pandas

You will also need a Java runtime installed for tabula-py to work.

Usage Example:
    from parse_pdf import parse_pdf

    result = parse_pdf("somefile.pdf")
    # result now has "text", "segmented_text", "tables", "images", "metadata"

Note:
- "images" is a list of fitz.Pixmap objects. If you need to save them, you can
  do so by calling pix.save("somefile.png") in your aggregator or further pipeline steps.
- "segmented_text" is a list of unstructured's DocumentElement objects (like
  Paragraph, Title, etc.). You can convert them to strings if desired.

"""

import fitz            # PyMuPDF
import camelot
import tabula
import pandas as pd
from unstructured.partition.pdf import partition_pdf

def _extract_with_pymupdf(pdf_path):
    """
    Uses PyMuPDF to extract:
      - raw_text (string)
      - metadata (dict from PyMuPDF)
      - images (list of fitz.Pixmap)
    Returns: (raw_text, metadata, images)
    """
    # Open the PDF with fitz
    doc = fitz.open(pdf_path)

    raw_text = []
    metadata = doc.metadata  # PDF-level metadata
    images = []

    # Iterate through pages
    for page in doc:
        # Extract text in "text" mode (basic layout)
        page_text = page.get_text("text")
        # Add to the raw_text list
        if page_text:
            raw_text.append(page_text)

        # Extract images from this page
        image_list = page.get_images(full=True)
        for img in image_list:
            xref = img[0]  # the image reference
            pix = fitz.Pixmap(doc, xref)
            # Convert CMYK pixmaps to RGB if needed
            if pix.n - (1 if pix.alpha else 0) >= 4:
                pix = fitz.Pixmap(fitz.csRGB, pix)
            images.append(pix)

    # Combine all page texts into one big string
    combined_text = "\n".join(raw_text)

    return combined_text, metadata, images

def _extract_unstructured_content(pdf_path):
    """
    Uses Unstructured (partition_pdf) to produce higher-level text segmentation.
    Returns a list of document elements (e.g. Paragraph, Title, etc.).
    Each element can be converted to a string if needed.
    """
    # partition_pdf returns a list of "Element" objects with .type, .text, etc.
    content_blocks = partition_pdf(filename=pdf_path)
    return content_blocks

def _extract_tables(pdf_path):
    """
    Uses both Camelot and Tabula-py to extract tables.
    Returns a list of Pandas DataFrames. We combine the results of both
    in case one library misses some tables.
    """
    tables_camelot = []
    tables_tabula = []

    # Camelot extraction
    try:
        camelot_tables = camelot.read_pdf(pdf_path, pages="all")
        # Convert each Camelot Table object to a DataFrame
        tables_camelot = [t.df for t in camelot_tables]
    except Exception as e:
        print(f"[parse_pdf] Camelot error on '{pdf_path}': {e}")
    
    # Tabula extraction
    try:
        # multiple_tables=True -> returns a list of DataFrames
        tabula_tables = tabula.read_pdf(pdf_path, pages="all", multiple_tables=True)
        tables_tabula = tabula_tables if tabula_tables else []
    except Exception as e:
        print(f"[parse_pdf] Tabula error on '{pdf_path}': {e}")

    # Combine them (this could result in duplicates if both libraries parse the same table)
    combined_tables = tables_camelot + tables_tabula
    return combined_tables

def parse_pdf(pdf_path: str) -> dict:
    """
    Main entry point to parse a PDF using:
      1) PyMuPDF for raw text, metadata, images
      2) Unstructured for segmented text blocks
      3) Camelot + Tabula for tables

    Returns a dictionary with keys:
      {
        "text":           str  (all pages' raw text combined),
        "segmented_text": list (Unstructured's content blocks),
        "tables":         list (each entry is a Pandas DataFrame),
        "images":         list (fitz.Pixmap objects),
        "metadata":       dict (from PyMuPDF)
      }
    """
    # Step 1: PyMuPDF for raw text, metadata, images
    raw_text, metadata, images = _extract_with_pymupdf(pdf_path)

    # Step 2: Unstructured for higher-level text blocks
    segmented_content = []
    try:
        segmented_content = _extract_unstructured_content(pdf_path)
    except Exception as e:
        print(f"[parse_pdf] Unstructured error on '{pdf_path}': {e}")

    # Step 3: Tables from Camelot + Tabula
    table_dfs = _extract_tables(pdf_path)

    # Construct the final dictionary
    result = {
        "text": raw_text,
        "segmented_text": segmented_content,   # list of blocks from Unstructured
        "tables": table_dfs,                   # list of DataFrames
        "images": images,                      # list of fitz.Pixmap
        "metadata": metadata                   # PDF-level metadata from PyMuPDF
    }

    return result
