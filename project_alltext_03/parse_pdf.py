"""
parse_pdf.py

This module extracts text, tables, and (optionally) embedded images from PDF files.
It uses:
  - PyMuPDF (fitz) for high-speed text extraction and metadata.
  - Camelot (or Tabula) for table detection/extraction if the PDF has tabular data.
  - (Optional) docTR or Tesseract for OCR on scanned pages or embedded images, if desired.

We return a structured dictionary:
{
  "text":   <string of combined textual content>,
  "tables": <list of 2D table data or CSV-like strings>,
  "images": <list of references or placeholders for embedded images>,
  "metadata": {
      "file_name": <pdf file name>,
      "page_count": <number of pages in the PDF>,
      ... (optionally more: creation date, author, etc.)
  }
}

Guiding Principles (as per discussion):
1. Offline, open-source solution: Use PyMuPDF, Camelot, etc., fully local.
2. Modular design: parse_pdf.py focuses solely on PDF ingestion.
3. Detailed & maintainable: Thorough comments for future devs to extend or debug.
4. Common return structure: "text", "tables", "images", "metadata" for uniform pipeline usage.

Notes on Tools:
- PyMuPDF (fitz) is extremely fast for text extraction from PDFs.
- Camelot or Tabula can parse table structures from PDFs that have lines or textual alignments:
   - Camelot: pip install camelot-py[cv]
   - Tabula: pip install tabula-py (requires Java installed).
- For scanned PDFs or images within PDFs, you might integrate OCR (docTR or Tesseract).
  This script includes a stub for image extraction references but doesn't do OCR automatically.

Usage Example:
    from parse_pdf import parse_pdf
    result = parse_pdf("myfile.pdf")
    # 'result' is a dict with keys "text", "tables", "images", "metadata".
"""

import os
import sys

# PyMuPDF for text extraction and PDF metadata
import fitz

# Optional: Camelot for table extraction
# If you want to also support Tabula, see the commented approach below
try:
    import camelot
    HAS_CAMELOT = True
except ImportError:
    HAS_CAMELOT = False
    # If Camelot isn't installed, you can fallback to Tabula or skip table parsing.

# If you wanted Tabula, you could do:
# try:
#     import tabula
#     HAS_TABULA = True
# except ImportError:
#     HAS_TABULA = False

def parse_pdf(file_path: str) -> dict:
    """
    Parse text, tables, and embedded images from a PDF file. Returns a structured dictionary.

    :param file_path: The full path to the PDF file on disk
    :type file_path: str

    :return: A dictionary with keys:
        "text": Combined textual content from all pages
        "tables": List of extracted tables; each table typically a list of rows (list of strings)
        "images": Placeholder list referencing embedded images (or empty if not extracted)
        "metadata": Additional info about the PDF (filename, page_count, etc.)
    :rtype: dict

    Workflow:
      1) Validate file existence
      2) Open PDF with PyMuPDF (fitz) and read:
         - all textual content from each page
         - metadata like creation date, author, etc.
         - embedded images references (if you plan to do OCR or store them)
      3) (Optional) Use Camelot to extract tables from each page if it has line-based tables
      4) Combine textual content into a single string
      5) Return the dictionary with 'text', 'tables', 'images', and 'metadata'

    Dependencies:
      pip install pymupdf
      pip install camelot-py[cv]  # (if using Camelot)
      # or pip install tabula-py  # (if using Tabula)
      # docTR or Tesseract are only needed if you want to OCR scanned PDFs or embedded images.

    Limitations:
      - If the PDF is scanned or if text extraction is incomplete, you may need OCR integration.
        This script currently doesn't do that automatically. See 'extract_embedded_images'
        for a stub to handle image xrefs if needed.
      - Camelot works best on PDFs with clearly delineated table lines. If your PDF doesn't have
        such lines or is a complex layout, consider Tabula or a fallback method.
    """

    # Step 1) Confirm file exists
    if not os.path.isfile(file_path):
        raise FileNotFoundError(
            f"parse_pdf: The file '{file_path}' does not exist or is not accessible."
        )

    # Step 2) Open PDF with PyMuPDF
    try:
        doc = fitz.open(file_path)
    except Exception as e:
        raise RuntimeError(f"parse_pdf: Failed to open PDF '{file_path}': {e}")

    # Extract PDF-level metadata from fitz (PyMuPDF)
    pdf_metadata = doc.metadata  # a dict with keys like 'title', 'author', etc.
    page_count = doc.page_count

    # We'll store each page's text content in a list, then join them.
    text_pages = []

    # Also, gather references to embedded images if you plan to do OCR on them later
    embedded_images = []

    for page_index in range(page_count):
        # get_text("text") returns the textual content with a basic layout
        # You might also consider "blocks" or "words" modes for advanced usage.
        page = doc.load_page(page_index)
        page_text = page.get_text("text")

        # If you'd like to gather references to embedded images, here's how:
        # images_info = page.get_images(full=True)
        # for img_info in images_info:
        #     # Each entry is (xref, smask, width, height, bpc, colorspace, ...)
        #     xref = img_info[0]
        #     embedded_images.append({
        #         "page_index": page_index,
        #         "xref": xref,
        #         # you could store width/height, etc.
        #     })
        # For simplicity, we'll skip storing them; you can uncomment above if needed.

        if page_text.strip():
            text_pages.append(page_text.strip())

    # Join all page texts with double newline for clarity
    combined_text = "\n\n".join(text_pages)

    # Step 3) Use Camelot or Tabula to extract tables
    # If you want table data, you can run Camelot if installed:
    extracted_tables = []
    if HAS_CAMELOT:
        try:
            # read_pdf returns a camelot.core.TableList
            # "pages=all" means process every page
            # If your PDF has no line-based tables, Camelot might return empty
            tables = camelot.read_pdf(file_path, pages="all")
            # Convert each Table object to a list of rows
            for t in tables:
                df = t.df  # a pandas DataFrame
                # Convert DataFrame rows to a list of lists (like CSV)
                table_as_list = df.values.tolist()
                extracted_tables.append(table_as_list)
        except Exception as e:
            # If Camelot fails (maybe no tables or PDF is scanned?), we skip table extraction
            print(f"parse_pdf: Camelot table extraction failed: {e}")
    else:
        # If Camelot not installed, you could optionally fallback to Tabula or skip
        # For example, with Tabula:
        # if HAS_TABULA:
        #     tabula_tables = tabula.read_pdf(file_path, pages="all", multiple_tables=True)
        #     # tabula_tables is a list of DataFrames
        #     for df in tabula_tables:
        #         table_as_list = df.values.tolist()
        #         extracted_tables.append(table_as_list)
        pass

    # Step 4) Build a consistent "images" list. For now, we skip actual image extraction and OCR.
    # We'll just keep it empty or store references if you uncomment the get_images logic.
    images_list = []  # e.g., you could do embedded_images if you want references

    # Step 5) Build a metadata dict with relevant info
    # We'll store the original filename, page_count, plus any doc metadata you find relevant.
    # doc.metadata might have keys: 'title', 'author', 'creationDate', 'modDate', 'producer', etc.
    # We'll add them if they're not None.
    metadata = {
        "file_name": os.path.basename(file_path),
        "page_count": page_count
    }
    # You can store more from pdf_metadata if desired:
    for key in ["title", "author", "creationDate", "modDate", "producer"]:
        val = pdf_metadata.get(key)
        if val:
            metadata[key] = val

    # Construct final result dictionary
    parse_result = {
        "text": combined_text,
        "tables": extracted_tables,
        "images": images_list,
        "metadata": metadata
    }

    # Close the doc to free resources
    doc.close()

    return parse_result


if __name__ == "__main__":
    """
    Testing parse_pdf.py directly from the command line:
    Usage:
        python parse_pdf.py <pdf_file>
    It will print out the first 500 chars of extracted text, the number of tables, etc.

    Example:
        python parse_pdf.py sample.pdf
    """
    if len(sys.argv) < 2:
        print("Usage: python parse_pdf.py <pdf_file>")
        sys.exit(1)

    pdf_file_path = sys.argv[1]
    try:
        result = parse_pdf(pdf_file_path)
        print(f"Successfully parsed '{pdf_file_path}'.")
        print("--- Extracted Text (first 500 chars) ---")
        if result["text"]:
            snippet = result["text"][:500]
            more_text = "..." if len(result["text"]) > 500 else ""
            print(snippet + more_text)
        else:
            print("[No textual content found or PDF might be scanned.]")

        # Print table info
        num_tables = len(result["tables"])
        print(f"\nFound {num_tables} table(s) via Camelot:")
        if num_tables > 0:
            print("First table (up to 5 rows):")
            for row in result["tables"][0][:5]:
                print(row)

        # images likely empty unless you handle them above
        print("\nImages extracted:", len(result["images"]))
        print("Metadata:", result["metadata"])
    except Exception as e:
        print(f"Error processing PDF '{pdf_file_path}': {e}")
