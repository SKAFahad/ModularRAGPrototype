"""
parse_pdf.py

Parses PDFs to produce:
  - Text chunks (each paragraph)
  - Table chunks (via Camelot/TAPAS flow)
  - Image chunks (actual files extracted from PDF)

We assume you're already using pdfplumber for text and page.images metadata.
We also demonstrate how to extract the embedded image bytes and save them.

Requires:
    pip install pdfplumber camelot-py
    # plus any dependencies (pillow, etc.)
"""

import os
import pdfplumber
import camelot
import uuid  # to create unique filenames if needed

def process_pdf_file(file_path: str):
    """
    Uses pdfplumber + Camelot to separate PDF into text, tables, and images.
    This version extracts each embedded PDF image to an actual .png file 
    so it can later be embedded with CLIP.

    Args:
        file_path (str): Path to the PDF file.

    Returns:
        chunks (list): A list of chunk dicts. Each dict has:
            {
                "chunk_id": str,
                "modality": "text" | "table" | "image",
                "content": str,  # text or file path
                "metadata": {...}
            }
    """
    chunks = []

    # We'll store extracted images in 'project/extracted_pdf_images'
    # (Assuming your project structure has such a folder.)
    # Make sure it exists:
    extracted_img_folder = "project/extracted_pdf_images"
    if not os.path.exists(extracted_img_folder):
        os.makedirs(extracted_img_folder)

    try:
        # 1) Extract text & images with pdfplumber
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                # Extract text
                page_text = page.extract_text()
                if page_text:
                    paragraphs = page_text.split('\n\n')
                    for p_i, paragraph in enumerate(paragraphs):
                        chunk_id = f"{os.path.basename(file_path)}_p{page_num}_par{p_i}"
                        chunks.append({
                            "chunk_id": chunk_id,
                            "modality": "text",
                            "content": paragraph.strip(),
                            "metadata": {
                                "file_name": os.path.basename(file_path),
                                "page_number": page_num
                            }
                        })

                # Extract images
                # page.images is a list of dictionaries describing each image object
                for img_i, pdf_img in enumerate(page.images):
                    # pdf_img might look like {"xref": 11, "width":..., "height":..., "top":..., ...}
                    xref = pdf_img.get("xref")
                    if xref is None:
                        # Some PDFs might not have xref references for images
                        continue

                    # Extract the raw image bytes using xref
                    img_dict = page.extract_image(xref)
                    # Example structure of img_dict:
                    # {
                    #   'ext': 'jpeg',
                    #   'width': 200,
                    #   'height': 100,
                    #   'image': <bytes of the extracted image>
                    # }
                    if not img_dict:
                        continue

                    img_ext = img_dict.get("ext", "png")  # default to png if no extension
                    img_bytes = img_dict.get("image", b"")

                    # Build a unique output filename
                    # e.g. 'myPDF_p0_img0.png'
                    base_filename = os.path.splitext(os.path.basename(file_path))[0]
                    out_filename = f"{base_filename}_p{page_num}_img{img_i}.{img_ext}"
                    out_path = os.path.join(extracted_img_folder, out_filename)

                    # Write bytes to disk
                    with open(out_path, "wb") as out_file:
                        out_file.write(img_bytes)

                    # Now create a chunk referencing the real image path
                    chunk_id = f"{os.path.basename(file_path)}_p{page_num}_img{img_i}"
                    chunks.append({
                        "chunk_id": chunk_id,
                        "modality": "image",
                        "content": out_path,  # The actual file path
                        "metadata": {
                            "file_name": os.path.basename(file_path),
                            "page_number": page_num,
                            "bbox": {
                                "x0": pdf_img["x0"],
                                "top": pdf_img["top"],
                                "x1": pdf_img["x1"],
                                "bottom": pdf_img["bottom"]
                            }
                        }
                    })

        # 2) Table extraction with Camelot
        #    If your PDF is not purely scanned, Camelot can parse textual tables.
        tables = camelot.read_pdf(file_path, pages="all")
        for t_i, table in enumerate(tables):
            df = table.df
            for r_i, row in df.iterrows():
                row_str = ", ".join([str(x) for x in row])
                chunk_id = f"{os.path.basename(file_path)}_table{t_i}_row{r_i}"
                chunks.append({
                    "chunk_id": chunk_id,
                    "modality": "table",
                    "content": row_str,
                    "metadata": {
                        "file_name": os.path.basename(file_path),
                        "table_index": t_i,
                        "row_index": r_i
                    }
                })

    except Exception as e:
        print(f"Error processing PDF file {file_path}: {e}")

    return chunks
