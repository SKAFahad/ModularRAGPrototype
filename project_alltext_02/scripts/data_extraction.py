"""
data_extraction.py

Aggregator script for "project_alltext_02."

It:
1) Reads files from "project_alltext_02/data/"
2) Identifies file types by extension
3) Dispatches to parse_pdf, parse_docx, parse_spreadsheet, parse_image, parse_text
   (all of which you've already written in separate modules)
4) Receives a dictionary of { 'text': ..., 'tables': ..., 'images': ... } 
   from each parser
5) Saves results in "project_alltext_02/preposed_data/" with separate subfolders
   for "text/", "table/", and "image/"

Directory Structure Example:
project_alltext_02/
├── data/
│   ├── file1.pdf
│   ├── doc2.docx
│   ├── ...
├── preposed_data/
│   ├── text/
│   ├── table/
│   └── image/
├── scripts/
│   ├── parse_pdf.py
│   ├── parse_docx.py
│   ├── parse_spreadsheet.py
│   ├── parse_image.py
│   ├── parse_text.py
│   └── data_extraction.py  <-- THIS SCRIPT
└── ...

Usage:
    python data_extraction.py

Dependencies:
    - The parse_* scripts must be in the same folder or otherwise importable.
    - Make sure your parse_* modules actually return dicts with 'text', 'tables', 'images'.
    - This script expects a folder "data/" with files to process,
      and will create "preposed_data" for outputs.

Note:
 - This script only orchestrates extraction & saving, not chunking or embedding.
 - Adjust naming or saving logic to suit your needs.
"""

import os
import shutil
import uuid

# Import your parser modules
from parse_pdf import parse_pdf
from parse_docx import parse_docx
from parse_spreadsheet import parse_spreadsheet
from parse_image import parse_image
from parse_text import parse_text_file

def extract_data_from_file(file_path: str) -> dict:
    """
    Identifies file extension and calls the relevant parser.
    Each parser should return a dictionary:
        {
          "text":   <string or list of strings>,
          "tables": [list of DataFrames or table-like structures],
          "images": [some image objects or references],
        }
    """
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    if ext == ".pdf":
        return parse_pdf(file_path)
    elif ext == ".docx":
        return parse_docx(file_path)
    elif ext in [".xlsx", ".xls"]:
        return parse_spreadsheet(file_path)
    elif ext in [".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff"]:
        return parse_image(file_path)
    elif ext == ".txt":
        return parse_text_file(file_path)
    else:
        print(f"[data_extraction] Unsupported file type: {os.path.basename(file_path)}")
        return {
            "text": "",
            "tables": [],
            "images": []
        }

def save_text_content(text_content, file_name, text_out_dir):
    """
    Saves extracted text into a .txt file in the text_out_dir.
    text_content could be a string or list of strings, depending on your parsers.
    """
    # If your parser returns a list of paragraphs or blocks, combine them.
    if isinstance(text_content, list):
        text_content = "\n".join(str(t) for t in text_content)
    # If it's already a string, no change needed

    # Build the output file path
    base_name = os.path.splitext(file_name)[0]
    out_file = os.path.join(text_out_dir, f"{base_name}_text.txt")

    # Write text to file
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(text_content)

    print(f"[data_extraction] Text saved to: {out_file}")

def save_table_data(table_list, file_name, table_out_dir):
    """
    Saves each table (typically a Pandas DataFrame) as a CSV file.
    table_list is the 'tables' list from the parser result.
    """
    if not table_list:
        return

    base_name = os.path.splitext(file_name)[0]

    # Each table is often a Pandas DataFrame, but your parser
    # might store them differently. Adjust accordingly.
    for idx, df in enumerate(table_list, start=1):
        csv_file = os.path.join(table_out_dir, f"{base_name}_table_{idx}.csv")
        try:
            df.to_csv(csv_file, index=False, header=False)
            print(f"[data_extraction] Table {idx} saved to: {csv_file}")
        except Exception as e:
            print(f"[data_extraction] Could not save table {idx} from {file_name}: {e}")

def save_image_data(image_list, file_name, image_out_dir):
    """
    Saves each extracted image in the image_out_dir.
    The type of 'images' can vary by parser:
      - PDF parser may return list of PyMuPDF Pixmap objects
      - docx parser might return base64 strings
      - spreadsheet parser might return references or PIL image objects
      - parse_image might be simpler
    So you have to handle each case accordingly.
    """
    if not image_list:
        return

    base_name = os.path.splitext(file_name)[0]

    # Example: If parse_pdf returns fitz.Pixmap or parse_docx returns base64, etc.
    # We'll do a minimal demonstration here. Adapt to your actual parse logic.
    count = 1
    for img_obj in image_list:
        # We'll create a filename. If there's no direct method, we fallback to .png
        out_img_path = os.path.join(image_out_dir, f"{base_name}_img_{count}.png")

        try:
            # Example case: if parse_pdf returns a fitz.Pixmap
            if hasattr(img_obj, "save"):
                img_obj.save(out_img_path)
                print(f"[data_extraction] Image {count} saved as: {out_img_path}")
            # Another example: if parse_docx returns { 'filename':..., 'b64':... }
            elif isinstance(img_obj, dict) and "b64" in img_obj:
                import base64
                b64data = img_obj["b64"]
                with open(out_img_path, "wb") as f:
                    f.write(base64.b64decode(b64data))
                print(f"[data_extraction] Base64 image {count} saved as: {out_img_path}")
            # If parse_image might just store a path, we could copy or rename
            elif isinstance(img_obj, dict) and "path" in img_obj:
                original_path = img_obj["path"]
                shutil.copyfile(original_path, out_img_path)
                print(f"[data_extraction] Image path {count} copied to: {out_img_path}")
            else:
                # If we get here, we might not know how to handle the image data
                print(f"[data_extraction] Unhandled image type for {file_name}, index {count}.")
        except Exception as e:
            print(f"[data_extraction] Could not save image {count} from {file_name}: {e}")

        count += 1

def main():
    """
    Main aggregator function:
     1) Loops over the "data/" folder in "project_alltext_02/"
     2) For each file, calls extract_data_from_file
     3) Saves text, tables, images into "preposed_data/text", "preposed_data/table", "preposed_data/image"
    """
    # Path setup
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_folder = os.path.join(project_root, "data")
    preposed_folder = os.path.join(project_root, "preposed_data")

    # Create subfolders if they don't exist
    text_out_dir = os.path.join(preposed_folder, "text")
    table_out_dir = os.path.join(preposed_folder, "table")
    image_out_dir = os.path.join(preposed_folder, "image")

    for d in [text_out_dir, table_out_dir, image_out_dir]:
        os.makedirs(d, exist_ok=True)

    if not os.path.isdir(data_folder):
        print(f"[data_extraction] '{data_folder}' does not exist or is not a directory.")
        return

    # Process each file in data_folder
    for file_name in os.listdir(data_folder):
        full_path = os.path.join(data_folder, file_name)
        if not os.path.isfile(full_path):
            continue

        print(f"[data_extraction] Processing file: {file_name}")

        # 1) Parse the file
        parse_result = extract_data_from_file(full_path)
        # parse_result expected to be: { 'text':..., 'tables':..., 'images':... }

        # 2) Save text
        text_content = parse_result.get("text", "")
        if text_content:
            save_text_content(text_content, file_name, text_out_dir)

        # 3) Save tables
        table_content = parse_result.get("tables", [])
        if table_content:
            save_table_data(table_content, file_name, table_out_dir)

        # 4) Save images
        image_content = parse_result.get("images", [])
        if image_content:
            save_image_data(image_content, file_name, image_out_dir)

    print("[data_extraction] Finished extracting data into 'preposed_data' folder.")

if __name__ == "__main__":
    main()
