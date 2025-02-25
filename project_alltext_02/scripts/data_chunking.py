"""
data_chunking.py

Aggregates the chunking of text, images, and tables from the 'preposed_data' folder,
producing a single JSON structure (e.g. chunked_data.json) with all chunk dicts.

Directory structure assumed:
  project_alltext_02/
  ├── preposed_data/
  │   ├── text/
  │   │   ├── ... .txt
  │   ├── image/
  │   │   ├── ... .png, .jpg, etc.
  │   └── table/
  │       ├── ... .csv
  ├── scripts/
  │   ├── data_chunking.py  <-- this script
  │   ├── chunk_text.py
  │   ├── chunk_image.py
  │   ├── chunk_table.py
  │   ...
  └── ...

For each file in these subfolders, we call:
 - chunk_text_file(...)  from chunk_text.py
 - chunk_image_file(...) from chunk_image.py
 - chunk_table_csv(...)  from chunk_table.py

We collect the resulting chunk dicts, grouped by file_name, then write them to a JSON file.

Final structure in chunked_data.json:
{
  "files": [
    {
      "file_name": "example.txt",
      "chunks": [
        {
          "chunk_id": "...",
          "modality": "text",
          "content": "...",
          "metadata": {...},
          "textual_modality": "wrapped_paragraph"
        },
        ...
      ]
    },
    {
      "file_name": "some_image.png",
      "chunks": [
        {
          "chunk_id": "...",
          "modality": "image",
          "content": "...",
          "metadata": {...},
          "textual_modality": "ocr_extracted"
        }
      ]
    },
    ...
  ]
}
"""

import os
import json

from chunk_text import chunk_text_file
from chunk_image import chunk_image_file
from chunk_table import chunk_table_csv

def chunk_data_folder(preposed_folder: str, output_json: str):
    """
    1) Looks for text/*.txt, image/*.(png|jpg|...), table/*.csv in preposed_folder.
    2) For each file, calls the relevant chunk_* function.
    3) Groups chunk dicts by file_name in the final JSON structure.
    4) Writes results to 'output_json'.
    """

    results = {"files": []}

    text_dir = os.path.join(preposed_folder, "text")
    image_dir = os.path.join(preposed_folder, "image")
    table_dir = os.path.join(preposed_folder, "table")

    # ----- Process TEXT files -----
    if os.path.isdir(text_dir):
        for file_name in os.listdir(text_dir):
            if file_name.lower().endswith(".txt"):
                full_path = os.path.join(text_dir, file_name)
                chunk_list = chunk_text_file(full_path, width=80)  # returns a list of chunk dicts
                results["files"].append({
                    "file_name": file_name,
                    "chunks": chunk_list
                })

    # ----- Process IMAGE files -----
    if os.path.isdir(image_dir):
        for file_name in os.listdir(image_dir):
            ext = os.path.splitext(file_name)[1].lower()
            if ext in [".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff"]:
                full_path = os.path.join(image_dir, file_name)
                chunk_list = chunk_image_file(full_path, width=80)
                results["files"].append({
                    "file_name": file_name,
                    "chunks": chunk_list
                })

    # ----- Process TABLE files (CSV) -----
    if os.path.isdir(table_dir):
        for file_name in os.listdir(table_dir):
            ext = os.path.splitext(file_name)[1].lower()
            if ext == ".csv":
                full_path = os.path.join(table_dir, file_name)
                chunk_list = chunk_table_csv(full_path)  # list of chunk dicts
                results["files"].append({
                    "file_name": file_name,
                    "chunks": chunk_list
                })

    # Write the aggregated results to JSON
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"[data_chunking] Created '{output_json}' with {len(results['files'])} file entries.")

def main():
    # Typically, your structure is: project_alltext_02/preposed_data/...
    project_root = os.path.dirname(os.path.abspath(__file__))
    # If data_chunking.py is in scripts/, go one level up
    root_parent = os.path.dirname(project_root)

    preposed_folder = os.path.join(root_parent, "preposed_data")
    output_json = os.path.join(root_parent, "chunked_data.json")

    chunk_data_folder(preposed_folder, output_json)

if __name__ == "__main__":
    main()
