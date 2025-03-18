"""
data_extraction.py

Extracts data from various file types (PDF, DOCX, XLSX, CSV, TXT, images, etc.)
using the parse_* scripts. Then **writes** the aggregated parse results to a JSON file
called "parse_results.json" (or any name you prefer, but run_pipeline.py references it).

This fixes the error "No such file or directory: 'parse_results.json'" in data_chunking.py,
because we now actually produce that file.

Guiding Principles:
-------------------
1) Keep local usage: no external cloud calls.
2) Write parse results into a single JSON file: parse_results.json, with a 
   top-level list structure:
     [
       {
         "file_name": <filename>,
         "parse_data": {
           "text": <str>,
           "tables": [...],
           "images": [...],
           "metadata": {...}
         }
       },
       ...
     ]
3) If any parse fails, we skip that file but log an error. The rest proceed.

Usage:
------
  python data_extraction.py
  # By default, it reads from the "data/" folder, calls parse_* scripts,
  # and writes "parse_results.json"

Implementation Steps:
---------------------
1) The main function `data_extraction` scans a 'data/' folder for files.
2) For each file, determine extension, dispatch to parse_* scripts.
3) Store the parse result in a Python list with {file_name, parse_data}.
4) After processing all files, dump that list to parse_results.json.
5) If run as a script, do the same. 
   So "run_script(data_extraction_py)" in run_pipeline will produce parse_results.json.
"""

import os
import json

# your parse_* imports
from parse_pdf import parse_pdf
from parse_docx import parse_docx
from parse_spreadsheet import parse_spreadsheet
from parse_text import parse_text_file
from parse_image import parse_image


def data_extraction(data_folder="data", output_json="parse_results.json"):
    """
    Orchestrates the extraction of data from various files in 'data_folder' 
    and writes them out to 'output_json'.

    :param data_folder: The folder containing input files to parse.
    :type data_folder: str
    :param output_json: The JSON file where parse results will be written.
    :type output_json: str

    :return: None, but writes parse results to output_json
    """

    # We expect a structure like:
    # parse_results = [
    #   {
    #       "file_name": "<filename>",
    #       "parse_data": {
    #           "text": <str>,
    #           "tables": [...],
    #           "images": [...],
    #           "metadata": {...}
    #       }
    #   },
    #   ...
    # ]

    parse_results = []

    if not os.path.isdir(data_folder):
        print(f"[data_extraction] '{data_folder}' does not exist or is not a directory.")
        return

    # scan the folder
    for file_name in os.listdir(data_folder):
        full_path = os.path.join(data_folder, file_name)
        if not os.path.isfile(full_path):
            # skip subfolders
            continue

        print(f"[data_extraction] Processing: {file_name}")
        # figure out extension
        ext = os.path.splitext(file_name)[1].lower()

        parse_result_entry = None

        try:
            if ext == ".pdf":
                parse_result = parse_pdf(full_path)
            elif ext == ".docx":
                parse_result = parse_docx(full_path)
            elif ext in [".xlsx", ".xls", ".csv"]:
                parse_result = parse_spreadsheet(full_path)
            elif ext == ".txt":
                parse_result = parse_text_file(full_path)
            elif ext in [".png", ".jpg", ".jpeg", ".gif", ".tiff"]:
                parse_result = parse_image(full_path)
            else:
                # skip unsupported
                print(f"[data_extraction] Skipping unsupported file type: {file_name}")
                continue

            # parse_result is something like:
            # {
            #   "text":   <str>,
            #   "tables": [list of tables],
            #   "images": [],
            #   "metadata": {...}
            # }

            # build the entry
            parse_result_entry = {
                "file_name": file_name,
                "parse_data": parse_result
            }

        except Exception as e:
            print(f"[data_extraction] Error parsing {file_name}: {e}")
            continue

        if parse_result_entry:
            parse_results.append(parse_result_entry)

    # Now we write parse_results to output_json
    try:
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(parse_results, f, indent=2)
        print(f"[data_extraction] Wrote parse results to '{output_json}' with {len(parse_results)} file entries.")
    except Exception as e:
        print(f"[data_extraction] Could not write to '{output_json}': {e}")


def main():
    """
    If run directly: python data_extraction.py
    We'll parse from 'data/' folder and write parse_results.json
    """
    data_folder = "data"
    output_json = "parse_results.json"

    data_extraction(data_folder=data_folder, output_json=output_json)


if __name__ == "__main__":
    main()
