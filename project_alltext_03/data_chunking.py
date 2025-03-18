"""
data_chunking.py

This module is responsible for transforming the parsed data (from data_extraction.py)
into smaller, more manageable "chunks" that can be embedded and stored in a graph or
vector database. Chunking ensures each segment of text, table row, or OCR-based content
is neither too large for an LLM context nor too small to lose context. The result is
a JSON-like structure that each downstream step (embedding, retrieval, etc.) can process.

Guiding Principles (from our RAG discussion):
1. **Focused Task**: data_chunking.py only handles chunk creation, not file parsing.
2. **Detailed Comments**: Provide comprehensive explanations for new developers.
3. **Consistent Return Format**: Return a dictionary with a "files" key, which is a list
   of file dictionaries. Each file dict has "file_name" and "chunks" (list of chunk dicts).
4. **Multiple Content Types**: We handle text, tables, and OCR-extracted text, using
   chunk_text, chunk_table, and chunk_image modules, respectively.

Typical Input Structure (from data_extraction.py):
[
  {
    "file_name": "some_file.pdf",
    "parse_data": {
       "text":   <string of text, if any>,
       "tables": [ <list of 2D tables> ],
       "images": [ ... depends on parse_image usage ... ],
       "metadata": {...}
    }
  },
  ...
]

Desired Output Structure:
{
  "files": [
    {
      "file_name": "some_file.pdf",
      "chunks": [
        {
          "chunk_id": "...",
          "modality": "text" or "table" or "image",
          "content": "...",
          "metadata": {...},
          "textual_modality": ...
        },
        ...
      ]
    },
    ...
  ]
}

Usage:
    python data_chunking.py <input_json> [output_json]

Where:
- <input_json> is a JSON file containing the list of parse results from data_extraction.
- [output_json] is an optional path to save the chunked result.
"""

import os
import json

# We'll import our chunking modules:
from chunk_text import chunk_text
from chunk_table import chunk_table_rows
from chunk_image import chunk_image_text


def chunk_data(parse_results: list, output_json: str = None) -> dict:
    """
    Given a list of parse results (each representing a file's extracted data),
    produce a chunked data structure that is easy to embed and store.

    :param parse_results: A list of dicts, each with:
        {
          "file_name": <string>,
          "parse_data": {
             "text": <str>,
             "tables": <list of 2D tables>,
             "images": <list of ??? if used>,
             "metadata": <dict> ...
          }
        }
    :type parse_results: list

    :param output_json: If provided, the resulting chunk structure is saved to this file.
    :type output_json: str or None

    :return: A dict with a "files" key, whose value is a list. Each element of that list
             is { "file_name": <str>, "chunks": [ list of chunk dicts ] }.
    :rtype: dict

    Steps:
      1) Initialize an output structure: {"files": []}.
      2) Loop over each entry in parse_results.
      3) For each file, gather:
         - text content -> chunk with chunk_text
         - table data -> chunk each row using chunk_table_rows
         - images / OCR text -> chunk_image_text if there's raw OCR data.
      4) Accumulate all chunk dicts in a single "chunks" list.
      5) Append { "file_name": file_name, "chunks": chunk_list } to "files".
      6) If output_json is given, write to disk. Return the final structure.

    Implementation Notes:
      - parse_data["text"] might be normal text or OCR text. If your pipeline
        differentiates them, you might do a check (like "is_ocr": True) and call
        chunk_image_text. Otherwise, chunk_text is safe for standard text.
      - parse_data["tables"] is a list of 2D arrays. We'll create row-based chunks with
        chunk_table_rows, giving each row or row-block its own chunk.
      - parse_data["images"] might be an array of references. Usually, parse_image.py
        places recognized text in parse_data["text"], so "images" might remain empty.
      - The chunking approach here is basic. For advanced usage, you might further
        split or combine text paragraphs if they are too big or too short.
    """
    final_result = {"files": []}

    for file_item in parse_results:
        file_name = file_item.get("file_name", "unknown_file")
        parse_data = file_item.get("parse_data", {})
        text_content = parse_data.get("text", "")
        table_list = parse_data.get("tables", [])
        images_data = parse_data.get("images", [])  # If used
        metadata = parse_data.get("metadata", {})

        # We'll gather chunk dicts here
        chunk_list = []

        # 1) Chunk the text content
        if text_content.strip():
            # For simplicity, we'll treat it as normal text chunking
            # (If you specifically need chunk_image_text for OCR text, do a check or a pipeline flag)
            text_chunks = chunk_text(
                text_content=text_content,
                file_name=file_name,
                wrap_width=80
            )
            chunk_list.extend(text_chunks)

        # 2) Chunk each table row
        for t_idx, table_data in enumerate(table_list):
            # We'll create row-based chunks for each table
            # We'll pass a custom file_name that indicates table index
            # so chunk_id doesn't overlap if there are multiple tables
            table_name = f"{file_name}_table_{t_idx}"
            table_chunks = chunk_table_rows(
                table_data=table_data,
                file_name=table_name,  # This ensures chunk_id references the correct table
                start_index=0
            )
            chunk_list.extend(table_chunks)

        # 3) If we have images with separate textual data, we might chunk them here
        #    Typically, parse_image.py puts recognized text in parse_data["text"], so
        #    images[] might not have direct text. But if it does:
        # for img_idx, img_content in enumerate(images_data):
        #     # if 'text' in img_content, we can chunk_image_text
        #     # or do other chunk logic as needed.
        #     pass

        # Append the aggregated chunks
        final_result["files"].append({
            "file_name": file_name,
            "chunks": chunk_list
        })

    # If output_json is provided, save to disk
    if output_json:
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(final_result, f, indent=2)
        print(f"[data_chunking] Wrote chunked data to '{output_json}'.")

    return final_result


if __name__ == "__main__":
    """
    If called as a script:
      python data_chunking.py <input_json> [output_json]

    <input_json> is expected to contain the parse_results from data_extraction,
    something like:
    [
      {
        "file_name": "example.txt",
        "parse_data": {
           "text": "...some text...",
           "tables": [],
           "images": [],
           "metadata": {...}
        }
      },
      ...
    ]

    The script will read <input_json>, produce chunked data, and if output_json is given,
    write the chunked structure to disk. Then it prints a summary of how many chunks
    each file produced.
    """
    import sys

    if len(sys.argv) < 2:
        print("Usage: python data_chunking.py <input_json> [output_json]")
        sys.exit(1)

    input_json_path = sys.argv[1]
    output_json_path = sys.argv[2] if len(sys.argv) > 2 else None

    # Load parse results from the specified JSON file
    try:
        with open(input_json_path, "r", encoding="utf-8") as f:
            parse_results_data = json.load(f)
    except Exception as e:
        print(f"[data_chunking] Error loading '{input_json_path}': {e}")
        sys.exit(1)

    # Chunk the data
    chunked_result = chunk_data(parse_results_data, output_json=output_json_path)

    # If no output JSON given, print a summary to stdout
    if not output_json_path:
        for file_obj in chunked_result["files"]:
            file_name = file_obj["file_name"]
            chunk_count = len(file_obj["chunks"])
            print(f"File: {file_name}, # of chunks: {chunk_count}")
