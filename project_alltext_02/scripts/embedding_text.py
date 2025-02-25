"""
embedding_text.py

A script that:
1) Reads input JSON structured as { "files": [ { "file_name":..., "chunks":[...] } ] }
2) For each chunk, if chunk["content"] is non-empty, we embed it (regardless of "modality").
3) We store the embedding in chunk["embedding"] (a list of floats), 
   then write the updated data to a new JSON file (never overwriting the original).

Usage:
    python embedding_text.py --input chunked_data.json --output embedded_data.json

Dependencies:
    pip install sentence-transformers
    # plus torch if not included

By default, we load "all-MiniLM-L6-v2". Use --model <modelname> to override.
"""

import json
import os
import argparse
from sentence_transformers import SentenceTransformer

def embed_all_chunks(input_json: str,
                     output_json: str,
                     model_name: str = "all-MiniLM-L6-v2") -> None:
    """
    Reads a JSON with structure:
      {
        "files": [
          {
            "file_name": "...",
            "chunks": [
              {
                "chunk_id": "...",
                "modality": "table"/"image"/"text",
                "content": "...some text..."
                ...
              },
              ...
            ]
          },
          ...
        ]
      }
    For each chunk, if 'content' is non-empty, we embed it with SentenceTransformer
    and store chunk["embedding"] = [ list_of_floats ].

    Writes output to 'output_json' as a new file, leaving 'input_json' unmodified.
    """
    if not os.path.isfile(input_json):
        raise FileNotFoundError(f"[embed_all] Input JSON not found: {input_json}")

    print(f"[embed_all] Loading model: {model_name}")
    model = SentenceTransformer(model_name)

    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    # We expect 'data' to be { "files": [ ... ] }
    if not isinstance(data, dict) or "files" not in data:
        raise ValueError("[embed_all] JSON must have 'files' key at top-level with a list of file objects.")

    files_list = data["files"]
    if not isinstance(files_list, list):
        raise ValueError("[embed_all] data['files'] must be a list.")

    count_embedded = 0
    count_skipped = 0

    # Iterate over each file entry
    for file_info in files_list:
        if "chunks" not in file_info or not isinstance(file_info["chunks"], list):
            print(f"[embed_all] Warning: no 'chunks' list in file_info for {file_info.get('file_name')}")
            continue

        # Go through each chunk
        for chunk in file_info["chunks"]:
            text_content = chunk.get("content", "")
            if text_content.strip():
                # If content is non-empty, embed it
                embedding = model.encode(text_content).tolist()
                chunk["embedding"] = embedding
                count_embedded += 1
            else:
                # If 'content' is empty or whitespace, skip
                count_skipped += 1

    # Write a new JSON, never overwriting input
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"[embed_all] Embedded {count_embedded} chunks, skipped {count_skipped} (empty content).")
    print(f"[embed_all] Output written to: {output_json}")

def main():
    parser = argparse.ArgumentParser(description="Embed all chunk content if non-empty, ignoring modality.")
    parser.add_argument("--input", type=str, default="chunked_data.json",
                        help="Path to input JSON with {files: [ {chunks: [...]}, ... ]}.")
    parser.add_argument("--output", type=str, default="embedded_data.json",
                        help="Where to write the updated JSON.")
    parser.add_argument("--model", type=str, default="all-MiniLM-L6-v2",
                        help="Which SentenceTransformer model to use.")
    args = parser.parse_args()

    embed_all_chunks(args.input, args.output, args.model)

if __name__ == "__main__":
    main()
