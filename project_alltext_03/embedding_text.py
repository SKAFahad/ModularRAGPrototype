"""
embedding_text.py

Embeds chunks from a JSON file (e.g. chunked_data.json) using SentenceTransformers,
writes the updated JSON (with an 'embedding' field for each chunk) to another file
(e.g. embedded_data.json).

We fix the issue where the script was mistakenly treating '--input' as the actual file,
by properly using argparse to parse '--input' and '--output' parameters.

Usage:
  python embedding_text.py --input chunked_data.json --output embedded_data.json [--model <model>]

Example:
  python embedding_text.py --input chunked_data.json --output embedded_data.json
  # uses default 'all-MiniLM-L6-v2' model

Implementation Steps:
---------------------
1) Parse command-line arguments (args.input, args.output, args.model).
2) Load chunked_data from 'args.input'.
3) For each chunk with non-empty 'content', embed it with SentenceTransformer.
4) Save updated data to 'args.output'.
"""

import os
import sys
import json
import argparse

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError(
        "embedding_text.py requires sentence-transformers.\n"
        "Install via: pip install sentence-transformers"
    )


def embed_all_chunks(input_json: str, output_json: str, model_name: str = "all-MiniLM-L6-v2") -> None:
    """
    Reads the chunked_data JSON from input_json, embeds each chunk's 'content',
    writes updated data with chunk["embedding"] to output_json.

    :param input_json: Path to chunked_data JSON
    :param output_json: Path to write embedded_data JSON
    :param model_name: HF SentenceTransformer model name
    """
    # Check if input exists
    if not os.path.isfile(input_json):
        raise FileNotFoundError(f"[embed_chunks] input file not found: {input_json}")

    # Load data
    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    # We expect data to be { "files": [...] }
    if not isinstance(data, dict) or "files" not in data:
        raise ValueError("[embed_chunks] JSON must have { 'files': [ ... ] } at top level.")

    files_list = data["files"]
    print(f"[embed_chunks] Found {len(files_list)} file entries in {input_json}.")

    # Load the embedding model
    print(f"[embed_chunks] Loading model: {model_name}")
    model = SentenceTransformer(model_name)

    count_embedded = 0
    count_skipped = 0

    # Iterate over each file
    for fobj in files_list:
        if "chunks" not in fobj or not isinstance(fobj["chunks"], list):
            continue
        # For each chunk
        for chunk in fobj["chunks"]:
            content = chunk.get("content", "")
            if content.strip():
                # embed
                embedding = model.encode(content).tolist()  # list of floats
                chunk["embedding"] = embedding
                count_embedded += 1
            else:
                # skip empty content
                count_skipped += 1

    print(f"[embed_chunks] Embedded {count_embedded} chunks, skipped {count_skipped} (empty content).")

    # Write output
    with open(output_json, "w", encoding="utf-8") as out_f:
        json.dump(data, out_f, indent=2)
    print(f"[embed_chunks] Wrote embedded data to '{output_json}'.")


def main():
    """
    Command-line entry point. Use argparse to parse:
      --input <input_json>
      --output <output_json>
      [--model <model_name>]
    If any are missing, we show an error or use defaults.
    """
    parser = argparse.ArgumentParser(description="Embed chunk content from an input JSON, write to output JSON.")
    parser.add_argument("--input", type=str, default="chunked_data.json",
                        help="Input JSON with {files: [ {chunks: [...]}, ... ]}.")
    parser.add_argument("--output", type=str, default="embedded_data.json",
                        help="Output JSON to write the updated data.")
    parser.add_argument("--model", type=str, default="all-MiniLM-L6-v2",
                        help="SentenceTransformer model name.")
    args = parser.parse_args()

    # Call the function
    embed_all_chunks(args.input, args.output, args.model)


if __name__ == "__main__":
    main()
