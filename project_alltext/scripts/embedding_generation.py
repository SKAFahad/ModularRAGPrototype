#!/usr/bin/env python3
"""
embedding_generation.py
-----------------------
This script reads 'chunked_data.json' (generated from data_ingestion.py),
embeds each chunk using a consistent text embedding model, and then outputs
'chunked_with_embeddings.json'.

IMPORTANT: The model here determines the dimension of your chunk embeddings.
If you plan to use 'all-MiniLM-L6-v2' (384-d), your RAG inference script must
also use 'all-MiniLM-L6-v2' to embed the user query, ensuring both are 384-d.
"""

import json
import os
from sentence_transformers import SentenceTransformer

def load_text_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """
    Loads a SentenceTransformers model that outputs 384-dimensional embeddings.

    Args:
        model_name (str): The name of the model to load.

    Returns:
        SentenceTransformer: The loaded embedding model.
    """
    print(f"[INFO] Loading embedding model '{model_name}'...")
    model = SentenceTransformer(model_name)
    print("[INFO] Model loaded successfully.")
    return model

def embed_text_chunks(input_json: str, output_json: str, model: SentenceTransformer):
    """
    Reads chunks from input_json, embeds each chunk's content, and writes
    an updated JSON with 'embedding' fields to output_json.

    Args:
        input_json (str): Path to the JSON containing chunked data.
        output_json (str): Path to save the JSON with embeddings added.
        model (SentenceTransformer): The model used to embed the text.
    """
    # Check if the input JSON file exists.
    if not os.path.isfile(input_json):
        print(f"[ERROR] Input file '{input_json}' not found.")
        return

    # Load chunks from the file.
    with open(input_json, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"[INFO] Loaded {len(chunks)} chunks from '{input_json}'.")

    # Embed each chunk.
    for i, chunk in enumerate(chunks):
        content = chunk.get("content", "")
        if content:
            embedding = model.encode(content, convert_to_numpy=True).tolist()
            chunk["embedding"] = embedding
        else:
            chunk["embedding"] = []

        # Optional: Log progress every 50 chunks
        if (i + 1) % 50 == 0:
            print(f"[INFO] Processed {i + 1} / {len(chunks)} chunks.")

    # Write the updated chunks to the output JSON file.
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)
    print(f"[INFO] Embedding generation complete. Saved to '{output_json}'.")

def main():
    # By default, we assume your chunked data is in project_alltext/chunked_data.json
    # and we want to produce project_alltext/chunked_with_embeddings.json
    input_path = "chunked_data.json"
    output_path = "chunked_with_embeddings.json"

    # Load the model. This script uses 'all-MiniLM-L6-v2' by default (384-d).
    model = load_text_model("all-MiniLM-L6-v2")

    # Embed and save.
    embed_text_chunks(input_path, output_path, model)

if __name__ == "__main__":
    main()
