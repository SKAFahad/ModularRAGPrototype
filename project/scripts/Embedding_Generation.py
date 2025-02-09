"""
Embedding_Generation.py

Orchestrates the multimodal embedding generation process by:
  1) Loading chunked data (produced by the data ingestion step).
  2) Loading three separate embedding modules (text, table, image).
  3) Iterating over each chunk and embedding it based on its modality.
  4) Saving the updated chunks (now with embedding vectors) to a JSON file.

This script is part of a modular RAG system, where each modality has 
its own embedding approach.
"""

import json
import os

from embed_text import load_text_model, embed_text_simcse
from embed_table import load_table_model, embed_table_tapas
from embed_image import load_image_model, embed_image_clip

def main():
    # 1) Load the chunked data (output from Data_Ingestion_and_Chunking.py)
    CHUNKED_DATA_PATH = "project/chunked_data.json"
    if not os.path.isfile(CHUNKED_DATA_PATH):
        print(f"Error: {CHUNKED_DATA_PATH} not found.")
        return

    with open(CHUNKED_DATA_PATH, "r") as f:
        chunks = json.load(f)
    print(f"Loaded {len(chunks)} chunks from '{CHUNKED_DATA_PATH}'.")

    # 2) Load models
    #    Each function returns a specialized embedding model (and tokenizer if needed).
    text_model = load_text_model()                # Supervised SimCSE
    table_tokenizer, table_model = load_table_model()  # TAPAS
    image_model, image_preprocess = load_image_model() # CLIP model + preprocess

    # 3) Embed each chunk by modality
    for chunk in chunks:
        modality = chunk.get("modality", "")
        content = chunk.get("content", "")

        if modality == "text":
            # Use SimCSE for text chunks
            emb = embed_text_simcse(content, text_model)
            chunk["embedding"] = emb

        elif modality == "table":
            # Use TAPAS for table chunks
            emb = embed_table_tapas(content, table_tokenizer, table_model)
            chunk["embedding"] = emb

        elif modality == "image":
            # Use CLIP for image chunks
            emb = embed_image_clip(content, image_model, image_preprocess)
            chunk["embedding"] = emb

        else:
            # If unknown or unsupported modality, store empty embedding
            print(f"Warning: Unrecognized modality '{modality}' for chunk_id '{chunk.get('chunk_id')}'.")
            chunk["embedding"] = []

        # Optional: Assign a manual attention score
        # chunk_id = chunk.get("chunk_id")
        # chunk["attention_score"] = ...
        # For now, you can assign 1.0 or read from a custom map if you like.

    # 4) Save the updated chunks (now with embeddings) to a new JSON
    OUTPUT_PATH = "project/chunked_with_embeddings.json"
    with open(OUTPUT_PATH, "w") as f:
        json.dump(chunks, f, indent=2)

    print(f"Embedding generation complete. {len(chunks)} chunks processed.")
    print(f"Embeddings saved to '{OUTPUT_PATH}'.")


if __name__ == "__main__":
    main()

