#!/usr/bin/env python3
"""
embedding_generation.py
-----------------------
This script performs text embedding on the text chunks produced by the parsing stage.
It reads a JSON file ("chunked_data.json") containing chunks that have been extracted from
various file types (PDF, DOCX, spreadsheets, images, and plain text), and then generates
an embedding vector for each chunk's text content using a pre-trained model.

Each chunk in the output JSON will contain:
  - chunk_id: A unique identifier for the chunk.
  - modality: "text" (since all input data has been converted to text).
  - content: The original text extracted from the file.
  - metadata: Additional information (e.g., file name, page number, etc.).
  - embedding: A list of floating-point numbers representing the text embedding vector.

The final output is saved as "chunked_with_embeddings.json".
"""

import json
import os
from sentence_transformers import SentenceTransformer

def load_text_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """
    Loads a pre-trained text embedding model using the SentenceTransformers library.

    Args:
        model_name (str): The name of the pre-trained model to use.
                          Default is "all-MiniLM-L6-v2", a compact yet powerful model.
    
    Returns:
        SentenceTransformer: The loaded text embedding model.

    The function prints messages to indicate the progress of model loading.
    """
    print(f"Loading text embedding model '{model_name}'...")
    model = SentenceTransformer(model_name)
    print("Model loaded successfully.")
    return model

def embed_text(text: str, model: SentenceTransformer) -> list:
    """
    Generates an embedding vector for a given text string using the provided model.

    Args:
        text (str): The text string to embed.
        model (SentenceTransformer): The pre-loaded text embedding model.

    Returns:
        list: A list of floats representing the embedding vector.

    Process:
        1. The text is passed to the model's encode() method.
        2. The method returns a NumPy array with the embedding.
        3. The NumPy array is converted to a Python list (for JSON serialization).
    """
    # Encode the text using the model to produce an embedding vector.
    embedding_vector = model.encode(text, convert_to_numpy=True)
    # Convert the NumPy array into a standard Python list.
    embedding_list = embedding_vector.tolist()
    return embedding_list

def generate_embeddings_for_chunks(input_path: str, output_path: str, model: SentenceTransformer):
    """
    Reads the text chunks from a JSON file, generates an embedding for each chunk's content,
    and writes the updated chunks (including embeddings) to a new JSON file.

    Args:
        input_path (str): Path to the input JSON file containing text chunks.
        output_path (str): Path where the output JSON file (with embeddings) will be saved.
        model (SentenceTransformer): The pre-loaded text embedding model.

    Process:
        - Load the JSON data from 'input_path'.
        - For each chunk, extract its "content" field.
        - Generate an embedding using embed_text() and store it in the chunk under "embedding".
        - Save the updated list of chunks to 'output_path'.
    """
    # Check if the input file exists.
    if not os.path.isfile(input_path):
        print(f"Error: Input file '{input_path}' not found.")
        return

    # Open and read the JSON file with all text chunks.
    with open(input_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"Loaded {len(chunks)} chunks from '{input_path}'.")

    # Process each chunk in the JSON list.
    for index, chunk in enumerate(chunks):
        # Retrieve the text content from the chunk.
        content = chunk.get("content", "")
        if content:  # Ensure there is text to embed.
            # Generate the embedding vector for this text.
            embedding = embed_text(content, model)
            # Store the embedding vector back into the chunk dictionary.
            chunk["embedding"] = embedding
        else:
            # If no content is found, assign an empty list as the embedding.
            chunk["embedding"] = []
        
        # Optional: Display progress every 10 chunks for large datasets.
        if (index + 1) % 10 == 0:
            print(f"Processed {index + 1} / {len(chunks)} chunks.")

    # Write the updated chunks (now including embeddings) to the output JSON file.
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)
    print(f"Embeddings generated for {len(chunks)} chunks and saved to '{output_path}'.")

def main():
    """
    Main function that orchestrates the embedding generation process.
    """
    # Define the input and output file paths.
    input_json = "project_alltext/chunked_data.json"
    output_json = "project_alltext/chunked_with_embeddings.json"

    # Load the pre-trained text embedding model.
    text_model = load_text_model()

    # Generate embeddings for all text chunks and save the results.
    generate_embeddings_for_chunks(input_json, output_json, text_model)

if __name__ == "__main__":
    main()
