"""
embed_text.py

This module handles text embedding using the Supervised SimCSE model from 
the SentenceTransformers library. The model is loaded once and can then be 
applied to individual strings or batches of text to produce sentence embeddings.

Model: princeton-nlp/sup-simcse-bert-base-uncased
"""

from sentence_transformers import SentenceTransformer
import numpy as np

def load_text_model():
    """
    Loads the Supervised SimCSE model (from Hugging Face) via SentenceTransformers.
    
    Returns:
        SentenceTransformer: An instance of the text embedding model, 
                             ready for encoding text chunks.
                             
    Example:
        text_model = load_text_model()
    """
    print("Loading Supervised SimCSE text model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model

def embed_text_simcse(text: str, model) -> list:
    """
    Generates an embedding for a given text string using a SentenceTransformer model.

    Args:
        text (str): The input text to embed.
        model (SentenceTransformer): The loaded SimCSE model instance.

    Returns:
        list: A Python list of floats representing the embedding vector.

    Example:
        embedding = embed_text_simcse("Hello world", text_model)
    """
    # The encode method typically returns either a NumPy array or a list of NumPy arrays.
    # Here we specify convert_to_numpy=True for convenience.
    embedding = model.encode(text, convert_to_numpy=True)
    
    # Convert NumPy array to a regular Python list for easier serialization (e.g. JSON).
    return embedding.tolist()
