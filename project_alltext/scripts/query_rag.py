#!/usr/bin/env python3
"""
query_rag.py
------------
This script performs a similarity search for a given user query against the text chunk embeddings
stored in a Neo4j database. It is a key component of a Retrieval-Augmented Generation (RAG) system,
allowing the system to retrieve the most relevant chunks based on their similarity to the query.

Process Overview:
  1. Prompt for or accept a user query.
  2. Use SentenceTransformers to encode the query text into an embedding vector.
  3. Connect to the Neo4j database and retrieve all :Chunk nodes along with their stored embeddings.
  4. Compute the cosine similarity between the query embedding and each chunk's embedding.
  5. Sort the chunks based on similarity and select the top-K results.
  6. Display the top-K results, including the similarity scores and metadata for each chunk.

Prerequisites:
  - Neo4j must be running and accessible.
  - The text chunks with embeddings should have been previously stored in Neo4j (e.g., via store_in_neo4j.py).
  - The SentenceTransformers model used here must be the same as the one used during embedding generation.
"""

import json
import os
import numpy as np
from neo4j import GraphDatabase, exceptions
from sentence_transformers import SentenceTransformer

# Neo4j connection configuration.
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASS = "Neo4j420"  # Replace with your actual Neo4j password

# Number of top results to retrieve.
TOP_K = 5

def load_text_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """
    Loads the pre-trained text embedding model from SentenceTransformers.

    Args:
      model_name (str): Name of the model to load.
                        Default is "all-MiniLM-L6-v2".
    
    Returns:
      SentenceTransformer: The loaded text embedding model.
    """
    print(f"Loading text embedding model '{model_name}'...")
    model = SentenceTransformer(model_name)
    print("Model loaded successfully.")
    return model

def embed_query(query: str, model: SentenceTransformer) -> np.ndarray:
    """
    Generates an embedding vector for the user query.

    Args:
      query (str): The user input query.
      model (SentenceTransformer): The pre-loaded text embedding model.
    
    Returns:
      np.ndarray: The query embedding vector.
    """
    print("Embedding user query...")
    # Generate embedding and ensure it is a numpy array.
    query_embedding = model.encode(query, convert_to_numpy=True)
    print("Query embedded successfully.")
    return query_embedding

def cosine_similarity(vec1, vec2) -> float:
    """
    Computes cosine similarity between two vectors.

    Args:
      vec1 (list or np.array): First embedding vector.
      vec2 (list or np.array): Second embedding vector.
    
    Returns:
      float: Cosine similarity value.
      
    The cosine similarity is computed as the dot product of the two vectors divided by
    the product of their Euclidean norms.
    """
    v1 = np.array(vec1, dtype=float)
    v2 = np.array(vec2, dtype=float)
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    return dot_product / (norm_v1 * norm_v2)

def retrieve_chunks_from_neo4j(driver):
    """
    Retrieves all :Chunk nodes from the Neo4j database along with their properties.

    Args:
      driver: An instance of the Neo4j driver.
    
    Returns:
      list: A list of dictionaries, each representing a chunk with keys like 'chunk_id',
            'content', 'embedding', and 'metadata'.
    
    The query assumes that each Chunk node stores its embedding as a property.
    """
    query = "MATCH (ch:Chunk) RETURN ch.chunk_id AS chunk_id, ch.content AS content, ch.embedding AS embedding, ch.modality AS modality"
    chunks = []
    with driver.session() as session:
        # Run the query to retrieve all chunks.
        results = session.run(query)
        for record in results:
            chunk = {
                "chunk_id": record["chunk_id"],
                "content": record["content"],
                "embedding": record["embedding"],
                "modality": record["modality"]
            }
            chunks.append(chunk)
    print(f"Retrieved {len(chunks)} chunks from Neo4j.")
    return chunks

def find_top_k_similar_chunks(query_embedding: np.ndarray, chunks: list, top_k: int = TOP_K):
    """
    Computes cosine similarity between the query embedding and each chunk's embedding,
    then returns the top-K most similar chunks.

    Args:
      query_embedding (np.ndarray): The embedding vector for the user query.
      chunks (list): List of chunk dictionaries retrieved from Neo4j.
      top_k (int): Number of top similar chunks to return.
    
    Returns:
      list: A list of tuples (chunk, similarity_score) sorted by similarity in descending order.
    """
    similarities = []
    
    # Iterate through each chunk.
    for chunk in chunks:
        # Ensure that the chunk has an embedding.
        if "embedding" not in chunk or not chunk["embedding"]:
            continue
        # Compute cosine similarity between the query embedding and the chunk's embedding.
        sim = cosine_similarity(query_embedding, chunk["embedding"])
        similarities.append((chunk, sim))
    
    # Sort the list of tuples by similarity score in descending order.
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # Return the top-K similar chunks.
    top_k_results = similarities[:top_k]
    return top_k_results

def main():
    """
    Main function to perform the query retrieval process.
    Steps:
      1. Load the text embedding model.
      2. Prompt for a user query (or use a preset query for testing).
      3. Embed the user query.
      4. Connect to Neo4j and retrieve all chunk nodes.
      5. Compute similarity between the query embedding and each chunk embedding.
      6. Display the top-K most similar chunks with their similarity scores.
    """
    # Load the text embedding model.
    model = load_text_model()

    # Prompt the user for a query. For testing, you can preset a query.
    user_query = input("Enter your query: ").strip()
    if not user_query:
        print("No query entered. Exiting.")
        return

    # Generate an embedding for the query.
    query_embedding = embed_query(user_query, model)

    # Connect to the Neo4j database.
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
    try:
        # Retrieve all chunks from Neo4j.
        chunks = retrieve_chunks_from_neo4j(driver)
        
        # Compute and retrieve the top-K similar chunks.
        top_chunks = find_top_k_similar_chunks(query_embedding, chunks, TOP_K)
        
        # Display the top results.
        print(f"\nTop {TOP_K} similar chunks for the query:\n")
        for i, (chunk, score) in enumerate(top_chunks, start=1):
            print(f"Result {i}:")
            print(f"  Chunk ID   : {chunk['chunk_id']}")
            print(f"  Similarity : {score:.4f}")
            print(f"  Content    : {chunk['content'][:200]}...")  # show first 200 characters
            print("-" * 50)
    except exceptions.AuthError as auth_err:
        print(f"Authentication error: {auth_err}. Please check your Neo4j credentials.")
    except exceptions.ServiceUnavailable as svc_err:
        print(f"Service unavailable: {svc_err}. Ensure Neo4j is running at {NEO4J_URI}.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        # Ensure the Neo4j driver is closed.
        driver.close()
        print("Neo4j connection closed.")

if __name__ == "__main__":
    main()
