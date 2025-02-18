#!/usr/bin/env python3
"""
compute_relationships.py
------------------------
This script computes pairwise cosine similarity between text chunk embeddings and
creates relationships in a Neo4j graph database for chunk pairs whose similarity is above
a specified threshold.

Process Overview:
  1. Load chunk data from a JSON file (e.g., "project_alltext/chunked_with_embeddings.json").
  2. For each pair of chunks, compute the cosine similarity between their embedding vectors.
  3. If the similarity is greater than or equal to a predefined threshold, create a relationship
     in Neo4j (e.g., a :SIMILAR_TEXT edge with a "score" property set to the similarity value).
  4. Log the number of relationships created.

Prerequisites:
  - The JSON file must contain a list of chunks, each with the following keys:
      "chunk_id": unique identifier,
      "embedding": a list of floats,
      "content": the text content,
      "metadata": additional file information.
  - Neo4j must be running and accessible at the configured URI.
  - The neo4j Python package must be installed.
"""

import json
import os
import numpy as np
from neo4j import GraphDatabase, exceptions

# Path to the JSON file containing the chunk data with embeddings.
CHUNKS_JSON_PATH = "project_alltext/chunked_with_embeddings.json"

# Neo4j connection configuration (adjust these as needed for your environment).
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASS = "Neo4j420"  # Replace with your actual Neo4j password

# Define the similarity threshold. Only pairs with a cosine similarity >= THRESHOLD will get an edge.
SIMILARITY_THRESHOLD = 0.70

def cosine_similarity(vec1, vec2):
    """
    Computes the cosine similarity between two vectors.

    Args:
      vec1 (list or np.array): First vector.
      vec2 (list or np.array): Second vector.

    Returns:
      float: Cosine similarity between vec1 and vec2.
      
    Explanation:
      - The dot product of the two vectors is computed.
      - Each vector's norm (magnitude) is computed.
      - The similarity is the dot product divided by the product of the norms.
      - If either vector is all zeros, returns 0.0 to avoid division by zero.
    """
    v1 = np.array(vec1, dtype=float)
    v2 = np.array(vec2, dtype=float)
    
    # Compute dot product
    dot_product = np.dot(v1, v2)
    
    # Compute the Euclidean norms of both vectors.
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    # Prevent division by zero by checking if either norm is zero.
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    
    # Calculate cosine similarity.
    similarity = dot_product / (norm_v1 * norm_v2)
    return similarity

def compute_and_store_relationships(chunks, driver):
    """
    Computes pairwise cosine similarity for the list of chunks and stores relationships
    in Neo4j for each pair with similarity >= SIMILARITY_THRESHOLD.

    Args:
      chunks (list): List of chunk dictionaries.
      driver: Neo4j driver instance for executing database queries.
      
    Process:
      - Iterate over each unique pair of chunks.
      - Calculate the cosine similarity between their embeddings.
      - If the similarity meets or exceeds the threshold, create a relationship in Neo4j.
      - The relationship is labeled :SIMILAR_TEXT and stores the similarity score as a property.
    """
    relationship_count = 0  # To keep track of the number of relationships created.
    
    # Open a session with the Neo4j database.
    with driver.session() as session:
        total_chunks = len(chunks)
        # Loop over each chunk using its index.
        for i in range(total_chunks):
            chunk1 = chunks[i]
            emb1 = chunk1.get("embedding", [])
            id1 = chunk1.get("chunk_id", "")
            
            # Iterate over subsequent chunks (to avoid duplicate comparisons).
            for j in range(i + 1, total_chunks):
                chunk2 = chunks[j]
                emb2 = chunk2.get("embedding", [])
                id2 = chunk2.get("chunk_id", "")
                
                # Compute cosine similarity between the two embedding vectors.
                sim = cosine_similarity(emb1, emb2)
                
                # If the computed similarity meets or exceeds the threshold, create a relationship.
                if sim >= SIMILARITY_THRESHOLD:
                    query = """
                    MATCH (a:Chunk {chunk_id: $id1}), (b:Chunk {chunk_id: $id2})
                    MERGE (a)-[r:SIMILAR_TEXT {score: $sim}]->(b)
                    """
                    # Execute the query with the current chunk IDs and similarity score.
                    session.run(query, {"id1": id1, "id2": id2, "sim": float(sim)})
                    relationship_count += 1
                    
    return relationship_count

def main():
    """
    Main function to orchestrate the relationship computation process.
    Steps:
      1. Check if the JSON file exists; if not, exit.
      2. Load the chunk data from the JSON file.
      3. Initialize the Neo4j driver.
      4. Compute pairwise similarities and store relationships in Neo4j.
      5. Log the total number of relationships created.
      6. Close the Neo4j connection.
    """
    # Verify that the input JSON file exists.
    if not os.path.isfile(CHUNKS_JSON_PATH):
        print(f"Error: File '{CHUNKS_JSON_PATH}' not found. Please generate embeddings first.")
        return

    # Load the chunks from the JSON file.
    with open(CHUNKS_JSON_PATH, "r", encoding="utf-8") as file:
        chunks = json.load(file)
    print(f"Loaded {len(chunks)} chunks from '{CHUNKS_JSON_PATH}'.")

    # Initialize the Neo4j driver using the configured URI and credentials.
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
    
    try:
        # Compute pairwise similarities and store relationships in the database.
        total_relationships = compute_and_store_relationships(chunks, driver)
        print(f"Created {total_relationships} SIMILAR_TEXT relationships (threshold: {SIMILARITY_THRESHOLD}).")
    except exceptions.AuthError as auth_err:
        print(f"Authentication error: {auth_err}. Please check your Neo4j credentials.")
    except exceptions.ServiceUnavailable as svc_err:
        print(f"Service unavailable: {svc_err}. Ensure Neo4j is running at {NEO4J_URI}.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        # Always close the driver to release resources.
        driver.close()
        print("Neo4j connection closed.")

if __name__ == "__main__":
    main()
