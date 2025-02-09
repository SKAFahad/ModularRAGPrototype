"""
BridgingAndComputeRelationships.py

A script that:
  1) Loads chunk data (with embeddings) from a JSON file (e.g. 'chunked_with_scores.json').
  2) Zero-pads or truncates embeddings to a single dimension (e.g. 768).
  3) Computes pairwise cosine similarity across chunks.
  4) Merges :SEMANTICALLY_RELATED relationships in Neo4j for chunk pairs 
     above a certain threshold.

Usage:
  python BridgingAndComputeRelationships.py

Assumptions:
  - chunked_with_scores.json has structure:
      [
        {
          "chunk_id": "...",
          "modality": "...",
          "content": "...",
          "embedding": [float, ...], 
          "attention_score": float, (optional)
          "metadata": {...} (optional)
        },
        ...
      ]
  - You've already run StoreInNeo4j.py to create the :Chunk nodes in Neo4j.
  - You want to unify dimension (dim) to 768 or your chosen dimension.

Why Combine?
  - This script is "modular" but merges bridging (zero-pad or truncate) 
    with relationship creation for convenience in a single step.
"""

import json
import os
import numpy as np
from neo4j import GraphDatabase

# Paths
CHUNKS_JSON = "project/chunked_with_scores.json"

# Neo4j config
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASS = "Neo4j420"  # your password

# Bridging/Similarity config
TARGET_DIM = 768  # unify all embeddings to 768
THRESHOLD = 0.75  # pairs with sim >= 0.75 get a relationship

def unify_dimension(embedding, target_dim):
    """
    Zero-pad or truncate to ensure embedding has length == target_dim.
    Example:
      if embedding has length 512, we pad 256 zeros
      if embedding has length 1024, we truncate to 768
    """
    emb_len = len(embedding)
    if emb_len == target_dim:
        return embedding
    elif emb_len > target_dim:
        return embedding[:target_dim]
    else:  # emb_len < target_dim
        pad_count = target_dim - emb_len
        return embedding + [0.0]*pad_count

def cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two same-dimension arrays."""
    v1 = np.array(vec1, dtype=float)
    v2 = np.array(vec2, dtype=float)
    dot = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)

def main():
    # 1) Load chunks from JSON
    if not os.path.isfile(CHUNKS_JSON):
        print(f"Error: {CHUNKS_JSON} not found.")
        return

    with open(CHUNKS_JSON, "r") as f:
        chunks = json.load(f)

    print(f"Loaded {len(chunks)} chunks from '{CHUNKS_JSON}'.")

    # 2) Bridge all embeddings to TARGET_DIM
    for chunk in chunks:
        emb = chunk.get("embedding", [])
        chunk["embedding"] = unify_dimension(emb, TARGET_DIM)

    # 3) Connect to Neo4j
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

    pair_count = 0
    with driver.session() as session:
        # do naive O(n^2)
        n = len(chunks)
        for i in range(n):
            c1 = chunks[i]
            id1 = c1.get("chunk_id", "")
            emb1 = c1["embedding"]  # now guaranteed length=TARGET_DIM
            for j in range(i+1, n):
                c2 = chunks[j]
                id2 = c2.get("chunk_id", "")
                emb2 = c2["embedding"]

                # compute similarity
                sim = cosine_similarity(emb1, emb2)
                if sim >= THRESHOLD:
                    # create MERGE in Neo4j
                    create_rel_query = """
                    MATCH (c1:Chunk {chunk_id: $id1}), (c2:Chunk {chunk_id: $id2})
                    MERGE (c1)-[r:SEMANTICALLY_RELATED {similarity: $sim}]->(c2)
                    """
                    session.run(create_rel_query, {
                        "id1": id1,
                        "id2": id2,
                        "sim": float(sim)
                    })
                    pair_count += 1

    driver.close()
    print(f"Bridged embeddings to dimension={TARGET_DIM}, computed similarity, and created {pair_count} relationships (threshold={THRESHOLD}).")

if __name__ == "__main__":
    main()
