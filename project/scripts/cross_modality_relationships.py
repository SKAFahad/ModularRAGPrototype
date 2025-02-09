"""
cross_modality_relationships.py

A Python script that:
  1) Loads chunk data from JSON (with embeddings and chunk_id).
  2) Optionally zero-pads or truncates embeddings to a uniform dimension.
  3) Only compares cross-modality pairs (text↔table, text↔image, table↔image).
  4) For pairs above a similarity threshold, merges a relationship in Neo4j:
       (chunk1)-[:CROSS_MODAL_RELATED {similarity: ...}]->(chunk2)

Usage:
  python cross_modality_relationships.py

Dependencies:
  - pip install neo4j numpy

Notes:
  - Ensure you've already run a script (e.g. StoreInNeo4j.py) to create nodes
    with chunk_id in the Neo4j database. This script only MERGEs edges.
  - If you want bridging (zero-padding/truncation), set DO_BRIDGING=True and
    specify TARGET_DIM. If your embeddings already match dimensions, you can
    set DO_BRIDGING=False.
  - Adjust THRESHOLD or skip dimension mismatch as you prefer.
"""

import json
import os
import numpy as np
from neo4j import GraphDatabase

### CONFIG ###

# JSON file with chunk data (each has { chunk_id, modality, embedding, ... })
CHUNKS_JSON = "project/chunked_with_scores.json"

# Neo4j Connection
NEO4J_URI  = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASS = "Neo4j420"

# Relationship threshold
THRESHOLD = 0.30

# Whether to zero-pad or truncate embeddings to a uniform dimension
DO_BRIDGING = True
TARGET_DIM  = 768  # The dimension we unify to if bridging

# If True, skip pairs with the same modality (only cross-modal)
ONLY_CROSS_MODALITY = True


### FUNCTIONS ###

def unify_dimension(embedding, target_dim):
    """
    Zero-pad or truncate embedding to exactly 'target_dim'.
    Example: if embedding is len=512 and target_dim=768, pad with 256 zeros.
             if len=1024, truncate to 768.
    """
    emb_len = len(embedding)
    if emb_len == target_dim:
        return embedding
    elif emb_len > target_dim:
        return embedding[:target_dim]
    else:
        # emb_len < target_dim
        pad_count = target_dim - emb_len
        return embedding + [0.0]*pad_count

def cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two same-length arrays/lists."""
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

    # 2) Optionally unify dimension
    if DO_BRIDGING:
        print(f"Zero-padding/truncating embeddings to dimension={TARGET_DIM} for bridging.")
        for chunk in chunks:
            emb = chunk.get("embedding", [])
            chunk["embedding"] = unify_dimension(emb, TARGET_DIM)

    # 3) Connect to Neo4j
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
    pair_count = 0

    # 4) Compare chunk pairs for cross-modality similarity
    with driver.session() as session:
        n = len(chunks)
        for i in range(n):
            c1 = chunks[i]
            id1 = c1.get("chunk_id", "")
            mod1 = c1.get("modality", "")
            emb1 = c1.get("embedding", [])

            for j in range(i+1, n):
                c2 = chunks[j]
                id2 = c2.get("chunk_id", "")
                mod2 = c2.get("modality", "")
                emb2 = c2.get("embedding", [])

                # skip same chunk
                if id1 == id2:
                    continue

                # 4.1) If ONLY_CROSS_MODALITY, skip same-modality
                if ONLY_CROSS_MODALITY and mod1 == mod2:
                    continue

                # 4.2) If dimension mismatch after bridging, skip
                if len(emb1) != len(emb2):
                    # (Should not happen if bridging is done, unless some chunk is empty)
                    continue

                # 4.3) Compute similarity
                sim = cosine_similarity(emb1, emb2)
                if sim >= THRESHOLD:
                    # MERGE the relationship in Neo4j
                    create_rel_query = """
                    MATCH (c1:Chunk {chunk_id: $id1}), (c2:Chunk {chunk_id: $id2})
                    MERGE (c1)-[r:CROSS_MODAL_RELATED {similarity: $sim}]->(c2)
                    """
                    session.run(create_rel_query, {
                        "id1": id1,
                        "id2": id2,
                        "sim": float(sim)
                    })
                    pair_count += 1

    driver.close()
    print(f"Created {pair_count} CROSS_MODAL_RELATED edges with sim >= {THRESHOLD}.")

if __name__ == "__main__":
    main()
