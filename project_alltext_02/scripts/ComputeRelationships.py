"""
ComputeRelationships.py

Creates chunk-chunk semantic edges in Neo4j using the same
user/password as your StoreInNeo4j script (neo4j / Neo4j420).

Steps:
 1) Reads embedded_data.json with the structure:
    {
      "files": [
        {
          "file_name": "...",
          "chunks": [
            {
              "chunk_id": "...",
              "embedding": [ float, ... ],
              ...
            },
            ...
          ]
        },
        ...
      ]
    }
 2) Collects all chunks that have a valid embedding (list of floats).
 3) Computes pairwise cosine similarity (O(n^2)).
 4) For pairs above THRESHOLD, merges a relationship in Neo4j:
    (c1:Chunk)-[:SIMILAR_TO { similarity: x }]->(c2:Chunk)

Usage:
  python ComputeRelationships.py

(You can set THRESHOLD or JSON path in code below.)
"""

import os
import sys
import json
import numpy as np
from neo4j import GraphDatabase, basic_auth


# Hard-coded Neo4j connection and config
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASS = "Neo4j420"

# Input JSON file and similarity threshold
INPUT_JSON = "embedded_data.json"
THRESHOLD = 0.75


def cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two float vectors."""
    v1 = np.array(vec1, dtype=float)
    v2 = np.array(vec2, dtype=float)
    dot = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)


def main():
    # 1) Read the JSON
    if not os.path.isfile(INPUT_JSON):
        print(f"[ComputeRelationships] File not found: {INPUT_JSON}")
        sys.exit(1)

    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    # We expect data["files"] -> list of file objects, each with file_name + chunks
    if "files" not in data or not isinstance(data["files"], list):
        print("[ComputeRelationships] JSON must have 'files' as a list.")
        sys.exit(1)

    # 2) Gather all (chunk_id, embedding) for chunks that have a non-empty embedding
    all_chunks = []
    for file_info in data["files"]:
        for chunk in file_info.get("chunks", []):
            cid = chunk.get("chunk_id")
            emb = chunk.get("embedding", [])
            # check it is a list of floats
            if cid and isinstance(emb, list) and len(emb) > 0:
                all_chunks.append((cid, emb))

    print(f"[ComputeRelationships] Found {len(all_chunks)} chunks with embeddings.")

    if len(all_chunks) < 2:
        print("[ComputeRelationships] Not enough chunks to form relationships. Exiting.")
        return

    # 3) Connect to Neo4j with the same user/password as store script
    print(f"[ComputeRelationships] Connecting to {NEO4J_URI} as '{NEO4J_USER}'...")
    driver = GraphDatabase.driver(NEO4J_URI, auth=basic_auth(NEO4J_USER, NEO4J_PASS))

    pair_count = 0
    n = len(all_chunks)

    # 4) O(n^2) pairwise similarity
    with driver.session() as session:
        for i in range(n):
            c1_id, emb1 = all_chunks[i]
            for j in range(i + 1, n):
                c2_id, emb2 = all_chunks[j]
                sim = cosine_similarity(emb1, emb2)
                if sim >= THRESHOLD:
                    # Create or merge the relationship
                    query = """
                    MATCH (c1:Chunk { chunk_id: $c1_id }), (c2:Chunk { chunk_id: $c2_id })
                    MERGE (c1)-[r:SIMILAR_TO { similarity: $sim }]->(c2)
                    """
                    session.run(query, {"c1_id": c1_id, "c2_id": c2_id, "sim": float(sim)})
                    pair_count += 1

    driver.close()
    print(f"[ComputeRelationships] Created {pair_count} SIMILAR_TO edges with sim >= {THRESHOLD}.")


if __name__ == "__main__":
    main()
