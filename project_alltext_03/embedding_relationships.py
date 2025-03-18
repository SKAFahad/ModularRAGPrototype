"""
embedding_relationships.py

This module handles creation of "EMBEDDING_SIM" relationships among chunk nodes in Neo4j
based on their vector embeddings. The embeddings themselves are stored in Neo4j under
each Chunk node as a property `embedding: [ float, ... ]`. We provide two main functions:

1) compute_embedding_similarity_topk(driver, k=5)
   - For each chunk, find the top-K nearest neighbors by cosine similarity 
     (naive O(N^2) approach). Create EMBEDDING_SIM edges with an 'embedding_similarity'
     property reflecting their similarity score.

2) compute_embedding_similarity_threshold(driver, threshold=0.75)
   - For each pair of chunks (again O(N^2)), if their similarity >= threshold, 
     create an EMBEDDING_SIM edge.

Guiding Principles (as per discussion):
- **Local usage**: We connect to an on-prem Neo4j with chunk embeddings.
- **Detailed commentary**: Each function is explained for new team members.
- **Efficient for moderate data**: For large data, consider approximate methods (e.g., FAISS).
- **Stored relationships**: For each new edge, we MERGE (c1)-[:EMBEDDING_SIM { embedding_similarity: <float> }]->(c2).
- **No duplication**: We'll do c1->c2 only, i<j or top-K from c1, so we don't create duplicates.

Typical usage within a bigger pipeline:
    from neo4j import GraphDatabase, basic_auth
    from embedding_relationships import (
        compute_embedding_similarity_topk,
        compute_embedding_similarity_threshold
    )

    driver = GraphDatabase.driver(...)
    compute_embedding_similarity_topk(driver, k=5)
    # or
    compute_embedding_similarity_threshold(driver, threshold=0.8)

Implementation Steps:
- Each function fetches chunk_id + embedding from Neo4j
- We store them in Python arrays for quick iteration
- We compute cosine similarity for each pair or for top-K
- We create EMBEDDING_SIM edges in Neo4j for relevant matches
"""

import numpy as np


def cosine_similarity(vec1, vec2):
    """
    Basic cosine similarity for 1D float arrays, returning a float in [-1,1].
    If either vector is zero or norm=0, returns 0.0 to avoid division by zero.
    """
    v1 = np.array(vec1, dtype=float)
    v2 = np.array(vec2, dtype=float)
    dot = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return dot / (norm1 * norm2)


def compute_embedding_similarity_topk(driver, k=5):
    """
    Connect each Chunk node to its top-K nearest neighbors in embedding space.
    This is an O(N^2) naive approach, suitable for moderate numbers of chunks.

    Steps:
      1) MATCH all chunks with a non-empty embedding from Neo4j.
      2) For each chunk (c1), compute similarity to all others (c2).
      3) Sort by descending similarity, pick top-K.
      4) Create a directed relationship in Neo4j:
         (c1)-[:EMBEDDING_SIM { embedding_similarity: <float> }]->(c2)
      5) Repeat for each chunk. 
         This means c2->c1 edges are only created if c2 is also in c1's top-K from its perspective.

    :param driver: A neo4j GraphDatabase driver
    :type driver: neo4j.Driver
    :param k: Number of nearest neighbors to link for each chunk
    :type k: int

    Usage Example:
        compute_embedding_similarity_topk(driver, k=5)
    """
    print(f"[embedding_relationships] EMBEDDING_SIM with top-K = {k}")

    # 1) Fetch chunk_id + embedding
    with driver.session() as session:
        query = """
        MATCH (c:Chunk)
        WHERE c.embedding IS NOT NULL AND size(c.embedding) > 0
        RETURN c.chunk_id AS chunk_id, c.embedding AS embedding
        """
        result = session.run(query)
        chunk_data = [(r["chunk_id"], r["embedding"]) for r in result]

    print(f"[topK] Retrieved {len(chunk_data)} chunks with embeddings.")

    if len(chunk_data) < 2:
        print("[topK] Not enough chunks to form relationships. Exiting.")
        return

    chunk_ids = [cd[0] for cd in chunk_data]
    embeddings = [np.array(cd[1], dtype=float) for cd in chunk_data]

    relationship_count = 0

    with driver.session() as session:
        # 2) For each chunk, compute similarity to all others
        for i, emb_i in enumerate(embeddings):
            sims = []
            for j, emb_j in enumerate(embeddings):
                if i == j:
                    continue
                sim_val = cosine_similarity(emb_i, emb_j)
                sims.append((sim_val, j))

            # 3) Sort by descending similarity, pick top-K
            sims.sort(key=lambda x: x[0], reverse=True)
            top_k = sims[:k]

            # 4) For each neighbor, MERGE an EMBEDDING_SIM edge
            for (sim_val, j_idx) in top_k:
                c1_id = chunk_ids[i]
                c2_id = chunk_ids[j_idx]
                merge_query = """
                MATCH (c1:Chunk { chunk_id: $c1_id }),
                      (c2:Chunk { chunk_id: $c2_id })
                MERGE (c1)-[:EMBEDDING_SIM { embedding_similarity: $sim }]->(c2)
                """
                session.run(merge_query, {
                    "c1_id": c1_id,
                    "c2_id": c2_id,
                    "sim": float(sim_val)
                })
                relationship_count += 1

    print(f"[topK] Created {relationship_count} EMBEDDING_SIM edges using top-K = {k}.")


def compute_embedding_similarity_threshold(driver, threshold=0.75):
    """
    Connect chunk pairs with similarity >= threshold. This is O(N^2) and 
    can create many edges if threshold is too low or chunk set is large.

    Steps:
      1) Fetch chunk_id + embedding from Neo4j
      2) For each pair (c1,c2), compute similarity
      3) If >= threshold, MERGE (c1)-[:EMBEDDING_SIM { embedding_similarity: <float> }]->(c2)

    :param driver: neo4j GraphDatabase driver
    :type driver: neo4j.Driver
    :param threshold: Minimum cosine similarity to link (e.g., 0.75)
    :type threshold: float

    Usage Example:
        compute_embedding_similarity_threshold(driver, threshold=0.8)
    """
    print(f"[embedding_relationships] EMBEDDING_SIM with threshold >= {threshold}")

    with driver.session() as session:
        query = """
        MATCH (c:Chunk)
        WHERE c.embedding IS NOT NULL AND size(c.embedding) > 0
        RETURN c.chunk_id AS chunk_id, c.embedding AS embedding
        """
        result = session.run(query)
        chunk_data = [(r["chunk_id"], r["embedding"]) for r in result]

    print(f"[threshold] Retrieved {len(chunk_data)} chunks with embeddings.")

    if len(chunk_data) < 2:
        print("[threshold] Not enough chunks to form relationships. Exiting.")
        return

    chunk_ids = [cd[0] for cd in chunk_data]
    embeddings = [np.array(cd[1], dtype=float) for cd in chunk_data]

    n = len(chunk_data)
    relationship_count = 0

    with driver.session() as session:
        # Compare all pairs
        for i in range(n):
            for j in range(i+1, n):
                sim_val = cosine_similarity(embeddings[i], embeddings[j])
                if sim_val >= threshold:
                    c1_id = chunk_ids[i]
                    c2_id = chunk_ids[j]
                    merge_query = """
                    MATCH (c1:Chunk { chunk_id: $c1_id }),
                          (c2:Chunk { chunk_id: $c2_id })
                    MERGE (c1)-[:EMBEDDING_SIM { embedding_similarity: $sim }]->(c2)
                    """
                    session.run(merge_query, {
                        "c1_id": c1_id,
                        "c2_id": c2_id,
                        "sim": float(sim_val)
                    })
                    relationship_count += 1

    print(f"[threshold] Created {relationship_count} EMBEDDING_SIM edges where sim >= {threshold}.")
