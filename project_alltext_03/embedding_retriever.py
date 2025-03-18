"""
embedding_retriever.py

This module focuses on retrieving relevant chunks from Neo4j by leveraging their 
embedding vectors. Given a user query embedding, we compute local cosine similarity 
for each chunk node (which has an 'embedding' property) and return the top-K matches.

Guiding Principles (from prior discussion):
1. **Local embedding usage**: We retrieve chunk embeddings from Neo4j, compute similarities 
   in Python. For large datasets, consider approximate indexing in Neo4j (version 5+) or 
   external ANN libraries like FAISS.
2. **Detailed commentary**: Each step is clearly described for future team members.
3. **Returns top-K**: We filter out all but the K highest-similarity chunks, returning them 
   in sorted order by their similarity to the user’s query embedding.
4. **Integration**: Typically used in a “hybrid” pipeline to also incorporate topic-based 
   expansions if needed, but that logic is handled in a separate module (e.g., `hybrid_retriever.py`).

Example Usage:
    from neo4j import GraphDatabase, basic_auth
    from embedding_retriever import retrieve_by_embedding

    driver = GraphDatabase.driver("bolt://localhost:7687", auth=basic_auth("neo4j", "password"))
    user_query_emb = <some numpy array of the user query>
    top_chunks = retrieve_by_embedding(driver, user_query_emb, top_k=10)
    # 'top_chunks' is a list of dictionaries, each chunk containing:
    #   { 'chunk_id': str, 'content': str, 'embedding': [...], 'topic_id': ???, 'sim': float }
"""

import numpy as np
from neo4j import Driver


def retrieve_by_embedding(driver: Driver, query_embedding: np.ndarray, top_k: int = 5) -> list:
    """
    Fetch chunk embeddings from Neo4j, compute cosine similarity to 'query_embedding',
    and return the top-K chunks with highest similarity. Each returned item includes
    a "sim" field indicating the computed similarity.

    :param driver: neo4j.Driver object for connecting to Neo4j
    :param query_embedding: a 1D numpy array or list representing the user query's embedding
    :param top_k: how many top results to return
    :return: list of dictionaries, each with keys:
       {
         'chunk_id': str,
         'content': str,
         'embedding': [...],
         'topic_id': <optional topic_id if stored>,
         'sim': float
       }
      The list is sorted by descending sim, up to 'top_k'.

    Steps:
      1) In a Neo4j session, MATCH all :Chunk nodes that have a non-empty 'embedding'.
         Return chunk_id, content, embedding, and optionally topic_id if it exists.
      2) In Python, compute local cosine similarity between the chunk embedding
         and the provided 'query_embedding'.
      3) Sort chunks by descending similarity, slice the top_k.
      4) Return that final list, each chunk dict includes 'sim' for the user to see
         how close it was to the query.

    Caveats:
      - This is an O(n) approach if you have 'n' chunks. For large data, you might 
        want a more sophisticated approach (like approximate nearest neighbor).
      - If you are using Neo4j 5+ with vector indexing, you could do a direct 
        KNN search in Cypher. This example demonstrates a simpler approach.

    Example:
       top_results = retrieve_by_embedding(driver, query_embedding, top_k=10)
       for r in top_results:
           print(r["chunk_id"], r["sim"], r["content"][:80])

    See also:
      * hybrid_retriever.py, which may combine these embedding results with 
        topic expansions for a more advanced approach.
    """

    # Basic local function for cosine similarity
    def cosine_similarity(v1, v2):
        v1 = np.array(v1, dtype=float)
        v2 = np.array(v2, dtype=float)
        dot = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0.0 or norm2 == 0.0:
            return 0.0
        return dot / (norm1 * norm2)

    # Step 1) Query Neo4j for chunk data
    with driver.session() as session:
        cypher = """
        MATCH (c:Chunk)
        WHERE c.embedding IS NOT NULL AND size(c.embedding) > 0
        RETURN c.chunk_id AS chunk_id,
               c.content AS content,
               c.embedding AS embedding,
               c.topic_id AS topic_id
        """
        result = session.run(cypher)
        chunk_data = [dict(record) for record in result]  # each record -> { 'chunk_id':..., etc. }

    # Step 2) For each chunk, compute local cos sim
    for c in chunk_data:
        emb = c["embedding"]
        # compute sim
        sim_val = cosine_similarity(query_embedding, emb)
        c["sim"] = sim_val

    # Step 3) sort by descending sim, slice top_k
    chunk_data.sort(key=lambda x: x["sim"], reverse=True)
    top_results = chunk_data[:top_k]

    # Step 4) return
    return top_results
