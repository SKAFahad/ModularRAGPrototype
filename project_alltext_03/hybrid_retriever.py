"""
hybrid_retriever.py

Implements a **hybrid retrieval** approach that combines:
1) **Embedding-based** chunk retrieval
2) **Topic-based** chunk expansion
using a weighted scoring mechanism:
    final_score = (1 - topic_weight)*embedding_similarity + topic_weight*topic_relevance

Where "topic_relevance" is a simple 0/1 indicator that the chunk is from a relevant topic.

Guiding Principles (from discussion):
-------------------------------------
1. **Local usage**: We connect to Neo4j, fetch chunk embeddings for embedding retrieval, 
   and chunk topics for expansions. No external hosting needed.
2. **Two-phase** approach:
   - Phase A: We do an embedding-based retrieval (using `embedding_retriever.retrieve_by_embedding`)
     to get top-K chunks for the user query.
   - Phase B: We look at the topics in the top few chunks from Phase A to guess which topics 
     are relevant. Then we retrieve additional chunks from those topics 
     (using `topic_retriever.retrieve_by_topic`).
3. **Scoring**: We unify the embedding-based chunks and the topic-based expansions 
   into one set. Each chunk is assigned:
      - `sim` = embedding similarity if it came from embedding retrieval (or 0.0 if only from topic)
      - `topic_rel` = 1 if it was found in the topic expansions or if it already had a relevant topic
4. **Weighted final score**: We compute 
      final_score = (1 - topic_weight)*sim + topic_weight*topic_rel
   and sort descending. Return the top-K final results.
5. **Extensibility**: You could incorporate additional signals (like lexical or metadata).
   This module is a reference implementation for combining two signals: embedding & topic.

Usage Example:
--------------
    from hybrid_retriever import hybrid_retrieve
    import numpy as np

    # Suppose 'driver' is a neo4j.Driver
    # Suppose 'query_vec' is a numpy array for the user query embedding
    final_chunks = hybrid_retrieve(
        driver, 
        query_embedding=query_vec,
        top_k=5, 
        top_n_topic=3, 
        topic_weight=0.3
    )
    # 'final_chunks' is a list of chunk dicts, sorted by 'final_score'.

Implementation Steps:
---------------------
1) Retrieve top_k by embedding (embedding_retriever).
2) Inspect top_n_topic from that set to determine relevant topic_ids 
   (topic_retriever.get_topic_ids_from_chunks).
3) Retrieve expansions from those topics (topic_retriever.retrieve_by_topic).
4) Combine them into a dictionary keyed by chunk_id:
     - if chunk is from embedding retrieval, store 'sim'
     - if chunk is from topic retrieval, store 'topic_rel=1'
     - if chunk is from both, it has both 'sim>0' and 'topic_rel=1'
     - otherwise sim=0, topic_rel=0
5) final_score = (1 - topic_weight)*sim + topic_weight*topic_rel
6) sort descending by final_score, return the top_k.

Note:
-----
- If we find no relevant topics (i.e., none in top_n_topic had a topic_id), 
  we skip the expansions. 
- If top_k=5 in embedding retrieval but you want to pass 10 for expansions, 
  you can parameterize further. 
- This is a minimal but general approach. You might refine how you choose relevant topics 
  or how many expansions to gather.
"""

from typing import List, Dict
import numpy as np

# local modules for retrieval
from embedding_retriever import retrieve_by_embedding
from topic_retriever import get_topic_ids_from_chunks, retrieve_by_topic


def hybrid_retrieve(
    driver,
    query_embedding: np.ndarray,
    top_k: int = 5,
    top_n_topic: int = 3,
    topic_weight: float = 0.3
) -> List[Dict]:
    """
    Perform a hybrid retrieval from Neo4j that merges:
      - embedding-based retrieval for top_k chunks,
      - expansions from their topics,
    and produces a final list sorted by a weighted score:
      final_score = (1 - topic_weight) * embedding_sim + topic_weight * topic_rel

    :param driver: A neo4j.Driver object
    :param query_embedding: The user query as a numpy vector
    :param top_k: How many chunks to return from embedding retrieval, 
                  and also how many final results to keep
    :param top_n_topic: Inspect the top 'top_n_topic' embedding results to guess topics
    :param topic_weight: fraction for the 'topic' portion of the final score. 
                        E.g. 0.3 means 70/30 weighting of embedding vs topic.
    :return: A list of chunk dictionaries, each with fields like:
               {
                  "chunk_id": ...,
                  "content": ...,
                  "sim": float,         # from embedding retrieval
                  "topic_id": ...,
                  "topic_rel": 0 or 1,  # 1 if chunk is from a relevant topic
                  "final_score": float,
                  ...
               }
             sorted descending by final_score, trimmed to top_k in the final return.
    """
    # 1) embedding retrieval
    embed_results = retrieve_by_embedding(driver, query_embedding, top_k=top_k)
    # embed_results => [ { "chunk_id", "content", "embedding", "topic_id", "sim" }, ... ]

    # 2) gather topic_ids from top_n_topic of embed_results
    # If embed_results is empty, we won't find topics
    relevant_topic_ids = get_topic_ids_from_chunks(embed_results, top_n=top_n_topic)

    # 3) retrieve expansions by topic
    # e.g. 5 expansions per topic? You can refine or param. We'll do 5 as a default
    expansions = retrieve_by_topic(driver, relevant_topic_ids, max_per_topic=5)

    # We'll unify them in a chunk_map keyed by chunk_id
    chunk_map = {}

    # step A: store embedding retrieval data
    for item in embed_results:
        cid = item["chunk_id"]
        chunk_map[cid] = {
            "chunk_id": cid,
            "content": item["content"],
            "topic_id": item.get("topic_id"),
            "sim": item["sim"],       # from embedding
            "topic_rel": 0           # default, might set to 1 if expansions also includes it
        }

    # step B: incorporate expansions
    for exp_item in expansions:
        cid = exp_item["chunk_id"]
        if cid not in chunk_map:
            # chunk wasn't in embedding top_k, so sim=0
            chunk_map[cid] = {
                "chunk_id": cid,
                "content": exp_item["content"],
                "topic_id": exp_item.get("topic_id"),
                "sim": 0.0,
                "topic_rel": 1
            }
        else:
            # chunk is in both sets, so topic_rel=1
            chunk_map[cid]["topic_rel"] = 1

    # 4) compute final_score = (1 - topic_weight)*sim + topic_weight*(topic_rel)
    final_list = []
    for cid, data in chunk_map.items():
        sim_val = data["sim"]
        t_rel = data["topic_rel"]
        # e.g. 70% embedding, 30% topic
        final_score = (1.0 - topic_weight)*sim_val + topic_weight*t_rel
        data["final_score"] = final_score
        final_list.append(data)

    # 5) sort by final_score desc
    final_list.sort(key=lambda x: x["final_score"], reverse=True)

    # return top_k from final
    return final_list[:top_k]
