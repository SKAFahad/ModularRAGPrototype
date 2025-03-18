"""
compute_relationships.py

This script is the **main orchestrator** for creating similarity relationships
between chunk nodes in Neo4j. It leverages two separate modules for clarity:

  1) `embedding_relationships.py`: 
      - compute_embedding_similarity_topk(driver, k=5)
      - compute_embedding_similarity_threshold(driver, threshold=0.75)

     These functions create :EMBEDDING_SIM edges among chunks, based on stored
     embeddings. For each chunk, you can connect to top-K nearest neighbors or
     connect all pairs above a chosen similarity threshold.

  2) `topic_relationships.py`:
      - compute_topic_similarity(driver, full_clique=True, top_k=5)

     This function creates :TOPIC_SIM edges among chunks that share the same
     topic_id (usually assigned by a topic model like BERTopic). You can create a
     full clique or only partial connections (e.g., top_k edges per chunk in that
     topic) to avoid large cliques.

By combining these approaches, you can maintain a Neo4j graph that supports
**hybrid retrieval**: search by embedding similarity and/or by topic membership.

Guiding Principles (from discussion):
- **Local usage**: Connect to a local or on-prem Neo4j instance storing chunk data.
- **Detailed commentary**: Thorough docstrings and inline comments for new devs.
- **Modular design**: Each similarity approach is in a separate module, making
  maintenance and enhancements simpler.
- **CLI**: This script reads command-line arguments to decide which relationships to
  compute and with what parameters.

Typical usage:
  python compute_relationships.py [--embedding threshold=0.8] [--topic fullClique]

Examples:
  - `python compute_relationships.py --embedding topK=5`
    => link each chunk to top-5 neighbors by embedding similarity

  - `python compute_relationships.py --embedding threshold=0.8`
    => link chunk pairs with embedding similarity >= 0.8

  - `python compute_relationships.py --topic fullClique`
    => link chunk pairs in each topic (fully connect them)

  - `python compute_relationships.py --topic topK=3`
    => link chunk pairs in each topic, each chunk to up to 3 others

You can combine them, for instance:
  python compute_relationships.py --embedding threshold=0.75 --topic topK=3

This script then:
  1) Connects to Neo4j.
  2) If --embedding is set, calls either compute_embedding_similarity_topk(...) or compute_embedding_similarity_threshold(...).
  3) If --topic is set, calls compute_topic_similarity(...).
  4) Closes the driver.

**Performance notes**: For large numbers of chunks, consider approximate indexing
(e.g., FAISS) for embeddings, and partial approach for topics to avoid big cliques.
"""


import argparse
import sys
from neo4j import GraphDatabase, basic_auth

# Local modules:
from embedding_relationships import (
    compute_embedding_similarity_topk,
    compute_embedding_similarity_threshold
)
from topic_relationships import compute_topic_similarity

# Hard-coded or external config for Neo4j:
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASS = "Neo4j420"  # <--- Replace with your actual password or load from environment


def main():
    """
    The main entry point. Parses command-line arguments to determine:
      - whether to compute embedding-based relationships
      - whether to compute topic-based relationships
      - the parameters (topK, threshold, fullClique, etc.)

    Then connects to Neo4j, calls the relevant functions, and exits.
    """

    parser = argparse.ArgumentParser(
        description="Compute chunk-chunk similarity relationships in Neo4j (embedding & topic)."
    )
    parser.add_argument("--embedding", action="store_true",
                        help="Compute EMBEDDING_SIM edges among chunks using stored embeddings.")
    parser.add_argument("--topic", action="store_true",
                        help="Compute TOPIC_SIM edges among chunks sharing the same topic_id.")

    # Additional parameters come as free-form tokens like "threshold=0.75", "topK=5", "fullClique"
    parser.add_argument("params", nargs="*", default=[],
                        help="Parameters: threshold=0.75, topK=5, fullClique, etc. See docs.")
    args = parser.parse_args()

    # Parse param tokens into a dict
    params_dict = {}
    for token in args.params:
        if "=" in token:
            # e.g. "threshold=0.8"
            key, val = token.split("=", 1)
            key = key.strip()
            val = val.strip()
            params_dict[key] = val
        else:
            # e.g. "fullClique"
            params_dict[token.strip()] = True

    # Connect to Neo4j
    print(f"[compute_relationships] Connecting to Neo4j: {NEO4J_URI} with user '{NEO4J_USER}'")
    driver = GraphDatabase.driver(NEO4J_URI, auth=basic_auth(NEO4J_USER, NEO4J_PASS))

    # EMBEDDING_SIM
    if args.embedding:
        # Check if we have topK or threshold in params
        if "topK" in params_dict:
            k_val = int(params_dict["topK"])
            compute_embedding_similarity_topk(driver, k=k_val)
        elif "threshold" in params_dict:
            thr_val = float(params_dict["threshold"])
            compute_embedding_similarity_threshold(driver, threshold=thr_val)
        else:
            # Default approach: threshold=0.75
            compute_embedding_similarity_threshold(driver, threshold=0.75)

    # TOPIC_SIM
    if args.topic:
        # Check if fullClique or partial approach
        if "fullClique" in params_dict or "fullclique" in params_dict:
            compute_topic_similarity(driver, full_clique=True)
        elif "topK" in params_dict:
            tk = int(params_dict["topK"])
            compute_topic_similarity(driver, full_clique=False, top_k=tk)
        else:
            # default is full clique
            compute_topic_similarity(driver, full_clique=True)

    driver.close()
    print("[compute_relationships] Done.")


if __name__ == "__main__":
    main()
