"""
topic_relationships.py

Handles creation of 'TOPIC_SIM' relationships among :Chunk nodes that share the same
topic_id in Neo4j. Typically, topic_id is assigned by a topic modeling process such as
BERTopic. The simplest approach is to connect all chunks belonging to the same topic
with a :TOPIC_SIM edge. We can optionally limit connections if a topic cluster is large.

Guiding Principles (from the discussion):
1. **Local usage**: We connect to a local Neo4j instance containing chunk nodes.
2. **Detailed comments**: Provide clarity for how and why edges are created.
3. **Two approaches**:
   - Full clique (full_clique=True): connect each chunk to every other chunk in that topic,
     adding a property like topic_similarity=1.
   - Partial approach (full_clique=False): each chunk is connected to only up to top_k
     other chunks in that topic, preventing huge cliques.
4. **Performance Consideration**: Large topics can lead to many edges, so partial approach
   is recommended if some topics have hundreds or thousands of chunks.

Usage:
    from topic_relationships import compute_topic_similarity

    driver = GraphDatabase.driver(...)  # from neo4j
    compute_topic_similarity(driver, full_clique=True)
    # or
    compute_topic_similarity(driver, full_clique=False, top_k=5)

Implementation steps:
 - Each chunk node is expected to have a 'topic_id' property. We:
   1) Fetch chunk_id, topic_id for each chunk with a topic_id.
   2) Group chunk_ids by topic_id in Python.
   3) For each group (i.e. each topic), either:
        - connect all pairs (full clique), or
        - connect each chunk to up to top_k neighbors, e.g. next top_k in the list
          or random selection
   4) MERGE relationships in Neo4j with relationship type :TOPIC_SIM 
      and property: topic_similarity=1 (or another score if advanced usage).
"""

from collections import defaultdict
from neo4j import Session


def compute_topic_similarity(driver, full_clique=True, top_k=5):
    """
    Creates :TOPIC_SIM edges among chunk nodes that share the same topic_id.

    :param driver: A neo4j GraphDatabase driver
    :type driver: neo4j.Driver

    :param full_clique: If True, for each topic_id we connect every chunk pair 
                        with an edge. Potentially large if many chunks share a topic.
    :type full_clique: bool

    :param top_k: If full_clique=False, we only connect each chunk to up to 
                  top_k other chunks in the same topic, limiting edge explosion.
    :type top_k: int

    Steps:
      1) MATCH all chunks that have a 'topic_id' property in Neo4j.
      2) Group chunk_ids by topic_id in Python.
      3) For each topic, retrieve the chunk list.
      4) If full_clique:
           create edges for all chunk pairs
         else:
           for each chunk i, connect it to next top_k chunks in the list.
      5) Each edge is stored as:
         (c1)-[:TOPIC_SIM { topic_similarity: 1 }]->(c2)
         or you could store a more nuanced similarity if you have distributions.

    Example usage:
        compute_topic_similarity(driver, full_clique=True)
        # or partial approach with top_k=3
        compute_topic_similarity(driver, full_clique=False, top_k=3)
    """
    print("[topic_relationships] Building :TOPIC_SIM edges from chunk nodes with topic_id")

    # 1) Fetch chunk_id and topic_id for all chunks that have a topic_id
    with driver.session() as session:
        query = """
        MATCH (c:Chunk)
        WHERE EXISTS(c.topic_id)
        RETURN c.chunk_id AS chunk_id, c.topic_id AS topic_id
        """
        result = session.run(query)

        chunk_topic_list = [(r["chunk_id"], r["topic_id"]) for r in result]

    num_chunks = len(chunk_topic_list)
    print(f"[topic_relationships] Found {num_chunks} chunk(s) that have a topic_id.")

    if num_chunks < 2:
        print("[topic_relationships] Not enough chunk nodes with topic_id to form edges.")
        return

    # 2) Group by topic_id
    topic_map = defaultdict(list)
    for cid, topic in chunk_topic_list:
        topic_map[topic].append(cid)

    relationship_count = 0

    # 3) For each topic, link the relevant chunk_ids
    with driver.session() as session:
        for topic_id, cids in topic_map.items():
            # If only one chunk in that topic, skip
            if len(cids) < 2:
                continue

            if full_clique:
                # connect all pairs in that topic
                for i in range(len(cids)):
                    for j in range(i+1, len(cids)):
                        c1_id = cids[i]
                        c2_id = cids[j]
                        merge_query = """
                        MATCH (c1:Chunk { chunk_id: $c1_id }),
                              (c2:Chunk { chunk_id: $c2_id })
                        MERGE (c1)-[:TOPIC_SIM { topic_similarity: 1 }]->(c2)
                        """
                        session.run(merge_query, {
                            "c1_id": c1_id,
                            "c2_id": c2_id
                        })
                        relationship_count += 1
            else:
                # partial approach: each chunk links to up to top_k neighbors
                for i in range(len(cids)):
                    c1_id = cids[i]
                    # pick up to top_k in the list after i
                    # you could do random but we do a stable approach
                    upper_bound = min(len(cids), i+1+top_k)
                    for j in range(i+1, upper_bound):
                        c2_id = cids[j]
                        merge_query = """
                        MATCH (c1:Chunk { chunk_id: $c1_id }),
                              (c2:Chunk { chunk_id: $c2_id })
                        MERGE (c1)-[:TOPIC_SIM { topic_similarity: 1 }]->(c2)
                        """
                        session.run(merge_query, {
                            "c1_id": c1_id,
                            "c2_id": c2_id
                        })
                        relationship_count += 1

    print(f"[topic_relationships] Created {relationship_count} :TOPIC_SIM edges.")
