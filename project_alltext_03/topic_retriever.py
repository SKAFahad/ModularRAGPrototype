"""
topic_retriever.py

This module helps retrieve relevant chunks from Neo4j based on their "topic_id"
property. Typically, each chunk node is assigned a topic_id from a topic modeling
step (like BERTopic). We define two main functions:

1) get_topic_ids_from_chunks(chunks, top_n=3)
   - Examines the top embedding-based chunks (or any chunk list), extracts the 
     topic_id from the first 'top_n' chunks, and returns a set of those topic_ids.
   - This is a heuristic for deciding which topics the user’s query is likely about.

2) retrieve_by_topic(driver, topic_ids, max_per_topic=5)
   - Given a set/list of topic_ids, queries Neo4j for chunks that match each topic_id,
     returning a limited number (max_per_topic) from each. This prevents huge floods 
     if a topic is large.

These functions are typically used in a "hybrid" retrieval scenario:
 - You do an embedding-based retrieval to get your top-K chunks.
 - From those chunks, you see which topic_ids appear.
 - Then you call retrieve_by_topic(...) to expand your candidate set with other chunks 
   in those same topics, even if they weren't in the top embedding results.

Note:
 - If you want a more advanced approach, you might measure how many of the top chunks 
   share a certain topic, or how "dominant" a topic is, or run a second query-based 
   classification step. This module just provides a simple, easily extended approach.
 
Usage:
    from topic_retriever import get_topic_ids_from_chunks, retrieve_by_topic

    relevant_topics = get_topic_ids_from_chunks(top_embedding_chunks, top_n=3)
    more_topic_chunks = retrieve_by_topic(driver, relevant_topics, max_per_topic=5)
"""

from typing import List, Dict, Set
from neo4j import Driver, Session


def get_topic_ids_from_chunks(chunks: List[Dict], top_n: int = 3) -> Set:
    """
    Inspect the first 'top_n' chunks (already retrieved by embedding or other means)
    to determine which topic_ids are relevant to the user’s query. We simply collect 
    the distinct topic_ids that appear in those top_n chunks and return them.

    :param chunks: A list of chunk dictionaries, each possibly with a 'topic_id' field.
                   Usually these come from an embedding-based retrieval (like top-K chunks).
                   E.g. [ { "chunk_id":..., "topic_id":..., "sim":..., ... }, ... ]
    :type chunks: list of dict

    :param top_n: How many chunks to look at. If the top_n chunks share a single topic,
                  that’s a strong indication that topic is relevant. 
    :type top_n: int

    :return: A set of distinct topic_ids found in the top_n chunks
    :rtype: set

    Steps:
      1) Slice the chunk list to the first top_n.
      2) Collect each chunk's topic_id (if any).
      3) Return a set of these topic_ids.

    Example:
        # Suppose your embedding-based retrieval returned 10 chunks
        top_chunks = [...]  # each has chunk_id, content, sim, topic_id
        relevant_topics = get_topic_ids_from_chunks(top_chunks, top_n=3)
        # returns a set like { 7, 12 }, meaning topic 7 and 12 appear in the top 3 chunks
    """
    # slice top_n
    subset = chunks[:top_n]
    topic_ids = set()
    for c in subset:
        tid = c.get("topic_id")
        if tid is not None:
            topic_ids.add(tid)
    return topic_ids


def retrieve_by_topic(driver: Driver, topic_ids, max_per_topic: int = 5) -> List[Dict]:
    """
    Given a set/list of topic_ids, retrieve up to 'max_per_topic' chunks for each 
    of those topics from Neo4j. This helps you "expand" your retrieval to include 
    thematically relevant chunks that might not have scored high in embedding similarity.

    :param driver: A neo4j.Driver to connect to the DB
    :type driver: neo4j.Driver

    :param topic_ids: The set or list of topics we want to retrieve. 
                      e.g. {7, 12} if those were determined relevant.
    :type topic_ids: set or list

    :param max_per_topic: The maximum number of chunks to retrieve per topic. 
                          If a topic has many chunks, we only pull up to this limit 
                          to avoid huge floods.
    :type max_per_topic: int

    :return: A list of chunk dicts from Neo4j, each with fields like 
             { "chunk_id":..., "content":..., "topic_id":... }, etc.
    :rtype: list of dict

    Steps:
      1) For each topic_id in topic_ids, run a Cypher query:
           MATCH (c:Chunk) WHERE c.topic_id = $tid RETURN ...
         with a LIMIT {max_per_topic}.
      2) Collect these results into a final list. 
      3) Return that list (it might have duplicates if the same chunk belongs to 
         multiple topics, but typically that’s rare unless your pipeline multi-labels).
    
    Example:
        relevant_topics = {7, 12}
        expansions = retrieve_by_topic(driver, relevant_topics, max_per_topic=5)
        # expansions -> e.g. [ {chunk_id:..., content:..., topic_id:7}, {...}, ... ]
    """
    if not topic_ids:
        return []

    results = []
    topic_ids_list = list(topic_ids)

    with driver.session() as session:
        for tid in topic_ids_list:
            cypher = f"""
            MATCH (c:Chunk)
            WHERE c.topic_id = $tid
            RETURN c.chunk_id AS chunk_id,
                   c.content AS content,
                   c.topic_id AS topic_id
            LIMIT {max_per_topic}
            """
            # We param bind 'tid', but the limit is inline
            recs = session.run(cypher, {"tid": tid})
            for rec in recs:
                chunk_info = dict(rec)
                results.append(chunk_info)

    return results
