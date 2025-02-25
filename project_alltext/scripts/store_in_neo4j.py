#!/usr/bin/env python3
"""
store_in_neo4j.py
-----------------
Reads 'chunked_with_embeddings.json' (containing chunk embeddings) and stores each chunk
as a :Chunk node in Neo4j. If a chunk's metadata contains file_name, a :Document node is
created and linked to the chunk via HAS_CHUNK.

Usage:
  python store_in_neo4j.py

Important:
  - Ensure 'chunked_with_embeddings.json' is in the same folder from which you run this script.
  - Neo4j should be running at the specified URI.
"""

import json
import os
from neo4j import GraphDatabase, exceptions

# Path to the JSON file with embedded chunks. 
# We're assuming it's in the same directory you run the script from.
CHUNKS_JSON_PATH = "chunked_with_embeddings.json"

# Neo4j connection info
NEO4J_URI  = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASS = "Neo4j420"  # As specified

# If True, clear existing data in Neo4j before storing new chunks
CLEAR_OLD_DATA = True

def store_chunks_in_neo4j(chunks, uri, user, password, clear_first=True):
    """
    Connects to Neo4j and stores each chunk as a :Chunk node. If metadata.file_name
    is present, creates a :Document node and links it via :HAS_CHUNK.
    """
    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        with driver.session() as session:
            # Optionally clear existing data
            if clear_first:
                print("[INFO] Clearing existing data from Neo4j...")
                session.run("MATCH (n) DETACH DELETE n")
                print("[INFO] Data cleared.")

            chunk_count = 0
            for c in chunks:
                chunk_id   = c.get("chunk_id", "")
                content    = c.get("content", "")
                embedding  = c.get("embedding", [])
                modality   = c.get("modality", "text")
                metadata   = c.get("metadata", {})
                file_name  = metadata.get("file_name", None)

                # Create or merge the :Chunk node
                create_chunk_query = """
                MERGE (ch:Chunk {chunk_id: $chunk_id})
                ON CREATE SET
                  ch.content   = $content,
                  ch.embedding = $embedding,
                  ch.modality  = $modality
                ON MATCH SET
                  ch.content   = $content,
                  ch.embedding = $embedding,
                  ch.modality  = $modality
                """
                session.run(create_chunk_query, {
                    "chunk_id": chunk_id,
                    "content": content,
                    "embedding": embedding,
                    "modality": modality
                })
                chunk_count += 1

                # If there's a file_name, create/merge a :Document node
                if file_name:
                    doc_query = """
                    MERGE (d:Document {doc_id: $file_name})
                    ON CREATE SET d.created_at = timestamp()
                    MERGE (d)-[:HAS_CHUNK]->(ch)
                    """
                    session.run(doc_query, {"file_name": file_name})

            print(f"[INFO] Stored or updated {chunk_count} chunk nodes in Neo4j.")

    except exceptions.AuthError as auth_err:
        print(f"[ERROR] Neo4j authentication failed: {auth_err}")
    except exceptions.ServiceUnavailable as svc_err:
        print(f"[ERROR] Could not connect to Neo4j: {svc_err}")
    except Exception as e:
        print(f"[ERROR] Unexpected error storing chunks: {e}")
    finally:
        driver.close()
        print("[INFO] Neo4j connection closed.")

def main():
    # 1) Check if the JSON file exists
    if not os.path.isfile(CHUNKS_JSON_PATH):
        print(f"[ERROR] File '{CHUNKS_JSON_PATH}' not found. Please generate embeddings first.")
        return

    # 2) Load the JSON data
    with open(CHUNKS_JSON_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"[INFO] Loaded {len(chunks)} chunks from '{CHUNKS_JSON_PATH}'.")

    # 3) Store in Neo4j
    store_chunks_in_neo4j(
        chunks,
        uri=NEO4J_URI,
        user=NEO4J_USER,
        password=NEO4J_PASS,
        clear_first=CLEAR_OLD_DATA
    )

if __name__ == "__main__":
    main()
