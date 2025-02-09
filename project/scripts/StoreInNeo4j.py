"""
StoreInNeo4j.py

Reads chunks from 'project/chunked_with_scores.json' and stores them in Neo4j as :Chunk nodes.
Optionally links them to a :Document node if metadata.file_name is present.

Usage:
  python StoreInNeo4j.py

The input JSON should have a structure like:
[
  {
    "chunk_id": "some_id",
    "modality": "text"|"table"|"image",
    "content": "...",
    "embedding": [0.12, 0.34, ...],
    "attention_score": 1.2,
    "metadata": {
      "file_name": "MyDoc.pdf",
      ...
    }
  },
  ...
]

Make sure Neo4j is running, and you have the correct user & password set below.
"""

import json
import os
from neo4j import GraphDatabase, exceptions

# Path to the JSON that has "embedding" and "attention_score"
CHUNKS_JSON_PATH = "project/chunked_with_scores.json"

# Neo4j connection config
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASS = "Neo4j420"  # Replace with your actual password

# If True, we remove existing data with MATCH (n) DETACH DELETE n
CLEAR_OLD_DATA = True

def store_chunks_in_neo4j(chunks, uri, user, password, clear_first=True):
    """
    Connects to Neo4j, optionally clears old data, 
    and creates :Chunk nodes for each chunk, storing chunk_id, modality, etc.
    If metadata.file_name is present, merges a :Document node and links (doc)-[:HAS_CHUNK]->(chunk).
    """
    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        with driver.session() as session:
            # 1) Optionally clear old data
            if clear_first:
                print("Clearing existing nodes in Neo4j...")
                session.run("MATCH (n) DETACH DELETE n")

            # 2) We can create a uniqueness constraint if desired
            # session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE")

            # We'll keep a count of how many chunks we create
            chunk_count = 0

            for c in chunks:
                chunk_id = c.get("chunk_id", "")
                modality = c.get("modality", "")
                content = c.get("content", "")
                embedding = c.get("embedding", [])
                attention = c.get("attention_score", 1.0)
                metadata = c.get("metadata", {})
                file_name = metadata.get("file_name", None)

                # 3) Create or merge the chunk node
                # Using MERGE or CREATE is up to you. MERGE ensures we don't duplicate chunk_id.
                create_chunk_query = """
                MERGE (chunk:Chunk {chunk_id: $chunk_id})
                ON CREATE SET chunk.modality = $modality,
                              chunk.content = $content,
                              chunk.embedding = $embedding,
                              chunk.attention_score = $attention_score
                ON MATCH SET chunk.modality = $modality,
                             chunk.content = $content,
                             chunk.embedding = $embedding,
                             chunk.attention_score = $attention_score
                """
                params = {
                    "chunk_id": chunk_id,
                    "modality": modality,
                    "content": content,
                    "embedding": embedding,
                    "attention_score": attention
                }
                session.run(create_chunk_query, params)
                chunk_count += 1

                # 4) If we want to link to a :Document node based on file_name
                if file_name:
                    # Merge doc
                    doc_query = """
                    MERGE (d:Document {doc_id: $file_name})
                    ON CREATE SET d.created_at = timestamp()
                    MERGE (d)-[:HAS_CHUNK]->(chunk)
                    """
                    session.run(doc_query, {"file_name": file_name, "chunk_id": chunk_id})

            print(f"Created or updated {chunk_count} chunk nodes in Neo4j.")

    except exceptions.AuthError as auth_err:
        print(f"Neo4j authentication failed. Check your credentials. Error:\n{auth_err}")
    except exceptions.ServiceUnavailable as svc_err:
        print(f"Could not connect to Neo4j at {uri}. Is Neo4j running? Error:\n{svc_err}")
    except Exception as e:
        print(f"Unexpected error while storing chunks: {e}")
    finally:
        driver.close()

def main():
    # 1) Load the chunk data
    if not os.path.isfile(CHUNKS_JSON_PATH):
        print(f"Error: {CHUNKS_JSON_PATH} does not exist.")
        return

    with open(CHUNKS_JSON_PATH, "r") as f:
        chunks = json.load(f)

    print(f"Loaded {len(chunks)} chunks from '{CHUNKS_JSON_PATH}'.")

    # 2) Store them in Neo4j
    store_chunks_in_neo4j(
        chunks,
        uri=NEO4J_URI,
        user=NEO4J_USER,
        password=NEO4J_PASS,
        clear_first=CLEAR_OLD_DATA
    )

if __name__ == "__main__":
    main()
