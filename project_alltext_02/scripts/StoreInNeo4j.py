"""
StoreInNeo4j.py

This script connects to Neo4j at a fixed URI with hard-coded user/password:
  NEO4J_USER = "neo4j"
  NEO4J_PASS = "Neo4j420"

It reads 'embedded_data.json' by default, merges Document and Chunk nodes,
and stores chunk properties (including embeddings).

Structure of input JSON:
{
  "files": [
    {
      "file_name": "...",
      "chunks": [
        {
          "chunk_id": "...",
          "modality": "...",
          "content": "...",
          "embedding": [...],
          "metadata": "...(or dictionary) ...",
          "textual_modality": "..."
        },
        ...
      ]
    },
    ...
  ]
}

Usage:
    python StoreInNeo4j.py

Dependencies:
    pip install neo4j
Requires a running Neo4j instance on bolt://localhost:7687,
with user=neo4j, pass=Neo4j420.
"""

import os
import json
from neo4j import GraphDatabase, basic_auth
import sys

# Hard-coded Neo4j credentials
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASS = "Neo4j420"

# If you'd like to optionally clear old data, set CLEAR_OLD_DATA to True:
CLEAR_OLD_DATA = False

# Path to your default input JSON (the final embedded data)
INPUT_JSON = "embedded_data.json"


def store_in_neo4j():
    """
    1) Reads the embedded JSON from INPUT_JSON.
    2) Connects to Neo4j using NEO4J_URI/USER/PASS.
    3) Optionally clears old data if CLEAR_OLD_DATA is True.
    4) Creates constraints for doc_id and chunk_id uniqueness.
    5) Merges Document nodes & merges Chunk nodes + a relationship:
          (Document)-[:HAS_CHUNK]->(Chunk)
    """
    if not os.path.isfile(INPUT_JSON):
        print(f"[StoreInNeo4j] Cannot find JSON file '{INPUT_JSON}'")
        sys.exit(1)

    # Load the data
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Validate structure
    if "files" not in data or not isinstance(data["files"], list):
        print("[StoreInNeo4j] JSON must have 'files' key with a list of file objects.")
        sys.exit(1)

    files_list = data["files"]

    print(f"[StoreInNeo4j] Connecting to {NEO4J_URI} with user '{NEO4J_USER}'...")
    driver = GraphDatabase.driver(NEO4J_URI, auth=basic_auth(NEO4J_USER, NEO4J_PASS))

    # Optionally clear old data
    if CLEAR_OLD_DATA:
        print("[StoreInNeo4j] Clearing old DB data (MATCH (n) DETACH DELETE n)")
        with driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")

    # Create constraints if not exist
    with driver.session() as session:
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (d:Document) REQUIRE d.doc_id IS UNIQUE")
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE")
    print("[StoreInNeo4j] Ensured constraints on doc_id and chunk_id.")

    doc_count = 0
    chunk_count = 0

    with driver.session() as session:
        for file_info in files_list:
            file_name = file_info.get("file_name", "")
            if not file_name.strip():
                continue

            # Merge the Document node
            merge_doc_query = """
            MERGE (d:Document { doc_id: $doc_id })
            ON CREATE SET d.created_at = timestamp()
            """
            session.run(merge_doc_query, {"doc_id": file_name})
            doc_count += 1

            # For each chunk
            for c in file_info.get("chunks", []):
                chunk_id = c.get("chunk_id", "")
                if not chunk_id.strip():
                    continue

                # We'll store or update these properties
                modality = c.get("modality", "")
                content = c.get("content", "")
                embedding = c.get("embedding", [])
                textual_modality = c.get("textual_modality", "")

                # For metadata, store as JSON string or direct map if small enough
                import json as pyjson
                metadata = c.get("metadata", {})
                metadata_str = pyjson.dumps(metadata, ensure_ascii=False)

                # Merge chunk
                merge_chunk_query = """
                MERGE (ch:Chunk { chunk_id: $chunk_id })
                ON CREATE SET 
                    ch.created_at = timestamp(),
                    ch.modality = $modality,
                    ch.content = $content,
                    ch.embedding = $embedding,
                    ch.textual_modality = $textual_modality,
                    ch.metadata = $metadata
                ON MATCH SET
                    ch.modality = $modality,
                    ch.content = $content,
                    ch.embedding = $embedding,
                    ch.textual_modality = $textual_modality,
                    ch.metadata = $metadata
                """
                session.run(merge_chunk_query, {
                    "chunk_id": chunk_id,
                    "modality": modality,
                    "content": content,
                    "embedding": embedding,
                    "textual_modality": textual_modality,
                    "metadata": metadata_str
                })

                # Link document to chunk
                link_query = """
                MATCH (d:Document { doc_id: $doc_id }),
                      (ch:Chunk { chunk_id: $chunk_id })
                MERGE (d)-[:HAS_CHUNK]->(ch)
                """
                session.run(link_query, {
                    "doc_id": file_name,
                    "chunk_id": chunk_id
                })

                chunk_count += 1

    driver.close()
    print(f"[StoreInNeo4j] Done. Created/updated {doc_count} Document nodes, {chunk_count} Chunk merges.")


if __name__ == "__main__":
    store_in_neo4j()
