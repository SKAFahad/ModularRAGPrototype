"""
store_in_neo4j.py

This module reads an embedded data JSON (produced by embedding_text.py) containing
files -> chunks with embeddings, and stores them in Neo4j. Each file becomes a
Document node, and each chunk becomes a Chunk node with properties:

- chunk_id
- modality
- content
- embedding (list of floats)
- textual_modality
- metadata (JSON or stringified dict)

We then create a relationship (Document)-[:HAS_CHUNK]->(Chunk). This sets the stage for
further computations (e.g., linking chunks with SIMILAR_TO edges, topic modeling, etc.).

Guiding Principles (from our RAG discussions):
1. **Offline & local**: Connect to a local Neo4j instance, no external services.
2. **Detailed commentary**: Provide future devs clarity on how/why data is merged.
3. **Consistent Data Model**: Each JSON file entry has "file_name" and "chunks". We
   create a Document node for the file, Chunk nodes for each chunk, and link them.
4. **Handling duplicates**: Use MERGE in Cypher for doc_id and chunk_id to avoid duplicates.
5. **CLI**: Typically run `python store_in_neo4j.py <embedded_data.json> [--clear]`
   to optionally clear old data before ingesting new.

Typical Embedded JSON (embedded_data.json):
{
  "files": [
    {
      "file_name": "somefile.txt",
      "chunks": [
        {
          "chunk_id": "somefile.txt_par_0",
          "modality": "text",
          "content": "some text chunk here...",
          "embedding": [0.123, -0.045, ...],
          "metadata": {...},
          "textual_modality": "wrapped_paragraph"
        },
        ...
      ]
    },
    ...
  ]
}

Resulting Graph in Neo4j:
(:Document {doc_id: file_name})
(:Chunk {chunk_id: chunk_id, content:..., embedding:..., ...})
(:Document)-[:HAS_CHUNK]->(:Chunk)

Usage Example:
    python store_in_neo4j.py embedded_data.json
    # Optionally, pass '--clear' to remove old data: python store_in_neo4j.py embedded_data.json --clear
"""

import os
import sys
import json
from neo4j import GraphDatabase, basic_auth


# Hard-coded or configurable
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASS = "Neo4j420"  # Replace with your actual password


def store_in_neo4j(
    input_json: str,
    clear_old_data: bool = False
):
    """
    Reads the JSON file at input_json, which should have the structure:
      {
        "files": [
          {
            "file_name": "somefile.txt",
            "chunks": [
              {
                "chunk_id": "somefile.txt_par_0",
                "modality": "text",
                "content": "...",
                "embedding": [...],
                "metadata": {... or string},
                "textual_modality": "wrapped_paragraph"
              },
              ...
            ]
          },
          ...
        ]
      }
    Connects to Neo4j, optionally clears old data, creates Document and Chunk nodes,
    and merges relationships.

    :param input_json: Path to embedded_data.json
    :type input_json: str

    :param clear_old_data: If True, runs MATCH (n) DETACH DELETE n to clear the entire DB.
    :type clear_old_data: bool

    :return: None
    """

    # 1) Load the JSON
    if not os.path.isfile(input_json):
        raise FileNotFoundError(f"[store_in_neo4j] Cannot find JSON: {input_json}")

    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Validate basic structure
    if "files" not in data or not isinstance(data["files"], list):
        raise ValueError("[store_in_neo4j] Input JSON must contain { 'files': [ ... ] }.")

    files_list = data["files"]

    # 2) Connect to Neo4j
    print(f"[store_in_neo4j] Connecting to {NEO4J_URI} with user '{NEO4J_USER}'...")
    driver = GraphDatabase.driver(NEO4J_URI, auth=basic_auth(NEO4J_USER, NEO4J_PASS))

    # 3) Optionally clear old data
    if clear_old_data:
        with driver.session() as session:
            print("[store_in_neo4j] Clearing all data in the database (MATCH (n) DETACH DELETE n)...")
            session.run("MATCH (n) DETACH DELETE n")

    # 4) Create constraints for doc_id and chunk_id
    with driver.session() as session:
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (d:Document) REQUIRE d.doc_id IS UNIQUE")
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE")

    doc_count = 0
    chunk_count = 0

    # 5) Merge Document and Chunk nodes
    with driver.session() as session:
        for file_info in files_list:
            file_name = file_info.get("file_name")
            if not file_name:
                # skip if no file_name
                continue

            doc_count += 1
            # MERGE the Document node
            merge_doc_query = """
            MERGE (d:Document { doc_id: $doc_id })
            ON CREATE SET d.created_at = timestamp()
            """
            session.run(merge_doc_query, {"doc_id": file_name})

            chunks = file_info.get("chunks", [])
            for ch in chunks:
                chunk_id = ch.get("chunk_id")
                if not chunk_id:
                    # skip if no chunk_id
                    continue

                # Properties we store or update
                modality = ch.get("modality", "")
                content = ch.get("content", "")
                embedding = ch.get("embedding", [])  # list of floats
                textual_modality = ch.get("textual_modality", "")
                metadata = ch.get("metadata", {})
                # We'll store metadata as a JSON string or map. For Neo4j < 5, you might store it as string
                # If new Neo4j versions allow maps, you can store directly. We'll do a string to be safe.
                import json as pyjson
                metadata_str = pyjson.dumps(metadata, ensure_ascii=False)

                # Merge chunk node
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

                # Link doc->chunk
                link_doc_chunk_query = """
                MATCH (d:Document { doc_id: $doc_id })
                MATCH (ch:Chunk { chunk_id: $chunk_id })
                MERGE (d)-[:HAS_CHUNK]->(ch)
                """
                session.run(link_doc_chunk_query, {
                    "doc_id": file_name,
                    "chunk_id": chunk_id
                })

                chunk_count += 1

    driver.close()

    print(f"[store_in_neo4j] Done. Created/updated {doc_count} Document nodes and {chunk_count} Chunk merges.")


if __name__ == "__main__":
    """
    CLI usage:
      python store_in_neo4j.py <embedded_data.json> [--clear]

    If --clear is provided, the script will delete all data from Neo4j
    before ingesting new. Use with caution.
    """
    clear_flag = False
    if len(sys.argv) < 2:
        print("Usage: python store_in_neo4j.py <embedded_data.json> [--clear]")
        sys.exit(1)

    input_file = sys.argv[1]
    if "--clear" in sys.argv:
        clear_flag = True

    try:
        store_in_neo4j(input_file, clear_old_data=clear_flag)
    except Exception as e:
        print(f"Error in store_in_neo4j: {e}")
        sys.exit(1)
