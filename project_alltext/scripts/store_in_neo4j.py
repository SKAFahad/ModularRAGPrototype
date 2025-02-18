#!/usr/bin/env python3
"""
store_in_neo4j.py
-----------------
This script reads a JSON file containing text chunks (with their embeddings and metadata)
and stores each chunk as a node in a Neo4j graph database. Additionally, if a chunk's metadata
contains a file name, a corresponding Document node is created and linked to the chunk via a
HAS_CHUNK relationship.

Each chunk node is stored with the following properties:
  - chunk_id: A unique identifier for the chunk.
  - content: The text content extracted from the source file.
  - embedding: The embedding vector (a list of floats) for the chunk.
  - modality: Although all data has been converted to text, this property can be used to store
              the original modality if desired.

Document nodes are created using the file_name from the chunk's metadata. These nodes are
linked to their corresponding chunk nodes.

Before running this script, ensure that:
  - Neo4j is running and accessible (e.g., bolt://localhost:7687).
  - The user credentials for Neo4j are correctly set.
  - The JSON file (e.g., "chunked_with_embeddings.json") is present and correctly formatted.
"""

import json
import os
from neo4j import GraphDatabase, exceptions

# Path to the JSON file that contains our chunks with embeddings
CHUNKS_JSON_PATH = "project_alltext/chunked_with_embeddings.json"

# Neo4j connection configuration: adjust these values to match your Neo4j instance
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASS = "Neo4j420"  # Replace with your actual Neo4j password

# Option to clear existing data from the database before inserting new data
CLEAR_OLD_DATA = True

def store_chunks_in_neo4j(chunks, uri, user, password, clear_first=True):
    """
    Connects to the Neo4j database and stores each chunk as a node with label :Chunk.
    Additionally, if a chunk has a file name in its metadata, a :Document node is created and
    linked to the chunk via a HAS_CHUNK relationship.

    Args:
        chunks (list): List of dictionaries, each representing a text chunk.
        uri (str): The connection URI for Neo4j (e.g., "bolt://localhost:7687").
        user (str): Neo4j username.
        password (str): Neo4j password.
        clear_first (bool): If True, clears all nodes in the database before inserting new data.

    The function logs progress to the console and handles connection closure properly.
    """
    # Create a driver to connect to the Neo4j database using provided credentials.
    driver = GraphDatabase.driver(uri, auth=(user, password))
    
    try:
        # Open a new session with the database.
        with driver.session() as session:
            # Optionally clear old data from the database.
            if clear_first:
                print("Clearing existing data from Neo4j...")
                session.run("MATCH (n) DETACH DELETE n")
                print("Old data cleared.")

            # Loop through each chunk in the list.
            for chunk in chunks:
                # Extract relevant properties from the chunk.
                chunk_id = chunk.get("chunk_id")
                content = chunk.get("content", "")
                embedding = chunk.get("embedding", [])
                modality = chunk.get("modality", "text")
                metadata = chunk.get("metadata", {})
                file_name = metadata.get("file_name", None)

                # Create or update the Chunk node with the chunk_id as a unique identifier.
                create_chunk_query = """
                MERGE (ch:Chunk {chunk_id: $chunk_id})
                ON CREATE SET ch.content = $content,
                              ch.embedding = $embedding,
                              ch.modality = $modality
                ON MATCH SET ch.content = $content,
                             ch.embedding = $embedding,
                             ch.modality = $modality
                """
                # Run the query with provided parameters.
                session.run(create_chunk_query, {
                    "chunk_id": chunk_id,
                    "content": content,
                    "embedding": embedding,
                    "modality": modality
                })

                # If the metadata includes a file name, create a Document node and link it.
                if file_name:
                    create_document_query = """
                    MERGE (doc:Document {doc_id: $file_name})
                    ON CREATE SET doc.created_at = timestamp()
                    MERGE (doc)-[:HAS_CHUNK]->(ch)
                    """
                    session.run(create_document_query, {
                        "file_name": file_name,
                        "chunk_id": chunk_id
                    })

            print(f"Stored {len(chunks)} chunk nodes in Neo4j.")

    except exceptions.AuthError as auth_err:
        print(f"Authentication error: {auth_err}. Please verify your Neo4j credentials.")
    except exceptions.ServiceUnavailable as svc_err:
        print(f"Service unavailable: {svc_err}. Ensure Neo4j is running at {uri}.")
    except Exception as e:
        print(f"An error occurred while storing chunks in Neo4j: {e}")
    finally:
        # Ensure the driver is closed even if an error occurs.
        driver.close()
        print("Neo4j connection closed.")

def main():
    """
    Main function that:
      1. Loads the JSON file with chunked data (including embeddings).
      2. Calls the store_chunks_in_neo4j() function to insert data into Neo4j.
    """
    # Check if the input JSON file exists.
    if not os.path.isfile(CHUNKS_JSON_PATH):
        print(f"Error: File '{CHUNKS_JSON_PATH}' not found. Please generate the embeddings first.")
        return

    # Load the JSON data from file.
    with open(CHUNKS_JSON_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"Loaded {len(chunks)} chunks from '{CHUNKS_JSON_PATH}'.")

    # Store the chunks in the Neo4j database.
    store_chunks_in_neo4j(chunks, NEO4J_URI, NEO4J_USER, NEO4J_PASS, clear_first=CLEAR_OLD_DATA)

if __name__ == "__main__":
    main()
