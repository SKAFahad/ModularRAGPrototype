#!/usr/bin/env python3
"""
rag_inference_multi.py
----------------------
A Retrieval-Augmented Generation (RAG) script that allows multiple queries in one session.

Process Overview:
  1. Loads the text embedding model ("all-MiniLM-L6-v2", 384-d).
  2. Connects to Neo4j, retrieves all chunk embeddings once.
  3. Enters a loop:
      - Prompts the user for a query (type "exit" or press enter on empty to quit).
      - Embeds the query.
      - Finds top-K similar chunks by cosine similarity.
      - Calls Ollama with "deepseek-r1:14b" to generate the final answer.
      - Prints the answer.
  4. When the user types "exit" or an empty query, the script ends.

Prerequisites:
  - chunked embeddings generated with "all-MiniLM-L6-v2" => 384-d.
  - Neo4j is running, with password "Neo4j420".
  - Ollama is installed, and the local model "deepseek-r1:14b" is accessible via:
      ollama run deepseek-r1:14b "Hello"
"""

import os
import subprocess
import numpy as np
from neo4j import GraphDatabase, exceptions
from sentence_transformers import SentenceTransformer

#############################################
#  CONFIGURATION
#############################################
NEO4J_URI  = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASS = "Neo4j420"
TOP_K      = 5
OLLAMA_MODEL = "deepseek-r1:14b"

#############################################
#  EMBEDDING & SIMILARITY
#############################################

def load_text_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """
    Loads a SentenceTransformers model for query embedding.
    """
    print(f"[INFO] Loading text embedding model '{model_name}'...")
    model = SentenceTransformer(model_name)
    print("[INFO] Embedding model loaded.")
    return model

def embed_query(query: str, model: SentenceTransformer) -> np.ndarray:
    """
    Embeds the user query into a 384-d vector with the loaded model.
    """
    print("[INFO] Embedding user query...")
    query_embedding = model.encode(query, convert_to_numpy=True)
    print(f"[INFO] Query embedding shape: {query_embedding.shape}")
    return query_embedding

def cosine_similarity(vec1, vec2) -> float:
    """
    Computes the cosine similarity between two numeric vectors.
    """
    v1 = np.array(vec1, dtype=float)
    v2 = np.array(vec2, dtype=float)
    dot_val = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_val / (norm1 * norm2)

#############################################
#  OLLAMA INTEGRATION
#############################################

def ollama_generate(prompt: str, model_name: str = OLLAMA_MODEL) -> str:
    """
    Calls Ollama (without the '-m' flag) in the form:
        ollama run deepseek-r1:14b "some prompt"
    """
    print(f"[INFO] Calling ollama with model '{model_name}'...")

    try:
        result = subprocess.run(
            ["ollama", "run", model_name, prompt],
            text=True,
            capture_output=True
        )
        if result.returncode != 0:
            print(f"[ERROR] Ollama returned exit code: {result.returncode}")
            print(f"[ERROR] stderr: {result.stderr}")
            return "Error generating response from Ollama."
        return result.stdout.strip()
    except FileNotFoundError:
        return "[ERROR] 'ollama' CLI not found. Please ensure Ollama is installed and on PATH."
    except Exception as e:
        return f"[ERROR] Unexpected error calling ollama: {e}"

def generate_answer_ollama(query: str, top_chunks: list) -> str:
    """
    Builds a prompt from top chunks + user query, then calls ollama_generate().
    """
    context_texts = []
    for i, (chunk, score) in enumerate(top_chunks, start=1):
        context_texts.append(f"--- Chunk {i} (similarity={score:.4f}) ---\n{chunk['content']}\n")

    combined_context = "\n".join(context_texts)

    prompt_text = f"""You are a helpful AI assistant. Below is relevant context, followed by the user query.

Context:
{combined_context}

User Query: {query}

Answer:
"""
    return ollama_generate(prompt_text)

#############################################
#  NEO4J CHUNK RETRIEVAL
#############################################

def retrieve_chunks_from_neo4j(uri: str, user: str, password: str):
    """
    Connects to Neo4j and retrieves chunk nodes + embeddings. Returns a list of dicts.
    """
    driver = GraphDatabase.driver(uri, auth=(user, password))
    query = """
    MATCH (ch:Chunk)
    RETURN ch.chunk_id AS chunk_id,
           ch.content AS content,
           ch.embedding AS embedding,
           ch.modality AS modality
    """
    chunks = []
    try:
        with driver.session() as session:
            results = session.run(query)
            for record in results:
                chunks.append({
                    "chunk_id": record["chunk_id"],
                    "content" : record["content"],
                    "embedding": record["embedding"],
                    "modality" : record["modality"]
                })
        print(f"[INFO] Retrieved {len(chunks)} chunks from Neo4j.")
    except exceptions.AuthError as e:
        print(f"[ERROR] Neo4j auth error: {e}")
    except exceptions.ServiceUnavailable as e:
        print(f"[ERROR] Neo4j service unavailable: {e}")
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
    finally:
        driver.close()
    return chunks

def find_top_k_chunks(query_embedding: np.ndarray, chunks: list, top_k: int = TOP_K):
    """
    Computes cosine similarity between query embedding and each chunk's embedding,
    returning top-K chunks in descending similarity order.
    """
    sims = []
    for c in chunks:
        emb = c.get("embedding", [])
        if not emb:
            continue
        sim_val = cosine_similarity(query_embedding, emb)
        sims.append((c, sim_val))

    sims.sort(key=lambda x: x[1], reverse=True)
    return sims[:top_k]

#############################################
#  MAIN - MULTI QUESTION LOOP
#############################################

def main():
    # 1) Load the embedding model (384-d)
    embedding_model = load_text_model("all-MiniLM-L6-v2")

    # 2) Retrieve chunks from Neo4j (just once)
    chunks = retrieve_chunks_from_neo4j(NEO4J_URI, NEO4J_USER, NEO4J_PASS)
    if not chunks:
        print("[ERROR] No chunks found in Neo4j. Exiting.")
        return

    # 3) Multi-query loop
    while True:
        user_query = input("\nEnter your query (type 'exit' or press Enter empty to quit): ").strip()
        if not user_query or user_query.lower() == "exit":
            print("[INFO] Exiting multi-query RAG session.")
            break

        # a) Embed the user query
        query_emb = embed_query(user_query, embedding_model)

        # b) Find top-K results
        top_results = find_top_k_chunks(query_emb, chunks, TOP_K)
        if not top_results:
            print("[INFO] No similar chunks found.")
            continue

        # c) Generate the answer via Ollama
        answer = generate_answer_ollama(user_query, top_results)

        # d) Print the final answer
        print("\n==================== RAG Answer (Ollama) ====================")
        print(answer)
        print("=============================================================")

if __name__ == "__main__":
    main()
