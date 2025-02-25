#!/usr/bin/env python3
"""
rag_inference.py
----------------
A Retrieval-Augmented Generation (RAG) script that:
  1. Prompts the user for a query.
  2. Embeds the query with a 384-dim model ("all-MiniLM-L6-v2").
  3. Retrieves top-K similar chunks from Neo4j.
  4. Calls Ollama to run the "deepseek-r1:14b" model,
     passing the final prompt as positional arguments (instead of -m).

Prerequisites:
  - Ollama is installed and the model "deepseek-r1:14b" is available.
  - You can run:   ollama run deepseek-r1:14b "Hello"   without errors.
"""

import os
import subprocess
import numpy as np
from neo4j import GraphDatabase, exceptions
from sentence_transformers import SentenceTransformer

#############################################
#     NEO4J CONFIG & GLOBAL SETTINGS        #
#############################################
NEO4J_URI  = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASS = "Neo4j420"  # your password
TOP_K      = 5           # how many chunks to retrieve
OLLAMA_MODEL = "deepseek-r1:14b"  # the local Ollama model name

#############################################
#     EMBEDDING & SIMILARITY FUNCTIONS      #
#############################################

def load_text_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """
    Loads a SentenceTransformers model that produces 384-d embeddings.
    """
    print(f"[INFO] Loading text embedding model '{model_name}'...")
    model = SentenceTransformer(model_name)
    print("[INFO] Model loaded successfully.")
    return model

def embed_query(query: str, model: SentenceTransformer) -> np.ndarray:
    """
    Embeds the user query using the loaded model (384-d).
    """
    print("[INFO] Embedding user query...")
    query_embedding = model.encode(query, convert_to_numpy=True)
    print(f"[INFO] Query embedding shape: {query_embedding.shape}")
    return query_embedding

def cosine_similarity(vec1, vec2) -> float:
    """
    Computes cosine similarity between two numeric vectors.
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
#        NEO4J CHUNK RETRIEVAL             #
#############################################

def retrieve_chunks_from_neo4j():
    """
    Connects to Neo4j, retrieves all chunk nodes + embeddings, returns list of dicts.
    """
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
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
                chunk = {
                    "chunk_id": record["chunk_id"],
                    "content" : record["content"],
                    "embedding": record["embedding"],
                    "modality": record["modality"]
                }
                chunks.append(chunk)
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
    Computes cosine similarity between query_embedding and each chunk's embedding,
    returns the top-K in descending order of similarity.
    """
    similarities = []
    for c in chunks:
        emb = c.get("embedding", [])
        if not emb:
            continue
        sim_val = cosine_similarity(query_embedding, emb)
        similarities.append((c, sim_val))

    # Sort by similarity desc
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]

#############################################
#     OLLAMA INTEGRATION                    #
#############################################

def ollama_generate(prompt: str, model_name: str = OLLAMA_MODEL) -> str:
    """
    Calls the Ollama CLI in a way that doesn't use the '-m' shorthand flag.
    Syntax: ollama run deepseek-r1:14b "some prompt text"

    Args:
      prompt (str): the text to feed to the model
      model_name (str): the local name for the model in Ollama
    Returns:
      str: the model's text response or an error
    """
    print(f"[INFO] Calling ollama with model '{model_name}'...")

    # We'll pass the model name as a positional argument, then the prompt as another argument.
    # e.g. ollama run deepseek-r1:14b "Hello"
    try:
        result = subprocess.run(
            ["ollama", "run", model_name, prompt],
            text=True,
            capture_output=True
        )
        if result.returncode != 0:
            print(f"[ERROR] Ollama returned a non-zero exit code: {result.returncode}")
            print(f"[ERROR] stderr: {result.stderr}")
            return "Error generating response from Ollama."
        return result.stdout.strip()
    except FileNotFoundError:
        return "[ERROR] 'ollama' CLI not found. Please ensure Ollama is installed and on PATH."
    except Exception as e:
        return f"[ERROR] Unexpected error calling ollama: {e}"

def generate_answer_ollama(query: str, top_chunks: list) -> str:
    """
    Builds a prompt from top chunks, calls ollama_generate, returns the text answer.
    """
    context_texts = []
    for i, (chunk, score) in enumerate(top_chunks, start=1):
        context_texts.append(f"--- Chunk {i} (similarity={score:.4f}) ---\n{chunk['content']}\n")
    combined_context = "\n".join(context_texts)

    prompt_text = f"""You are a helpful AI assistant. Below is relevant context, followed by a user query:

Context:
{combined_context}

User Query: {query}

Answer:
"""
    return ollama_generate(prompt_text)

#############################################
#                 MAIN                      #
#############################################

def main():
    user_query = input("Enter your query: ").strip()
    if not user_query:
        print("[INFO] No query provided. Exiting.")
        return

    # 1) Load the SentenceTransformers model for a 384-d embedding
    embedding_model = load_text_model("all-MiniLM-L6-v2")
    # 2) Embed the query
    query_emb = embed_query(user_query, embedding_model)
    # 3) Retrieve chunks from Neo4j
    chunks = retrieve_chunks_from_neo4j()
    if not chunks:
        print("[ERROR] No chunks found. Exiting.")
        return
    # 4) Find top-K chunk matches
    top_results = find_top_k_chunks(query_emb, chunks, TOP_K)
    if not top_results:
        print("[INFO] No similar chunks found. Exiting.")
        return
    # 5) Generate an answer using Ollama
    answer = generate_answer_ollama(user_query, top_results)
    # 6) Print the final answer
    print("\n==================== RAG Answer (Ollama) ====================")
    print(answer)
    print("=============================================================")

if __name__ == "__main__":
    main()
