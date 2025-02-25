"""
rag_query.py

An interactive Retrieval-Augmented Generation (RAG) pipeline that:
 1) Prompts the user for queries in a loop.
 2) Embeds each query using SentenceTransformer ("all-MiniLM-L6-v2" by default).
 3) Retrieves top-K similar chunks (from Neo4j) based on cosine similarity.
 4) Builds a prompt with context from those chunks.
 5) Invokes Ollama (using "ollama run <model>" with the prompt piped to STDIN)
    with the model "deepseek-r1:32b" (or another model if updated).
 6) Displays the LLM's answer.
 
Type "exit" or "quit" to end the session.

Dependencies:
    pip install sentence-transformers neo4j numpy
    Ensure Ollama is installed with the model "deepseek-r1:32b"
    Ensure Neo4j is running with the proper credentials.
"""

import os
import sys
import json
import argparse
import subprocess
import numpy as np

from neo4j import GraphDatabase, basic_auth
from sentence_transformers import SentenceTransformer

# Hard-coded Neo4j credentials (update if needed)
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASS = "Neo4j420"

# Ollama model name to use
OLLAMA_MODEL = "deepseek-r1:32b"
# SentenceTransformer model for embedding queries
EMBED_MODEL = "all-MiniLM-L6-v2"
# Number of top similar chunks to retrieve
TOP_K = 5

def cosine_similarity(vec1, vec2):
    """
    Compute cosine similarity between two float vectors.
    Returns a float between -1 and 1.
    """
    v1 = np.array(vec1, dtype=float)
    v2 = np.array(vec2, dtype=float)
    dot = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)

def fetch_all_chunks_from_neo4j():
    """
    Connects to Neo4j and retrieves all Chunk nodes that have an 'embedding' property.
    Returns a list of dictionaries with keys:
      chunk_id, embedding, content, modality, textual_modality.
    """
    driver = GraphDatabase.driver(NEO4J_URI, auth=basic_auth(NEO4J_USER, NEO4J_PASS))
    chunks = []
    with driver.session() as session:
        query = """
        MATCH (c:Chunk)
        WHERE c.embedding IS NOT NULL
        RETURN c.chunk_id AS chunk_id,
               c.embedding AS embedding,
               c.content AS content,
               c.modality AS modality,
               c.textual_modality AS textual_modality
        """
        result = session.run(query)
        for record in result:
            cdict = {
                "chunk_id": record["chunk_id"],
                "embedding": record["embedding"],
                "content": record["content"] or "",
                "modality": record["modality"],
                "textual_modality": record["textual_modality"]
            }
            chunks.append(cdict)
    driver.close()
    return chunks

def retrieve_top_k(query_embedding, all_chunks, k=TOP_K):
    """
    Computes cosine similarity between query_embedding and each chunk's embedding.
    Returns the top k chunk dictionaries sorted by descending similarity.
    """
    scored = []
    for ch in all_chunks:
        sim = cosine_similarity(query_embedding, ch["embedding"])
        scored.append((sim, ch))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for (sim, c) in scored[:k]]

def build_prompt(top_chunks, user_query):
    """
    Builds a prompt string with the top chunks as context, followed by the user's query.
    """
    context_lines = []
    for i, chunk in enumerate(top_chunks):
        context_lines.append(f"CHUNK #{i+1} (modality={chunk['modality']}):\n{chunk['content']}\n")
    context_str = "\n".join(context_lines)
    prompt = (
        "You are a helpful AI using the following context to answer the question.\n\n"
        f"CONTEXT:\n{context_str}\n"
        f"QUESTION: {user_query}\n\n"
        "ANSWER:"
    )
    return prompt

def call_ollama_stdin(prompt, model=OLLAMA_MODEL):
    """
    Calls the Ollama CLI with the specified model.
    The prompt is passed to STDIN (i.e., piped in) as Ollama expects.
    
    The correct invocation is:
      echo "<prompt>" | ollama run <model>
    """
    cmd = ["ollama", "run", model]
    result = subprocess.run(cmd, input=prompt, capture_output=True, text=True)
    if result.returncode != 0:
        print("[call_ollama_stdin] Ollama command failed:", result.stderr)
        return None
    return result.stdout.strip()

def interactive_session():
    """
    Starts an interactive loop, allowing the user to ask multiple queries.
    The user can type "exit" or "quit" to end the session.
    """
    print("=== Interactive RAG Query Session ===")
    print("Type your query, or 'exit' to quit.")

    # Load the query embedding model once
    print(f"[rag_query] Loading embedding model: {EMBED_MODEL}")
    emb_model = SentenceTransformer(EMBED_MODEL)

    # Fetch all chunk data from Neo4j once (we assume it doesn't change during the session)
    print("[rag_query] Fetching chunk data from Neo4j...")
    all_chunks = fetch_all_chunks_from_neo4j()
    print(f"[rag_query] Retrieved {len(all_chunks)} chunks from DB.")

    if not all_chunks:
        print("[rag_query] No chunks found. Exiting.")
        return

    while True:
        user_query = input("\nEnter your query (or type 'exit' to quit): ").strip()
        if user_query.lower() in ("exit", "quit", ""):
            print("Exiting interactive session.")
            break

        # Embed the user query
        print("[rag_query] Embedding user query...")
        query_emb = emb_model.encode(user_query).tolist()

        # Retrieve top-K similar chunks
        top_k = retrieve_top_k(query_emb, all_chunks, k=TOP_K)
        print(f"[rag_query] Using top {len(top_k)} chunks as context...")

        # Build prompt
        prompt_text = build_prompt(top_k, user_query)

        # Call Ollama with the prompt piped via stdin
        print(f"[rag_query] Invoking Ollama with model '{OLLAMA_MODEL}'...")
        answer = call_ollama_stdin(prompt_text, model=OLLAMA_MODEL)

        if answer is None:
            print("[rag_query] Ollama call failed or returned no result.")
        else:
            print("\n=== LLM Answer ===")
            print(answer)
            print("==================")

def main():
    parser = argparse.ArgumentParser(description="RAG Query using Ollama (interactive mode).")
    parser.add_argument("--query", type=str, nargs="?",
                        help="User query. If provided, a single query will be processed and the program will exit. Otherwise, an interactive session starts.")
    args = parser.parse_args()

    if args.query:
        # Process a single query and exit
        user_query = args.query.strip()
        if not user_query:
            print("Empty query. Exiting.")
            return

        print(f"[rag_query] Loading embedding model: {EMBED_MODEL}")
        emb_model = SentenceTransformer(EMBED_MODEL)
        print("[rag_query] Embedding user query...")
        query_emb = emb_model.encode(user_query).tolist()
        print("[rag_query] Fetching chunk data from Neo4j...")
        all_chunks = fetch_all_chunks_from_neo4j()
        if not all_chunks:
            print("[rag_query] No chunks found. Exiting.")
            return
        top_k = retrieve_top_k(query_emb, all_chunks, k=TOP_K)
        print(f"[rag_query] Using top {len(top_k)} chunks as context...")
        prompt_text = build_prompt(top_k, user_query)
        print(f"[rag_query] Invoking Ollama with model '{OLLAMA_MODEL}'...")
        answer = call_ollama_stdin(prompt_text, model=OLLAMA_MODEL)
        if answer is None:
            print("[rag_query] Ollama call failed or returned no result.")
        else:
            print("\n=== LLM Answer ===")
            print(answer)
            print("==================")
    else:
        # Start interactive session
        interactive_session()

if __name__ == "__main__":
    main()
