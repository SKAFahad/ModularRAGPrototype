"""
rag_query.py

Provides an **interactive question-answering** (Q&A) session using:
  - A local Neo4j database that stores chunk nodes (with embeddings, topics, etc.).
  - A local LLM (DeepSeek R1 via Ollama) to produce final answers.

Key Steps:
1) The user types a question at the prompt.
2) We embed the question locally (or we can skip if you store query embeddings in Neo4j 5+).
3) We do retrieval from Neo4j to get top-K chunk nodes. In a basic approach, 
   we might just do an O(N) loop to compute similarities. 
   If you prefer a more advanced approach (like "hybrid retrieval" with topic), 
   you can adapt that or import from a separate `hybrid_retriever.py`.
4) Build a text prompt combining these retrieved chunks plus the user question.
5) Call the local LLM (DeepSeek R1) using Ollama CLI. 
6) Show the generated answer, then let the user ask another question.
7) Repeat until "exit" or "quit" is typed.

You'll need:
 - `neo4j` Python driver
 - `sentence_transformers` (unless skipping local query embedding)
 - `ollama` installed for running local LLM commands (or adapt to your environment).

Usage:
------
  python rag_query.py
  # Type queries, type "exit" or "quit" to end.
"""

import os
import sys
import subprocess
import numpy as np
from neo4j import GraphDatabase, basic_auth

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    # If you do a direct vector index approach or skip local embedding, 
    # you can remove or handle this gracefully.
    pass

# Hard-coded or environment-based credentials for Neo4j
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASS = "Neo4j420"  # Adjust to your actual password or load from env

############################
# LLM call (DeepSeek R1) 
############################
def call_ollama(prompt: str, model="deepseek-r1:32b") -> str:
    """
    Calls the local LLM via the Ollama CLI with the chosen model 
    (e.g. deepseek-r1:32b). Adjust if you have a different approach to 
    calling the LLM.

    :param prompt: The text prompt (context + question)
    :param model: The local model name, e.g. 'deepseek-r1:32b'
    :return: The LLM's textual answer
    """
    cmd = ["ollama", "run", model]
    # We'll pipe the prompt into STDIN
    result = subprocess.run(cmd, input=prompt, capture_output=True, text=True)
    if result.returncode != 0:
        err_msg = f"[call_ollama] LLM call error: {result.stderr}"
        print(err_msg)
        return "Error calling the local LLM. Check logs."
    return result.stdout.strip()

############################
# Embedding query locally 
############################
def embed_query(user_question, model_name="all-MiniLM-L6-v2"):
    """
    Simple function to embed the user query using a local SentenceTransformer 
    model. The script expects you to have installed sentence-transformers.

    :param user_question: The text typed by the user
    :param model_name: e.g. "all-MiniLM-L6-v2"
    :return: np array for the query embedding
    """
    model = SentenceTransformer(model_name)
    emb = model.encode([user_question])[0]  # shape (embedding_dim,)
    return emb

############################
# Retrieving chunks from Neo4j
############################
def retrieve_topk_chunks(driver, query_emb, k=5):
    """
    Example approach to fetch all chunk embeddings from Neo4j, compute local 
    cosine similarity to the user query, and return top-K matches. 
    For large data, consider a vector index or a separate 'hybrid' approach.

    :param driver: The Neo4j driver
    :param query_emb: np array of shape (dim,) for user question
    :param k: how many top chunks to return
    :return: a list of (chunk_id, content, sim)
    """
    def cos_sim(a, b):
        a = np.array(a, dtype=float)
        b = np.array(b, dtype=float)
        dot = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    with driver.session() as session:
        q = """
        MATCH (c:Chunk)
        WHERE c.embedding IS NOT NULL AND size(c.embedding) > 0
        RETURN c.chunk_id AS chunk_id, c.content AS content, c.embedding AS embedding
        """
        result = session.run(q)
        rows = [(r["chunk_id"], r["content"], r["embedding"]) for r in result]

    chunk_sims = []
    for cid, content, emb in rows:
        sim_val = cos_sim(query_emb, emb)
        chunk_sims.append((cid, content, sim_val))

    # sort by descending sim, pick top-k
    chunk_sims.sort(key=lambda x: x[2], reverse=True)
    return chunk_sims[:k]

############################
# Prompt building
############################
def build_prompt(top_chunks, user_question):
    """
    Combine top chunk contents plus the user question into a single text prompt 
    for the LLM.

    :param top_chunks: a list of (chunk_id, content, sim)
    :param user_question: the user's query string
    :return: string prompt
    """
    context_str = ""
    for i, (cid, content, sim_val) in enumerate(top_chunks):
        context_str += (f"CHUNK #{i+1} (id={cid}, sim={sim_val:.3f}):\n"
                        f"{content}\n\n")

    prompt_text = (
        "You are a helpful AI. Use ONLY the context below to answer the question.\n\n"
        f"CONTEXT:\n{context_str}"
        f"QUESTION: {user_question}\n\n"
        "ANSWER:"
    )
    return prompt_text

############################
# Interactive loop
############################
def interactive_session(driver, embedding_model="all-MiniLM-L6-v2"):
    """
    Repeatedly ask the user for queries, run the pipeline for each:
    1) embed query
    2) retrieve top-5 chunks
    3) build prompt
    4) call LLM
    5) print answer

    Type 'exit' or 'quit' to end.
    """
    print("=== Interactive RAG Q&A Session ===")
    print("(Type 'exit' or 'quit' to end)")

    while True:
        user_q = input("\nYour question: ").strip()
        if user_q.lower() in ("exit", "quit"):
            print("[rag_query] Exiting session.")
            break

        # 1) embed
        qvec = embed_query(user_q, model_name=embedding_model)

        # 2) retrieve top-5
        top_k = retrieve_topk_chunks(driver, qvec, k=5)

        # 3) build prompt
        prompt_txt = build_prompt(top_k, user_q)

        # 4) call LLM
        llm_answer = call_ollama(prompt_txt, model="deepseek-r1:32b")

        # 5) print
        print("\n=== LLM Answer ===")
        print(llm_answer)
        print("===")

def main():
    """
    Entry point: connect to Neo4j, start an interactive Q&A loop.
    After user ends, close the driver.
    """
    print(f"[rag_query] Connecting to Neo4j at {NEO4J_URI} with user '{NEO4J_USER}'")
    driver = GraphDatabase.driver(NEO4J_URI, auth=basic_auth(NEO4J_USER, NEO4J_PASS))

    interactive_session(driver, embedding_model="all-MiniLM-L6-v2")

    driver.close()
    print("[rag_query] Done.")


if __name__ == "__main__":
    main()
