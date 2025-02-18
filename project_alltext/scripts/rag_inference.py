#!/usr/bin/env python3
"""
rag_inference.py
----------------
A modular script demonstrating an end-to-end Retrieval-Augmented Generation (RAG) flow:
  1. Load a text embedding model (SentenceTransformers).
  2. Prompt the user for a query (or accept it as a function argument).
  3. Embed the query, then retrieve top-K similar chunks from Neo4j.
  4. Concatenate those chunks into a prompt.
  5. Invoke a large language model (deepseek-r1:14b or another LLM) to generate a final answer.

Prerequisites:
  - The chunked data has already been ingested and embedded (embedding_generation.py).
  - The chunk nodes and embeddings have been stored in Neo4j (store_in_neo4j.py).
  - Relationship scripts (compute_relationships.py) are optional but can enhance the knowledge graph.
  - The 'deepseek-r1:14b' model is locally available and can be loaded via Hugging Face or another interface.
  - The user has installed 'transformers' (pip install transformers) or an alternative library that can load the LLM.

Usage:
  python rag_inference.py
"""

import os
import numpy as np
from neo4j import GraphDatabase, exceptions
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM  # or the relevant loader for deepseek-r1:14b

#############################################
#          CONFIGURATION SECTION            #
#############################################

# Neo4j connection settings
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASS = "Neo4j420"

# LLM settings: adjust model_name to your local or HF-hub reference for deepseek-r1:14b
LLM_MODEL_NAME = "deepseek-r1:14b"  
# e.g. "path/to/deepseek-r1-14b", or a Hugging Face model ID if it’s published.

# Text embedding model used for both chunks and query
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Number of chunks to retrieve from the knowledge graph
TOP_K = 5

#############################################
#            HELPER FUNCTIONS               #
#############################################

def load_text_model(model_name: str = EMBEDDING_MODEL_NAME):
    """
    Loads the SentenceTransformers model used to embed text. This must match the model
    used to generate embeddings for your chunks, so the vector spaces align.
    """
    print(f"[INFO] Loading text embedding model '{model_name}'...")
    model = SentenceTransformer(model_name)
    print("[INFO] Embedding model loaded.")
    return model

def embed_query(query: str, model):
    """
    Embeds the user query using the SentenceTransformers model.
    Returns a NumPy array representing the query embedding.
    """
    embedding_vector = model.encode(query, convert_to_numpy=True)
    return embedding_vector

def cosine_similarity(vec1, vec2):
    """
    Computes the cosine similarity between two vectors (lists or np.ndarrays).
    """
    v1 = np.array(vec1, dtype=float)
    v2 = np.array(vec2, dtype=float)
    dot = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)

def retrieve_chunks_from_neo4j(driver):
    """
    Retrieves all :Chunk nodes from Neo4j, including their embeddings.
    Returns a list of dicts with keys: chunk_id, content, embedding, modality.
    """
    query = """
    MATCH (ch:Chunk)
    RETURN ch.chunk_id AS chunk_id,
           ch.content AS content,
           ch.embedding AS embedding,
           ch.modality AS modality
    """
    chunks = []
    with driver.session() as session:
        results = session.run(query)
        for record in results:
            chunk = {
                "chunk_id": record["chunk_id"],
                "content": record["content"],
                "embedding": record["embedding"],
                "modality": record["modality"]
            }
            chunks.append(chunk)
    return chunks

def find_top_k_chunks(query_embedding, chunks, top_k=TOP_K):
    """
    Computes similarity between the query embedding and each chunk's embedding,
    then returns the top-K chunks sorted by descending similarity.
    """
    similarities = []
    for c in chunks:
        emb = c.get("embedding", [])
        if not emb:
            continue
        sim = cosine_similarity(query_embedding, emb)
        similarities.append((c, sim))
    # Sort by similarity descending
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]

def load_llm(model_name: str = LLM_MODEL_NAME):
    """
    Loads the large language model (deepseek-r1:14b or similar) using Hugging Face Transformers.
    Adjust the code if you have a different local loading mechanism (e.g., Olla CLI).
    """
    print(f"[INFO] Loading LLM: '{model_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto") 
    # device_map="auto" tries to place the model on GPU if available, CPU otherwise.
    print("[INFO] LLM loaded successfully.")
    return tokenizer, model

def generate_answer(query: str, top_chunks: list, tokenizer, model):
    """
    Given a user query and a list of top chunks, constructs a prompt and generates an answer
    using the loaded LLM. This function can be adapted to your specific prompt engineering strategy.

    :param query: The user question (string).
    :param top_chunks: List of (chunk_dict, similarity_score) tuples.
    :param tokenizer: The tokenizer for the LLM.
    :param model: The loaded LLM model.
    :return: The model-generated answer (string).
    """
    # Construct a "context" string from the top chunks.
    context_texts = []
    for i, (chunk, score) in enumerate(top_chunks, start=1):
        # Add chunk content with a short label or marker
        context_texts.append(f"--- Chunk {i} (similarity={score:.4f}) ---\n{chunk['content']}\n")

    # Combine all chunk texts into a single context block.
    combined_context = "\n".join(context_texts)

    # Example Prompt (very simplistic). Adjust as needed:
    prompt = f"""You are a helpful AI assistant. You have access to the following context:

{combined_context}

User Query: {query}

Please provide the best possible answer using the above context. If the context doesn't contain an answer, say so.
"""

    # Tokenize the prompt for the LLM
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    # Generate output
    output_tokens = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.7,
        do_sample=True,
        top_p=0.9
    )
    # Decode the tokens to get the answer text
    answer = tokenizer.decode(output_tokens[0], skip_special_tokens=True)

    return answer

#############################################
#               MAIN SCRIPT                #
#############################################

def main():
    """
    Orchestrates the full RAG query flow:
      1) Prompt user for a query.
      2) Embed the query with SentenceTransformers.
      3) Retrieve top-K relevant chunks from Neo4j.
      4) Load the deepseek-r1:14b (or other) LLM locally.
      5) Construct a prompt using the top chunks as context.
      6) Generate and display the final answer.
    """
    # 1) Prompt for user query
    user_query = input("Enter your query: ").strip()
    if not user_query:
        print("No query provided. Exiting.")
        return

    # 2) Load embedding model & embed the query
    embedding_model = load_text_model()
    query_emb = embed_query(user_query, embedding_model)

    # 3) Connect to Neo4j & retrieve chunks
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
    try:
        chunks = retrieve_chunks_from_neo4j(driver)
        print(f"[INFO] Retrieved {len(chunks)} chunks from Neo4j.")
    except exceptions.AuthError as auth_err:
        print(f"[ERROR] Neo4j authentication failed: {auth_err}")
        return
    except exceptions.ServiceUnavailable as svc_err:
        print(f"[ERROR] Neo4j service unavailable: {svc_err}")
        return
    except Exception as e:
        print(f"[ERROR] Unexpected error retrieving chunks: {e}")
        return
    finally:
        # Close the driver to free resources
        driver.close()

    # 4) Find top-K chunks by similarity
    top_results = find_top_k_chunks(query_emb, chunks, TOP_K)
    if not top_results:
        print("No chunks found or no embeddings available.")
        return

    # 5) Load the LLM (deepseek-r1:14b or your chosen model)
    tokenizer, llm_model = load_llm(LLM_MODEL_NAME)

    # 6) Generate an answer by combining user query + top chunk context
    answer = generate_answer(user_query, top_results, tokenizer, llm_model)

    # 7) Print the final answer
    print("\n==================== RAG Answer ====================")
    print(answer)
    print("====================================================")

if __name__ == "__main__":
    main()
