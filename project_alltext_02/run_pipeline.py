"""
run_pipeline.py

This script orchestrates the entire pipeline from scratch and then launches an interactive
RAG query session. The pipeline steps are:

 1. Data Extraction:
    - Process raw files in "data/" and save output to "preposed_data/".

 2. Data Chunking:
    - Convert preposed_data files into chunked data (saved as chunked_data.json).

 3. Text Embedding:
    - Embed the chunks using SentenceTransformer and write to embedded_data.json.

 4. Store in Neo4j:
    - Read embedded_data.json and store Document and Chunk nodes in Neo4j.

 5. Compute Relationships:
    - Compute semantic relationships among chunks in Neo4j (SIMILAR_TO edges).

 6. Querying for RAG:
    - Launch the interactive RAG query session (via rag_query.py) so you can ask questions.

Usage:
    python run_pipeline.py

Make sure:
  - Your scripts/ folder contains the necessary modules.
  - Neo4j is running and accessible with the hard-coded credentials.
  - Ollama and the "deepseek-r1:32b" model are set up for querying.
"""

import os
import sys
import subprocess

def run_script(script_path, args=None):
    """
    Executes a Python script using the current Python interpreter.
    :param script_path: Absolute path to the Python script.
    :param args: List of additional command-line arguments.
    :return: True if the script runs successfully, False otherwise.
    """
    if args is None:
        args = []
    command = [sys.executable, script_path] + args
    print(f"\nRunning: {' '.join(command)}")
    result = subprocess.run(command, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        return False
    return True

def main():
    # Determine project root: assume run_pipeline.py is in project root.
    project_root = os.path.dirname(os.path.abspath(__file__))
    scripts_dir = os.path.join(project_root, "scripts")
    
    # Step 1: Data Extraction
    print("=== Step 1: Data Extraction ===")
    if not run_script(os.path.join(scripts_dir, "data_extraction.py")):
        print("Data Extraction failed.")
        sys.exit(1)
    
    # Step 2: Data Chunking
    print("\n=== Step 2: Data Chunking ===")
    if not run_script(os.path.join(scripts_dir, "data_chunking.py")):
        print("Data Chunking failed.")
        sys.exit(1)
    
    # Step 3: Text Embedding
    print("\n=== Step 3: Text Embedding ===")
    if not run_script(os.path.join(scripts_dir, "embedding_text.py")):
        print("Text Embedding failed.")
        sys.exit(1)
    
    # Step 4: Store in Neo4j
    print("\n=== Step 4: Store in Neo4j ===")
    if not run_script(os.path.join(scripts_dir, "StoreInNeo4j.py")):
        print("Storing in Neo4j failed.")
        sys.exit(1)
    
    # Step 5: Compute Relationships
    print("\n=== Step 5: Compute Relationships ===")
    if not run_script(os.path.join(scripts_dir, "ComputeRelationships.py")):
        print("Computing Relationships failed.")
        sys.exit(1)
    
    # Step 6: Interactive Querying for RAG
    # Here we run the rag_query.py script to allow interactive Q&A.
    print("\n=== Step 6: Interactive RAG Query Session ===")
    print("Launching interactive session. Type 'exit' or 'quit' to end.")
    
    # Instead of running and capturing output (which would block), we simply
    # execute the rag_query.py script in interactive mode.
    # This assumes rag_query.py is designed to work interactively.
    rag_query_script = os.path.join(scripts_dir, "rag_query.py")
    # Use run_script without capturing output so that interactive input works:
    subprocess.run([sys.executable, rag_query_script])
    
    print("\nPipeline executed successfully!")

if __name__ == "__main__":
    main()
