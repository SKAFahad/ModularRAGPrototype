"""
run_pipeline.py

This script orchestrates your entire RAG pipeline end-to-end and **always** launches
the interactive session (`rag_query.py`) at the end. The key difference is that we
treat `rag_query.py` as an interactive script, so we **do not capture** its output or
input in Python. Instead, we let it run directly so the user can see and type in the
console.

Problem Addressed:
------------------
In previous attempts, using `subprocess.run(..., capture_output=True, text=True)` 
caused the interactive session not to appear, because the prompt and user input 
were being captured (not forwarded to the user). Hence, no interactive Q&A was visible.

Solution:
---------
- We define one helper for normal scripts (`capture_output=True`) and 
  one helper for the interactive `rag_query.py` call (`capture_output=False`). 
- That way, rag_query can take over the terminal input/output properly.

Pipeline Steps:
--------------
1) data_extraction.py
2) data_chunking.py parse_results.json chunked_data.json
3) embedding_text.py --input chunked_data.json --output embedded_data.json
4) store_in_neo4j.py embedded_data.json
5) compute_relationships.py (unless we skip with --skip-relationships)
6) ALWAYS run rag_query.py in interactive mode (no capturing). The user can 
   type queries, type "exit"/"quit" to leave.

Usage:
------
    python run_pipeline.py [--skip-relationships]

No further flags. It automatically starts the interactive Q&A once steps are done.

If any step fails, we print an error and stop immediately, 
so partial data doesn't cause confusion.

Ensure your scripts exist in the same directory:
  - data_extraction.py
  - data_chunking.py
  - embedding_text.py
  - store_in_neo4j.py
  - compute_relationships.py
  - rag_query.py
plus your 'data/' folder for inputs, etc.

Let's begin:
"""

import sys
import os
import argparse
import subprocess


def run_script_normal(script_path, args=None) -> bool:
    """
    Runs a Python script in a 'normal' mode, capturing stdout/stderr so we can 
    display them here. If there's an error, returns False. If success, True.

    :param script_path: e.g. /path/to/data_extraction.py
    :type script_path: str
    :param args: CLI arguments for that script
    :type args: list
    """
    if args is None:
        args = []

    command = [sys.executable, script_path] + args
    print(f"\n[run_pipeline] Running (normal): {' '.join(command)}")

    # capture_output=True so we see logs in this pipeline's stdout
    result = subprocess.run(command, capture_output=True, text=True)
    print(result.stdout)  # print any standard output

    if result.returncode != 0:
        print(f"*** Error running {script_path}: ***\n{result.stderr}")
        return False

    return True


def run_script_interactive(script_path, args=None) -> bool:
    """
    Runs a Python script in *interactive* mode, meaning we do NOT capture 
    its stdout/stderr. Instead, we pass them through. This is crucial for 
    scripts like rag_query.py that prompt the user for input.

    :param script_path: e.g. /path/to/rag_query.py
    :param args: CLI arguments for that script
    :return: True if the script exits with code 0, else False
    """
    if args is None:
        args = []

    command = [sys.executable, script_path] + args
    print(f"\n[run_pipeline] Launching interactive script: {' '.join(command)}")

    # Here, we do NOT set capture_output. We let it attach to this process's 
    # stdin/stdout, so the user sees the prompt and can type.
    result = subprocess.run(command)
    # No result.stdout or result.stderr to print, because we want direct pass-through.

    if result.returncode != 0:
        print(f"*** Error running interactive script {script_path}, returncode={result.returncode}.")
        return False

    return True


def main():
    """
    The pipeline:
      1) data_extraction.py -> parse_results.json
      2) data_chunking.py parse_results.json chunked_data.json
      3) embedding_text.py --input chunked_data.json --output embedded_data.json
      4) store_in_neo4j.py embedded_data.json
      5) (optional) compute_relationships.py if not skipping
      6) ALWAYS run rag_query.py in interactive mode.

    Usage:
      python run_pipeline.py [--skip-relationships]
    """
    parser = argparse.ArgumentParser(
        description="Run entire pipeline then ALWAYS open interactive rag_query."
    )
    parser.add_argument("--skip-relationships", action="store_true",
                        help="Skip compute_relationships step.")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Pipeline scripts
    data_extraction_py       = os.path.join(script_dir, "data_extraction.py")
    data_chunking_py         = os.path.join(script_dir, "data_chunking.py")
    embedding_text_py        = os.path.join(script_dir, "embedding_text.py")
    store_in_neo4j_py        = os.path.join(script_dir, "store_in_neo4j.py")
    compute_relationships_py = os.path.join(script_dir, "compute_relationships.py")
    rag_query_py             = os.path.join(script_dir, "rag_query.py")

    # Filenames for each step
    parse_results_json = "parse_results.json"
    chunked_data_json  = "chunked_data.json"
    embedded_data_json = "embedded_data.json"

    # Step 1) data_extraction
    if not run_script_normal(data_extraction_py):
        print("[run_pipeline] data_extraction failed. Stopping.")
        return

    # Step 2) chunking
    if not run_script_normal(data_chunking_py, args=[parse_results_json, chunked_data_json]):
        print("[run_pipeline] data_chunking failed. Stopping.")
        return

    # Step 3) embedding_text
    if not run_script_normal(embedding_text_py, args=["--input", chunked_data_json, 
                                                      "--output", embedded_data_json]):
        print("[run_pipeline] embedding_text failed. Stopping.")
        return

    # Step 4) store_in_neo4j
    if not run_script_normal(store_in_neo4j_py, args=[embedded_data_json]):
        print("[run_pipeline] store_in_neo4j failed. Stopping.")
        return

    # Step 5) compute_relationships if not skipping
    if not args.skip_relationships:
        if not run_script_normal(compute_relationships_py, args=[]):
            print("[run_pipeline] compute_relationships failed. Stopping.")
            return
    else:
        print("[run_pipeline] Skipping compute_relationships step as requested.")

    print("[run_pipeline] Pipeline steps completed successfully!")
    print("[run_pipeline] Now launching rag_query.py for interactive Q&A session.\n")

    # Step 6) ALWAYS launch rag_query in interactive mode
    if not run_script_interactive(rag_query_py, args=[]):
        print("[run_pipeline] rag_query ended with errors.")
    else:
        print("[run_pipeline] rag_query ended normally.")

    print("[run_pipeline] Done. The entire pipeline (including interactive session) is complete.")


if __name__ == "__main__":
    main()
