"""
CalculateAttentionScores.py

A separate script that loads chunked embeddings, applies
a rule-based or custom logic to assign an `attention_score`
to each chunk, and saves the updated JSON.

Usage:
  python CalculateAttentionScores.py

Outputs:
  project/chunked_with_scores.json  (or you can overwrite the same JSON)
"""

import json
import os

# Input/Output paths
INPUT_JSON = "project/chunked_with_embeddings.json"
OUTPUT_JSON = "project/chunked_with_scores.json"

def assign_rule_based_attention(chunk):
    """
    Rule-based logic for assigning attention_score 
    to a chunk based on chunk['modality'].
    You can replace this with more advanced logic or ML-based approach.
    """
    # If chunk already has an attention_score, skip or override
    if "attention_score" in chunk:
        return  # or comment out if you want to override anyway

    modality = chunk.get("modality", "")
    if modality == "text":
        chunk["attention_score"] = 1.0
    elif modality == "table":
        chunk["attention_score"] = 1.2  # slightly prefer tables
    elif modality == "image":
        chunk["attention_score"] = 0.9
    else:
        chunk["attention_score"] = 1.0  # default

def main():
    # 1) Load the chunk data from file
    if not os.path.isfile(INPUT_JSON):
        print(f"Error: {INPUT_JSON} not found. Run Embedding Generation first.")
        return

    with open(INPUT_JSON, "r") as f:
        chunks = json.load(f)

    print(f"Loaded {len(chunks)} chunks from '{INPUT_JSON}'.")

    # 2) Assign attention scores
    for chunk in chunks:
        assign_rule_based_attention(chunk)

    # 3) Save the updated chunks
    with open(OUTPUT_JSON, "w") as f:
        json.dump(chunks, f, indent=2)

    print(f"Saved updated chunks with attention_score to '{OUTPUT_JSON}'.")

if __name__ == "__main__":
    main()
