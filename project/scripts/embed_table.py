"""
embed_table.py

Handles table embedding using Google's TAPAS model (google/tapas-base).
TAPAS was designed for QA over tabular data, but here we adapt it to
produce a single embedding vector for each table chunk.

NEW: The tokenizer now expects a pandas DataFrame, so we'll build
a DataFrame from the chunk string.
"""

import torch
import pandas as pd
from transformers import TapasTokenizer, TapasModel
from typing import List

def load_table_model():
    """
    Loads the TAPAS tokenizer & model from Hugging Face.

    Returns:
        (TapasTokenizer, TapasModel): The tokenizer and model objects for TAPAS.
        
    Example:
        tokenizer, model = load_table_model()
    """
    print("Loading TAPAS model & tokenizer...")
    tokenizer = TapasTokenizer.from_pretrained("google/tapas-base")
    model = TapasModel.from_pretrained("google/tapas-base")
    return tokenizer, model


def embed_table_tapas(table_str: str, tokenizer, model) -> List[float]:
    """
    Creates a single-row DataFrame from the chunk string "ColumnA: valA, ColumnB: valB"
    and uses TAPAS to produce an embedding by:
      1) building a DataFrame,
      2) tokenizing with a dummy question,
      3) average-pooling the last hidden states.

    Args:
        table_str (str): e.g. "City: NYC, Population: 8.3M, Year: 2022"
        tokenizer (TapasTokenizer): TAPAS tokenizer instance.
        model (TapasModel): TAPAS model instance.

    Returns:
        List[float]: The pooled embedding as a Python list of floats.
    """
    # 1) Parse the chunk string into columns/values
    #    Example: "ColumnA: valA, ColumnB: valB"
    columns_values = [col.strip() for col in table_str.split(',')]
    headers = []
    row_data = []

    # Attempt to extract "Column: Value" pairs
    for cv in columns_values:
        if ':' in cv:
            col_name, val = cv.split(':', 1)
            headers.append(col_name.strip())
            row_data.append(val.strip())
        else:
            # fallback if the format is not "Column: Value"
            headers.append("Unknown")
            row_data.append(cv.strip())

    # 2) Build a one-row DataFrame
    #    e.g. columns=["City","Population","Year"], row_data=["NYC","8.3M","2022"]
    df = pd.DataFrame([row_data], columns=headers)

    # 3) Prepare a dummy question to feed the tokenizer
    question = "What is in the table?"

    # 4) Tokenize with the DataFrame
    inputs = tokenizer(
        table=df,
        queries=question, 
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    # 5) Forward pass through TAPAS
    outputs = model(**inputs)
    hidden_states = outputs.last_hidden_state  # shape: (batch_size, seq_len, hidden_dim)

    # 6) Average pool across the seq_len dimension
    #    shape: (1, hidden_dim) after pooling
    pooled = torch.mean(hidden_states, dim=1)

    # 7) Convert to a Python list of floats
    return pooled.squeeze(0).tolist()
