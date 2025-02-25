"""
chunk_text.py

Reads a text file, splits into paragraphs, sentence-tokenizes them with NLTK,
wraps lines with textwrap, and returns a list of chunk dictionaries. Each chunk
dictionary contains:

{
  "chunk_id": str,               # e.g. "myfile_par_0"
  "modality": "text",            # always "text" here
  "content": str,                # the chunk's text
  "metadata": {
    "file_name": "...",          # base name of file
    "paragraph_index": int,      # which paragraph
    ...
  },
  "textual_modality": "wrapped_paragraph"
}

Dependencies:
    pip install nltk
Also ensure NLTK's 'punkt' data is downloaded for sentence tokenization:
    import nltk
    nltk.download('punkt')
"""

import os
import textwrap
import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt', quiet=True)


def read_text_file(file_path: str) -> str:
    """
    Reads the entire text file into a single string.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def reflow_paragraphs(text: str, width: int = 80) -> list:
    """
    Reflows text into a list of 'wrapped' paragraphs.

    Steps:
      1) Split text by double newlines => paragraphs
      2) Clean whitespace per paragraph
      3) Sentence-tokenize to ensure correct sentence boundaries
      4) Rejoin sentences and use textwrap.fill(...) to wrap lines to 'width'
      5) Return the list of reflowed paragraphs (strings)
    """
    paragraphs = text.split("\n\n")
    formatted_paragraphs = []

    for para in paragraphs:
        # Remove stray newlines/whitespace
        clean_para = " ".join(para.split())
        if not clean_para:
            continue

        # Sentence-tokenize for clarity
        sentences = sent_tokenize(clean_para)
        combined = " ".join(sentences)

        # Wrap lines to 'width'
        wrapped = textwrap.fill(combined, width=width)
        formatted_paragraphs.append(wrapped)

    return formatted_paragraphs


def chunk_text_file(file_path: str, width: int = 80) -> list:
    """
    Main function for chunking a text file into a list of chunk dicts.
    Each dict has: {chunk_id, modality, content, metadata, textual_modality}.

    Example usage:
        chunks = chunk_text_file("myfile.txt", width=80)
        # chunks is a list of dicts, one per paragraph
    """
    raw_text = read_text_file(file_path)
    reflowed_pars = reflow_paragraphs(raw_text, width=width)

    chunk_list = []
    base_name = os.path.basename(file_path)

    for i, paragraph_text in enumerate(reflowed_pars):
        chunk_id = f"{base_name}_par_{i}"

        chunk_dict = {
            "chunk_id": chunk_id,
            "modality": "text",
            "content": paragraph_text,
            "metadata": {
                "file_name": base_name,
                "paragraph_index": i
            },
            "textual_modality": "wrapped_paragraph"
        }
        chunk_list.append(chunk_dict)

    return chunk_list


if __name__ == "__main__":
    # Example usage / standalone test
    # Replace these with your actual file paths
    input_file = "input.txt"
    print(f"[chunk_text] Processing: {input_file}")

    chunks = chunk_text_file(input_file, width=80)

    print(f"[chunk_text] Generated {len(chunks)} chunks.")
    for c in chunks[:3]:  # Show first 3 chunks
        print(c)
