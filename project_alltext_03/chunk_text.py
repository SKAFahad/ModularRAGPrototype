"""
chunk_text.py

This module handles chunking plain text content into smaller, more manageable segments.
In a Retrieval-Augmented Generation (RAG) pipeline, splitting large text into chunks
makes it easier to embed and store in a vector DB or graph database. Also, it helps
avoid exceeding context-window limits in LLMs.

Guiding Principles (from our discussion):
1. **Focus**: chunk_text.py should only handle chunking text, not parse files from disk.
2. **Detailed Comments**: Provide enough documentation for new team members.
3. **Consistent Return Format**: Return a list of dictionaries, each representing
   a chunk, including metadata and a textual_modality.
4. **Flexible**: The chunk size or style (paragraph-based, sentence-based, or
   token-based) can be customized as needed.

Recommended Return Structure (list of chunk dicts):
[
  {
    "chunk_id": <unique ID for this chunk>,
    "modality": "text",
    "content": <the chunk's textual content>,
    "metadata": {
      "file_name": <optional, name of source text or doc>,
      "paragraph_index": <the index or sequence number of this chunk>
      ...
    },
    "textual_modality": "wrapped_paragraph"
  },
  ...
]

Usage Example:
    from chunk_text import chunk_text

    big_text = \"\"\"This is a long piece of text...
                   possibly multiple paragraphs or lines...\"\"\"

    chunk_list = chunk_text(
                    text_content=big_text,
                    file_name="example.txt",     # optional
                    wrap_width=80
                 )
    # chunk_list will be a list of chunk dicts, each representing a paragraph chunk.
"""

import textwrap
import nltk
from nltk.tokenize import sent_tokenize

# Make sure the 'punkt' tokenizer is available
nltk.download('punkt', quiet=True)


def chunk_text(
    text_content: str,
    file_name: str = "",
    wrap_width: int = 80
) -> list:
    """
    Splits a large text string into paragraph-based chunks, further ensuring lines
    are wrapped at 'wrap_width' characters. Each chunk is returned in a dictionary.

    :param text_content: The entire text to be chunked.
    :type text_content: str

    :param file_name: An optional identifier (e.g., the file name where text came from),
                      stored in the chunk's metadata.
    :type file_name: str

    :param wrap_width: The approximate character width for wrapping each chunk.
    :type wrap_width: int

    :return: A list of chunk dictionaries with keys: chunk_id, modality, content, metadata,
             textual_modality.
    :rtype: list

    Detailed Steps:
      1) Split text by double newlines to identify paragraphs. This approach
         preserves larger conceptual breaks.
      2) Clean up extra spaces or empty paragraphs.
      3) For each paragraph, optionally tokenize sentences (to refine boundaries).
      4) Wrap lines using textwrap.fill(..., width=wrap_width) to avoid overly
         long lines. This makes the text more readable and can help some LLMs
         with context boundaries.
      5) Build chunk dictionaries, each including:
          - chunk_id: e.g. <file_name>_par_<index>
          - modality: "text"
          - content: the final wrapped text
          - metadata: includes file_name, paragraph_index
          - textual_modality: "wrapped_paragraph"
      6) Return the list of chunk dicts.

    Example:
        text = \"\"\"First paragraph of text.

                    Second paragraph of text,
                    possibly multiple lines...\"\"\"

        result = chunk_text(text, file_name="mydoc.txt")
        # result might look like:
        # [
        #   {
        #     "chunk_id": "mydoc.txt_par_0",
        #     "modality": "text",
        #     "content": "First paragraph of text.",
        #     "metadata": { "file_name": "mydoc.txt", "paragraph_index": 0 },
        #     "textual_modality": "wrapped_paragraph"
        #   },
        #   {
        #     "chunk_id": "mydoc.txt_par_1",
        #     ...
        #   }
        # ]
    """

    # If there's no text, return empty list
    if not text_content.strip():
        return []

    # Step 1) Split the text by double newlines to get paragraphs
    raw_paragraphs = text_content.split("\n\n")
    cleaned_paragraphs = []

    for para in raw_paragraphs:
        # Remove stray whitespace
        clean_para = " ".join(para.split())
        if clean_para:
            cleaned_paragraphs.append(clean_para)

    # The final list of chunk dictionaries
    chunk_list = []
    paragraph_counter = 0

    # Step 2) For each paragraph, we can also split into sentences if needed.
    #         Then rejoin them. This helps ensure each chunk is a coherent block.
    for paragraph_text in cleaned_paragraphs:
        # Sentence-tokenize
        sentences = sent_tokenize(paragraph_text)
        # Rejoin for final chunk text
        combined_text = " ".join(sentences)

        # Step 3) Wrap lines to avoid extremely long lines
        wrapped_text = textwrap.fill(combined_text, width=wrap_width)

        # Build a chunk_id using the file_name if provided
        # If file_name isn't given, it'll just be "_par_" plus index
        base_name = file_name if file_name else "unknown_source"
        chunk_id = f"{base_name}_par_{paragraph_counter}"

        chunk_dict = {
            "chunk_id": chunk_id,
            "modality": "text",
            "content": wrapped_text,
            "metadata": {
                "file_name": base_name,
                "paragraph_index": paragraph_counter
            },
            "textual_modality": "wrapped_paragraph"
        }
        chunk_list.append(chunk_dict)

        paragraph_counter += 1

    return chunk_list


# Optional: if you want a function that reads from a file and then calls chunk_text,
# you can define chunk_text_file. In many pipelines, parsing is separate, 
# but here's an example if needed:
def chunk_text_file(file_path: str, wrap_width: int = 80) -> list:
    """
    Reads a text file from disk, then calls chunk_text(...) to split it into
    paragraph-based chunks. Each chunk is returned in a dict with metadata.

    :param file_path: Path to the .txt file on disk
    :type file_path: str

    :param wrap_width: The approximate character width for wrapping lines
    :type wrap_width: int

    :return: A list of chunk dictionaries (see chunk_text for details).
    :rtype: list
    """
    import os

    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"chunk_text_file: '{file_path}' not found or invalid.")

    with open(file_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    file_name = os.path.basename(file_path)
    return chunk_text(raw_text, file_name=file_name, wrap_width=wrap_width)


if __name__ == "__main__":
    """
    Simple testing if run as a script:
      python chunk_text.py [path_to_text_file]
    """
    import sys
    if len(sys.argv) > 1:
        text_file = sys.argv[1]
        try:
            chunks = chunk_text_file(text_file, wrap_width=80)
            print(f"Chunked {len(chunks)} paragraphs from '{text_file}':")
            for ch in chunks[:3]:  # show only first 3 for brevity
                print("chunk_id:", ch["chunk_id"])
                print("content:", ch["content"][:100], "...")
                print("metadata:", ch["metadata"])
                print("--")
        except Exception as e:
            print(f"Error reading file '{text_file}': {e}")
    else:
        # Demo chunk_text with a sample string
        sample_text = """This is a sample text.

        We have multiple paragraphs. This is the second paragraph,
        which might span multiple lines. And some more lines here.

        Lastly, we have a third paragraph. End of sample."""
        results = chunk_text(sample_text, file_name="sample.txt", wrap_width=50)
        print(f"Created {len(results)} chunk(s) from sample_text.\n")
        for r in results:
            print("Chunk ID:", r["chunk_id"])
            print("Content (first 80 chars):", r["content"][:80], "..." if len(r["content"])>80 else "")
            print("Metadata:", r["metadata"])
            print("------------------------------------")
