"""
parse_text.py

Reads a plain .txt file. We treat the entire file content as 'text'
(or you could separate by lines/paragraphs if you prefer).
Returns a dictionary with keys: 'text', 'tables', 'images', 'metadata'.
"""

import os

def parse_text_file(file_path: str) -> dict:
    """
    Reads the entire .txt file content. 
    Optionally, we can store each line separately, but here we'll do one big string.

    Returns:
    {
      'text': "entire text content",
      'tables': [],
      'images': [],
      'metadata': { 'file_name': ... }
    }
    """
    result = {
        "text": "",
        "tables": [],
        "images": [],
        "metadata": {}
    }
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            file_content = f.read()

        result["text"] = file_content
        result["metadata"] = {
            "file_name": os.path.basename(file_path)
        }
    except Exception as e:
        print(f"[parse_text_file] Error processing '{file_path}': {e}")

    return result
