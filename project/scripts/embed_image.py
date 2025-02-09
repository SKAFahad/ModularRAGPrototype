
"""
embed_image.py

Handles image/chart embedding using OpenAI's CLIP model.
CLIP (Contrastive Language–Image Pretraining) is well-suited for 
generating a single embedding vector that represents the image content.

Installation:
    pip install git+https://github.com/openai/CLIP.git
or:
    pip install clip-anytorch

Usage:
    from embed_image import load_image_model, embed_image_clip

Steps:
1) load_image_model() - Loads a chosen CLIP model (e.g., "ViT-B/32") 
   and a preprocessing pipeline.
2) embed_image_clip() - Accepts an image path, preprocesses it, and 
   runs it through CLIP's image encoder to get a vector.
"""

import torch
import clip
from PIL import Image
from typing import List

def load_image_model(model_name="ViT-B/32", device="cuda"):
    """
    Loads a CLIP model and its corresponding preprocess function.

    Args:
        model_name (str): The model variant to load (e.g., "ViT-B/32", "ViT-L/14").
        device (str): Either "cuda" (GPU) or "cpu".

    Returns:
        (model, preprocess):
            model: The CLIP model loaded on the specified device.
            preprocess: A function that preprocesses PIL images for CLIP.
    
    Example:
        model, preprocess = load_image_model("ViT-B/32", device="cuda")
    """
    print(f"Loading CLIP model '{model_name}' on device '{device}'...")
    # model and preprocess are loaded from clip
    model, preprocess_fn = clip.load(model_name, device=device)
    return model, preprocess_fn


def embed_image_clip(image_path: str, model, preprocess, device="cuda") -> List[float]:
    """
    Generates an embedding for an image (PNG, JPG, etc.) using CLIP.

    Args:
        image_path (str): Path to the image file on disk.
        model: The CLIP model instance returned by load_image_model.
        preprocess: The preprocessing function from load_image_model.
        device (str): "cuda" or "cpu", matching how you loaded the model.

    Returns:
        List[float]: A list of floats representing the image embedding.

    Process:
        1. Read the image from disk using PIL.
        2. Preprocess the image according to CLIP requirements (resize, normalize).
        3. Forward pass through model.encode_image() to get a feature vector.
        4. (Optional) Normalize the vector (common in CLIP workflows).
        5. Convert the tensor to a Python list of floats (for easy JSON/db storage).

    Example:
        embedding = embed_image_clip("sample.png", model, preprocess, device="cuda")
    """
    # 1) Open and convert the image to RGB
    img = Image.open(image_path).convert("RGB")

    # 2) Preprocess the image (resizing, normalization, etc.)
    img_tensor = preprocess(img).unsqueeze(0).to(device)

    # 3) Forward pass through the image encoder to get features
    with torch.no_grad():
        image_features = model.encode_image(img_tensor)

    # 4) (Optional) normalize the features
    #    This step is common to keep image embeddings on a unit sphere for similarity.
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    # 5) Convert to CPU numpy array and then to list
    image_embedding = image_features.squeeze(0).cpu().numpy().tolist()

    return image_embedding
