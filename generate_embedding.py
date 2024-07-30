# generate_embedding.py

import os
import json
import numpy as np
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# Load BLIP model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def get_image_embedding(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model.get_image_features(**inputs)
    return outputs.cpu().numpy().flatten()

def generate_embeddings(image_folder, output_file):
    embeddings = {}
    if not os.path.exists(image_folder):
        raise FileNotFoundError(f"The directory {image_folder} does not exist.")
    
    for image_file in os.listdir(image_folder):
        image_path = os.path.join(image_folder, image_file)
        if os.path.isfile(image_path):
            embedding = get_image_embedding(image_path)
            embeddings[image_file] = embedding.tolist()  # Convert numpy array to list for JSON serialization

    with open(output_file, 'w') as f:
        json.dump(embeddings, f)

if __name__ == "__main__":
    image_folder = "../images"  # Update this path
    output_file = "image_embeddings.json"
    generate_embeddings(image_folder, output_file)
