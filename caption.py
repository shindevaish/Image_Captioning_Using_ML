import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import requests

# Load the pre-trained BLIP model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Function to preprocess the image
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    return image

# Function to generate caption
def generate_caption(image_path):
    image = preprocess_image(image_path)
    inputs = processor(image, return_tensors="pt", padding=True)  # Ensure padding is applied
    
    # Generate caption
    outputs = model.generate(**inputs)
    caption = processor.decode(outputs[0], skip_special_tokens=True)
    return caption

# Example usage
img_path = '../IMG_20240609_055837.jpg'
caption = generate_caption(img_path)
print('Caption:', caption)
