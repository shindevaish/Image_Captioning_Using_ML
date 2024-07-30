from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
import json
import numpy as np
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from scipy.spatial.distance import cosine
from io import BytesIO
import requests
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Load embeddings
with open('embeddings.json', 'r') as f:
    embeddings = json.load(f)

# Initialize BLIP model and processor
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def generate_embedding(image):
    inputs = blip_processor(images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = blip_model.get_image_features(**inputs)
    embedding = outputs.pooler_output.numpy().flatten()  # Flatten the embedding
    return embedding

def search(query_image):
    query_embedding = generate_embedding(query_image)
    
    # Compare with stored embeddings
    distances = {}
    for img_file, emb in embeddings.items():
        stored_embedding = np.array(emb)
        distance = cosine(query_embedding, stored_embedding)
        distances[img_file] = distance
    
    # Sort images by distance
    sorted_images = sorted(distances.items(), key=lambda x: x[1])
    
    return sorted_images

@app.get("/")
async def main():
    return HTMLResponse(open("static/index.html").read())

@app.post("/search/")
async def search_endpoint(request: Request):
    data = await request.json()
    query_url = data.get("query")
    
    # Here we assume `query` is a URL of an image
    response = requests.get(query_url)
    image = Image.open(BytesIO(response.content)).convert("RGB")
    
    results = search(image)
    
    img_list = [{"file_name": img_file, "distance": distance} for img_file, distance in results]
    
    return JSONResponse(content={"images": img_list})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
