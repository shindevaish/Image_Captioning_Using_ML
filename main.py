from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from PIL import Image
import json
import numpy as np
import torch
import pandas as pd
import requests
import os
import uvicorn

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

image_caption=pd.read_csv("captions.csv")

@app.get("/", response_class=HTMLResponse)
async def index():
    index_file = Path("static/index.html")
    if index_file.exists():
        with open(index_file) as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>Index file not found</h1>")

# Load embeddings
# with open('image_embeddings.json', 'r') as f:
#     embeddings = json.load(f)

# Initialize BLIP model and processor
# blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
# blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# def search(query_image):
#     query_embedding = generate_embedding(query_image)
    
#     # Compare with stored embeddings
#     distances = {}
#     for img_file, emb in embeddings.items():
#         stored_embedding = np.array(emb)
#         distance = cosine(query_embedding, stored_embedding)
#         distances[img_file] = distance
    
#     # Sort images by distance
#     sorted_images = sorted(distances.items(), key=lambda x: x[1])
    
#     return sorted_images[:10]

@app.post("/search/")
async def search_endpoint(request: Request):
    data = await request.json()
    query = data.get("query")
    
    # Here we assume `query` is a URL of an image
    # response = requests.get(query_url)
    # image = Image.open(BytesIO(response.content)).convert("RGB")
    
    # results = search(image)
    
    # img_list = [{"file_name": img_file, "distance": distance} for img_file, distance in results]

    if not query:
        return JSONResponse(content={'message' : "No matching image found"})
    else:
        match=image_caption[image_caption['caption'].str.contains(fr'\b{query}\b', case=False, na=False)]
        lst=[{"file_name" : row['image_id'], 'caption' : row['caption']} for _, row in match.iterrows()]

    # folder="static/images"
    # files= sorted(os.listdir(folder))[:10]
    # lst=[{"file_name":file,"url":f"static/images/{file}"} for file in files]

    return JSONResponse(content={"images": lst})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
