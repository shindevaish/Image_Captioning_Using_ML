from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pycocotools.coco import COCO
from elasticsearch import Elasticsearch
import uvicorn
import json
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import requests
from io import BytesIO

app = FastAPI()

# Path to COCO annotations
coco_annotation_path = '/Users/vaishnavishinde/Desktop/cocoapi/annotations/instances_train2017.json'

# Initialize COCO API
coco = COCO(coco_annotation_path)

# Initialize Elasticsearch client
es = Elasticsearch([{'host': 'localhost', 'port': 9200, 'scheme': 'http'}])

# Mount the static directory to serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load BLIP model and processor
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def generate_caption(image_url):
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content)).convert("RGB")
    image = image.resize((256, 256))  # Resize the image to a consistent size
    inputs = blip_processor(images=image, return_tensors="pt", padding=True)

    out = blip_model.generate(**inputs)
    caption = blip_processor.decode(out[0], skip_special_tokens=True)
    return caption

@app.get("/")
async def main():
    return HTMLResponse(open("static/index.html").read())

@app.post("/search/")
async def search(request: Request):
    try:
        data = await request.json()
        query = data.get("query")
        
        # Find images that contain a specific category
        cat_ids = coco.getCatIds(catNms=[query])
        if not cat_ids:
            raise HTTPException(status_code=404, detail="Category not found")

        img_ids = coco.getImgIds(catIds=cat_ids)
        imgs = coco.loadImgs(img_ids)

        # Generate captions for each image
        img_list = []
        for img in imgs:
            img_url = f"http://images.cocodataset.org/train2017/{img['file_name']}"
            caption = generate_caption(img_url)
            img_list.append({"url": img_url, "caption": caption})

        return JSONResponse(content={"images": img_list})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
