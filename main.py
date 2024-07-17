from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pycocotools.coco import COCO
from elasticsearch import Elasticsearch
import uvicorn
import json

app = FastAPI()

# Path to COCO annotations
coco_annotation_path = '/Users/vaishnavishinde/Desktop/cocoapi/annotations/instances_train2017.json'

# Initialize COCO API
coco = COCO(coco_annotation_path)

# Initialize Elasticsearch client
es = Elasticsearch([{'host': 'localhost', 'port': 9200, 'scheme': 'http'}])

# Mount the static directory to serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def main():
    return HTMLResponse(open("static/index.html").read())

@app.post("/search/")
async def search(request: Request):
    try:
        data = await request.json()
        query = data.get("query")
        print(f"Received query: {query}")

        # Find images that contain a specific category
        cat_ids = coco.getCatIds(catNms=[query])
        if not cat_ids:
            raise HTTPException(status_code=404, detail="Category not found")

        img_ids = coco.getImgIds(catIds=cat_ids)
        imgs = coco.loadImgs(img_ids)

        # Convert image data to JSON for response
        img_list = [json.dumps(img) for img in imgs]
        print(f"Found images: {img_list}")

        return JSONResponse(content={"images": img_list})
    except Exception as e:
        print(f"Error: {str(e)}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
