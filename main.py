from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pycocotools.coco import COCO
import uvicorn
import json

app = FastAPI()

# Path to COCO annotations
coco_annotation_path = '/Users/vaishnavishinde/Desktop/cocoapi/annotations/instances_train2017.json'

# Initialize COCO API
coco = COCO(coco_annotation_path)

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
        
        # Find images that contain a specific category
        cat_ids = coco.getCatIds(catNms=[query])
        if not cat_ids:
            return JSONResponse(content={"images": [], "error": "Category not found"})

        img_ids = coco.getImgIds(catIds=cat_ids)
        imgs = coco.loadImgs(img_ids)

        # Convert image data to JSON for response
        img_list = [{"url": img['coco_url'], "caption": img['file_name']} for img in imgs]  # Use 'file_name' as caption

        return JSONResponse(content={"images": img_list})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/categories/")
async def get_categories():
    categories = coco.loadCats(coco.getCatIds())
    category_names = [cat['name'] for cat in categories]
    return JSONResponse(content={"categories": category_names})

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
