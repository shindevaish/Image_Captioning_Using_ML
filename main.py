from fastapi import FastAPI, Request, HTTPException, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pycocotools.coco import COCO
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import uvicorn
import json
import io

app = FastAPI()

# Path to COCO annotations
coco_annotation_path = "/Users/vaishnavishinde/Desktop/cocoapi/annotations/instances_train2017.json"

# Initialize COCO API
coco = COCO(coco_annotation_path)

# Initialize BLIP model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

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
            raise HTTPException(status_code=404, detail="Category not found")

        img_ids = coco.getImgIds(catIds=cat_ids)
        imgs = coco.loadImgs(img_ids)

        # Convert image data to JSON for response
        img_list = [{"url": img['coco_url'], "caption": img['id']} for img in imgs]  # Use actual fields

        return JSONResponse(content={"images": img_list})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/caption/")
async def caption(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # Process the image using BLIP model
        inputs = processor(images=image, return_tensors="pt")
        outputs = model.generate(**inputs)
        caption = processor.decode(outputs[0], skip_special_tokens=True)

        return {"filename": file.filename, "caption": caption}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
