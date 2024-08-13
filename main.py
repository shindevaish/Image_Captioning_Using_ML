from fastapi import FastAPI, Request,File,UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from transformers import BlipProcessor, BlipForConditionalGeneration
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


@app.post("/search/")
async def search_endpoint(request: Request):
    data = await request.json()
    query = data.get("query")
    
    if not query:
        return JSONResponse(content={'message' : "No matching image found"})
    else:
        match=image_caption[image_caption['caption'].str.contains(fr'\b{query}\b', case=False, na=False)]
        lst=[{"file_name" : row['image_id'], 'caption' : row['caption']} for _, row in match.iterrows()]

    # folder="static/images"
    # files= sorted(os.listdir(folder))[:10]
    # lst=[{"file_name":file,"url":f"static/images/{file}"} for file in files]

    return JSONResponse(content={"images": lst})

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return HTMLResponse(content="", status_code=204)  # Empty response with no content


#Initialisation of blip model 

processor=BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model=BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

data=[]

@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    try:
        image=Image.open(file.file).convert("RGB")
        inputs = processor(image, return_tensors="pt", padding=True)  # Ensure padding is applied
        outputs = model.generate(**inputs)
        caption = processor.decode(outputs[0], skip_special_tokens=True)
        words=caption.split()

        for word in words:
            match=image_caption[image_caption['caption'].str.contains(fr'\b{word}\b', case=False, na=False)]
            lst=[{"file_name" : row['image_id'], 'caption' : row['caption']} for _, row in match.iterrows()]
        
        lst = [dict(t) for t in {tuple(d.items()) for d in lst}]

        return JSONResponse(content={"images": lst})


    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/add_image/")
async def add_image(file: UploadFile = File(...)):
    try:
        # Save the uploaded file
        image = Image.open(file.file).convert("RGB")
        file_name = file.filename
        image_path = f"static/image/{file_name}"
        image.save(image_path)

        # Generate the caption for the image
        inputs = processor(image, return_tensors="pt", padding=True)
        outputs = model.generate(**inputs)
        caption = processor.decode(outputs[0], skip_special_tokens=True)

        # Append the new image data to the CSV file
        new_data = pd.DataFrame([[file_name, caption]], columns=["image_id", "caption"])
        new_data.to_csv("captions.csv", mode='a', header=False, index=False)

        return JSONResponse(content={"file_name": file_name, "caption": caption})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
