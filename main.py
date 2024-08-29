from fastapi import FastAPI, Request,File,UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from transformers import BlipProcessor, BlipForConditionalGeneration
from pathlib import Path
from PIL import Image
from collections import defaultdict
from IPython.display import display, HTML
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import contractions
import string
import json
import numpy as np
import torch
import pandas as pd
import time
import re
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

def preprocess(text):
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words("english"))
    expanded_words = [contractions.fix(word) for word in text.split()]
    text = ' '.join(expanded_words)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    tokens = text.lower().split()
    tokens = [stemmer.stem(token) for token in tokens if token not in stop_words]
    return tokens

def construct_inverted_index(df):
    dictionary = {} # inverted index
 
    for index, row in df.iterrows():
        tokens = preprocess(row['caption'])
        for token in tokens:
            if token not in dictionary:
                dictionary[token] = [index]
            else:
                dictionary[token].append(index)
    dictionary = {k: set(v) for k, v in dictionary.items()}
    return dictionary

inverted_index = construct_inverted_index(image_caption)
print("Size of inverted index ", len(inverted_index))

def tokenize_infix_expression(expression):
    return expression.split()

def infix_to_postfix(tokens):
    precedence = {"AND": 2, "NOT": 3, "OR":1 } # set the precedence of operators for postfix expression
    stack = []
    postfix = []
    for token in tokens:
        if token in precedence: # add operands first and then operators
            while stack and precedence.get(stack[-1], 0) >= precedence[token]:
                postfix.append(stack.pop())
            stack.append(token)
        elif token == '(':
            stack.append(token)
        elif token == ')':
            while stack and stack[-1] != '(':
                postfix.append(stack.pop())
            stack.pop()
        else:
            postfix.append(token)
    while stack:
        postfix.append(stack.pop())
    return postfix

def evaluate_postfix(postfix):
    stemmer = PorterStemmer()
    stack = []
    for token in postfix:
        if token == "AND": # take intersection of postings
            set2 = stack.pop()
            set1 = stack.pop()
            result = set1.intersection(set2)
            stack.append(result)
        elif token == "NOT": # finding all documents that are not in the postings list
            set1 = stack.pop()
            stack.append(set(range(len(image_caption))).difference(set1))
        elif token == "OR": # take union of postings
            set1 = stack.pop()
            set2 = stack.pop()
            stack.append(set1.union(set2))  # Convert token to a set
        else: # retrive the posting of the stemmed token
          stack.append(inverted_index.get(stemmer.stem(token), set()))
       
    return stack[0]


@app.post("/search/")
async def search_endpoint(request: Request):
    data = await request.json()
    query = data.get("query")

    print(f"Received query: {query}")
    # Tokenize the expression
    tokens = tokenize_infix_expression(query)
    postfix_expression = infix_to_postfix(tokens)
    print("Postfix Expression: ",postfix_expression)
    # # Evaluate the postfix expression
    result = evaluate_postfix(postfix_expression)
    print(f"Search result indices: {result}")
    titles = []
    abstracts = []
    for res in list(result)[:10]:
        titles.append(image_caption.iloc[res].image_id)
        abstracts.append(image_caption.iloc[res].caption)
    results = [{"Title": title, "Abstract": abstract} for title, abstract in zip(titles, abstracts)]
    
    return JSONResponse(content={"results": results})

    
    # if not query:
    #     return JSONResponse(content={'message' : "No matching image found"})
    # else:
    #     match=image_caption[image_caption['caption'].str.contains(fr'\b{query}\b', case=False, na=False)]
    #     lst=[{"file_name" : row['image_id'], 'caption' : row['caption']} for _, row in match.iterrows()]

    # # folder="static/images"
    # # files= sorted(os.listdir(folder))[:10]
    # # lst=[{"file_name":file,"url":f"static/images/{file}"} for file in files]

    # return JSONResponse(content={"images": lst})

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
        image = Image.open(file.file).convert("RGB")
        file_name = file.filename
        image_path = f"static/image/{file_name}"
        image.save(image_path)
        inputs = processor(image, return_tensors="pt", padding=True)
        outputs = model.generate(**inputs)
        caption = processor.decode(outputs[0], skip_special_tokens=True)

        new_data = pd.DataFrame([[file_name, caption]], columns=["image_id", "caption"])
        new_data.to_csv("captions.csv", mode='a', header=False, index=False)

        return JSONResponse(content={"file_name": file_name, "caption": caption})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
