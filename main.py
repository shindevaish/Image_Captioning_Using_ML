from fastapi import FastAPI, Request,File,UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from transformers import BlipProcessor, BlipForConditionalGeneration, AutoTokenizer, AutoModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
from PIL import Image
from collections import defaultdict, Counter
from IPython.display import display, HTML
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import math
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

def boolean_search(query):
    
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
        dictionary = {} 

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
        precedence = {"and": 2, "not": 3, "or":1, "AND": 2, "NOT": 3, "OR":1 }
        stack = []
        postfix = []
        for token in tokens:
            if token in precedence: 
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
            if token == "AND" or token=="and":
                set2 = stack.pop()
                set1 = stack.pop()
                result = set1.intersection(set2)
                stack.append(result)
            elif token == "NOT" or token=="not": 
                set1 = stack.pop()
                stack.append(set(range(len(image_caption))).difference(set1))
            elif token == "OR" or token=="or":
                set1 = stack.pop()
                set2 = stack.pop()
                stack.append(set1.union(set2)) 
            else: 
                stack.append(inverted_index.get(stemmer.stem(token), set()))
        
        return stack[0]
    
    tokens = tokenize_infix_expression(query)
    postfix_expression = infix_to_postfix(tokens)
    print("Postfix Expression: ", postfix_expression)

    result = evaluate_postfix(postfix_expression)
    return result

def semantic_search_tfidf(query):
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words("english"))
    corpus = []

    for summary in image_caption['caption'].to_list():
        tokens = summary.lower().split()

        tokens = [stemmer.stem(token) for token in tokens if token not in stop_words]
        corpus.append(" ".join(tokens))
        

    def compute_tf(corpus):
        tf_corpus=[]
        for doc in corpus:
            tf_doc=Counter(doc)
            tf_corpus.append(tf_doc)
        
        return tf_corpus

    def compute_df(corpus):
        df=defaultdict(int)
        for doc in corpus:
            terms=set(doc)
            for term in terms:
                df[term] +=1
        return df
    
    def compute_idf(corpus,df):
        N=len(corpus)
        idf={}
        for term,freq in df.items():
            idf[term]=math.log(N/(1+freq))
        return idf

    def compute_tfidf(tf_corpus, idf):
        tfidf_corpus = []
        for tf_doc in tf_corpus:
            tfidf_doc = {}
            for term, tf in tf_doc.items():
                tfidf_doc[term] = tf * idf[term]  # TF * IDF
            tfidf_corpus.append(tfidf_doc)
        return tfidf_corpus 
    
    tf_corpus = compute_tf(corpus)
    df = compute_df(corpus)
    idf = compute_idf(corpus, df)
    print(compute_tfidf(tf_corpus, idf))

    # create a TF-IDF index based on the whole corpus
    vectorizer = TfidfVectorizer(sublinear_tf=True, max_features=5000)
    documents_tfidf_features = vectorizer.fit_transform(corpus)
    # Print features used for TF-IDF vectorization
    query_tokens = query.lower().split()
    query_tokens = [stemmer.stem(token) for token in query_tokens if token not in stop_words]
    query_processed = " ".join(query_tokens)

    query_tfidf_features = vectorizer.transform([query_processed])
    similarities = cosine_similarity(documents_tfidf_features, query_tfidf_features).flatten()
    TopK = 15
    top_indices = similarities.argsort()[::-1][:TopK]# Pick TopK document ids having highest cosine similarity
    
    # Display the relevant documents
    # results = []
    # for index in top_indices:
    #     if similarities[index] > 0:  # Only include results with non-zero similarity
    #         row = image_caption.iloc[index]
            # results.append({"file_name": row['image_id'], 'caption': row['caption'], 'similarity': similarities[index]})
    
    return top_indices

tokenizer_bert = AutoTokenizer.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')
model_bert = AutoModel.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')

def bert_model(query):
    query_tokens = tokenizer_bert([query], max_length=128, truncation=True, padding='max_length', return_tensors='pt')
    query_outputs = model_bert(**query_tokens)
    
    query_embeddings = query_outputs.last_hidden_state
    query_mask = query_tokens['attention_mask'].unsqueeze(-1).expand(query_embeddings.size()).float()
    query_masked_embeddings = query_embeddings * query_mask
    query_summed = torch.sum(query_masked_embeddings, 1)
    query_counted = torch.clamp(query_mask.sum(1), min=1e-9)
    query_mean_pooled = query_summed / query_counted

    query_embedding = query_mean_pooled.detach().numpy()

    caption_embeddings =  caption_embeddings = np.load("caption_embeddings.npy")

    similarities = cosine_similarity(query_embedding, caption_embeddings).flatten()

    TopK = 15
    top_indices = similarities.argsort()[::-1][:TopK]

    return top_indices

def search_with_dot_product(query):
    tokens = tokenizer_bert(query, max_length=128, truncation=True, padding='max_length', return_tensors='pt')
    print(tokens['input_ids'].shape)
    outputs = model_bert(**tokens)
    
    embeddings = outputs.last_hidden_state
    mask = tokens['attention_mask'].unsqueeze(-1).expand(embeddings.size()).float()
    masked_embeddings = embeddings * mask
    summed = torch.sum(masked_embeddings, 1)
    counted = torch.clamp(mask.sum(1), min=1e-9)
    query_embedding = summed / counted
    
    query_embedding = query_embedding.detach().numpy()

    caption_embeddings = np.load("caption_embeddings.npy")
    
    dot_products = np.dot(caption_embeddings, query_embedding.T).flatten()
    
    TopK = 15
    top_indices = dot_products.argsort()[::-1][:TopK]
    
    return top_indices


@app.post("/search/")
async def search_endpoint(request: Request):
    try:
        data = await request.json()
        query = data.get("query")
        algorithm = data.get("algorithm")

        if not query:
            return JSONResponse(content={"detail": "Query is missing"}, status_code=400)
        
        if not algorithm:
            return JSONResponse(content={"detail": "Algorithm is missing"}, status_code=400)

        if algorithm == "boolean":
            result = boolean_search(query)
        elif algorithm == "semantic":
            result = semantic_search_tfidf(query)
        elif algorithm == "bert":
            result = bert_model(query)
        elif algorithm == "bert_dot_product":
            result=search_with_dot_product(query)
        else:
            return JSONResponse(content={"detail": "Invalid algorithm selected"}, status_code=400)

        lst = []
        for index in result:
            row = image_caption.iloc[index]
            lst.append({"file_name": row['image_id'], 'caption': row['caption']})    

        return JSONResponse(content={"results": lst})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return HTMLResponse(content="", status_code=204) 



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
