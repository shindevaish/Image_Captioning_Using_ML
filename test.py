import json
import numpy as np
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration, BertTokenizer, BertModel

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

with open('image_embeddings.json', 'r') as f:
    image_embeddings = json.load(f)

def get_text_embedding(query):
    inputs = tokenizer(query, return_tensors="pt")
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).cpu().numpy().flatten()

def search(query):
    query_embedding = get_text_embedding(query)
    results = []
    for image_file, image_embedding in image_embeddings.items():
        distance = np.linalg.norm(query_embedding - np.array(image_embedding))
        results.append((image_file, distance))

    results.sort(key=lambda x: x[1])
    return results

if __name__ == "__main__":
    query = "dog"  
    results = search(query)
    for img_file, distance in results:
        print(f"Image: {img_file}, Distance: {distance}")
