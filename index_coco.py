from elasticsearch import Elasticsearch, helpers
from pycocotools.coco import COCO
import os

# Initialize Elasticsearch client
es = Elasticsearch([{'host': 'localhost', 'port': 9200, 'scheme': 'http'}])

# Path to COCO annotations
coco_annotation_path = '/Users/vaishnavishinde/Desktop/cocoapi/annotations/instances_train2017.json'

# Initialize COCO API
coco = COCO(coco_annotation_path)

# Create an index in Elasticsearch
index_name = 'coco_images'
if not es.indices.exists(index=index_name):
    es.indices.create(index=index_name)

# Index COCO images
def index_coco_images():
    img_ids = coco.getImgIds()
    imgs = coco.loadImgs(img_ids)
    actions = [
        {
            "_index": index_name,
            "_id": img['id'],
            "_source": {
                "file_name": img['file_name'],
                "coco_url": img['coco_url'],
                "height": img['height'],
                "width": img['width'],
                "id": img['id']
            }
        }
        for img in imgs
    ]
    helpers.bulk(es, actions)

if __name__ == "__main__":
    index_coco_images()
