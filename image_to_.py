import torch
from ultralytics import YOLO
import base64
from io import BytesIO
from PIL import Image
import numpy as np

def load_class_names(file_path="coco.names"):
    with open(file_path, "r") as f:
        return f.read().strip().split("\n")

def detect_objects_and_scene(image_path):

    base64_data = image_path.split(",")[1]
    
    # Decode it to an image
    image_data = base64.b64decode(base64_data)
    image = Image.open(BytesIO(image_data))
    model = YOLO("yolov8n.pt") 
    results = model(image)  

    class_names = load_class_names("coco.names")

    
    detected_objects = [
        class_names[int(class_id)]
        for res in results
        for class_id in res.boxes.cls
    ]

    return detected_objects



