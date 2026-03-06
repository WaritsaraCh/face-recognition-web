import json
import os
import shutil
import cv2
import numpy as np
from fastapi import APIRouter, File, UploadFile, Request
from fastapi.templating import Jinja2Templates
from google.cloud import vision
from ultralytics import YOLO
from insightface.app import FaceAnalysis

router = APIRouter()
templates = Jinja2Templates(directory="templates")

UPLOAD_DIR = "data/uploaded_images"
CROPPED_DIR = "data/pictures"
EMBEDDINGS_DIR = "data/embeddings"

for directory in [UPLOAD_DIR, CROPPED_DIR, EMBEDDINGS_DIR]:
    os.makedirs(directory, exist_ok=True)

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "orcsofta-b7dbda7ce938.json"


client = vision.ImageAnnotatorClient()
model = YOLO('id_card_detect_model.pt')
face_model = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
face_model.prepare(ctx_id=0, det_size=(640,640))

def extract_text_from_image(image_bytes):
    vision_image = vision.Image(content=image_bytes)
    response = client.text_detection(image=vision_image)
    texts = response.text_annotations
    return texts[0].description.strip().replace("\n", "_") if texts else ""

@router.get("/upload")
async def upload_form(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})

@router.post("/upload")
async def handle_upload(request: Request, file: UploadFile = File(...)):
    file_location = f"{UPLOAD_DIR}/{file.filename}"
    
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

 
    results = model(file_location)
    image = cv2.imread(file_location)
    
    eng_name, picture_path = None, None
    saved_data = []

    cropped_objects = []
    if results and len(results[0].boxes) > 0:
        for result in results[0].boxes:
            box = result.xyxy[0].cpu().numpy().astype(int)
            class_id = int(result.cls[0].cpu().numpy())
            class_name = model.names[class_id].lower()
            
            x1, y1, x2, y2 = box
            cropped_image = image[y1:y2, x1:x2]
            cropped_objects.append({"class_name": class_name, "image": cropped_image})


    for obj in cropped_objects:
        if obj["class_name"] == "eng_name":
            _, encoded_image = cv2.imencode('.jpg', obj["image"])
            eng_name = extract_text_from_image(encoded_image.tobytes())
            break

 
    success = False
    if eng_name:
        for obj in cropped_objects:
            if obj["class_name"] == "picture":
               
                filename = f"{eng_name}.jpg"
                picture_path = os.path.join(CROPPED_DIR, filename).replace("\\", "/")
                cv2.imwrite(picture_path, obj["image"])
                saved_data.append((eng_name, picture_path))

            
                faces = face_model.get(obj["image"])
                if faces:
                    embedding = faces[0].embedding
                    embedding = embedding / np.linalg.norm(embedding)
                    
                  
                    embedding_data = {
                        "name": eng_name,
                        "embedding": embedding.tolist(),
                        "picture_path": picture_path 
                    }
                    
                    with open(os.path.join(EMBEDDINGS_DIR, f"{eng_name}.json"), 'w', encoding='utf-8') as f:
                        json.dump(embedding_data, f, ensure_ascii=False, indent=2)
        
       
        if saved_data:
            success = True

    return templates.TemplateResponse("upload.html", {
        "request": request,
        "filename": file.filename,
        "success": success,
        "saved_data": saved_data
    })