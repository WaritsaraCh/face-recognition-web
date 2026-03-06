from fastapi.templating import Jinja2Templates
from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect
import cv2
import numpy as np
import json
import os
import base64
import asyncio
import glob
from collections import deque
from insightface.app import FaceAnalysis

import mediapipe.python.solutions.face_mesh as mp_face_mesh
import mediapipe.python.solutions.drawing_utils as mp_drawing

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        elif isinstance(obj, np.floating): return float(obj)
        elif isinstance(obj, np.ndarray): return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

router = APIRouter()
templates = Jinja2Templates(directory="templates")

face_app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
face_app.prepare(ctx_id=0, det_size=(640, 640))

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False, 
    max_num_faces=1, 
    refine_landmarks=True, 
    min_detection_confidence=0.5
)

landmark_history = deque(maxlen=10)
ear_history = deque(maxlen=5) 
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

EMBEDDINGS_DIR = "data/embeddings"
face_database = {}

def load_face_database():
    """โหลด embeddings โดยเก็บ URL รูปภาพไว้ด้วยเพื่อส่งให้เว็บแสดงผล"""
    global face_database
    face_database = {}
    if not os.path.exists(EMBEDDINGS_DIR):
        return
    
    json_files = glob.glob(os.path.join(EMBEDDINGS_DIR, "*.json"))
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                name = data['name']
           
                img_url = f"/{data['picture_path']}" if 'picture_path' in data else ""
                
                face_database[name] = {
                    'embedding': np.array(data['embedding']),
                    'image_url': img_url
                }
        except Exception as e:
            print(f"Error loading DB: {e}")

def cosine_similarity(embedding1, embedding2):
    return np.dot(embedding1, embedding2)

def find_best_match(face_embedding, threshold=0.55):
    if not face_database: return None, 0.0, ""
    
    best_match, best_score, best_image = None, 0.0, ""
    for name, db_data in face_database.items():
        similarity = cosine_similarity(face_embedding, db_data['embedding'])
        if similarity > best_score and similarity > threshold:
            best_score, best_match, best_image = similarity, name, db_data['image_url']
            
    return best_match, best_score, best_image

def compute_motion_score(landmarks):
    if len(landmark_history) < 2: return 0.0
    return np.linalg.norm(np.array(landmark_history[-2]) - np.array(landmarks))

def eye_aspect_ratio(eye_landmarks):
    A = np.linalg.norm(np.array(eye_landmarks[1]) - np.array(eye_landmarks[5]))
    B = np.linalg.norm(np.array(eye_landmarks[2]) - np.array(eye_landmarks[4]))
    C = np.linalg.norm(np.array(eye_landmarks[0]) - np.array(eye_landmarks[3]))
    return (A + B) / (2.0 * C)

def process_frame(frame):
    try:
        faces = face_app.get(frame)
        results = []
        for face in faces:
            bbox = face.bbox.astype(int)
            embedding = face.embedding / np.linalg.norm(face.embedding)
            name, confidence, image_url = find_best_match(embedding)
            
            results.append({
                'bbox': bbox.tolist(),
                'name': name if name else 'Unknown',
                'confidence': float(confidence),
                'image_url': image_url
            })
            
            color = (0, 255, 0) if name else (0, 0, 255)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            cv2.putText(frame, f"{name}: {confidence:.2f}" if name else "Unknown", 
                        (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Liveness
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results_mesh = face_mesh.process(rgb_frame)
        is_real = False
        
        if results_mesh.multi_face_landmarks:
            landmarks = results_mesh.multi_face_landmarks[0]
            coords = [(lm.x, lm.y, lm.z) for lm in landmarks.landmark]
            landmark_history.append(coords)

            motion_score = compute_motion_score(coords)
            if motion_score > 0.002: is_real = True

            ih, iw, _ = frame.shape
            points = [(int(lm.x * iw), int(lm.y * ih)) for lm in landmarks.landmark]
            left_ear = eye_aspect_ratio([points[i] for i in LEFT_EYE_IDX])
            right_ear = eye_aspect_ratio([points[i] for i in RIGHT_EYE_IDX])
            avg_ear = (left_ear + right_ear) / 2.0
            
            ear_history.append(avg_ear)
            if avg_ear < 0.20: is_real = True 

            label = "Live" if is_real else "Fake/Static"
            cv2.putText(frame, f"Liveness: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        return frame, results
    except Exception as e:
        print(f"Frame Error: {e}")
        return frame, []

@router.get("/face")
async def face_recog(request: Request):
    load_face_database()
    return templates.TemplateResponse("face.html", {"request": request})

@router.websocket("/ws/face")
async def websocket_face_recognition(websocket: WebSocket):
    await websocket.accept()
    load_face_database()
    try:
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            processed_frame, recognition_results = process_frame(frame)
            _, buffer = cv2.imencode('.jpg', processed_frame)
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            
            await websocket.send_text(json.dumps({
                'frame': frame_base64,
                'faces': recognition_results
            }, cls=NumpyEncoder))
            
            await asyncio.sleep(0.03) 
    except Exception as e: 
        print(f"WS Disconnected: {e}")
    finally: 
        cap.release()

@router.post("/reload_database")
async def reload_db():
    load_face_database()
    return {"status": "success"}

@router.get("/face_count")
async def get_count():
    return {"count": len(face_database), "names": list(face_database.keys())}