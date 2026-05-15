import os
import json
import base64
import requests
import numpy as np
import cv2
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from deepface import DeepFace
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="EEMC Face-ID AI Service")

# Cấu hình CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cấu hình Model
MODEL_NAME = "Facenet"
DETECTOR_BACKEND = "opencv"

print(f"Khởi tạo mô hình {MODEL_NAME}...")
try:
    DeepFace.build_model(MODEL_NAME)
    print("Mô hình đã sẵn sàng!")
except Exception as e:
    print(f"Lỗi khởi tạo: {e}")

class FaceRequest(BaseModel):
    username: Optional[str] = None
    image: str
    callback_url: Optional[str] = None

def decode_base64_image(base64_str):
    try:
        if "," in base64_str:
            base64_str = base64_str.split(",")[1]
        img_data = base64.b64decode(base64_str)
        nparr = np.frombuffer(img_data, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception as e:
        raise HTTPException(status_code=400, detail="Lỗi giải mã ảnh")

def get_face_embedding(image):
    results = DeepFace.represent(
        img_path=image, 
        model_name=MODEL_NAME,
        detector_backend=DETECTOR_BACKEND,
        enforce_detection=True,
        align=True
    )
    if not results:
        raise Exception("Không tìm thấy khuôn mặt")
    return results[0]["embedding"]

@app.post("/register")
async def register(req: FaceRequest):
    try:
        img = decode_base64_image(req.image)
        embedding = get_face_embedding(img)
        vector_str = json.dumps(embedding)
        
        api_url = req.callback_url if req.callback_url else "https://eemc.com.vn/calendar/backend/api_face.php"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Content-Type": "application/json"
        }
        
        resp = requests.post(api_url, json={"user_id": req.username, "face_token": vector_str}, headers=headers, timeout=10)
        return {"status": "success", "message": resp.text[:100]}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/verify")
async def verify(req: FaceRequest):
    """Endpoint dành cho Đăng nhập: Chỉ trả về Vector"""
    try:
        img = decode_base64_image(req.image)
        embedding = get_face_embedding(img)
        return {"status": "success", "embedding": embedding}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/")
async def root():
    return {"message": "AI is running", "mode": "FastAPI"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
