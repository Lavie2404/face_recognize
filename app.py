import os
import json
import base64
import requests
import face_recognition
import numpy as np
import cv2
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from deepface import DeepFace
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="EEMC Face Recognition Microservice")

# Cấu hình CORS để cho phép Frontend từ Mắt Bão gọi tới
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Tải trước model để tăng tốc xử lý
MODEL_NAME = "Facenet"
DETECTOR_BACKEND = "opencv"

print(f"Đang tải model {MODEL_NAME}...")
DeepFace.build_model(MODEL_NAME)
print("Model đã sẵn sàng!")

class FaceRequest(BaseModel):
    username: Optional[str] = None
    image: str

def decode_base64_image(base64_str):
    try:
        if "," in base64_str:
            base64_str = base64_str.split(",")[1]
        img_data = base64.b64decode(base64_str)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Lỗi giải mã ảnh: {str(e)}")

def send_vector_to_matbao(user_id, image):
    """Trích xuất vector và gửi về server Mắt Bão"""
    # Chuyển sang RGB cho face_recognition
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Trích xuất vector (128 dims)
    encodings = face_recognition.face_encodings(rgb_image)
    
    if len(encodings) > 0:
        vector_str = json.dumps(encodings[0].tolist())
        
        # Gửi tới API trên Mắt Bão
        api_url = "https://eemc.com.vn/calendar/backend/api_face.php"
        payload = {
            "user_id": user_id,
            "face_token": vector_str
        }
        
        try:
            response = requests.post(api_url, data=payload, timeout=10)
            result = response.json()
            return result.get("message", "Cập nhật thành công")
        except Exception as e:
            return f"Lỗi gọi API Mắt Bão: {str(e)}"
            
    return "Không tìm thấy khuôn mặt trong ảnh"

@app.post("/register")
async def register(req: FaceRequest):
    """Endpoint nhận ảnh, trích xuất vector và đẩy về Mắt Bão"""
    img = decode_base64_image(req.image)
    message = send_vector_to_matbao(req.username, img)
    
    if "thành công" in message.lower():
        return {"status": "success", "message": message}
    else:
        return {"status": "error", "message": message}

@app.get("/health")
async def health():
    return {"status": "healthy", "model": MODEL_NAME}

if __name__ == "__main__":
    print("Đang khởi động Server xử lý ảnh tại http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
