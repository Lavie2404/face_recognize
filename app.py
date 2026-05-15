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

app = FastAPI(title="EEMC Face-ID Microservice (DeepFace Version)")

# Cấu hình CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cấu hình Model - Facenet cho kết quả 128-d ổn định
MODEL_NAME = "Facenet"
DETECTOR_BACKEND = "opencv"

print(f"Đang khởi tạo mô hình {MODEL_NAME}...")
try:
    DeepFace.build_model(MODEL_NAME)
    print("Mô hình đã sẵn sàng!")
except Exception as e:
    print(f"Lưu ý: Không thể build model ngay. Lỗi: {e}")

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
    """Sử dụng DeepFace để trích xuất vector và gửi về Mắt Bão dạng JSON"""
    try:
        results = DeepFace.represent(
            img_path=image, 
            model_name=MODEL_NAME,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=True,
            align=True
        )
        
        if len(results) > 0:
            vector = results[0]["embedding"]
            vector_str = json.dumps(vector)
            
            # Payload gửi đi
            api_url = "https://eemc.com.vn/calendar/backend/api_face.php"
            payload = {
                "user_id": user_id,
                "face_token": vector_str
            }
            
            try:
                response = requests.post(api_url, json=payload, timeout=10)
                # In ra log để debug nếu Mắt Bão trả về lỗi không phải JSON
                print(f"Mắt Bão Response: {response.text}")
                result = response.json()
                return result.get("message", "Cập nhật thành công")
            except Exception as e:
                return f"Lỗi gọi API Mắt Bão: {str(e)} | Nội dung gốc: {response.text[:100]}"
        
        return "Không tìm thấy khuôn mặt rõ ràng"
        
    except Exception as e:
        return f"Lỗi xử lý AI: {str(e)}"

@app.post("/register")
async def register(req: FaceRequest):
    img = decode_base64_image(req.image)
    message = send_vector_to_matbao(req.username, img)
    
    if "thành công" in message.lower():
        return {"status": "success", "message": message}
    else:
        return {"status": "error", "message": message}

@app.get("/")
async def root():
    return {
        "message": "EEMC Face-ID AI Microservice is running!",
        "status": "Mô hình đã sẵn sàng!",
        "engine": "DeepFace (Facenet)"
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "engine": "DeepFace", "model": MODEL_NAME}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
