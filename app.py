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

app = FastAPI(title="EEMC Face-ID Microservice (Final Fix)")

# Cấu hình CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cấu hình Model - Facenet
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
    callback_url: Optional[str] = None

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

def send_vector_to_matbao(user_id, image, custom_url=None):
    """Sử dụng DeepFace để trích xuất vector và gửi về Mắt Bão"""
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
            
            # Địa chỉ đích (ưu tiên từ frontend)
            api_url = custom_url if custom_url else "https://eemc.com.vn/calendar/backend/api_face.php"
            
            payload = {
                "user_id": user_id,
                "face_token": vector_str
            }

            # Giả lập trình duyệt để tránh bị chặn User-Agent
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Content-Type": "application/json"
            }
            
            try:
                print(f"Đang gửi dữ liệu về: {api_url}")
                response = requests.post(api_url, json=payload, headers=headers, timeout=10)
                print(f"Mắt Bão Response ({response.status_code}): {response.text[:500]}")
                result = response.json()
                return result.get("message", "Cập nhật thành công")
            except Exception as e:
                raw_text = response.text[:200] if 'response' in locals() else "Không có phản hồi"
                return f"Lỗi API: {str(e)} | Nội dung từ server: {raw_text}"
        
        return "Không tìm thấy khuôn mặt"
        
    except Exception as e:
        return f"Lỗi AI: {str(e)}"

@app.post("/register")
async def register(req: FaceRequest):
    img = decode_base64_image(req.image)
    message = send_vector_to_matbao(req.username, img, req.callback_url)
    
    if "thành công" in message.lower():
        return {"status": "success", "message": message}
    else:
        return {"status": "error", "message": message}

@app.get("/")
async def root():
    return {"message": "EEMC Face-ID AI Microservice is running!", "mode": "Dynamic Site Support"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
