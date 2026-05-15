import os
import json
import base64
import requests
import face_recognition
import numpy as np
import cv2
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from deepface import DeepFace
from pydantic import BaseModel
import uvicorn

app = FastAPI()

# Cấu hình CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cho phép tất cả các nguồn (hoặc chỉ định ["http://127.0.0.1:5500"])
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Đường dẫn dữ liệu
DB_FILE = "database.json"
STATIC_DIR = "static"

# Tải trước model để tăng tốc xử lý cho các lần gọi sau
MODEL_NAME = "Facenet"
DETECTOR_BACKEND = "opencv"

print(f"Đang tải model {MODEL_NAME}...")
DeepFace.build_model(MODEL_NAME)
print("Model đã sẵn sàng!")

# Đảm bảo các thư mục tồn tại
if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR)

# Khởi tạo database nếu chưa có
if not os.path.exists(DB_FILE):
    with open(DB_FILE, "w", encoding="utf-8") as f:
        json.dump({}, f)

def load_db():
    try:
        with open(DB_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, ValueError):
        return {}

def save_db(data):
    with open(DB_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

class FaceRequest(BaseModel):
    username: Optional[str] = None
    image: str

def decode_base64_image(base64_str):
    try:
        # Loại bỏ tiền tố data:image/jpeg;base64, nếu có
        if "," in base64_str:
            base64_str = base64_str.split(",")[1]
        img_data = base64.b64decode(base64_str)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Lỗi giải mã ảnh: {str(e)}")

def send_vector_to_matbao(user_id, image):
    # Chuyển từ BGR (OpenCV) sang RGB (face_recognition)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 1. Trích xuất vector (128 dims)
    encodings = face_recognition.face_encodings(rgb_image)
    
    if len(encodings) > 0:
        # Chuyển vector thành chuỗi JSON
        vector_str = json.dumps(encodings[0].tolist())
        
        # 2. Gửi dữ liệu tới link API trên Mắt Bão
        api_url = "https://eemc.com.vn/calendar/backend/api_face.php"
        payload = {
            "user_id": user_id,
            "face_token": vector_str
        }
        
        try:
            response = requests.post(api_url, data=payload, timeout=10)
            result = response.json()
            return result["message"]
        except Exception as e:
            return f"Lỗi gọi API: {str(e)}"
            
    return "Không tìm thấy khuôn mặt"

@app.post("/register")
async def register(req: FaceRequest):
    img = decode_base64_image(req.image)
    
    # Sử dụng hàm gửi vector tới Mắt Bão
    message = send_vector_to_matbao(req.username, img)
    
    if "thành công" in message.lower():
        return {"status": "success", "message": message}
    else:
        return {"status": "error", "message": message}

@app.post("/verify")
async def verify(req: FaceRequest):
    img = decode_base64_image(req.image)
    db = load_db()
    
    if not db:
        return {"status": "error", "message": "Chưa có dữ liệu khuôn mặt nào được đăng ký"}
    
    try:
        # Trích xuất vector hiện tại
        result = DeepFace.represent(
            img, 
            model_name=MODEL_NAME, 
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=True
        )
        current_embedding = result[0]["embedding"]
        current_vec = np.array(current_embedding)
        
        # Tìm kiếm người dùng có độ tương đồng cao nhất
        best_match = None
        min_distance = 1.0  # Khoảng cách Cosine tối đa là 1.0
        
        # Ngưỡng (Threshold) tùy thuộc vào model
        # Facenet thường là 0.40, VGG-Face là 0.40
        threshold = 0.40
        
        for user, stored_embedding in db.items():
            stored_vec = np.array(stored_embedding)
            # Tính khoảng cách Cosine
            distance = 1 - np.dot(stored_vec, current_vec) / (np.linalg.norm(stored_vec) * np.linalg.norm(current_vec))
            
            if distance < min_distance:
                min_distance = distance
                best_match = user
        
        if best_match and min_distance < threshold:
            return {
                "status": "success", 
                "message": f"Xin chào {best_match}!", 
                "username": best_match,
                "distance": float(min_distance)
            }
        else:
            return {"status": "error", "message": "Không nhận diện được khuôn mặt này", "distance": float(min_distance)}
            
    except Exception as e:
        return {"status": "error", "message": f"Lỗi xác thực: {str(e)}"}

# Phục vụ file tĩnh
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")

if __name__ == "__main__":
    print("Đang khởi động Server tại http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
