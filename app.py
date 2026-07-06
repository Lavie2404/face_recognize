"""
EEMC Face-ID v2 - Dich vu trich xuat dac trung khuon mat
Feature: 001-faceid-rework | Hop dong: specs/001-faceid-rework/contracts/ai-service.md

Nguyen tac:
- CHI trich xuat embedding + cham chat luong. KHONG so khop, KHONG biet danh sach nhan vien.
- Tu choi la mac dinh: khong mat / mat nho / tin cay thap / nghi gia mao -> rejected.
- KHONG luu anh, KHONG log noi dung anh.
"""
import base64
import time
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

MODEL_VERSION = "facenet512-retinaface-v1"
MODEL_NAME = "Facenet512"
DETECTOR = "retinaface"
MIN_FACE_PX = 80
MIN_CONFIDENCE = 0.90

from deepface import DeepFace  # noqa: E402  (import cham, de sau cac hang so)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm-up: nap san model embedding + detector + anti-spoofing de request dau khong phai tai
    DeepFace.build_model(MODEL_NAME)
    dummy = np.zeros((160, 160, 3), dtype=np.uint8)
    try:
        DeepFace.extract_faces(
            img_path=dummy, detector_backend=DETECTOR,
            enforce_detection=False, align=True, anti_spoofing=True,
        )
    except Exception:
        pass  # anh den khong co mat - chi can nap trong so
    yield


app = FastAPI(lifespan=lifespan)

# CORS: chi can cho /health (ping danh thuc tu trinh duyet eemc.com.vn);
# /embed duoc goi server-to-server tu PHP nen khong can CORS.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://eemc.com.vn"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


class EmbedRequest(BaseModel):
    image: str
    client: str = ""


def _decode_image(data_url: str):
    """Giai ma base64 data-URL JPEG -> anh BGR (numpy). Tra None neu hong."""
    try:
        raw = data_url.split(",", 1)[1] if "," in data_url else data_url
        buf = np.frombuffer(base64.b64decode(raw), dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


@app.get("/health")
def health():
    return {"status": "ok", "model_version": MODEL_VERSION}


@app.post("/embed")
def embed(req: EmbedRequest):
    t0 = time.time()
    img = _decode_image(req.image)
    if img is None or img.size == 0:
        return {"status": "error", "message": "Khong giai ma duoc anh"}

    # 1) Detect + align + anti-spoofing (tu choi la mac dinh)
    try:
        faces = DeepFace.extract_faces(
            img_path=img,
            detector_backend=DETECTOR,
            enforce_detection=True,
            align=True,
            anti_spoofing=True,
        )
    except ValueError:
        return {"status": "rejected", "reason": "no_face",
                "model_version": MODEL_VERSION}
    except Exception as exc:  # loi he thong (thieu tai nguyen...)
        return {"status": "error", "message": str(exc)[:200]}

    faces = [f for f in faces if f.get("confidence", 0) > 0.5]
    if len(faces) == 0:
        return {"status": "rejected", "reason": "no_face",
                "model_version": MODEL_VERSION}
    if len(faces) > 1:
        return {"status": "rejected", "reason": "multiple_faces",
                "model_version": MODEL_VERSION}

    face = faces[0]
    area = face.get("facial_area", {})
    confidence = float(face.get("confidence", 0.0))
    is_real = bool(face.get("is_real", False))
    antispoof_score = float(face.get("antispoof_score", 0.0))

    if area.get("w", 0) < MIN_FACE_PX or area.get("h", 0) < MIN_FACE_PX:
        return {"status": "rejected", "reason": "face_too_small",
                "facial_area": {k: area.get(k) for k in ("x", "y", "w", "h")},
                "model_version": MODEL_VERSION}
    if confidence < MIN_CONFIDENCE:
        return {"status": "rejected", "reason": "low_confidence",
                "face_confidence": round(confidence, 4),
                "model_version": MODEL_VERSION}
    if not is_real:
        return {"status": "rejected", "reason": "spoof_suspect",
                "antispoof_score": round(antispoof_score, 4),
                "model_version": MODEL_VERSION}

    # 2) Embedding tren khuon mat DA duoc detect + align (khong detect lai)
    aligned = (face["face"] * 255).astype(np.uint8)  # extract_faces tra RGB 0..1
    aligned_bgr = cv2.cvtColor(aligned, cv2.COLOR_RGB2BGR)
    try:
        reps = DeepFace.represent(
            img_path=aligned_bgr,
            model_name=MODEL_NAME,
            detector_backend="skip",
            enforce_detection=False,
            align=False,
        )
    except Exception as exc:
        return {"status": "error", "message": str(exc)[:200]}

    if not reps or "embedding" not in reps[0]:
        return {"status": "error", "message": "Khong trich xuat duoc dac trung"}

    return {
        "status": "success",
        "embedding": reps[0]["embedding"],
        "facial_area": {k: area.get(k) for k in ("x", "y", "w", "h")},
        "face_confidence": round(confidence, 4),
        "antispoof_passed": True,
        "antispoof_score": round(antispoof_score, 4),
        "model_version": MODEL_VERSION,
        "processing_ms": int((time.time() - t0) * 1000),
    }
