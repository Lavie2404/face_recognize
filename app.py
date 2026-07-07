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

# CORS cho trinh duyet eemc.com.vn: /health (ping GET) va /verify (cac trang
# diem danh cu goi POST truc tiep). /embed goi server-to-server tu PHP, khong can CORS.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://eemc.com.vn"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class EmbedRequest(BaseModel):
    image: str
    client: str = ""
    username: str = ""  # cac trang diem danh cu gui kem (khong bat buoc)


def _decode_image(data_url: str):
    """Giai ma base64 data-URL JPEG -> anh BGR (numpy). Tra None neu hong."""
    try:
        raw = data_url.split(",", 1)[1] if "," in data_url else data_url
        buf = np.frombuffer(base64.b64decode(raw), dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


@app.get("/")
def root():
    """Trang chu Space: xac nhan dich vu dang chay va liet ke endpoint."""
    return {
        "service": "EEMC Face-ID v2",
        "status": "ok",
        "model_version": MODEL_VERSION,
        "endpoints": ["/health", "/embed", "/verify"],
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_version": MODEL_VERSION}


def _detect_and_embed(img, do_antispoof):
    """
    Detect + align (RetinaFace) roi trich xuat embedding Facenet512.
    Tra ve tuple (result_dict, ok_bool):
      ok=True  -> result la {'embedding','facial_area','face_confidence','is_real','antispoof_score'}
      ok=False -> result la {'status':'rejected'|'error', 'reason'|'message', ...}
    """
    if img is None or img.size == 0:
        return ({"status": "error", "message": "Khong giai ma duoc anh"}, False)

    # Dò mặt: thử RetinaFace trước, không thấy thì thử các detector dự phòng.
    faces = None
    last_exc = None
    for det in (DETECTOR, "mtcnn", "opencv"):
        try:
            faces = DeepFace.extract_faces(
                img_path=img, detector_backend=det,
                enforce_detection=True, align=True, anti_spoofing=do_antispoof,
            )
            if faces:
                print(f"[embed] detector={det} tim thay {len(faces)} mat")
                break
        except ValueError as ve:
            last_exc = ve
            print(f"[embed] detector={det}: khong tim thay mat")
            faces = None
        except Exception as exc:
            last_exc = exc
            print(f"[embed] detector={det} loi: {exc}")
            faces = None

    if not faces:
        print(f"[embed] TAT CA detector khong thay mat (loi cuoi: {last_exc})")
        return ({"status": "rejected", "reason": "no_face", "model_version": MODEL_VERSION}, False)

    # Chon mat lon nhat; canh bao nhieu mat chi khi co >=2 mat cỡ tuong duong
    def _area(f):
        a = f.get("facial_area", {})
        return a.get("w", 0) * a.get("h", 0)
    faces = sorted(faces, key=_area, reverse=True)
    if len(faces) > 1 and _area(faces[1]) > 0.5 * _area(faces[0]):
        return ({"status": "rejected", "reason": "multiple_faces", "model_version": MODEL_VERSION}, False)

    face = faces[0]
    area = face.get("facial_area", {})
    # confidence co the thieu tuy phien ban DeepFace -> mac dinh cao (da qua enforce_detection)
    confidence = float(face.get("confidence", 0.99) or 0.99)
    is_real = bool(face.get("is_real", True))
    antispoof_score = float(face.get("antispoof_score", 0.0) or 0.0)

    if area.get("w", 0) < MIN_FACE_PX or area.get("h", 0) < MIN_FACE_PX:
        return ({"status": "rejected", "reason": "face_too_small",
                 "facial_area": {k: area.get(k) for k in ("x", "y", "w", "h")},
                 "model_version": MODEL_VERSION}, False)
    if do_antispoof and not is_real:
        return ({"status": "rejected", "reason": "spoof_suspect",
                 "antispoof_score": round(antispoof_score, 4), "model_version": MODEL_VERSION}, False)

    aligned = (face["face"] * 255).astype(np.uint8)  # extract_faces tra RGB 0..1
    aligned_bgr = cv2.cvtColor(aligned, cv2.COLOR_RGB2BGR)
    try:
        reps = DeepFace.represent(
            img_path=aligned_bgr, model_name=MODEL_NAME,
            detector_backend="skip", enforce_detection=False, align=False,
        )
    except Exception as exc:
        return ({"status": "error", "message": str(exc)[:200]}, False)

    if not reps or "embedding" not in reps[0]:
        return ({"status": "error", "message": "Khong trich xuat duoc dac trung"}, False)

    return ({
        "embedding": reps[0]["embedding"],
        "facial_area": {k: area.get(k) for k in ("x", "y", "w", "h")},
        "face_confidence": round(confidence, 4),
        "is_real": is_real,
        "antispoof_score": round(antispoof_score, 4),
    }, True)


@app.post("/embed")
def embed(req: EmbedRequest):
    """Face-ID v2: strict (co anti-spoofing), tra ve embedding + chi so chat luong."""
    t0 = time.time()
    res, ok = _detect_and_embed(_decode_image(req.image), do_antispoof=True)
    if not ok:
        return res
    return {
        "status": "success",
        "embedding": res["embedding"],
        "facial_area": res["facial_area"],
        "face_confidence": res["face_confidence"],
        "antispoof_passed": True,
        "antispoof_score": res["antispoof_score"],
        "model_version": MODEL_VERSION,
        "processing_ms": int((time.time() - t0) * 1000),
    }


@app.post("/verify")
def verify(req: EmbedRequest):
    """
    Tuong thich NGUOC voi cac trang diem danh cu goi Space nay (POST /verify
    {image, username} -> {status:'success', embedding}). Giu detect+align chuan
    nhung KHONG bat anti-spoofing (giu hanh vi cu de khong vo luong diem danh).
    """
    res, ok = _detect_and_embed(_decode_image(req.image), do_antispoof=False)
    if not ok:
        # Trang cu chi doc res.message khi status != success
        reason = res.get("reason")
        msg = res.get("message") or (
            "Face could not be detected" if reason == "no_face" else
            "Phat hien nhieu khuon mat" if reason == "multiple_faces" else
            "Anh chua dat chat luong"
        )
        return {"status": "error", "message": msg}
    return {"status": "success", "embedding": res["embedding"]}
