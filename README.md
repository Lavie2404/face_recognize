---
title: EEMC Face v2
emoji: 🔒
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# EEMC Face-ID v2 — Dịch vụ trích xuất đặc trưng khuôn mặt

Dịch vụ AI cho hệ thống EEMC Calendar (feature `001-faceid-rework`). Hợp đồng API:
`specs/001-faceid-rework/contracts/ai-service.md` trong repo chính.

## Nguyên tắc

- CHỈ trích xuất embedding (Facenet512, 512 chiều) + chấm chất lượng — KHÔNG so khớp, KHÔNG biết danh sách nhân viên, KHÔNG lưu ảnh.
- Từ chối là mặc định: không tìm thấy mặt / ≥2 mặt / mặt < 80px / độ tin cậy detector < 0.90 / nghi giả mạo (MiniFASNet) → `status: rejected`.
- Detect + align bắt buộc bằng RetinaFace (`enforce_detection=True`) — đây là điểm sửa gốc rễ lỗi "hồ sơ hút" của hệ thống cũ.

## Triển khai lên Hugging Face Space (CẬP NHẬT SPACE GỐC)

Cập nhật thẳng vào Space gốc `lavie2404/deepface` (KHÔNG tạo Space mới), URL:
`https://lavie2404-deepface.hf.space`.

1. Mở Space `lavie2404/deepface` trên Hugging Face → tab **Files**.
2. Thay/thêm 4 file: `README.md`, `Dockerfile`, `requirements.txt`, `app.py`.
   - Nếu Space gốc đang dùng SDK khác (Gradio…), cần đổi sang **SDK: Docker** trong Settings để chạy được FastAPI này (metadata `sdk: docker` đã có trong README).
3. Chờ build (~10 phút lần đầu — tải trọng số RetinaFace + Facenet512 + MiniFASNet).
4. Kiểm tra: `GET https://lavie2404-deepface.hf.space/health` → `{"status":"ok","model_version":"facenet512-retinaface-v1"}`.

**Tương thích ngược**: app.py giữ endpoint `POST /verify` (trả `{status:'success', embedding}`)
để các trang điểm danh cũ đang gọi Space này không bị vỡ; đồng thời thêm `POST /embed`
(strict, có anti-spoofing) cho Face-ID v2 và `GET /health`.

## Lưu ý phiên bản

Bất kỳ thay đổi model/detector nào PHẢI tăng `MODEL_VERSION` trong `app.py` — backend PHP
từ chối so khớp hồ sơ khác version (buộc đăng ký lại, tránh so khớp chéo hai không gian vector).
