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

## Triển khai lên Hugging Face Space

1. Đăng nhập tài khoản `lavie2404`, tạo Space mới tên `eemc-face-v2`, SDK: **Docker** (KHÔNG đè Space cũ đang phục vụ hệ thống hiện tại).
2. Upload 4 file: `README.md`, `Dockerfile`, `requirements.txt`, `app.py`.
3. Chờ build (~10 phút lần đầu — tải trọng số RetinaFace + Facenet512 + MiniFASNet).
4. Kiểm tra: `GET https://lavie2404-eemc-face-v2.hf.space/health` → `{"status":"ok","model_version":"facenet512-retinaface-v1"}`.

## Lưu ý phiên bản

Bất kỳ thay đổi model/detector nào PHẢI tăng `MODEL_VERSION` trong `app.py` — backend PHP
từ chối so khớp hồ sơ khác version (buộc đăng ký lại, tránh so khớp chéo hai không gian vector).
