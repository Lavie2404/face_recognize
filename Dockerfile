FROM python:3.10-slim

# Thu vien he thong cho opencv-headless + deepface
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Trong so model tai ve thu muc nay khi khoi dong lan dau
ENV DEEPFACE_HOME=/app/.deepface
RUN mkdir -p /app/.deepface && chmod 777 /app/.deepface

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

# HF Spaces (Docker SDK) yeu cau port 7860
EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
