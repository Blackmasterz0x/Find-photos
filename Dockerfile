FROM python:3.10-slim

WORKDIR /app

# ติดตั้ง system dependencies ที่จำเป็น
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt-get/lists/*

# คัดลอกและติดตั้ง python packages ก่อน
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ดาวน์โหลดโมเดล InsightFace
ARG MODEL_NAME=buffalo_sc
ENV MODEL_NAME=$MODEL_NAME OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
RUN python -c "from insightface.app import FaceAnalysis; FaceAnalysis(name='$MODEL_NAME', allowed_modules=['detection', 'recognition'], providers=['CPUExecutionProvider']).prepare(ctx_id=0)"

COPY . .

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]