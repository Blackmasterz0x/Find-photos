FROM python:3.10-slim

WORKDIR /app

# ติดตั้ง system dependencies ทั้งหมดที่ OpenCV และ ONNXRuntime บน Linux จำเป็นต้องใช้
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt-get/lists/*

# ติดตั้ง numpy<2 ล่วงหน้าเพื่อป้องกันปัญหา NumPy 2.x ขัดแย้งกับ ONNXRuntime
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir "numpy<2" && \
    pip install --no-cache-dir -r requirements.txt

# ดาวน์โหลดโมเดล InsightFace
ARG MODEL_NAME=buffalo_sc
ENV MODEL_NAME=$MODEL_NAME OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
RUN python -c "from insightface.app import FaceAnalysis; FaceAnalysis(name='$MODEL_NAME', allowed_modules=['detection', 'recognition'], providers=['CPUExecutionProvider']).prepare(ctx_id=0)"

COPY . .

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]