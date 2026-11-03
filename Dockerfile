FROM python:3.10-slim
RUN apt-get update && apt-get install -y --no-install-recommends build-essential libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# buffalo_sc = เล็ก ใช้ RAM น้อย (พอสำหรับโฮสต์ฟรี 512MB) | buffalo_l = แม่นกว่า ต้องใช้ RAM ~2GB
ARG MODEL_NAME=buffalo_sc
ENV MODEL_NAME=$MODEL_NAME OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
RUN python -c "from insightface.app import FaceAnalysis; FaceAnalysis(name='$MODEL_NAME', allowed_modules=['detection','recognition'], providers=['CPUExecutionProvider']).prepare(ctx_id=0)"
COPY . .
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
