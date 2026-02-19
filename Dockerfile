FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    fonts-unifont \
    fonts-liberation \
    && rm -rf /var/lib/apt/lists/*

# Build uses requirements-docker.txt only; torch installed separately. See requirements.txt for full project deps.
COPY requirements-docker.txt ./

RUN pip install --no-cache-dir --upgrade pip && \
    (pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu torch==2.1.0+cpu || pip install --no-cache-dir "torch==2.1.0") && \
    echo "torch==$(pip show torch | sed -n 's/^Version: //p')" > /tmp/torch-constraint.txt && \
    pip install --no-cache-dir -r requirements-docker.txt -c /tmp/torch-constraint.txt

RUN playwright install chromium

COPY . .

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTORCH_ENABLE_MPS_FALLBACK=1
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV TORCH_NUM_THREADS=1
ENV MALLOC_TRIM_THRESHOLD_=100000
ENV MALLOC_MMAP_THRESHOLD_=100000

EXPOSE 8000

ENV PORT=8000
ENV LOW_MEMORY_MODE=true
ENV LAZY_LOAD_MODELS=true
ENV ENABLE_PLAYWRIGHT=false
ENV ENABLE_OCR=true
ENV EMBEDDING_CACHE_MAX_SIZE=1000
ENV MAX_CONCURRENT_ANALYZE=1

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD wget -q -O /dev/null --tries=1 http://localhost:8000/health || exit 1
CMD ["sh", "-c", "python -m uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1 --limit-concurrency 10"]

