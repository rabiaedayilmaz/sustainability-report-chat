# syntax=docker/dockerfile:1.6
#
# Single-image build for the NTT DATA Sustainability RAG service.
# - Embedding model + PaddleOCR models are downloaded on first run and cached
#   under /root/.cache/huggingface and /root/.paddlex (mount as volumes in
#   docker-compose to persist across rebuilds).
# - Qdrant runs externally (Qdrant Cloud); no service container needed.
# - Ollama runs as a sibling container (see docker-compose.yml).
#
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/root/.cache/huggingface \
    PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=true

# System libs required by paddlepaddle, PyMuPDF, OpenCV (paddleocr's runtime dep)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libgl1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first so they cache between code changes.
COPY requirements.txt .
RUN pip install -r requirements.txt

# Application code
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY app.py .

# Make sure cache dirs exist (volume mounts will overlay them).
RUN mkdir -p /root/.cache/huggingface /root/.paddlex

EXPOSE 8000

# Healthcheck hits /live — cheap, dependency-free, just confirms the process is responding.
# Use /ready if you want the orchestrator to also wait on Qdrant + Ollama.
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD python -c "import urllib.request,sys; \
        r=urllib.request.urlopen('http://localhost:8000/live', timeout=5); \
        sys.exit(0 if r.status==200 else 1)" || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
