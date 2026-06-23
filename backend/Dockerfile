# syntax=docker/dockerfile:1
#
# LogSense backend (FastAPI + the Module 1-4 pipeline in ../src)
# Build context is the PROJECT ROOT (see docker-compose.yml), so that we can
# install both the root requirements.txt and backend/requirements.txt and copy
# the shared src/ package the backend imports from.
#
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    # Cache the downloaded sentence-transformer weights inside the image/volume
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

# 1) Install dependencies first for better layer caching.
#    Root requirements = pipeline libs (drain3, scikit-learn, faiss, torch, ...)
#    backend requirements = the web layer (fastapi, uvicorn, ...)
COPY requirements.txt ./requirements.txt
COPY backend/requirements.txt ./backend/requirements.txt
# Install the CPU-only PyTorch build first, from PyTorch's CPU wheel index.
# This is critical: the default PyPI torch wheel pulls in several GB of CUDA
# libraries (nvidia-*, triton, cuda-toolkit) that are useless in this CPU-only
# container. Pre-installing the +cpu build satisfies the torch requirement so
# the subsequent `-r requirements.txt` resolve does not drag CUDA in.
RUN pip install --upgrade pip \
    && pip install --index-url https://download.pytorch.org/whl/cpu torch \
    && pip install -r requirements.txt \
    && pip install -r backend/requirements.txt

# 2) Copy application code.
COPY src/ ./src/
COPY backend/ ./backend/

# 3) Runtime directories (also bind-mounted from the host via compose so that
#    generated models/indexes/processed data persist across restarts).
RUN mkdir -p models data/processed data/raw "$HF_HOME"

# The backend resolves PROJECT_ROOT as the parent of the backend/ folder and
# runs `uvicorn main:app`, so we launch from inside /app/backend.
WORKDIR /app/backend

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
