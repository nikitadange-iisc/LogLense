# ── Stage 1: Build React frontend ─────────────────────────────────────────
FROM node:20-slim AS frontend-build
WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --ignore-scripts
COPY frontend/ ./
RUN npm run build

# ── Stage 2: Python backend ────────────────────────────────────────────────
FROM python:3.11-slim
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY backend/requirements.txt ./backend_requirements.txt
RUN pip install --no-cache-dir -r backend_requirements.txt

# Pre-download embedding model for fast cold starts
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Application source
COPY src/ ./src/
COPY backend/ ./backend/

# Built React app from stage 1 — FastAPI serves it via StaticFiles
COPY --from=frontend-build /frontend/dist ./frontend/dist

# Runtime directories (models and uploads persisted in Railway volume or ephemeral)
RUN mkdir -p /app/models/faiss_index /app/models/drain_state /app/data/raw/uploads

ENV PYTHONPATH=/app/src
ENV PORT=8000

EXPOSE 8000

WORKDIR /app/backend
CMD ["sh", "-c", "exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --timeout-graceful-shutdown 5"]
