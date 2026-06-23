FROM python:3.11-slim

WORKDIR /app

# System deps required by faiss-cpu, sentence-transformers, and scikit-learn
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# Install ML pipeline deps first (large, cached layer)
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install backend-specific deps
COPY backend/requirements.txt ./backend_requirements.txt
RUN pip install --no-cache-dir -r backend_requirements.txt

# Pre-download the embedding model so cold starts don't hit the network
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy application code
COPY src/ ./src/
COPY backend/ ./backend/

# Create model dirs — populated at runtime (gitignored, not in build context)
RUN mkdir -p /app/models/faiss_index /app/models/drain_state

ENV PYTHONPATH=/app/src
ENV PORT=8000

EXPOSE 8000

WORKDIR /app/backend
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
