"""
LogSense FastAPI backend.

Wraps the Python pipeline (Modules 1-4) and exposes HTTP endpoints
for the React frontend.

Run:
    cd backend
    python run.py          ← recommended (handles Ctrl+C cleanly)
    uvicorn main:app --reload --port 8000   ← also works
"""
import os

# Must be set before torch/transformers are imported anywhere.
# OMP_NUM_THREADS=1 stops OpenMP from spawning worker threads that block Ctrl+C on Windows.
# TOKENIZERS_PARALLELISM=false stops the HuggingFace fast tokenizer from doing the same.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import sys
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _preload_embedder():
    try:
        from embedder import SessionEmbedder
        from state import app_state
        logger.info("Pre-loading sentence-transformer model…")
        app_state.embedder = SessionEmbedder(model_name="all-MiniLM-L6-v2")
        logger.info("Embedder ready: %s (dim=%d)", app_state.embedder.model_name, app_state.embedder.dimension)
    except Exception as exc:
        logger.warning("Embedder pre-load failed (%s) — will load on first upload", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    from state import app_state
    app_state.load_from_disk()
    _preload_embedder()
    yield


app = FastAPI(
    title="LogSense API",
    description="RAG-powered log anomaly analysis",
    version="1.0.0",
    lifespan=lifespan,
)

# In production (Railway) frontend and backend share the same origin,
# so CORS is only needed for local dev. Override via ALLOWED_ORIGINS env var.
_origins = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173",
).split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _origins],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from routes.upload import router as upload_router
from routes.sessions import router as sessions_router
from routes.chat import router as chat_router
from routes.history import router as history_router
from routes.logs import router as logs_router

app.include_router(upload_router,   prefix="/api")
app.include_router(sessions_router, prefix="/api")
app.include_router(chat_router,     prefix="/api")
app.include_router(history_router,  prefix="/api")
app.include_router(logs_router,     prefix="/api")


@app.get("/api/health")
async def health():
    return {"status": "ok"}


# Serve the React SPA for all non-API routes.
# StaticFiles with html=True returns index.html for any path that isn't a
# real file — this is what enables client-side routing (/, /dashboard, etc.)
# to work after a hard refresh on Railway.
# Must be mounted LAST so /api/* routes above take priority.
_FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist"
if _FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=str(_FRONTEND_DIST), html=True), name="spa")
    logger.info("Serving React build from %s", _FRONTEND_DIST)
else:
    logger.info("No frontend/dist found — API-only mode (run 'npm run build' in frontend/)")
