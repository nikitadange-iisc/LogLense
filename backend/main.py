"""
LogSense FastAPI backend.

Wraps the Python pipeline (Modules 1-4) and exposes HTTP endpoints
for the React frontend.

Run:
    cd backend
    uvicorn main:app --reload --port 8000
"""
import sys
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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
    """
    Load the sentence-transformer model weights at startup so the first
    upload doesn't pay the cold-load cost (~2-5s for MiniLM).
    Stores the loaded SessionEmbedder in app_state.embedder.
    """
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
    _preload_embedder()
    yield


app = FastAPI(
    title="LogSense API",
    description="RAG-powered log anomaly analysis",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from routes.upload import router as upload_router
from routes.sessions import router as sessions_router
from routes.chat import router as chat_router
from routes.history import router as history_router

app.include_router(upload_router, prefix="/api")
app.include_router(sessions_router, prefix="/api")
app.include_router(chat_router, prefix="/api")
app.include_router(history_router, prefix="/api")


@app.get("/api/health")
async def health():
    return {"status": "ok"}
