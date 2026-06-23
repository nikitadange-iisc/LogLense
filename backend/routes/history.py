"""
Session history routes (ChatGPT-style past sessions).

GET  /api/history                      — list past completed pipeline runs
POST /api/history/{session_id}/activate — load a past session as the active one
"""
import sys
import logging
import types
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from state import app_state

logger = logging.getLogger(__name__)
router = APIRouter()


def _deserialize_session(d: dict):
    return types.SimpleNamespace(
        session_id    = d["session_id"],
        raw_lines     = d.get("raw_lines", []),
        events        = d.get("events", []),
        label         = d.get("label"),
        anomaly_score = d.get("anomaly_score"),
        line_range    = tuple(d["line_range"]) if d.get("line_range") else (0, 0),
    )


def _load_session_sync(record: dict):
    """Blocking — run in threadpool."""
    from embedder import SessionEmbedder
    from vector_store import FAISSVectorStore
    from rag_pipeline import RAGPipeline

    dataset   = record["dataset"]
    index_dir = record["index_dir"]

    # Reuse already-loaded embedder if available, otherwise create one
    if app_state.embedder is not None:
        embedder = app_state.embedder
    else:
        embedder = SessionEmbedder(model_name="all-MiniLM-L6-v2")

    store = FAISSVectorStore(dimension=embedder.dimension)
    store.load(index_dir)

    rag = RAGPipeline(
        embedder=embedder,
        vector_store=store,
        dataset=dataset,
        llm_provider="auto",
    )

    sessions = [_deserialize_session(d) for d in record.get("sessions_data", [])]

    return sessions, rag, embedder, store


@router.get("/history")
async def list_history():
    """Return metadata for all past pipeline runs (newest first)."""
    return [
        {
            "session_id": r["session_id"],
            "filename":   r["filename"],
            "dataset":    r["dataset"],
            "created_at": r["created_at"],
            "stats":      r["stats"],
        }
        for r in app_state.session_history
    ]


@router.post("/history/{session_id}/activate")
async def activate_session(session_id: str):
    """Load a past session as the active pipeline (enables dashboard + chat)."""
    record = next(
        (r for r in app_state.session_history if r["session_id"] == session_id), None
    )
    if record is None:
        raise HTTPException(status_code=404, detail="Session not found in history.")

    # Already active — nothing to do
    if app_state.active_session_id == session_id and app_state.rag_pipeline is not None:
        return {"status": "already_active", "session_id": session_id}

    try:
        sessions, rag, embedder, store = await run_in_threadpool(_load_session_sync, record)
    except Exception as exc:
        logger.error("Failed to activate session %s: %s", session_id, exc)
        raise HTTPException(status_code=500, detail=f"Failed to load session: {exc}")

    app_state.sessions         = sessions
    app_state.rag_pipeline     = rag
    app_state.embedder         = embedder
    app_state.vector_store     = store
    app_state.analysis_cache   = dict(record.get("analysis_cache", {}))
    app_state.active_session_id = session_id
    app_state.index_stats      = {
        "size":            record["stats"].get("index_size", 0),
        "dataset":         record["dataset"],
        "embedding_model": "all-MiniLM-L6-v2",
        "llm_provider":    rag.provider,
        "llm_model":       rag.model,
    }

    # Reflect ready state in the job dict so dashboard renders correctly
    app_state.set_ready(record["stats"])

    logger.info("Activated session %s (%s, %d sessions)",
                session_id, record["filename"], len(sessions))

    return {
        "status":     "activated",
        "session_id": session_id,
        "filename":   record["filename"],
        "dataset":    record["dataset"],
        "stats":      record["stats"],
    }
