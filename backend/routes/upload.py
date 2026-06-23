"""
Upload, status, cancel, reset, and tryout routes.

POST   /api/upload   — accept log/csv file, run pipeline in background
GET    /api/status   — poll job progress
DELETE /api/pipeline — cancel the running pipeline
POST   /api/reset    — reset pipeline state back to idle
POST   /api/tryout   — quick demo on first 5 000 lines of HDFS.log (if present)
"""
import sys
import shutil
import logging
import traceback
import types
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from state import app_state

logger = logging.getLogger(__name__)
router = APIRouter()

UPLOAD_DIR = PROJECT_ROOT / "data" / "raw" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# For tryout demo only — not used for normal uploads
_DEMO_FILE    = PROJECT_ROOT / "data" / "raw" / "HDFS.log"
_DEMO_DATASET = "hdfs"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _count_lines(file_path: Path) -> int:
    count = 0
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            count += chunk.count(b"\n")
    return count


def _cancelled() -> bool:
    return app_state.cancel_requested


def _serialize_session(s) -> dict:
    """Convert a Session (dataclass or SimpleNamespace) to a plain dict for history storage."""
    return {
        "session_id":    s.session_id,
        "raw_lines":     list(s.raw_lines or []),
        "events":        list(s.events or []),
        "label":         s.label,
        "anomaly_score": getattr(s, "anomaly_score", None),
        "line_range":    list(s.line_range) if s.line_range else None,
    }


def _deserialize_session(d: dict):
    """Reconstruct a SimpleNamespace session from a stored dict."""
    return types.SimpleNamespace(
        session_id    = d["session_id"],
        raw_lines     = d.get("raw_lines", []),
        events        = d.get("events", []),
        label         = d.get("label"),
        anomaly_score = d.get("anomaly_score"),
        line_range    = tuple(d["line_range"]) if d.get("line_range") else (0, 0),
    )


# ---------------------------------------------------------------------------
# Core pipeline runner
# ---------------------------------------------------------------------------

def _run_pipeline(file_path: Path, dataset: str, max_lines: int = None,
                  session_id: str = None, original_filename: str = None):
    """
    Blocking pipeline — meant to run inside a BackgroundTask thread.
    """
    if session_id is None:
        session_id = str(uuid4())

    session_index_dir = PROJECT_ROOT / "models" / "faiss_index" / session_id
    session_index_dir.mkdir(parents=True, exist_ok=True)

    app_state.active_session_id = session_id
    app_state.reset_for_new_run()

    try:
        from module1_ingest_parse import run_module1
        from module2_session_anomaly import run_module2
        from module3_embed_index import run_module3
        from rag_pipeline import RAGPipeline

        suffix   = file_path.suffix.lower()
        csv_path = None

        # ── Module 1 ────────────────────────────────────────────────────────
        if suffix in (".log", ".txt"):
            app_state.set_step("parsing", "Counting lines in log file…", 5)
            total_lines = _count_lines(file_path)
            app_state.job.update({
                "parsing_total":     total_lines,
                "parsing_scanned":   0,
                "parsing_kept":      0,
                "parsing_templates": 0,
                "parsing_rate":      0.0,
            })
            app_state.set_step("parsing", f"Parsing {total_lines:,} lines with Drain3…", 8)
            logger.info("Module 1: parsing %s (%d lines)", file_path, total_lines)

            def on_m1_progress(scanned, kept, templates, elapsed, current_line=''):
                if _cancelled():
                    return
                rate = kept / elapsed if elapsed > 0 else 0
                pct  = 8 + int(min(scanned / total_lines, 1.0) * 17)
                app_state.job.update({
                    "parsing_scanned":      scanned,
                    "parsing_kept":         kept,
                    "parsing_templates":    templates,
                    "parsing_rate":         round(rate),
                    "parsing_current_line": current_line[:300],
                    "progress_pct":         pct,
                    "message":              f"Parsing — {scanned:,} / {total_lines:,} lines",
                })

            m1 = run_module1(
                str(file_path),
                dataset=dataset,
                max_lines=max_lines,
                on_progress=on_m1_progress,
                should_cancel=_cancelled,
            )
            if m1 is None or _cancelled():
                app_state.set_error("Cancelled by user.")
                return

            csv_path = m1["csv_path"]
            app_state.job.update({
                "parsing_scanned":   m1.get("total_lines", total_lines),
                "parsing_kept":      m1.get("dedup_lines", 0),
                "parsing_templates": m1.get("unique_event_ids", 0),
            })
            app_state.set_step(
                "detecting",
                f"Parsed {m1.get('total_lines', total_lines):,} lines "
                f"→ {m1.get('unique_event_ids', 0)} templates",
                25,
            )

        elif suffix == ".csv":
            csv_path = str(file_path)
            app_state.set_step("detecting", "CSV file — skipping parse step.", 25)
        else:
            raise ValueError(f"Unsupported file type: {suffix}")

        if _cancelled():
            app_state.set_error("Cancelled by user.")
            return

        # ── Module 2 ────────────────────────────────────────────────────────
        logger.info("Module 2: anomaly detection on %s", csv_path)

        def on_m2_stage(stage, message):
            if _cancelled():
                return
            _M2_PCT = {
                "loading":      26,
                "sessionizing": 30,
                "vectorizing":  38,
                "training":     44,
                "scoring":      52,
            }
            app_state.job["m2_stage"] = stage
            app_state.set_step("detecting", message, _M2_PCT.get(stage, 35))

        m2 = run_module2(csv_path, dataset=dataset, on_stage=on_m2_stage)

        if _cancelled():
            app_state.set_error("Cancelled by user.")
            return

        n_anomalous = m2["anomalous_sessions"]
        app_state.set_step(
            "detecting",
            f"Found {n_anomalous:,} anomalous sessions out of {m2['total_sessions']:,}",
            55,
        )

        # ── Module 3 ────────────────────────────────────────────────────────
        anomalous_sessions = m2["anomalous"]
        app_state.job.update({"embed_done": 0, "embed_total": len(anomalous_sessions)})
        app_state.set_step(
            "embedding",
            f"Embedding {len(anomalous_sessions):,} anomalous sessions…",
            58,
        )
        logger.info("Module 3: embedding %d anomalous sessions", len(anomalous_sessions))

        def on_m3_progress(done, total):
            if _cancelled():
                return
            pct = 58 + int(done / total * 27) if total else 85
            app_state.job.update({
                "embed_done":   done,
                "embed_total":  total,
                "progress_pct": pct,
                "message":      f"Embedding sessions — {done:,} / {total:,}",
            })

        m3 = run_module3(
            sessions=anomalous_sessions,
            dataset=dataset,
            embedding_model="all-MiniLM-L6-v2",
            index_dir=str(session_index_dir),
            on_progress=on_m3_progress,
            embedder=app_state.embedder,
        )

        if _cancelled():
            app_state.set_error("Cancelled by user.")
            return

        app_state.set_step("embedding", "Building RAG pipeline…", 88)
        rag = RAGPipeline(
            embedder=m3["embedder"],
            vector_store=m3["vector_store"],
            dataset=dataset,
            llm_provider="auto",
        )

        stats = {
            "total_sessions":     m2["total_sessions"],
            "anomalous_sessions": n_anomalous,
            "anomaly_rate_pct":   m2["anomaly_rate_pct"],
            "index_size":         m3["index_size"],
            "dataset":            dataset,
            "llm_provider":       rag.provider,
            "llm_model":          rag.model,
        }

        # Persist active state
        app_state.sessions       = anomalous_sessions
        app_state.rag_pipeline   = rag
        app_state.embedder       = m3["embedder"]
        app_state.vector_store   = m3["vector_store"]
        app_state.analysis_cache = {}
        app_state.index_stats    = {
            "size":            m3["index_size"],
            "dataset":         dataset,
            "embedding_model": m3["embedding_model"],
            "llm_provider":    rag.provider,
            "llm_model":       rag.model,
        }

        # Save to ChatGPT-style history
        fname = original_filename or file_path.name
        app_state.save_to_history({
            "session_id":   session_id,
            "filename":     fname,
            "dataset":      dataset,
            "created_at":   datetime.now(timezone.utc).isoformat(),
            "stats":        stats,
            "index_dir":    str(session_index_dir),
            "sessions_data": [_serialize_session(s) for s in anomalous_sessions],
            "analysis_cache": {},
        })

        app_state.set_ready(stats)
        logger.info("Pipeline complete — %d sessions indexed (session=%s)",
                    m3["index_size"], session_id)

    except Exception as exc:
        logger.error("Pipeline failed: %s", exc)
        logger.debug(traceback.format_exc())
        app_state.set_error(str(exc))


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post("/upload")
async def upload_log(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    dataset: str = Form(...),
):
    if dataset not in ("hdfs", "bgl", "thunderbird"):
        raise HTTPException(status_code=422, detail="dataset must be hdfs, bgl, or thunderbird")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in (".log", ".txt", ".csv"):
        raise HTTPException(status_code=422, detail="Only .log, .txt, or .csv files are accepted")

    if app_state.job["step"] not in ("idle", "ready", "error"):
        raise HTTPException(status_code=409, detail="A pipeline is already running.")

    dest = UPLOAD_DIR / file.filename
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    session_id = str(uuid4())
    app_state.set_step("parsing", f"File '{file.filename}' received. Starting pipeline…", 5)
    background_tasks.add_task(
        _run_pipeline, dest, dataset, None, session_id, file.filename
    )

    return {"status": "accepted", "filename": file.filename, "dataset": dataset,
            "session_id": session_id}


@router.get("/status")
async def get_status():
    return app_state.job


@router.delete("/pipeline")
async def cancel_pipeline():
    if app_state.job["step"] in ("idle", "ready", "error"):
        raise HTTPException(status_code=400, detail="No pipeline is currently running.")
    app_state.cancel_requested = True
    app_state.job["message"] = "Cancellation requested — stopping after current step…"
    return {"status": "cancelling"}


@router.post("/reset")
async def reset_pipeline():
    """Reset pipeline state back to idle so a new file can be uploaded."""
    if app_state.job["step"] not in ("idle", "ready", "error"):
        raise HTTPException(status_code=409, detail="Cannot reset while pipeline is running.")
    app_state.job.update({
        "step":     "idle",
        "status":   "idle",
        "message":  "No file uploaded yet.",
        "progress_pct": 0,
        "stats":    None,
        "error":    None,
        "parsing_total": 0, "parsing_scanned": 0,
        "parsing_kept": 0,  "parsing_templates": 0,
        "parsing_rate": 0.0, "parsing_current_line": "",
        "m2_stage": "",
        "embed_done": 0,    "embed_total": 0,
    })
    return {"status": "reset"}


@router.post("/tryout")
async def tryout(background_tasks: BackgroundTasks):
    """Quick demo: run on first 5 000 lines of HDFS.log (if present)."""
    if not _DEMO_FILE.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Demo file not found: {_DEMO_FILE}. "
                   "Place HDFS.log in data/raw/ to enable the tryout mode."
        )
    if app_state.job["step"] not in ("idle", "ready", "error"):
        raise HTTPException(status_code=409, detail="A pipeline is already running.")

    session_id = str(uuid4())
    app_state.set_step("parsing", "Try-out mode: sampling 5 000 lines from HDFS.log…", 5)
    background_tasks.add_task(
        _run_pipeline, _DEMO_FILE, _DEMO_DATASET, 5_000, session_id, "HDFS.log (demo)"
    )
    return {"status": "accepted", "mode": "tryout", "max_lines": 5_000,
            "session_id": session_id}
