"""
In-memory singleton that holds pipeline state across requests.
Session history is persisted to disk so it survives server restarts.
"""
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)
_HISTORY_FILE = Path(__file__).resolve().parent.parent / "data" / "sessions_db.json"


@dataclass
class AppState:
    job: dict = field(default_factory=lambda: {
        "status":   "idle",
        "step":     "idle",
        "message":  "No file uploaded yet.",
        "progress_pct": 0,
        "stats":    None,
        "error":    None,
        # Module 1 parsing counters
        "parsing_total":        0,
        "parsing_scanned":      0,
        "parsing_kept":         0,
        "parsing_templates":    0,
        "parsing_rate":         0.0,
        "parsing_current_line": "",
        # Module 2 sub-stage label
        "m2_stage": "",
        # Module 3 embedding counters
        "embed_done":  0,
        "embed_total": 0,
        # Set to the step name when an error occurs, so the UI can show ✗
        "failed_at_step": None,
    })
    cancel_requested: bool = False
    sessions: list = field(default_factory=list)
    rag_pipeline: Any = None
    index_stats: dict = field(default_factory=dict)
    analysis_cache: dict = field(default_factory=dict)
    embedder: Any = None
    vector_store: Any = None
    csv_path: str = ""        # path to Module 1 structured CSV
    raw_log_path: str = ""    # path to the original uploaded file (for raw log viewer)
    total_log_lines: int = 0  # cached line count for paginator
    score_distribution: list = field(default_factory=list)  # [{score, label, is_anomalous}]

    # ChatGPT-style session history (newest first, in-memory per server run)
    session_history: list = field(default_factory=list)
    active_session_id: str = ""

    def set_step(self, step: str, message: str, progress_pct: int):
        self.job["step"]         = step
        self.job["message"]      = message
        self.job["progress_pct"] = progress_pct
        self.job["status"]       = "running" if step not in ("idle", "ready", "error") else step
        self.job["error"]        = None

    def set_ready(self, stats: dict):
        self.job.update({
            "step":         "ready",
            "status":       "ready",
            "message":      "Pipeline complete.",
            "progress_pct": 100,
            "stats":        stats,
            "error":        None,
        })

    def set_error(self, message: str):
        self.job.update({
            "failed_at_step": self.job.get("step"),  # capture which step failed
            "step":   "error",
            "status": "error",
            "message": message,
            "error":   message,
        })

    def reset_for_new_run(self):
        self.cancel_requested = False
        self.csv_path = ""
        self.raw_log_path = ""
        self.total_log_lines = 0
        self.score_distribution = []
        self.job.update({
            "parsing_total": 0, "parsing_scanned": 0,
            "parsing_kept": 0,  "parsing_templates": 0,
            "parsing_rate": 0.0, "parsing_current_line": "",
            "m2_stage": "",
            "embed_done": 0,    "embed_total": 0,
            "stats": None,      "error": None,
            "failed_at_step": None,
        })

    def save_to_history(self, record: dict):
        """Prepend a completed session record (upsert by session_id) and persist to disk."""
        self.session_history = [
            r for r in self.session_history
            if r["session_id"] != record["session_id"]
        ]
        self.session_history.insert(0, record)
        self._save_to_disk()

    def _save_to_disk(self):
        try:
            _HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(_HISTORY_FILE, "w", encoding="utf-8") as f:
                json.dump({"sessions": self.session_history}, f, indent=2)
        except Exception as exc:
            logger.warning("Could not save session history to disk: %s", exc)

    def load_from_disk(self):
        """Load persisted session history on server startup."""
        if not _HISTORY_FILE.exists():
            return
        try:
            with open(_HISTORY_FILE, encoding="utf-8") as f:
                data = json.load(f)
            self.session_history = data.get("sessions", [])
            logger.info("Loaded %d past sessions from disk", len(self.session_history))
        except Exception as exc:
            logger.warning("Could not load session history from disk: %s", exc)


app_state = AppState()
