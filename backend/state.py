"""
In-memory singleton that holds pipeline state across requests.
"""
from dataclasses import dataclass, field
from typing import Any


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
    })
    cancel_requested: bool = False
    sessions: list = field(default_factory=list)
    rag_pipeline: Any = None
    index_stats: dict = field(default_factory=dict)
    analysis_cache: dict = field(default_factory=dict)
    embedder: Any = None
    vector_store: Any = None

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
            "step":   "error",
            "status": "error",
            "message": message,
            "error":   message,
        })

    def reset_for_new_run(self):
        self.cancel_requested = False
        self.job.update({
            "parsing_total": 0, "parsing_scanned": 0,
            "parsing_kept": 0,  "parsing_templates": 0,
            "parsing_rate": 0.0, "parsing_current_line": "",
            "m2_stage": "",
            "embed_done": 0,    "embed_total": 0,
            "stats": None,      "error": None,
        })

    def save_to_history(self, record: dict):
        """Prepend a completed session record (upsert by session_id)."""
        self.session_history = [
            r for r in self.session_history
            if r["session_id"] != record["session_id"]
        ]
        self.session_history.insert(0, record)


app_state = AppState()
