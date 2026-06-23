"""
Session list and detail routes.

GET /api/sessions           — list of anomalous sessions
GET /api/sessions/{id}      — full detail with raw lines
"""
from fastapi import APIRouter, HTTPException

from state import app_state

router = APIRouter()


def _session_summary(s) -> dict:
    cached = app_state.analysis_cache.get(s.session_id, {})
    return {
        "session_id":    s.session_id,
        "anomaly_score": getattr(s, "anomaly_score", None),
        "label":         s.label,
        "num_lines":     len(s.raw_lines or []),
        "line_range":    list(s.line_range) if s.line_range else None,
        "severity":      cached.get("severity"),
        "analyzed":      bool(cached),
    }


@router.get("/sessions")
async def list_sessions():
    if not app_state.sessions:
        return []
    sessions = sorted(app_state.sessions, key=lambda s: getattr(s, "anomaly_score", 0) or 0)
    return [_session_summary(s) for s in sessions]


@router.get("/scores")
async def get_scores():
    """Score distribution for all sessions — used by the histogram chart."""
    return app_state.score_distribution


@router.get("/sessions/{session_id}")
async def get_session(session_id: str):
    session = next(
        (s for s in app_state.sessions if s.session_id == session_id), None
    )
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    cached = app_state.analysis_cache.get(session_id, {})
    events = getattr(session, "events", []) or []
    event_ids = [
        e.get("event_template", "") if isinstance(e, dict) else str(e)
        for e in events
    ]

    return {
        "session_id":    session.session_id,
        "anomaly_score": getattr(session, "anomaly_score", None),
        "label":         session.label,
        "line_range":    list(session.line_range) if session.line_range else None,
        "raw_lines":     (session.raw_lines or [])[:200],
        "event_sequence": event_ids[:50],
        "analysis":      cached or None,
    }
