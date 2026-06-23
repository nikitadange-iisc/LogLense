"""
Analyze and chat routes.

POST /api/analyze   — run Module 4 RAG on a single session
POST /api/chat      — free-form LLM Q&A with anomaly context
"""
import json
import logging
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from state import app_state

logger = logging.getLogger(__name__)
router = APIRouter()


class AnalyzeRequest(BaseModel):
    session_id: str
    top_k: int = 3


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    question: str
    session_id: Optional[str] = None
    history: List[ChatMessage] = []


@router.post("/analyze")
async def analyze_session(req: AnalyzeRequest):
    if app_state.rag_pipeline is None:
        raise HTTPException(status_code=400, detail="No pipeline loaded. Upload a log file first.")

    session = next(
        (s for s in app_state.sessions if s.session_id == req.session_id), None
    )
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    if req.session_id in app_state.analysis_cache:
        return app_state.analysis_cache[req.session_id]

    try:
        rag = app_state.rag_pipeline
        if rag.provider == "offline":
            result = rag.analyze_offline(session, top_k=req.top_k)
        else:
            result = rag.analyze(session, top_k=req.top_k)
        app_state.analysis_cache[req.session_id] = result
        return result
    except Exception as exc:
        logger.error("Analysis failed for %s: %s", req.session_id, exc)
        raise HTTPException(status_code=500, detail=str(exc))


def _build_chat_context(session_id: Optional[str]) -> str:
    stats = app_state.index_stats
    lines = [
        "You are an expert log analyst assistant for the LogSense system.",
        "Format answers in short paragraphs. If you answer a general question, put any LogSense-specific caveat on a new paragraph.",
        f"The current dataset contains {len(app_state.sessions)} anomalous log sessions.",
        f"Dataset: {stats.get('dataset', 'unknown').upper()}",
        f"Vector index size: {stats.get('size', 0)} sessions",
        "",
    ]

    if app_state.sessions:
        lines.append("Top anomalous sessions (by score):")
        for s in sorted(app_state.sessions, key=lambda x: getattr(x, "anomaly_score", 0) or 0)[:5]:
            cached = app_state.analysis_cache.get(s.session_id, {})
            sev   = cached.get("severity", "unanalyzed")
            score = getattr(s, "anomaly_score", None)
            score_str = f"{score:.4f}" if score is not None else "n/a"
            lines.append(f"  - {s.session_id}: score={score_str}, severity={sev}")
        lines.append("")

    if session_id:
        session = next((s for s in app_state.sessions if s.session_id == session_id), None)
        if session:
            lines.append(f"Currently focused session: {session_id}")
            lines.append(f"Anomaly score: {session.anomaly_score}")
            lines.append(f"Raw lines (first 30):")
            for ln in (session.raw_lines or [])[:30]:
                lines.append(f"  {ln}")
            lines.append("")

            cached = app_state.analysis_cache.get(session_id)
            if cached and "root_cause" in cached:
                lines.append("Cached analysis for this session:")
                lines.append(f"  Root cause: {cached.get('root_cause', 'n/a')}")
                lines.append(f"  Severity:   {cached.get('severity', 'n/a')}")
                lines.append(f"  Confidence: {cached.get('confidence', 0):.2f}")
                lines.append(f"  Explanation: {cached.get('explanation', 'n/a')[:400]}")

    return "\n".join(lines)


def _format_chat_answer(answer: str) -> str:
    """Keep short direct answers readable when the model appends a caveat."""
    answer = (answer or "").strip()
    for marker in (" However,", " However "):
        if marker in answer and "\n" not in answer[:answer.index(marker)]:
            return answer.replace(marker, "\n\n" + marker.strip(), 1)
    return answer


@router.post("/chat")
async def chat(req: ChatRequest):
    if app_state.rag_pipeline is None:
        raise HTTPException(status_code=400, detail="No pipeline loaded. Upload a log file first.")

    rag = app_state.rag_pipeline
    if rag.provider == "offline" or rag._client is None:
        return {
            "answer": (
                "No LLM API key configured. Set ANTHROPIC_API_KEY or OPENAI_API_KEY "
                "in your .env file and restart the server."
            ),
            "session_id": req.session_id,
        }

    system_ctx = _build_chat_context(req.session_id)

    messages = [{"role": m.role, "content": m.content} for m in req.history]
    messages.append({"role": "user", "content": req.question})

    try:
        if rag.provider == "claude":
            response = rag._client.messages.create(
                model=rag.model,
                max_tokens=1024,
                system=system_ctx,
                messages=messages,
                temperature=0.3,
            )
            answer = _format_chat_answer(response.content[0].text)

        else:  # openai
            openai_msgs = [{"role": "system", "content": system_ctx}] + messages
            response = rag._client.chat.completions.create(
                model=rag.model,
                messages=openai_msgs,
                temperature=0.3,
                max_tokens=1024,
            )
            answer = _format_chat_answer(response.choices[0].message.content)

        return {"answer": answer, "session_id": req.session_id}

    except Exception as exc:
        logger.error("Chat failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
