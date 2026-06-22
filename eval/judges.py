"""
LLM-as-Judge chains for the LogLense evaluation pipeline.

Each judge is a LangChain chain:
    ChatPromptTemplate | ChatAnthropic | JsonOutputParser

All judges return {"score": 0|1, "reason": str}.

The judge model is intentionally a cheaper/faster model (haiku) to keep
evaluation costs low while still catching clear failures.
"""

import logging
import os
from typing import Optional

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from eval.metrics import JUDGE_DIMENSIONS, JUDGE_RUBRICS

load_dotenv()

logger = logging.getLogger(__name__)

JUDGE_MODEL = os.getenv("EVAL_JUDGE_MODEL", "claude-haiku-4-5-20251001")
JUDGE_TEMPERATURE = 0.0  # deterministic scoring
JUDGE_MAX_TOKENS = 256    # rubrics only ask for a small JSON object


def _build_judge_chain(dimension: str) -> object:
    """Build a LangChain chain for a single judge dimension."""
    rubric = JUDGE_RUBRICS[dimension]
    prompt_text = rubric["rubric"]

    # Wrap the raw rubric template in a ChatPromptTemplate.
    # The rubric already has {placeholder} variables — ChatPromptTemplate.from_template
    # will pick those up automatically.
    prompt = ChatPromptTemplate.from_template(prompt_text)

    llm = ChatAnthropic(
        model=JUDGE_MODEL,
        temperature=JUDGE_TEMPERATURE,
        max_tokens=JUDGE_MAX_TOKENS,
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
    )

    return prompt | llm | JsonOutputParser()


# Build all six chains at import time (lazy init inside function to avoid
# import-time failures when ANTHROPIC_API_KEY is absent).
_CHAINS: dict = {}


def _get_chain(dimension: str):
    if dimension not in _CHAINS:
        _CHAINS[dimension] = _build_judge_chain(dimension)
    return _CHAINS[dimension]


# ── Per-dimension judge functions ─────────────────────────────────────────────

def _safe_invoke(chain, inputs: dict, dimension: str) -> dict:
    """Invoke a chain and return {score, reason} on success or {score: 0, reason: error} on failure."""
    try:
        result = chain.invoke(inputs)
        score = int(result.get("score", 0))
        reason = str(result.get("reason", ""))
        return {"score": score, "reason": reason}
    except Exception as e:
        logger.warning(f"Judge '{dimension}' failed: {e}")
        return {"score": 0, "reason": f"Judge error: {e}"}


def judge_root_cause_specificity(
    session_id: str,
    root_cause: str,
    log_lines: list,
) -> dict:
    chain = _get_chain("root_cause_specificity")
    inputs = {
        "session_id": session_id,
        "root_cause": root_cause,
        "log_lines": "\n".join(log_lines[:15]),
    }
    return _safe_invoke(chain, inputs, "root_cause_specificity")


def judge_grounding(
    log_lines: list,
    failure_trace: list,
) -> dict:
    chain = _get_chain("grounding")
    # failure_trace is a list of dicts with "line" and "annotation"
    trace_text = "\n".join(
        item["line"] if isinstance(item, dict) else str(item)
        for item in (failure_trace or [])
    )
    inputs = {
        "log_lines": "\n".join(log_lines),
        "failure_trace": trace_text or "(empty — no failure trace provided)",
    }
    return _safe_invoke(chain, inputs, "grounding")


def judge_completeness(
    log_lines: list,
    explanation: str,
) -> dict:
    chain = _get_chain("completeness")
    inputs = {
        "log_lines": "\n".join(log_lines),
        "explanation": explanation or "(empty)",
    }
    return _safe_invoke(chain, inputs, "completeness")


def judge_severity_calibration(
    severity: str,
    root_cause: str,
    log_lines: list,
) -> dict:
    chain = _get_chain("severity_calibration")
    inputs = {
        "severity": severity or "unknown",
        "root_cause": root_cause or "(empty)",
        "log_lines": "\n".join(log_lines),
    }
    return _safe_invoke(chain, inputs, "severity_calibration")


def judge_actionability(
    session_id: str,
    recommended_action: str,
    root_cause: str,
) -> dict:
    chain = _get_chain("actionability")
    inputs = {
        "session_id": session_id,
        "recommended_action": recommended_action or "(empty)",
        "root_cause": root_cause or "(empty)",
    }
    return _safe_invoke(chain, inputs, "actionability")


def judge_retrieval_relevance(
    log_lines: list,
    retrieved_examples: list,
) -> dict:
    chain = _get_chain("retrieval_relevance")
    if not retrieved_examples:
        examples_text = "(no examples retrieved)"
    else:
        parts = []
        for i, ex in enumerate(retrieved_examples[:3], 1):
            if isinstance(ex, dict):
                rc = ex.get("root_cause", "Unknown")
                sid = ex.get("session_id", "?")
                parts.append(f"Example {i} [{sid}]: {rc}")
            else:
                parts.append(f"Example {i}: {ex}")
        examples_text = "\n".join(parts)

    inputs = {
        "log_lines": "\n".join(log_lines[:10]),
        "retrieved_examples": examples_text,
    }
    return _safe_invoke(chain, inputs, "retrieval_relevance")


# ── Full session judging ──────────────────────────────────────────────────────

def judge_all_dimensions(
    analysis_result: dict,
    raw_log_lines: list,
    session_id: Optional[str] = None,
) -> dict:
    """
    Run all 6 judge dimensions for a single analysis result.

    Args:
        analysis_result: Dict returned by RAGPipeline.analyze().
        raw_log_lines:   The session's actual raw log lines (not from the result).
        session_id:      Override session ID (defaults to analysis_result["session_id"]).

    Returns:
        {dimension: {"score": 0|1, "reason": str}, ...}
    """
    sid = session_id or analysis_result.get("session_id", "unknown")
    root_cause = analysis_result.get("root_cause", "")
    explanation = analysis_result.get("explanation", "")
    failure_trace = analysis_result.get("failure_trace", [])
    severity = analysis_result.get("severity", "")
    recommended_action = analysis_result.get("recommended_action", "")
    retrieved_examples = analysis_result.get("retrieved_examples", [])

    scores = {}
    scores["root_cause_specificity"] = judge_root_cause_specificity(
        session_id=sid,
        root_cause=root_cause,
        log_lines=raw_log_lines,
    )
    scores["grounding"] = judge_grounding(
        log_lines=raw_log_lines,
        failure_trace=failure_trace,
    )
    scores["completeness"] = judge_completeness(
        log_lines=raw_log_lines,
        explanation=explanation,
    )
    scores["severity_calibration"] = judge_severity_calibration(
        severity=severity,
        root_cause=root_cause,
        log_lines=raw_log_lines,
    )
    scores["actionability"] = judge_actionability(
        session_id=sid,
        recommended_action=recommended_action,
        root_cause=root_cause,
    )
    scores["retrieval_relevance"] = judge_retrieval_relevance(
        log_lines=raw_log_lines,
        retrieved_examples=retrieved_examples,
    )

    logger.info(
        f"Judged session {sid}: "
        + " | ".join(f"{d}={scores[d]['score']}" for d in JUDGE_DIMENSIONS)
    )
    return scores
