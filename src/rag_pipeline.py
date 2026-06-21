"""
Stage 6: Retrieval-Augmented Generation for Log Root Cause Analysis

Embeds a flagged session, retrieves top-K similar historical anomalies from
FAISS, assembles a prompt with the flagged lines + retrieved examples, and
calls an LLM (Claude or OpenAI) for structured root cause analysis.

Provider selection (auto-detect order):
  1. ANTHROPIC_API_KEY  → Claude  (claude-sonnet-4-6 default)
  2. OPENAI_API_KEY     → OpenAI  (gpt-4o-mini default)
  3. Neither            → offline (retrieval + prompt only, no LLM call)
"""

import os
import json
import logging
import time
from typing import Optional

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# ── Dataset-aware system prompts ───────────────────────────────────────────

_SYSTEM_BASE = """You are an expert system log analyst specialising in root cause analysis of distributed system failures.

Your task:
1. Analyse the flagged anomalous log session carefully.
2. Use the similar historical failure examples as supporting context.
3. Identify the specific root cause of the failure.
4. Provide a line-level failure trace showing how the issue progressed.

Respond ONLY with valid JSON in this exact format:
{
    "root_cause": "Brief one-sentence description of the root cause",
    "affected_line_range": [start_line, end_line],
    "confidence": 0.0,
    "explanation": "Detailed explanation of the failure and how you identified it",
    "failure_trace": [
        {"line": "exact log line", "annotation": "what this line indicates"}
    ],
    "severity": "critical|high|medium|low",
    "recommended_action": "Concrete steps to resolve or investigate further"
}"""

_SYSTEM_HDFS = _SYSTEM_BASE + """

Domain context — HDFS distributed filesystem:
- Sessions are grouped by Block ID (e.g. blk_-3102267849859399193).
- Key actors: NameNode (metadata), DataNode (storage), PacketResponder (replication).
- Common failure modes: replication pipeline errors, block corruption, DataNode disconnects,
  write pipeline timeouts, packet loss during block transfer.
- Anomaly score: more negative = more anomalous (Isolation Forest output).
"""

_SYSTEM_BGL = _SYSTEM_BASE + """

Domain context — BlueGene/L supercomputer (BGL):
- Sessions are grouped by Node ID (e.g. R36-M0-N1-C:J15-U01).
- Key actors: kernel, RAS (Reliability/Availability/Serviceability), ciod (I/O daemon).
- Common failure modes: machine check exceptions (hardware errors), memory ECC errors,
  core dumps, FATAL kernel messages, I/O daemon connection failures, link training errors.
- FATAL-level events are hardware faults; multiple in one session = serious node failure.
- Anomaly score: more negative = more anomalous (Isolation Forest output).
"""

_SYSTEM_THUNDERBIRD = _SYSTEM_BASE + """

Domain context — Thunderbird supercomputer:
- Sessions are sliding windows of consecutive log lines.
- Common failure modes: kernel panics, hardware errors, network timeouts, daemon crashes.
- Anomaly score: more negative = more anomalous (Isolation Forest output).
"""

_SYSTEM_PROMPTS = {
    "hdfs":        _SYSTEM_HDFS,
    "bgl":         _SYSTEM_BGL,
    "thunderbird": _SYSTEM_THUNDERBIRD,
}


def _system_prompt(dataset: str) -> str:
    return _SYSTEM_PROMPTS.get(dataset.lower(), _SYSTEM_BASE)


# ── Provider detection ─────────────────────────────────────────────────────

def _detect_provider() -> str:
    """Return 'claude', 'openai', or 'offline' based on available API keys."""
    if os.getenv("ANTHROPIC_API_KEY"):
        return "claude"
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    return "offline"


_DEFAULT_MODELS = {
    "claude": "claude-sonnet-4-6",
    "openai": "gpt-4o-mini",
}


# ── RAGPipeline ────────────────────────────────────────────────────────────

class RAGPipeline:
    """Retrieval-Augmented Generation pipeline for log root cause analysis."""

    def __init__(self, embedder, vector_store,
                 dataset: str = "hdfs",
                 llm_provider: str = "auto",
                 model: str = None,
                 api_key: str = None):
        """
        Args:
            embedder      : SessionEmbedder — must match the model used to build the index.
            vector_store  : FAISSVectorStore — pre-loaded with anomalous sessions.
            dataset       : "hdfs" | "bgl" | "thunderbird" — selects domain system prompt.
            llm_provider  : "auto" | "claude" | "openai" | "offline".
                            "auto" tries ANTHROPIC_API_KEY then OPENAI_API_KEY.
            model         : Override default model (claude-sonnet-4-6 / gpt-4o-mini).
            api_key       : Explicit API key (overrides environment variable).
        """
        load_dotenv()

        self.embedder     = embedder
        self.vector_store = vector_store
        self.dataset      = dataset.lower()

        # Resolve provider
        self.provider = _detect_provider() if llm_provider == "auto" else llm_provider
        self.model    = model or _DEFAULT_MODELS.get(self.provider, "")
        self._client  = None

        if self.provider == "claude":
            key = api_key or os.getenv("ANTHROPIC_API_KEY")
            if key:
                from anthropic import Anthropic
                self._client = Anthropic(api_key=key)
            else:
                logger.warning("ANTHROPIC_API_KEY not set — falling back to offline mode")
                self.provider = "offline"

        elif self.provider == "openai":
            key = api_key or os.getenv("OPENAI_API_KEY")
            if key:
                from openai import OpenAI
                self._client = OpenAI(api_key=key)
            else:
                logger.warning("OPENAI_API_KEY not set — falling back to offline mode")
                self.provider = "offline"

        logger.info("RAGPipeline ready (provider=%s, model=%s, dataset=%s)",
                    self.provider, self.model, self.dataset)

    # ── Retrieval ──────────────────────────────────────────────────────────

    def retrieve_similar(self, session, top_k: int = 3,
                         exclude_self: bool = True,
                         extra_candidates: int = 5) -> list:
        """
        Embed session and retrieve top-K nearest anomalies from FAISS.

        Returns:
            List of (metadata_dict, distance) tuples sorted by distance.
        """
        embedding = self.embedder.embed_session(session, mode="hybrid")
        results   = self.vector_store.search(embedding, top_k=top_k)
        logger.debug("Retrieved %d examples for session %s",
                     len(results), session.session_id)
        return results

    # ── Prompt building ────────────────────────────────────────────────────

    def build_prompt(self, session, retrieved_examples: list) -> str:
        """
        Assemble the user prompt from the flagged session + retrieved examples.
        Uses actual stored metadata fields (event_sequence, anomaly_score, etc.)
        """
        score = getattr(session, "anomaly_score", None)
        score_str = f"{score:.4f}" if score is not None else "n/a"

        events = getattr(session, "events", []) or []
        event_ids = [
            e.get("event_template", "") if isinstance(e, dict) else str(e)
            for e in events
        ]
        event_str = " → ".join(event_ids[:30]) or "n/a"

        flagged_lines = "\n".join((session.raw_lines or [])[:100])

        prompt = f"""## FLAGGED ANOMALOUS LOG SESSION

Session ID    : {session.session_id}
Dataset       : {self.dataset.upper()}
Anomaly score : {score_str}  (more negative = more anomalous)
Line range    : {getattr(session, 'line_range', 'n/a')}
Event sequence: {event_str}
Total lines   : {len(session.raw_lines or [])}

### Raw Log Lines:
```
{flagged_lines}
```
"""

        if retrieved_examples:
            prompt += "\n## SIMILAR HISTORICAL ANOMALIES (from vector index)\n\n"
            for i, (meta, distance) in enumerate(retrieved_examples, 1):
                ex_lines   = meta.get("raw_lines", [])
                ex_text    = "\n".join(ex_lines[:30]) if ex_lines else "(no lines stored)"
                ex_events  = " → ".join(meta.get("event_sequence", [])[:20]) or "n/a"
                ex_score   = meta.get("anomaly_score")
                ex_score_s = f"{ex_score:.4f}" if ex_score is not None else "n/a"
                ex_label   = meta.get("label", "unknown")

                prompt += f"""### Example {i}  (L2 distance={distance:.4f})
Session ID    : {meta.get('session_id', 'n/a')}
Label         : {ex_label}
Anomaly score : {ex_score_s}
Event sequence: {ex_events}
```
{ex_text}
```

"""

        prompt += """## TASK
Analyse the flagged session above. Use the historical examples as context clues.
Identify the root cause and respond with the required JSON structure.
"""
        return prompt

    # ── LLM call ──────────────────────────────────────────────────────────

    def _call_llm(self, prompt: str) -> str:
        """Call the configured LLM and return the raw response text."""
        if self.provider == "offline" or self._client is None:
            raise RuntimeError(
                "No LLM client available. Set ANTHROPIC_API_KEY or OPENAI_API_KEY, "
                "or use analyze_offline()."
            )

        sys_prompt = _system_prompt(self.dataset)

        if self.provider == "claude":
            response = self._client.messages.create(
                model=self.model,
                max_tokens=2048,
                system=sys_prompt,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            return response.content[0].text

        else:  # openai
            response = self._client.chat.completions.create(
                model=self.model,
                max_tokens=2048,
                temperature=0.2,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user",   "content": prompt},
                ],
                response_format={"type": "json_object"},
            )
            return response.choices[0].message.content

    # ── Analysis methods ───────────────────────────────────────────────────

    @staticmethod
    def _parse_llm_json(response_text: str) -> dict:
        """Parse JSON returned directly or inside a markdown code fence."""
        text = response_text.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        return json.loads(text)

    def analyze(self, session, top_k: int = 3) -> dict:
        """
        Full RAG analysis for a single session: retrieve → prompt → LLM → parse.

        Returns:
            Dict with root_cause, confidence, severity, explanation,
            failure_trace, recommended_action, plus session metadata.
        """
        retrieved    = self.retrieve_similar(session, top_k=top_k)
        prompt       = self.build_prompt(session, retrieved)
        logger.info("Calling %s (%s) for session %s ...",
                    self.provider, self.model, session.session_id)
        raw_response = self._call_llm(prompt)

        try:
            # Claude may wrap JSON in markdown fences — strip them
            text = raw_response.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            result = json.loads(text.strip())
        except json.JSONDecodeError:
            logger.warning("LLM response was not valid JSON — storing as plain text")
            result = {
                "root_cause":         raw_response[:500],
                "confidence":         0.5,
                "explanation":        "LLM response was not valid JSON.",
                "failure_trace":      [],
                "severity":           "unknown",
                "recommended_action": "Review raw LLM response.",
            }

        result["session_id"]              = session.session_id
        result["line_range"]              = getattr(session, "line_range", None)
        result["anomaly_score"]           = getattr(session, "anomaly_score", None)
        result["retrieved_examples_count"] = len(retrieved)
        result["llm_provider"]            = self.provider
        result["llm_model"]               = self.model
        return result

    def analyze_batch(self, sessions: list, top_k: int = 3) -> list:
        """Analyse multiple sessions, isolating errors per session."""
        results = []
        for i, session in enumerate(sessions, 1):
            logger.info("Analysing session %d/%d: %s", i, len(sessions), session.session_id)
            try:
                results.append(self.analyze(session, top_k=top_k))
            except Exception as e:
                logger.error("Failed to analyse %s: %s", session.session_id, e)
                results.append({
                    "session_id":  session.session_id,
                    "error":       str(e),
                    "llm_provider": self.provider,
                    "llm_model":   self.model,
                })
        logger.info("Batch analysis complete — %d sessions", len(results))
        return results

    def analyze_offline(self, session, top_k: int = 3) -> dict:
        """
        Offline mode: retrieve similar sessions and build the prompt without
        calling an LLM. Useful for inspection, testing, or when no API key
        is available.

        Returns:
            Dict with retrieved examples, built prompt, and session metadata.
        """
        started = time.time()
        retrieved = self.retrieve_similar(session, top_k=top_k)
        prompt    = self.build_prompt(session, retrieved)

        return {
            "session_id":    session.session_id,
            "line_range":    getattr(session, "line_range", None),
            "anomaly_score": getattr(session, "anomaly_score", None),
            "num_lines":     len(session.raw_lines or []),
            "retrieved_examples": [
                {
                    "session_id":    meta.get("session_id"),
                    "distance":      dist,
                    "label":         meta.get("label"),
                    "anomaly_score": meta.get("anomaly_score"),
                    "event_sequence": meta.get("event_sequence", []),
                    "first_line":    (meta.get("raw_lines") or [""])[0],
                }
                for meta, dist in retrieved
            ],
            "prompt":        prompt,
            "llm_provider":  "offline",
            "llm_model":     None,
            "note": (
                "Offline mode — LLM not called. "
                "Pass the prompt to any LLM for root cause analysis."
            ),
        }
