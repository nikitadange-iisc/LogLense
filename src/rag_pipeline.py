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

Respond ONLY with valid JSON in this exact format — no text before or after the JSON:
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
}

CRITICAL JSON rules — violations make the output unusable:
- All string values must have internal double-quotes escaped as \"
- All backslashes must be escaped as \\
- No raw newlines inside string values — use \\n if needed
- Keep each failure_trace line value short (under 120 chars); truncate with ... if needed
- Output raw JSON only — do NOT wrap in markdown code fences"""

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

    def _call_llm(self, prompt: str) -> tuple:
        """
        Call the configured LLM.

        Returns:
            (response_text, usage_dict) where usage_dict has keys
            input_tokens and output_tokens.
        """
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
            usage = {
                "input_tokens":  response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            }
            return response.content[0].text, usage

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
            usage = {
                "input_tokens":  response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
            }
            return response.choices[0].message.content, usage

    # ── Analysis methods ───────────────────────────────────────────────────

    @staticmethod
    def _parse_llm_json(response_text: str) -> dict:
        """Parse JSON from a response that may contain markdown fences or preamble text."""
        import re
        text = response_text.strip()

        # Strip opening ```json or ``` fence anchored at the start of the text
        open_fence = re.match(r"```(?:json)?\s*\n?", text)
        if open_fence:
            text = text[open_fence.end():]
            # Strip the matching closing fence at the end
            close_fence = re.search(r"\n?```\s*$", text)
            if close_fence:
                text = text[:close_fence.start()]
            text = text.strip()

        # 1. Try direct parse (works when response is clean JSON or fence was stripped)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # 2. Slice from first '{' to last '}' — handles stray preamble/postamble
        start = text.find("{")
        end   = text.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass

        # 3. Regex field extraction — recovers key fields when failure_trace contains
        #    unescaped quotes or other characters that break JSON syntax
        extracted = RAGPipeline._extract_fields_regex(text)
        if extracted:
            return extracted

        raise json.JSONDecodeError("No valid JSON found in LLM response", text, 0)

    @staticmethod
    def _extract_fields_regex(text: str) -> dict:
        """
        Last-resort field extractor using regex when json.loads fails entirely.
        Pulls scalar fields reliably; skips failure_trace (too fragile to regex-parse).
        Returns None if the minimum required fields are not found.
        """
        import re

        def _str_field(name: str) -> str:
            m = re.search(rf'"{name}"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
            return m.group(1) if m else None

        def _num_field(name: str):
            m = re.search(rf'"{name}"\s*:\s*([0-9.]+)', text)
            return float(m.group(1)) if m else None

        def _arr_field(name: str):
            m = re.search(rf'"{name}"\s*:\s*\[([^\]]*)\]', text)
            if not m:
                return None
            try:
                return json.loads("[" + m.group(1) + "]")
            except json.JSONDecodeError:
                return None

        root_cause = _str_field("root_cause")
        if not root_cause:
            return None

        return {
            "root_cause":          root_cause,
            "severity":            _str_field("severity") or "unknown",
            "confidence":          _num_field("confidence") or 0.5,
            "explanation":         _str_field("explanation") or "",
            "recommended_action":  _str_field("recommended_action") or "",
            "affected_line_range": _arr_field("affected_line_range"),
            "failure_trace":       [],
            "_parse_note":         "regex-extracted: failure_trace skipped (malformed JSON)",
        }

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
        raw_response, usage = self._call_llm(prompt)
        logger.info("Tokens — input: %d  output: %d  total: %d",
                    usage["input_tokens"], usage["output_tokens"],
                    usage["input_tokens"] + usage["output_tokens"])

        try:
            result = self._parse_llm_json(raw_response)
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

        result["session_id"]               = session.session_id
        result["line_range"]               = getattr(session, "line_range", None)
        result["anomaly_score"]            = getattr(session, "anomaly_score", None)
        result["retrieved_examples_count"] = len(retrieved)
        result["llm_provider"]             = self.provider
        result["llm_model"]                = self.model
        result["token_usage"]              = usage
        return result

    def analyze_batch(self, sessions: list, top_k: int = 3) -> list:
        """Analyse multiple sessions, isolating errors per session."""
        results = []
        total_in = total_out = 0
        for i, session in enumerate(sessions, 1):
            logger.info("Analysing session %d/%d: %s", i, len(sessions), session.session_id)
            try:
                r = self.analyze(session, top_k=top_k)
                u = r.get("token_usage", {})
                total_in  += u.get("input_tokens", 0)
                total_out += u.get("output_tokens", 0)
                results.append(r)
            except Exception as e:
                logger.error("Failed to analyse %s: %s", session.session_id, e)
                results.append({
                    "session_id":   session.session_id,
                    "error":        str(e),
                    "llm_provider": self.provider,
                    "llm_model":    self.model,
                })
        logger.info("Batch complete — %d sessions | tokens in=%d out=%d total=%d",
                    len(results), total_in, total_out, total_in + total_out)
        self.last_batch_tokens = {"input": total_in, "output": total_out, "total": total_in + total_out}
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
