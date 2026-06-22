"""
Stage 6: Retrieval-Augmented Agentic Reasoning

At inference, embeds flagged sessions and retrieves top-K similar historical
failures from FAISS. Assembles a prompt with flagged lines + retrieved examples
and passes to Anthropic for root cause identification.
"""

import os
import json
import logging
import time
from typing import Optional

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert system log analyst specializing in root cause analysis of distributed system failures. You analyze anomalous log sessions to identify the root cause of failures.

Your task:
1. Analyze the flagged anomalous log lines carefully.
2. Consider the similar historical failure examples provided for context.
3. Identify the root cause of the failure.
4. Provide a line-level failure trace showing the progression of the issue.
5. Separate direct evidence from hypotheses. Do not overclaim when evidence is weak.

You MUST respond with valid JSON in the following format:
{
    "root_cause": "Brief description of the root cause",
    "affected_line_range": [start_line, end_line],
    "confidence": 0.0 to 1.0,
    "explanation": "Detailed explanation of the failure and how it was identified",
    "evidence": [
        {
            "line_reference": "line number or range",
            "observation": "specific log evidence",
            "supports": "how this supports the root cause"
        }
    ],
    "failure_trace": [
        {
            "line": "the log line",
            "annotation": "what this line indicates about the failure"
        }
    ],
    "severity": "critical|high|medium|low",
    "recommended_action": "What should be done to resolve this issue",
    "retrieval_assessment": "How relevant the retrieved examples are",
    "limitations": "Any uncertainty or missing context"
}"""


class RAGPipeline:
    """Retrieval-Augmented Generation pipeline for log root cause analysis."""

    def __init__(self, embedder, vector_store, model: str = "claude-haiku-4-5",
                 api_key: str = None, provider: str = None):
        """
        Args:
            embedder: SessionEmbedder instance.
            vector_store: FAISSVectorStore instance.
            model: LLM model name.
            api_key: Provider API key (or loaded from .env).
            provider: "anthropic". If omitted, loaded from LLM_PROVIDER.
        """
        load_dotenv()

        self.embedder = embedder
        self.vector_store = vector_store
        self.model = model or os.getenv("LLM_MODEL", "claude-haiku-4-5")
        self.provider = (provider or os.getenv("LLM_PROVIDER") or "anthropic").lower()
        if self.provider != "anthropic":
            raise ValueError(f"Unsupported LLM provider: {self.provider}")

        self.client = None
        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            logger.warning("No Anthropic API key found. LLM calls will fail.")
        else:
            try:
                from anthropic import Anthropic
                self.client = Anthropic(api_key=api_key)
            except ImportError as exc:
                raise RuntimeError("anthropic package is required for LLM calls.") from exc

        logger.info(f"RAGPipeline initialized (provider={self.provider}, model={self.model})")

    def retrieve_similar(self, session, top_k: int = 3,
                         exclude_self: bool = True,
                         extra_candidates: int = 5) -> list:
        """
        Retrieve top-K similar historical failures for a session.

        Args:
            session: Session object to query.
            top_k: Number of similar examples to retrieve.
            exclude_self: Whether to exclude examples with the same session_id.
            extra_candidates: Extra candidates to fetch before filtering.

        Returns:
            List of (metadata_dict, distance) tuples.
        """
        embedding = self.embedder.embed_session(session)
        search_k = top_k + extra_candidates if exclude_self else top_k
        results = self.vector_store.search(embedding, top_k=search_k)

        if exclude_self:
            session_id = getattr(session, "session_id", None)
            results = [
                (meta, dist)
                for meta, dist in results
                if meta.get("session_id") != session_id
            ]

        filtered = results[:top_k]
        logger.info(
            f"Retrieved {len(filtered)} similar examples for session {session.session_id}"
        )
        return filtered

    @staticmethod
    def _event_template_sequence(session, max_events: int = 30) -> list:
        """Return a compact deduplicated event-template sequence."""
        events = getattr(session, "events", None) or []
        sequence = []
        previous = None
        for event in events:
            template = event.get("event_template", "")
            event_id = event.get("event_template_id", "")
            item = f"E{event_id}: {template}" if event_id != "" else template
            if item and item != previous:
                sequence.append(item)
                previous = item
            if len(sequence) >= max_events:
                break
        return sequence

    @staticmethod
    def _severity_summary(session) -> dict:
        """Count log levels in a session."""
        counts = {}
        for event in getattr(session, "events", None) or []:
            level = event.get("level", "UNKNOWN")
            counts[level] = counts.get(level, 0) + 1
        return counts

    @staticmethod
    def _retrieval_quality(retrieved_examples: list, requested_top_k: int) -> dict:
        """Summarize retrieval quality for the LLM and output metadata."""
        if not retrieved_examples:
            return {
                "status": "empty",
                "message": "No similar historical examples were retrieved.",
            }

        distances = [dist for _, dist in retrieved_examples]
        labels = [meta.get("root_cause") or meta.get("label") for meta, _ in retrieved_examples]
        return {
            "status": "ok" if len(retrieved_examples) >= requested_top_k else "partial",
            "retrieved": len(retrieved_examples),
            "requested": requested_top_k,
            "best_distance": min(distances),
            "worst_distance": max(distances),
            "labels": labels,
        }

    def build_prompt(self, session, retrieved_examples: list, top_k: int = 3) -> str:
        """
        Build the LLM prompt with flagged lines and retrieved examples.

        Args:
            session: Flagged anomalous Session object.
            retrieved_examples: List of (metadata, distance) tuples.

        Returns:
            Formatted prompt string.
        """
        flagged_lines = "\n".join(
            f"{i + 1}: {line}" for i, line in enumerate(session.raw_lines[:100])
        )
        event_sequence = self._event_template_sequence(session)
        event_block = "\n".join(event_sequence) if event_sequence else "Not available"
        severity_summary = self._severity_summary(session)
        anomaly_score = getattr(session, "anomaly_score", None)
        retrieval_quality = self._retrieval_quality(retrieved_examples, top_k)

        prompt = f"""## FLAGGED ANOMALOUS LOG SESSION

Session ID: {session.session_id}
Line Range: {session.line_range}
Number of log lines: {len(session.raw_lines)}
Anomaly Score: {anomaly_score if anomaly_score is not None else "Not available"}
Severity Counts: {json.dumps(severity_summary, sort_keys=True)}

### Event Template Sequence:
```
{event_block}
```

### Anomalous Log Lines:
```
{flagged_lines}
```

### Retrieval Quality:
{json.dumps(retrieval_quality, indent=2)}
"""

        # Retrieved similar historical failures
        if retrieved_examples:
            prompt += "\n## SIMILAR HISTORICAL FAILURES\n\n"
            for i, (meta, distance) in enumerate(retrieved_examples, 1):
                example_lines = meta.get("raw_lines", [])
                if isinstance(example_lines, list):
                    example_text = "\n".join(example_lines[:50])
                else:
                    example_text = str(example_lines)[:2000]

                root_cause = meta.get("root_cause", "Unknown")
                example_templates = meta.get("event_sequence", [])
                if isinstance(example_templates, list):
                    example_templates = "\n".join(example_templates[:30])
                else:
                    example_templates = str(example_templates)
                prompt += f"""### Historical Example {i} (similarity distance: {distance:.4f})
Session ID: {meta.get('session_id', 'N/A')}
Known Root Cause: {root_cause}
Line Range: {meta.get('line_range', 'N/A')}
Anomaly Score: {meta.get('anomaly_score', 'N/A')}
Event Sequence:
```
{example_templates or "Not available"}
```
Raw Lines:
```
{example_text}
```

"""

        prompt += """## INSTRUCTIONS
Analyze the flagged anomalous log session above. Use the retrieved examples as context, but do not assume they prove the same root cause.

Rules:
- Base the root cause primarily on the flagged log lines and event sequence.
- Use retrieved examples only when they are clearly relevant.
- If evidence is weak, lower confidence and state the limitation.
- Cite concrete line numbers from the flagged session in the evidence and failure trace.
- Do not invent components, timestamps, or failures not present in the logs.
- Return only valid JSON in the specified format.
"""

        return prompt

    def _call_llm(self, prompt: str) -> str:
        """
        Call the configured LLM API with the assembled prompt.

        Args:
            prompt: User prompt string.

        Returns:
            LLM response text.
        """
        if self.client is None:
            raise RuntimeError(f"{self.provider} client not initialized. Provide an API key.")

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2000,
                temperature=0.2,
                system=SYSTEM_PROMPT,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            prompt
                            + "\n\nReturn only valid JSON. Do not include markdown fences."
                        ),
                    }
                ],
            )
            return "".join(
                block.text
                for block in response.content
                if getattr(block, "type", None) == "text"
            )

        except Exception as e:
            logger.error(f"LLM API call failed: {e}")
            raise

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
        Full analysis pipeline for a single session.

        Args:
            session: Flagged anomalous Session object.
            top_k: Number of similar examples to retrieve.

        Returns:
            Dict with root_cause, affected_line_range, confidence,
            explanation, failure_trace, retrieved_examples_count.
        """
        started = time.time()
        retrieved = self.retrieve_similar(session, top_k=top_k)
        retrieval_quality = self._retrieval_quality(retrieved, top_k)

        prompt = self.build_prompt(session, retrieved, top_k=top_k)

        logger.info(f"Analyzing session {session.session_id} with {self.model}...")
        response_text = self._call_llm(prompt)

        try:
            result = self._parse_llm_json(response_text)
        except json.JSONDecodeError:
            logger.warning("Failed to parse LLM response as JSON, wrapping as text")
            result = {
                "root_cause": response_text,
                "confidence": 0.5,
                "explanation": "Response was not valid JSON",
            }

        result["session_id"] = session.session_id
        result["retrieved_examples_count"] = len(retrieved)
        result["retrieved_examples"] = [
            {
                "session_id": meta.get("session_id"),
                "distance": dist,
                "root_cause": meta.get("root_cause", "Unknown"),
            }
            for meta, dist in retrieved
        ]
        result["retrieval_quality"] = retrieval_quality
        result["line_range"] = session.line_range
        result["latency_sec"] = round(time.time() - started, 3)

        logger.info(f"Analysis complete for session {session.session_id}: "
                     f"{result.get('root_cause', 'N/A')}")

        return result

    def analyze_batch(self, sessions: list, top_k: int = 3) -> list:
        """
        Analyze multiple flagged sessions.

        Args:
            sessions: List of anomalous Session objects.
            top_k: Number of similar examples to retrieve per session.

        Returns:
            List of analysis result dicts.
        """
        results = []
        for i, session in enumerate(sessions, 1):
            logger.info(f"Analyzing session {i}/{len(sessions)}: {session.session_id}")
            started = time.time()
            try:
                result = self.analyze(session, top_k=top_k)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to analyze session {session.session_id}: {e}")
                results.append({
                    "session_id": session.session_id,
                    "error": str(e),
                    "latency_sec": round(time.time() - started, 3),
                })

        logger.info(f"Batch analysis complete: {len(results)} sessions analyzed")
        return results

    def analyze_offline(self, session, top_k: int = 3) -> dict:
        """
        Offline analysis (no LLM call) — returns retrieved examples and prompt
        for manual review or when API key is not available.

        Args:
            session: Flagged anomalous Session object.
            top_k: Number of similar examples to retrieve.

        Returns:
            Dict with prompt, retrieved_examples, and session info.
        """
        started = time.time()
        retrieved = self.retrieve_similar(session, top_k=top_k)
        prompt = self.build_prompt(session, retrieved, top_k=top_k)
        retrieval_quality = self._retrieval_quality(retrieved, top_k)

        return {
            "session_id": session.session_id,
            "line_range": session.line_range,
            "num_lines": len(session.raw_lines),
            "retrieved_examples_count": len(retrieved),
            "retrieval_quality": retrieval_quality,
            "retrieved_examples": [
                {
                    "session_id": meta.get("session_id"),
                    "distance": dist,
                    "root_cause": meta.get("root_cause", "Unknown"),
                }
                for meta, dist in retrieved
            ],
            "prompt": prompt,
            "latency_sec": round(time.time() - started, 3),
            "note": "Offline mode — LLM not called. Use the prompt above with any LLM.",
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    print("RAG Pipeline module loaded. Use via pipeline.py for end-to-end execution.")
