"""
Stage 6: Retrieval-Augmented Agentic Reasoning

At inference, embeds flagged sessions and retrieves top-K similar historical
failures from FAISS. Assembles a prompt with flagged lines + retrieved examples
and passes to Anthropic for root cause identification.
"""

import os
import json
import logging
from typing import Optional

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert system log analyst specializing in root cause analysis of distributed system failures. You analyze anomalous log sessions to identify the root cause of failures.

Your task:
1. Analyze the flagged anomalous log lines carefully.
2. Consider the similar historical failure examples provided for context.
3. Identify the root cause of the failure.
4. Provide a line-level failure trace showing the progression of the issue.

You MUST respond with valid JSON in the following format:
{
    "root_cause": "Brief description of the root cause",
    "affected_line_range": [start_line, end_line],
    "confidence": 0.0 to 1.0,
    "explanation": "Detailed explanation of the failure and how it was identified",
    "failure_trace": [
        {
            "line": "the log line",
            "annotation": "what this line indicates about the failure"
        }
    ],
    "severity": "critical|high|medium|low",
    "recommended_action": "What should be done to resolve this issue"
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

    def retrieve_similar(self, session, top_k: int = 3) -> list:
        """
        Retrieve top-K similar historical failures for a session.

        Args:
            session: Session object to query.
            top_k: Number of similar examples to retrieve.

        Returns:
            List of (metadata_dict, distance) tuples.
        """
        embedding = self.embedder.embed_session(session)
        results = self.vector_store.search(embedding, top_k=top_k)
        logger.info(f"Retrieved {len(results)} similar examples for session {session.session_id}")
        return results

    def build_prompt(self, session, retrieved_examples: list) -> str:
        """
        Build the LLM prompt with flagged lines and retrieved examples.

        Args:
            session: Flagged anomalous Session object.
            retrieved_examples: List of (metadata, distance) tuples.

        Returns:
            Formatted prompt string.
        """
        # Flagged log lines
        flagged_lines = "\n".join(session.raw_lines[:100])  # Limit to 100 lines

        prompt = f"""## FLAGGED ANOMALOUS LOG SESSION

Session ID: {session.session_id}
Line Range: {session.line_range}
Number of log lines: {len(session.raw_lines)}

### Anomalous Log Lines:
```
{flagged_lines}
```
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
                prompt += f"""### Historical Example {i} (similarity distance: {distance:.4f})
Session ID: {meta.get('session_id', 'N/A')}
Known Root Cause: {root_cause}
```
{example_text}
```

"""

        prompt += """## INSTRUCTIONS
Analyze the flagged anomalous log session above. Use the similar historical failures as context.
Identify the root cause, provide a line-level failure trace, and respond in the specified JSON format.
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
        # Step 1: Retrieve similar historical failures
        retrieved = self.retrieve_similar(session, top_k=top_k)

        # Step 2: Build prompt
        prompt = self.build_prompt(session, retrieved)

        # Step 3: Call LLM
        logger.info(f"Analyzing session {session.session_id} with {self.model}...")
        response_text = self._call_llm(prompt)

        # Step 4: Parse response
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
        result["line_range"] = session.line_range

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
            try:
                result = self.analyze(session, top_k=top_k)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to analyze session {session.session_id}: {e}")
                results.append({
                    "session_id": session.session_id,
                    "error": str(e),
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
        retrieved = self.retrieve_similar(session, top_k=top_k)
        prompt = self.build_prompt(session, retrieved)

        return {
            "session_id": session.session_id,
            "line_range": session.line_range,
            "num_lines": len(session.raw_lines),
            "retrieved_examples_count": len(retrieved),
            "retrieved_examples": [
                {
                    "session_id": meta.get("session_id"),
                    "distance": dist,
                    "root_cause": meta.get("root_cause", "Unknown"),
                }
                for meta, dist in retrieved
            ],
            "prompt": prompt,
            "note": "Offline mode — LLM not called. Use the prompt above with any LLM.",
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    print("RAG Pipeline module loaded. Use via pipeline.py for end-to-end execution.")
