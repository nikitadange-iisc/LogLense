"""
Stage 5a: Session Embedding

Transforms flagged anomalous sessions into embeddings using a
sentence-transformer model for vector similarity search.

Improvements over v1:
  - Default model upgraded to all-mpnet-base-v2 (768-dim) for higher
    retrieval quality; falls back to all-MiniLM-L6-v2 if unavailable.
  - Template-aware embedding mode: embeds the *event template sequence*
    instead of (or alongside) raw lines, for better semantic matching.
  - Smart truncation: keeps both head and tail of long sessions so
    failure indicators at the end are not silently dropped.
  - Severity-weighted prefix: prepends a severity summary to give the
    embedding model signal about error density.
"""

import logging
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Models ranked by retrieval quality (descending).  The first one that
# loads successfully is used.
_MODEL_PRIORITY = [
    "all-mpnet-base-v2",      # 768-dim, best quality
    "all-MiniLM-L6-v2",       # 384-dim, fast fallback
]


class SessionEmbedder:
    """Embeds log sessions using sentence-transformers."""

    def __init__(self, model_name: str = "all-mpnet-base-v2"):
        """
        Args:
            model_name: Sentence-transformer model name.  If the
                        requested model cannot be loaded the embedder
                        automatically falls back through a priority list,
                        and ultimately to a TF-IDF fallback if no
                        sentence-transformer model is available.
        """
        self._use_tfidf = False

        try:
            from sentence_transformers import SentenceTransformer

            # Try requested model first, then fall through priority list
            models_to_try = [model_name] + [
                m for m in _MODEL_PRIORITY if m != model_name
            ]

            for name in models_to_try:
                try:
                    logger.info(f"Loading sentence-transformer model: {name}")
                    self.model = SentenceTransformer(name)
                    self.model_name = name
                    self.dimension = self.model.get_sentence_embedding_dimension()
                    logger.info(
                        f"Model loaded: {self.model_name} — "
                        f"embedding dimension: {self.dimension}"
                    )
                    return  # success
                except Exception as e:
                    logger.warning(f"Could not load {name}: {e}")

        except ImportError:
            logger.warning("sentence-transformers not installed")

        # ── TF-IDF fallback ────────────────────────────────────────────
        logger.warning(
            "No sentence-transformer model available — "
            "falling back to TF-IDF embeddings"
        )
        self._init_tfidf_fallback()

    def _init_tfidf_fallback(self):
        """Initialise a TF-IDF + SVD pipeline as an offline fallback."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD
        from sklearn.pipeline import Pipeline

        self._use_tfidf = True
        self.model_name = "tfidf-svd-256"
        self.dimension = 256
        self._tfidf = TfidfVectorizer(
            max_features=10_000,
            sublinear_tf=True,
            analyzer="word",
            token_pattern=r"(?u)\b\w[\w./:_-]+\b",
        )
        self._svd = TruncatedSVD(n_components=self.dimension, random_state=42)
        self._fitted = False
        self.model = None  # no SentenceTransformer
        logger.info(
            f"TF-IDF fallback initialised (dimension={self.dimension})"
        )

    # ── Text preparation ───────────────────────────────────────────────

    @staticmethod
    def _prepare_session_text(
        session,
        mode: str = "hybrid",
        max_chars: int = 6000,
    ) -> str:
        """
        Convert a session into a single text string for embedding.

        Args:
            session: Session object with .raw_lines and optionally .events.
            mode: "raw"      — concatenate raw log lines.
                  "template" — use event template sequences only.
                  "hybrid"   — severity summary + template sequence +
                               head/tail raw lines (default).
            max_chars: Maximum character budget.

        Returns:
            Prepared text string.
        """
        # ── Severity summary prefix ────────────────────────────────────
        severity_prefix = ""
        events = getattr(session, "events", None)
        if events:
            level_counts: dict = {}
            for ev in events:
                lvl = ev.get("level", "UNKNOWN")
                level_counts[lvl] = level_counts.get(lvl, 0) + 1
            if any(k in level_counts for k in ("ERROR", "FATAL", "CRITICAL",
                                                 "WARN", "WARNING")):
                parts = [f"{k}:{v}" for k, v in sorted(level_counts.items())]
                severity_prefix = "Severity: " + " ".join(parts) + "\n"

        # ── Body ───────────────────────────────────────────────────────
        if mode == "template" and events:
            # Deduplicate consecutive identical templates
            templates = []
            prev = None
            for ev in events:
                t = ev.get("event_template", "")
                if t != prev:
                    templates.append(t)
                    prev = t
            body = "\n".join(templates)

        elif mode == "hybrid" and events:
            # Template sequence (compact) + head/tail raw lines
            templates = []
            prev = None
            for ev in events:
                t = ev.get("event_template", "")
                if t != prev:
                    templates.append(t)
                    prev = t
            template_block = "\n".join(templates)

            # Keep first + last raw lines for concrete detail
            raw = session.raw_lines
            head = raw[:10]
            tail = raw[-5:] if len(raw) > 15 else []
            raw_block = "\n".join(head + (["..."] if tail else []) + tail)

            body = template_block + "\n---\n" + raw_block

        else:
            # Default: raw lines
            body = "\n".join(session.raw_lines)

        text = severity_prefix + body

        # ── Smart truncation (keep head + tail) ────────────────────────
        if len(text) > max_chars:
            half = max_chars // 2 - 20
            text = text[:half] + "\n...[truncated]...\n" + text[-half:]

        return text

    # ── Embedding methods ──────────────────────────────────────────────

    def embed_text(self, text: str) -> np.ndarray:
        """Encode a single text string into an embedding vector."""
        if self._use_tfidf:
            return self._tfidf_embed_texts([text])[0]
        return self.model.encode(text, convert_to_numpy=True)

    def embed_session(
        self, session, mode: str = "hybrid"
    ) -> np.ndarray:
        """
        Embed a single session.

        Args:
            session: Session object with .raw_lines (and optionally .events).
            mode: Text preparation mode — "raw", "template", or "hybrid".

        Returns:
            Numpy embedding vector.
        """
        text = self._prepare_session_text(session, mode=mode)
        return self.embed_text(text)

    def embed_batch(
        self, sessions: list, mode: str = "hybrid"
    ) -> np.ndarray:
        """
        Embed multiple sessions at once for efficiency.

        Args:
            sessions: List of Session objects.
            mode: Text preparation mode.

        Returns:
            Numpy array of shape (n_sessions, embedding_dim).
        """
        texts = [
            self._prepare_session_text(s, mode=mode) for s in sessions
        ]

        logger.info(f"Batch embedding {len(texts)} sessions ({self.model_name})...")

        if self._use_tfidf:
            embeddings = self._tfidf_embed_texts(texts)
        else:
            embeddings = self.model.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=len(texts) > 100,
                batch_size=32,
            )

        logger.info(f"Batch embedding complete — shape: {embeddings.shape}")
        return embeddings

    # ── TF-IDF helpers ─────────────────────────────────────────────────

    def _tfidf_embed_texts(self, texts: list) -> np.ndarray:
        """Embed texts using TF-IDF + SVD (offline fallback)."""
        if not self._fitted:
            # Fit on the provided texts (first batch)
            tfidf_matrix = self._tfidf.fit_transform(texts)
            # Handle case where n_features < n_components
            n_components = min(self.dimension, tfidf_matrix.shape[1], tfidf_matrix.shape[0])
            if n_components < self._svd.n_components:
                self._svd.n_components = n_components
                self.dimension = n_components
            self._svd.fit(tfidf_matrix)
            self._fitted = True
            embeddings = self._svd.transform(tfidf_matrix)
        else:
            tfidf_matrix = self._tfidf.transform(texts)
            embeddings = self._svd.transform(tfidf_matrix)

        # Pad to target dimension if needed
        if embeddings.shape[1] < self.dimension:
            pad = np.zeros((embeddings.shape[0], self.dimension - embeddings.shape[1]))
            embeddings = np.hstack([embeddings, pad])

        return embeddings.astype(np.float32)

