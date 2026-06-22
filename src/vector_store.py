"""
Stage 5b: FAISS Vector Store

Manages a FAISS index for storing and retrieving anomalous session embeddings.
Only anomalous session embeddings are indexed — normal sessions are never stored.
"""

import os
import pickle
import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import faiss

logger = logging.getLogger(__name__)


class FAISSVectorStore:
    """FAISS-based vector store for anomalous session embeddings."""

    def __init__(self, dimension: int = 768, index_type: str = "flat",
                 index_path: str = "models/faiss_index"):
        """
        Args:
            dimension: Embedding dimension (must match the embedding model).
                       768 for all-mpnet-base-v2, 384 for all-MiniLM-L6-v2.
            index_type: "flat" for IndexFlatL2, "ivf" for IndexIVFFlat.
            index_path: Directory for saving/loading the index.
        """
        self.dimension = dimension
        self.index_type = index_type
        self.index_path = Path(index_path)
        self.metadata: List[dict] = []

        if index_type == "flat":
            self.index = faiss.IndexFlatL2(dimension)
        elif index_type == "ivf":
            quantizer = faiss.IndexFlatL2(dimension)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, 100)
        else:
            raise ValueError(f"Unknown index type: {index_type}")

        logger.info(f"FAISS index created (type={index_type}, dim={dimension})")

    def add(self, embeddings: np.ndarray, metadata_list: List[dict]):
        """
        Add embeddings and corresponding metadata to the index.

        Args:
            embeddings: Numpy array of shape (n, dimension).
            metadata_list: List of metadata dicts (one per embedding) with
                          keys like session_id, raw_lines, line_numbers, root_cause.
        """
        if len(embeddings) != len(metadata_list):
            raise ValueError("Embeddings and metadata must have the same length")

        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

        # Train IVF index if needed
        if self.index_type == "ivf" and not self.index.is_trained:
            logger.info("Training IVF index...")
            self.index.train(embeddings)

        self.index.add(embeddings)
        self.metadata.extend(metadata_list)

        logger.info(f"Added {len(embeddings)} vectors to FAISS index "
                     f"(total: {self.index.ntotal})")

    def search(self, query_embedding: np.ndarray, top_k: int = 3) -> List[Tuple[dict, float]]:
        """
        Search for top-K nearest neighbors.

        Args:
            query_embedding: Query embedding vector.
            top_k: Number of results to return.

        Returns:
            List of (metadata_dict, distance) tuples, sorted by distance.
        """
        if self.index.ntotal == 0:
            logger.warning("FAISS index is empty, no results to return")
            return []

        query = np.ascontiguousarray(query_embedding.reshape(1, -1), dtype=np.float32)
        top_k = min(top_k, self.index.ntotal)

        distances, indices = self.index.search(query, top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx >= 0 and idx < len(self.metadata):
                results.append((self.metadata[idx], float(dist)))

        logger.info(f"FAISS search returned {len(results)} results")
        return results

    def save(self, path: str = None):
        """Save FAISS index and metadata to disk."""
        save_dir = Path(path) if path else self.index_path
        save_dir.mkdir(parents=True, exist_ok=True)

        index_file = save_dir / "index.faiss"
        meta_file = save_dir / "metadata.pkl"

        faiss.write_index(self.index, str(index_file))
        with open(meta_file, "wb") as f:
            pickle.dump(self.metadata, f)

        logger.info(f"FAISS index saved to {save_dir} ({self.index.ntotal} vectors)")

    def load(self, path: str = None):
        """Load FAISS index and metadata from disk."""
        load_dir = Path(path) if path else self.index_path
        index_file = load_dir / "index.faiss"
        meta_file = load_dir / "metadata.pkl"

        if not index_file.exists():
            raise FileNotFoundError(f"FAISS index not found: {index_file}")

        self.index = faiss.read_index(str(index_file))
        with open(meta_file, "rb") as f:
            self.metadata = pickle.load(f)

        logger.info(f"FAISS index loaded from {load_dir} ({self.index.ntotal} vectors)")

    def reset(self) -> None:
        """Discard all vectors and metadata, rebuilding an empty index."""
        if self.index_type == "flat":
            self.index = faiss.IndexFlatL2(self.dimension)
        elif self.index_type == "ivf":
            quantizer = faiss.IndexFlatL2(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, 100)
        self.metadata = []
        logger.info("FAISS index reset (dimension=%d)", self.dimension)

    def size(self) -> int:
        """Return the number of vectors in the index."""
        return self.index.ntotal

