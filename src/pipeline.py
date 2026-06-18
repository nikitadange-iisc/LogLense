"""
LogSense End-to-End Pipeline Orchestration

Orchestrates all 6 stages of the LogSense pipeline:
  1. Streaming Ingestion & Deduplication
  2. Log Parsing with Drain
  3. Session Grouping & Vectorization
  4. Isolation Forest Anomaly Gate
  5. Embedding & FAISS Vector Store
  6. Retrieval-Augmented Agentic Reasoning
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np

from ingestion import stream_deduplicated, deduplicate_stream
from log_parser import LogParser
from sessionizer import Sessionizer, Session
from anomaly_gate import AnomalyGate
from embedder import SessionEmbedder
from vector_store import FAISSVectorStore
from rag_pipeline import RAGPipeline

logger = logging.getLogger(__name__)


class LogSensePipeline:
    """End-to-end orchestration of the LogSense pipeline."""

    def __init__(self, config: dict = None):
        """
        Args:
            config: Optional configuration dict with keys:
                - dataset_type: "hdfs", "bgl", or "thunderbird"
                - model_dir: directory for saved models
                - embedding_model: sentence-transformer model name
                - llm_model: LLM model name
                - top_k: number of similar examples to retrieve
                - contamination: Isolation Forest contamination parameter
                - window_size: sliding window size (for BGL/Thunderbird)
                - step_size: sliding window step size
        """
        self.config = config or {}
        self.dataset_type = self.config.get("dataset_type", "hdfs")
        self.model_dir = Path(self.config.get("model_dir", "models"))
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components (lazy — created when needed)
        self._parser = None
        self._sessionizer = None
        self._anomaly_gate = None
        self._embedder = None
        self._vector_store = None
        self._rag_pipeline = None
        self._vocabulary = None

        logger.info(f"LogSense pipeline initialized (dataset={self.dataset_type})")

    @property
    def parser(self):
        if self._parser is None:
            self._parser = LogParser(
                dataset=self.dataset_type,
                state_dir=str(self.model_dir / "drain_state"),
            )
        return self._parser

    @property
    def sessionizer(self):
        if self._sessionizer is None:
            method = "block_id" if self.dataset_type == "hdfs" else "sliding_window"
            self._sessionizer = Sessionizer(
                method=method,
                window_size=self.config.get("window_size", 50),
                step_size=self.config.get("step_size", 25),
            )
        return self._sessionizer

    @property
    def anomaly_gate(self):
        if self._anomaly_gate is None:
            self._anomaly_gate = AnomalyGate(
                model_path=str(self.model_dir / "isolation_forest.joblib"),
                contamination=self.config.get("contamination", 0.1),
            )
        return self._anomaly_gate

    @property
    def embedder(self):
        if self._embedder is None:
            self._embedder = SessionEmbedder(
                model_name=self.config.get("embedding_model", "all-mpnet-base-v2")
            )
        return self._embedder

    @property
    def vector_store(self):
        if self._vector_store is None:
            dimension = self.embedder.dimension
            self._vector_store = FAISSVectorStore(
                dimension=dimension,
                index_path=str(self.model_dir / "faiss_index"),
            )
        return self._vector_store

    @property
    def rag_pipeline(self):
        if self._rag_pipeline is None:
            self._rag_pipeline = RAGPipeline(
                embedder=self.embedder,
                vector_store=self.vector_store,
                model=self.config.get("llm_model", "gpt-4o-mini"),
            )
        return self._rag_pipeline

    # ── Stage 1: Ingestion & Deduplication ──────────────────────────────

    def run_ingestion(self, input_path: str, output_path: str = None) -> dict:
        """Run Stage 1: Streaming ingestion and deduplication."""
        logger.info("=" * 60)
        logger.info("STAGE 1: Streaming Ingestion & Deduplication")
        logger.info("=" * 60)

        start = time.time()
        stats = deduplicate_stream(input_path, output_path)
        stats["duration_sec"] = round(time.time() - start, 2)

        logger.info(f"Stage 1 complete in {stats['duration_sec']}s")
        return stats

    # ── Stage 2: Parsing ────────────────────────────────────────────────

    def run_parsing(self, input_path: str, max_lines: int = None) -> list:
        """Run Stage 2: Log parsing with Drain."""
        logger.info("=" * 60)
        logger.info("STAGE 2: Log Parsing with Drain")
        logger.info("=" * 60)

        start = time.time()
        parsed_events = []

        for parsed in self.parser.parse_stream(stream_deduplicated(input_path)):
            parsed_events.append(parsed)
            if max_lines and len(parsed_events) >= max_lines:
                break

        duration = round(time.time() - start, 2)
        logger.info(f"Stage 2 complete in {duration}s — "
                     f"{len(parsed_events)} events, "
                     f"{self.parser.get_template_count()} templates")

        self.parser.save_state()
        return parsed_events

    # ── Stage 3: Session Grouping & Vectorization ───────────────────────

    def run_sessionization(self, parsed_events: list,
                           label_path: str = None) -> tuple:
        """Run Stage 3: Session grouping and vectorization."""
        logger.info("=" * 60)
        logger.info("STAGE 3: Session Grouping & Vectorization")
        logger.info("=" * 60)

        start = time.time()

        sessions = self.sessionizer.create_sessions(iter(parsed_events))

        if label_path:
            sessions = self.sessionizer.load_labels(label_path, sessions)

        sessions, vocabulary = self.sessionizer.vectorize_all(sessions)
        self._vocabulary = vocabulary

        duration = round(time.time() - start, 2)
        logger.info(f"Stage 3 complete in {duration}s — "
                     f"{len(sessions)} sessions, vocab size {len(vocabulary)}")

        return sessions, vocabulary

    # ── Stage 4: Anomaly Gate ───────────────────────────────────────────

    def run_anomaly_gate(self, sessions: list, train: bool = True) -> list:
        """
        Run Stage 4: Isolation Forest anomaly detection.

        Args:
            sessions: List of Session objects with vectors.
            train: If True, train the model. If False, load existing model.

        Returns:
            List of anomalous sessions.
        """
        logger.info("=" * 60)
        logger.info("STAGE 4: Isolation Forest Anomaly Gate")
        logger.info("=" * 60)

        start = time.time()

        if train:
            # Use labeled normal sessions for training if available
            labeled_normal = [s for s in sessions if s.label and s.label.lower() == "normal"]

            if labeled_normal:
                normal_vectors = np.array([s.vector for s in labeled_normal])
                logger.info(f"Training on {len(labeled_normal)} labeled normal sessions")
                self.anomaly_gate.train(normal_vectors)
            else:
                # Train on all data with contamination parameter
                all_vectors = np.array([s.vector for s in sessions])
                logger.info(f"No labels found — training on all {len(sessions)} sessions")
                self.anomaly_gate.train(all_vectors, all_vectors)
        else:
            self.anomaly_gate.load_model()

        # Filter anomalous sessions
        anomalous = self.anomaly_gate.filter_anomalous(sessions)

        # Evaluate if labels are available
        labeled = [s for s in sessions if s.label is not None]
        if labeled:
            metrics = self.anomaly_gate.evaluate(sessions)
            logger.info(f"Evaluation metrics: {json.dumps(metrics, indent=2)}")

        duration = round(time.time() - start, 2)
        logger.info(f"Stage 4 complete in {duration}s — "
                     f"{len(anomalous)} anomalous sessions flagged")

        return anomalous

    # ── Stage 5: Embedding & FAISS Index ────────────────────────────────

    def run_embedding_indexing(self, anomalous_sessions: list,
                                load_existing: bool = False) -> None:
        """
        Run Stage 5: Embed anomalous sessions and build FAISS index.

        Args:
            anomalous_sessions: List of anomalous Session objects.
            load_existing: If True, load existing FAISS index instead of building.
        """
        logger.info("=" * 60)
        logger.info("STAGE 5: Embedding & FAISS Vector Store")
        logger.info("=" * 60)

        start = time.time()

        if load_existing:
            self.vector_store.load()
            logger.info(f"Loaded existing FAISS index with {self.vector_store.size()} vectors")
        else:
            if not anomalous_sessions:
                logger.warning("No anomalous sessions to embed")
                return

            # Batch embed
            embeddings = self.embedder.embed_batch(anomalous_sessions)

            # Build metadata
            metadata_list = []
            for session in anomalous_sessions:
                metadata_list.append({
                    "session_id": session.session_id,
                    "raw_lines": session.raw_lines[:100],  # Limit stored lines
                    "line_range": session.line_range,
                    "label": session.label,
                    "root_cause": session.label if session.label else "Unknown",
                })

            # Add to FAISS index
            self.vector_store.add(embeddings, metadata_list)
            self.vector_store.save()

        duration = round(time.time() - start, 2)
        logger.info(f"Stage 5 complete in {duration}s — "
                     f"{self.vector_store.size()} vectors in index")

    # ── Stage 6: RAG Analysis ───────────────────────────────────────────

    def run_rag_analysis(self, sessions: list, top_k: int = 3,
                          offline: bool = False) -> list:
        """
        Run Stage 6: Retrieval-augmented analysis.

        Args:
            sessions: List of anomalous Session objects to analyze.
            top_k: Number of similar examples to retrieve.
            offline: If True, generate prompts without calling LLM.

        Returns:
            List of analysis result dicts.
        """
        logger.info("=" * 60)
        logger.info("STAGE 6: Retrieval-Augmented Agentic Reasoning")
        logger.info("=" * 60)

        start = time.time()
        top_k = self.config.get("top_k", top_k)

        if offline:
            results = []
            for session in sessions:
                result = self.rag_pipeline.analyze_offline(session, top_k=top_k)
                results.append(result)
        else:
            results = self.rag_pipeline.analyze_batch(sessions, top_k=top_k)

        duration = round(time.time() - start, 2)
        logger.info(f"Stage 6 complete in {duration}s — "
                     f"{len(results)} sessions analyzed")

        return results

    # ── Full Pipeline ───────────────────────────────────────────────────

    def run_full_pipeline(self, input_path: str, label_path: str = None,
                           max_lines: int = None, train_model: bool = True,
                           offline_llm: bool = False,
                           max_analyze: int = 10) -> dict:
        """
        Run the complete end-to-end pipeline.

        Args:
            input_path: Path to raw log file.
            label_path: Optional path to ground-truth labels.
            max_lines: Maximum lines to process (for testing).
            train_model: Whether to train the anomaly model.
            offline_llm: If True, skip LLM calls.
            max_analyze: Maximum number of sessions to analyze with LLM.

        Returns:
            Dict with results from all stages.
        """
        pipeline_start = time.time()
        logger.info("=" * 70)
        logger.info("LOGSENSE PIPELINE — FULL EXECUTION")
        logger.info(f"Input: {input_path}")
        logger.info(f"Dataset type: {self.dataset_type}")
        logger.info("=" * 70)

        results = {"input_path": input_path, "dataset_type": self.dataset_type}

        # Stage 1: Ingestion
        ingestion_stats = self.run_ingestion(input_path)
        results["stage1_ingestion"] = ingestion_stats

        # Stage 2: Parsing
        parsed_events = self.run_parsing(input_path, max_lines=max_lines)
        results["stage2_parsing"] = {
            "total_events": len(parsed_events),
            "template_count": self.parser.get_template_count(),
        }

        # Stage 3: Sessionization
        sessions, vocabulary = self.run_sessionization(parsed_events, label_path)
        results["stage3_sessionization"] = {
            "total_sessions": len(sessions),
            "vocabulary_size": len(vocabulary),
        }

        # Stage 4: Anomaly Gate
        anomalous = self.run_anomaly_gate(sessions, train=train_model)
        results["stage4_anomaly_gate"] = {
            "total_sessions": len(sessions),
            "anomalous_sessions": len(anomalous),
            "compression_ratio": f"{(1 - len(anomalous)/max(len(sessions),1))*100:.1f}%",
        }

        # Stage 5: Embedding & Indexing
        self.run_embedding_indexing(anomalous)
        results["stage5_embedding"] = {
            "indexed_vectors": self.vector_store.size(),
        }

        # Stage 6: RAG Analysis (on a subset)
        sessions_to_analyze = anomalous[:max_analyze]
        if sessions_to_analyze:
            analyses = self.run_rag_analysis(
                sessions_to_analyze,
                offline=offline_llm
            )
            results["stage6_analysis"] = {
                "sessions_analyzed": len(analyses),
                "results": analyses,
            }
        else:
            results["stage6_analysis"] = {"sessions_analyzed": 0, "results": []}

        # Pipeline summary
        total_duration = round(time.time() - pipeline_start, 2)
        results["total_duration_sec"] = total_duration

        compression = len(parsed_events) - len(sessions_to_analyze) if parsed_events else 0
        results["summary"] = {
            "total_lines_processed": len(parsed_events),
            "sessions_created": len(sessions),
            "anomalous_sessions": len(anomalous),
            "sessions_sent_to_llm": len(sessions_to_analyze),
            "compression_efficiency": f"{(compression/max(len(parsed_events),1))*100:.1f}%",
            "total_duration_sec": total_duration,
        }

        logger.info("=" * 70)
        logger.info("PIPELINE COMPLETE")
        logger.info(f"Summary: {json.dumps(results['summary'], indent=2)}")
        logger.info("=" * 70)

        return results

    def save_results(self, results: dict, output_path: str = None):
        """Save pipeline results to a JSON file."""
        if output_path is None:
            output_path = f"results_{self.dataset_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        # Convert non-serializable objects
        def make_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, (tuple,)):
                return list(obj)
            return str(obj)

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=make_serializable)

        logger.info(f"Results saved to {output_path}")


def main():
    arg_parser = argparse.ArgumentParser(
        description="LogSense: Agentic AI Framework for Root Cause Analysis"
    )
    arg_parser.add_argument("input_file", help="Path to raw log file")
    arg_parser.add_argument("-l", "--labels", help="Path to ground-truth labels file",
                            default=None)
    arg_parser.add_argument("-d", "--dataset", choices=["hdfs", "bgl", "thunderbird"],
                            default="hdfs", help="Dataset type")
    arg_parser.add_argument("-n", "--max-lines", type=int, default=None,
                            help="Max lines to process (for testing)")
    arg_parser.add_argument("--max-analyze", type=int, default=10,
                            help="Max sessions to analyze with LLM")
    arg_parser.add_argument("--offline", action="store_true",
                            help="Offline mode — skip LLM calls")
    arg_parser.add_argument("--no-train", action="store_true",
                            help="Load existing model instead of training")
    arg_parser.add_argument("-o", "--output", help="Output results file path",
                            default=None)
    arg_parser.add_argument("--contamination", type=float, default=0.1,
                            help="Isolation Forest contamination parameter")
    arg_parser.add_argument("--window-size", type=int, default=50,
                            help="Sliding window size (BGL/Thunderbird)")
    arg_parser.add_argument("--top-k", type=int, default=3,
                            help="Number of similar examples to retrieve")
    arg_parser.add_argument("-v", "--verbose", action="store_true",
                            help="Enable verbose logging")

    args = arg_parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"logsense_{args.dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        ]
    )

    # Configuration
    config = {
        "dataset_type": args.dataset,
        "contamination": args.contamination,
        "window_size": args.window_size,
        "top_k": args.top_k,
    }

    # Run pipeline
    pipeline = LogSensePipeline(config=config)

    results = pipeline.run_full_pipeline(
        input_path=args.input_file,
        label_path=args.labels,
        max_lines=args.max_lines,
        train_model=not args.no_train,
        offline_llm=args.offline,
        max_analyze=args.max_analyze,
    )

    # Save results
    pipeline.save_results(results, args.output)

    # Print summary
    print("\n" + "=" * 60)
    print("LOGSENSE PIPELINE RESULTS")
    print("=" * 60)
    print(json.dumps(results.get("summary", {}), indent=2))


if __name__ == "__main__":
    main()

