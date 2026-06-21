"""
Module 2b - Isolation Forest Anomaly Gate
==========================================

Trains an Isolation Forest on session count vectors and filters out
normal sessions, passing only anomalous ones to downstream stages.

Training modes:
  - With ground-truth labels (HDFS / BGL / Thunderbird all supported now):
      train on labeled-normal sessions only, contamination='auto'.
  - Without labels:
      train on all sessions with the configured contamination parameter.
"""

import logging
from pathlib import Path

import numpy as np
import joblib
from sklearn.ensemble import IsolationForest

logger = logging.getLogger(__name__)


class AnomalyGate:
    """Isolation Forest-based anomaly detection gate."""

    def __init__(self, model_path: str = "models/isolation_forest.joblib",
                 contamination: float = 0.03, n_estimators: int = 200,
                 random_state: int = 42):
        """
        Args:
            model_path    : Path to save/load the trained model.
            contamination : Expected fraction of anomalies (for unsupervised mode).
                            Default 0.03 matches paper (approx 3% anomaly rate).
            n_estimators  : Number of trees in the forest.
            random_state  : Seed for reproducibility.
        """
        self.model_path   = Path(model_path)
        self.contamination = contamination
        self.n_estimators  = n_estimators
        self.random_state  = random_state
        self.model         = None

        logger.info("AnomalyGate initialized (contamination=%s, n_estimators=%d)",
                    contamination, n_estimators)

    # ── Training ────────────────────────────────────────────────────────

    def train(self, normal_vectors: np.ndarray,
              all_vectors: np.ndarray = None,
              save: bool = True) -> None:
        """
        Fit the Isolation Forest.

        Args:
            normal_vectors : Vectors for labeled-normal sessions.
                             If this is the only argument, trains with
                             contamination='auto' (unsupervised-friendly).
            all_vectors    : If provided, train on all data using the
                             configured contamination parameter instead.
            save           : Write the model to disk after training.
        """
        if all_vectors is not None:
            training_data = all_vectors
            contam = self.contamination
            logger.info("Training on all sessions (%d) with contamination=%s",
                        len(training_data), contam)
        else:
            training_data = normal_vectors
            contam = self.contamination
            logger.info("Training on labeled-normal sessions (%d), contamination=%s",
                        len(training_data), contam)

        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=contam,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.model.fit(training_data)
        logger.info("Isolation Forest training complete")

        if save:
            self.save_model()

    # ── Persistence ─────────────────────────────────────────────────────

    def save_model(self, path: str = None) -> None:
        save_path = Path(path) if path else self.model_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, save_path)
        logger.info("Model saved to %s", save_path)

    def load_model(self, path: str = None) -> None:
        load_path = Path(path) if path else self.model_path
        if not load_path.exists():
            raise FileNotFoundError(f"Model not found: {load_path}")
        self.model = joblib.load(load_path)
        logger.info("Model loaded from %s", load_path)

    # ── Inference ───────────────────────────────────────────────────────

    def predict(self, vectors: np.ndarray) -> np.ndarray:
        """Return predictions: +1 = normal, -1 = anomaly."""
        if self.model is None:
            raise RuntimeError("Call train() or load_model() first.")
        return self.model.predict(vectors)

    def score(self, vectors: np.ndarray) -> np.ndarray:
        """Return decision-function scores (lower = more anomalous)."""
        if self.model is None:
            raise RuntimeError("Call train() or load_model() first.")
        return self.model.decision_function(vectors)

    def filter_anomalous(self, sessions: list) -> list:
        """
        Return only sessions flagged as anomalous.

        Runs predict() exactly once — does not call get_gate_statistics()
        to avoid a second full inference pass.
        """
        if not sessions:
            return []

        vectors     = np.array([s.vector for s in sessions])
        predictions = self.predict(vectors)

        anomalous       = [s for s, p in zip(sessions, predictions) if p == -1]
        normal_count    = int(np.sum(predictions == 1))
        anomalous_count = len(anomalous)
        total           = len(sessions)

        logger.info(
            "Anomaly gate: %d/%d sessions flagged (%.1f%%), %d discarded",
            anomalous_count, total,
            (anomalous_count / total * 100) if total else 0.0,
            normal_count,
        )
        return anomalous

    def get_gate_statistics(self, sessions: list) -> dict:
        """Compute gate statistics (runs one predict pass)."""
        if not sessions:
            return {"total_sessions": 0, "normal_count": 0,
                    "anomalous_count": 0, "anomaly_percentage": 0.0}

        vectors     = np.array([s.vector for s in sessions])
        predictions = self.predict(vectors)

        normal_count    = int(np.sum(predictions == 1))
        anomalous_count = int(np.sum(predictions == -1))
        total           = len(sessions)

        return {
            "total_sessions":    total,
            "normal_count":      normal_count,
            "anomalous_count":   anomalous_count,
            "anomaly_percentage": (anomalous_count / total * 100) if total else 0.0,
        }

    # ── Evaluation ──────────────────────────────────────────────────────

    def evaluate(self, sessions: list) -> dict:
        """
        Evaluate against ground-truth labels.

        Sessions must have .label set to "Normal" or "Anomaly".
        Returns precision, recall, F1, accuracy plus confusion matrix cells.
        """
        labeled = [s for s in sessions if s.label is not None]
        if not labeled:
            logger.warning("No labeled sessions for evaluation")
            return {}

        vectors     = np.array([s.vector for s in labeled])
        predictions = self.predict(vectors)

        pred_binary = (predictions == -1).astype(int)
        true_binary = np.array(
            [1 if s.label.lower() == "anomaly" else 0 for s in labeled]
        )

        tp = int(np.sum((pred_binary == 1) & (true_binary == 1)))
        fp = int(np.sum((pred_binary == 1) & (true_binary == 0)))
        fn = int(np.sum((pred_binary == 0) & (true_binary == 1)))
        tn = int(np.sum((pred_binary == 0) & (true_binary == 0)))

        precision = tp / (tp + fp)   if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn)   if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)
        accuracy  = (tp + tn) / len(labeled) if labeled else 0.0

        metrics = {
            "precision":      round(precision, 4),
            "recall":         round(recall, 4),
            "f1_score":       round(f1, 4),
            "accuracy":       round(accuracy, 4),
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "true_negatives": tn,
            "total_labeled":  len(labeled),
        }
        logger.info("Evaluation — P=%.3f  R=%.3f  F1=%.3f  Acc=%.3f",
                    precision, recall, f1, accuracy)
        return metrics
