"""
Stage 4: Isolation Forest Anomaly Gate

Trains an Isolation Forest on normal session vectors using LogHub ground-truth
labels. At inference, scores each session vector and discards normal sessions,
passing only flagged anomalous sessions downstream.
"""

import os
import logging
import argparse
from pathlib import Path

import numpy as np
import joblib
from sklearn.ensemble import IsolationForest

logger = logging.getLogger(__name__)


class AnomalyGate:
    """Isolation Forest-based anomaly detection gate."""

    def __init__(self, model_path: str = "models/isolation_forest.joblib",
                 contamination: float = 0.1, n_estimators: int = 200,
                 random_state: int = 42):
        """
        Args:
            model_path: Path to save/load the trained model.
            contamination: Expected proportion of anomalies (for training).
            n_estimators: Number of trees in the forest.
            random_state: Random seed for reproducibility.
        """
        self.model_path = Path(model_path)
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model = None

        logger.info(f"AnomalyGate initialized (contamination={contamination}, "
                     f"n_estimators={n_estimators})")

    def train(self, normal_session_vectors: np.ndarray,
              all_session_vectors: np.ndarray = None):
        """
        Train Isolation Forest on normal session vectors.

        Args:
            normal_session_vectors: Numpy array of normal session count vectors.
            all_session_vectors: Optional. If provided, trains on all data with
                                 contamination parameter. Otherwise trains on
                                 normal data only with contamination='auto'.
        """
        if all_session_vectors is not None:
            training_data = all_session_vectors
            contam = self.contamination
            logger.info(f"Training on all sessions ({len(training_data)}) "
                        f"with contamination={contam}")
        else:
            training_data = normal_session_vectors
            contam = "auto"
            logger.info(f"Training on normal sessions only ({len(training_data)})")

        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=contam,
            random_state=self.random_state,
            n_jobs=-1,
        )

        self.model.fit(training_data)
        logger.info("Isolation Forest training complete")

        self.save_model()

    def save_model(self, path: str = None):
        """Save the trained model to disk."""
        save_path = Path(path) if path else self.model_path
        save_path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.model, save_path)
        logger.info(f"Model saved to {save_path}")

    def load_model(self, path: str = None):
        """Load a trained model from disk."""
        load_path = Path(path) if path else self.model_path
        if not load_path.exists():
            raise FileNotFoundError(f"Model file not found: {load_path}")

        self.model = joblib.load(load_path)
        logger.info(f"Model loaded from {load_path}")

    def predict(self, session_vectors: np.ndarray) -> np.ndarray:
        """
        Predict anomaly labels for session vectors.

        Args:
            session_vectors: Numpy array of session count vectors.

        Returns:
            Numpy array of predictions: 1 for normal, -1 for anomaly.
        """
        if self.model is None:
            raise RuntimeError("Model not trained or loaded. Call train() or load_model() first.")
        return self.model.predict(session_vectors)

    def score(self, session_vectors: np.ndarray) -> np.ndarray:
        """
        Compute anomaly scores for session vectors.

        Args:
            session_vectors: Numpy array of session count vectors.

        Returns:
            Numpy array of anomaly scores (lower = more anomalous).
        """
        if self.model is None:
            raise RuntimeError("Model not trained or loaded. Call train() or load_model() first.")
        return self.model.decision_function(session_vectors)

    def filter_anomalous(self, sessions: list) -> list:
        """
        Filter sessions, returning only those flagged as anomalous.

        Args:
            sessions: List of Session objects (with .vector attribute).

        Returns:
            List of anomalous Session objects.
        """
        if not sessions:
            return []

        vectors = np.array([s.vector for s in sessions])
        predictions = self.predict(vectors)

        anomalous = [s for s, pred in zip(sessions, predictions) if pred == -1]

        stats = self.get_gate_statistics(sessions)
        logger.info(
            f"Anomaly gate: {stats['anomalous_count']}/{stats['total_sessions']} "
            f"sessions flagged ({stats['anomaly_percentage']:.1f}%), "
            f"{stats['normal_count']} discarded"
        )

        return anomalous

    def get_gate_statistics(self, sessions: list) -> dict:
        """
        Compute gate statistics for a set of sessions.

        Args:
            sessions: List of Session objects with vectors.

        Returns:
            Dict with total_sessions, normal_count, anomalous_count,
            anomaly_percentage.
        """
        if not sessions:
            return {
                "total_sessions": 0,
                "normal_count": 0,
                "anomalous_count": 0,
                "anomaly_percentage": 0.0,
            }

        vectors = np.array([s.vector for s in sessions])
        predictions = self.predict(vectors)

        normal_count = int(np.sum(predictions == 1))
        anomalous_count = int(np.sum(predictions == -1))
        total = len(sessions)

        return {
            "total_sessions": total,
            "normal_count": normal_count,
            "anomalous_count": anomalous_count,
            "anomaly_percentage": (anomalous_count / total * 100) if total else 0.0,
        }

    def evaluate(self, sessions: list) -> dict:
        """
        Evaluate anomaly detection against ground-truth labels.

        Args:
            sessions: List of Session objects with .label and .vector attributes.

        Returns:
            Dict with precision, recall, f1_score, accuracy.
        """
        labeled = [s for s in sessions if s.label is not None]
        if not labeled:
            logger.warning("No labeled sessions found for evaluation")
            return {}

        vectors = np.array([s.vector for s in labeled])
        predictions = self.predict(vectors)

        # Convert to binary: 1=anomaly, 0=normal
        pred_binary = (predictions == -1).astype(int)
        true_binary = np.array([1 if s.label.lower() == "anomaly" else 0 for s in labeled])

        tp = int(np.sum((pred_binary == 1) & (true_binary == 1)))
        fp = int(np.sum((pred_binary == 1) & (true_binary == 0)))
        fn = int(np.sum((pred_binary == 0) & (true_binary == 1)))
        tn = int(np.sum((pred_binary == 0) & (true_binary == 0)))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / len(labeled) if labeled else 0.0

        metrics = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "accuracy": accuracy,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "true_negatives": tn,
            "total_labeled": len(labeled),
        }

        logger.info(f"Evaluation — P: {precision:.3f}, R: {recall:.3f}, "
                     f"F1: {f1:.3f}, Acc: {accuracy:.3f}")
        return metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    # Quick test with synthetic data
    print("Testing AnomalyGate with synthetic data...")

    np.random.seed(42)
    normal_data = np.random.randn(100, 10)
    anomaly_data = np.random.randn(10, 10) + 5  # Shifted anomalies

    gate = AnomalyGate(model_path="models/test_isolation_forest.joblib")
    gate.train(normal_data)

    # Test predictions
    all_data = np.vstack([normal_data, anomaly_data])
    predictions = gate.predict(all_data)
    scores = gate.score(all_data)

    print(f"Normal predictions (first 10): {predictions[:10]}")
    print(f"Anomaly predictions (last 10): {predictions[-10:]}")
    print(f"Score range: [{scores.min():.3f}, {scores.max():.3f}]")

