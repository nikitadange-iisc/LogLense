"""
Module 2 - Session Grouping & Isolation Forest Anomaly Detection
================================================================

Second step of the LogSense pipeline. Reads the structured CSV produced by
Module 1, groups events into sessions, vectorizes them with a fixed-length
event-count vector, trains an Isolation Forest, and writes the flagged
anomalous sessions to JSON for downstream embedding and RAG stages.

Sessionization strategy:
    hdfs        -- group by Block ID (from ParameterList / Content)
    bgl         -- group by Node column (concentrates fault bursts per node)
    thunderbird -- sliding window of window_size events

Label strategy:
    hdfs        -- optional anomaly_label.csv (BlockId,Label header)
    bgl / tb    -- derived automatically from the 'Label' column written
                   by Module 1 (any event with Label != '-' -> Anomaly)

Files produced:
    models/event_columns.json                  vocabulary (template_id -> index)
    models/isolation_forest.joblib             trained Isolation Forest
    data/processed/<stem>_anomalies.json       flagged session list + metrics

How to run:
    python src/module2_session_anomaly.py data/processed/HDFS_structured.csv --dataset hdfs
    python src/module2_session_anomaly.py data/processed/HDFS_structured.csv --dataset hdfs --label-path data/raw/anomaly_label.csv
    python src/module2_session_anomaly.py data/processed/BGL_structured.csv  --dataset bgl
    python src/module2_session_anomaly.py data/processed/Thunderbird_structured.csv --dataset thunderbird
    python src/module2_session_anomaly.py data/processed/HDFS_structured.csv --dataset hdfs --max-sessions 500
"""

import sys
import json
import time
import logging
import argparse
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = str(PROJECT_ROOT / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from sessionizer import Sessionizer, load_events_from_csv
from anomaly_gate import AnomalyGate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR  = PROJECT_ROOT / "models"

# Sessionization method per dataset
_SESSION_METHOD = {
    "hdfs":        "block_id",
    "bgl":         "node",
    "thunderbird": "sliding_window",
    "default":     "sliding_window",
}

# Vectorization weighting per dataset
# hdfs/thunderbird: count (raw counts work well; keep existing behaviour)
# bgl: tfidf (upweights rare fault templates like E19/E4 vs dominant E1)
_SESSION_WEIGHTING = {
    "hdfs":        "count",
    "bgl":         "tfidf",
    "thunderbird": "count",
    "default":     "count",
}


def run_module2(
    csv_path: str,
    dataset: str,
    label_path: str = None,
    window_size: int = 20,
    step_size: int = 10,
    contamination: float = 0.03,
    n_estimators: int = 200,
    max_sessions: int = None,
    output_json: str = None,
    vocab_path: str = None,
    model_path: str = None,
    train: bool = True,
    weighting: str = None,
    on_stage=None,
) -> dict:
    """
    Run all of Module 2 on a structured CSV produced by Module 1.

    Args:
        csv_path      : Path to Module 1 output CSV.
        dataset       : "hdfs", "bgl", "thunderbird", or "default".
        label_path    : HDFS anomaly_label.csv path (optional; ignored for BGL/TB).
        window_size   : Events per sliding window (Thunderbird).
        step_size     : Stride between windows.
        contamination : Isolation Forest contamination fraction.
        n_estimators  : Number of IF trees.
        max_sessions  : Cap number of sessions (for quick tests).
        output_json   : Custom output path (auto-named if None).
        vocab_path    : Custom vocabulary JSON path (auto-named if None).
        model_path    : Custom IF model path (auto-named if None).
        train         : Train a new model; if False, load existing model.
        weighting     : "count" or "tfidf"; defaults to per-dataset setting.
        on_stage      : Optional callback ``on_stage(stage, message)`` invoked at
                        each major step ("loading", "sessionizing",
                        "vectorizing", "training", "scoring"). Used by the API
                        layer to surface progress in the UI.

    Returns:
        Dict with paths, counts, evaluation metrics, and anomalous sessions.
    """
    def _stage(stage, message):
        logger.info(message)
        if on_stage:
            on_stage(stage, message)

    csv_path  = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    stem        = csv_path.stem.replace("_structured", "")
    vocab_path  = Path(vocab_path)  if vocab_path  else MODEL_DIR / "event_columns.json"
    model_path  = Path(model_path)  if model_path  else MODEL_DIR / "isolation_forest.joblib"
    output_json = Path(output_json) if output_json else OUTPUT_DIR / f"{stem}_anomalies.json"

    vocab_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    method    = _SESSION_METHOD.get(dataset, "sliding_window")
    weighting = weighting or _SESSION_WEIGHTING.get(dataset, "count")

    logger.info("Module 2: Session Grouping & Isolation Forest")
    logger.info("Dataset   : %s", dataset)
    logger.info("CSV       : %s", csv_path)
    logger.info("Method    : %s", method)
    logger.info("Weighting : %s", weighting)

    t0 = time.time()

    # ── Step 1: Load events from Module 1 CSV ───────────────────────────
    _stage("loading", "Loading events from structured CSV…")
    events = load_events_from_csv(str(csv_path), dataset)

    # ── Step 2: Sessionize ──────────────────────────────────────────────
    _stage("sessionizing", "Grouping events into sessions…")
    sessionizer = Sessionizer(
        method=method,
        window_size=window_size,
        step_size=step_size,
        weighting=weighting,
    )
    sessions = sessionizer.create_sessions(iter(events))

    if max_sessions:
        sessions = sessions[:max_sessions]
        logger.info("Capped at %d sessions (--max-sessions)", max_sessions)

    # Fail fast with an actionable message instead of letting an empty feature
    # matrix reach Isolation Forest (which raises an opaque
    # "Expected 2D array, got 1D array instead" error). Zero sessions almost
    # always means the selected dataset type does not match the log format.
    if not sessions:
        _hint = {
            "hdfs": "HDFS grouping needs Block IDs like 'blk_-1608999687919862906' "
                    "on each line.",
            "bgl": "BGL grouping needs node identifiers in each line.",
            "thunderbird": "Thunderbird uses a sliding window over events.",
        }.get(dataset, "")
        raise ValueError(
            f"No sessions could be formed from {len(events):,} events using the "
            f"'{dataset}' method ('{method}'). This usually means the selected "
            f"dataset type does not match the uploaded log format. {_hint}".strip()
        )

    # ── Step 3: Assign ground-truth labels ──────────────────────────────
    if label_path and dataset == "hdfs":
        sessions = sessionizer.load_labels(label_path, sessions)
    elif dataset in ("bgl", "thunderbird"):
        sessions = sessionizer.assign_labels_from_events(sessions)
    # else: no labels; evaluation will be skipped

    # ── Step 4: Vectorize & persist vocabulary ──────────────────────────
    _stage("vectorizing", "Vectorizing sessions…")
    sessions, vocabulary = sessionizer.vectorize_all(sessions)
    idf = (sessionizer.build_idf(sessions, vocabulary)
           if weighting == "tfidf" else None)
    sessionizer.save_vocabulary(vocabulary, str(vocab_path), idf=idf)

    # ── Step 5: Train / load Isolation Forest ───────────────────────────
    _stage("training", "Training Isolation Forest…")
    gate = AnomalyGate(
        model_path=str(model_path),
        contamination=contamination,
        n_estimators=n_estimators,
    )

    if train:
        labeled_normal = [s for s in sessions
                          if s.label and s.label.lower() == "normal"]
        if labeled_normal:
            normal_vecs = np.array([s.vector for s in labeled_normal])
            logger.info("Training on %d labeled-normal sessions", len(labeled_normal))
            gate.train(normal_vecs, save=True)
        else:
            all_vecs = np.array([s.vector for s in sessions])
            logger.info("No labels — training on all %d sessions (contamination=%.3f)",
                        len(sessions), contamination)
            gate.train(all_vecs, all_vectors=all_vecs, save=True)
    else:
        gate.load_model()

    # ── Step 6: Filter anomalous sessions ───────────────────────────────
    _stage("scoring", "Scoring sessions for anomalies…")
    anomalous = gate.filter_anomalous(sessions)

    # ── Step 7: Compute anomaly scores for anomalous sessions ───────────
    anomaly_scores = {}
    if anomalous:
        vecs   = np.array([s.vector for s in anomalous])
        scores = gate.score(vecs)
        for s, sc in zip(anomalous, scores):
            anomaly_scores[s.session_id] = float(sc)
            # Also store the score on the Session object itself so downstream
            # consumers (Module 3 metadata, the API /sessions list, Module 4
            # sorting) can read it via s.anomaly_score instead of this dict.
            s.anomaly_score = float(sc)

    # ── Step 8: Evaluate if labels are available ─────────────────────────
    labeled = [s for s in sessions if s.label is not None]
    evaluation = gate.evaluate(sessions) if labeled else {}

    elapsed = time.time() - t0

    # ── Step 9: Build and save output JSON ──────────────────────────────
    session_records = []
    for s in anomalous:
        event_seq = [f"E{e['event_template_id']}" for e in s.events]
        session_records.append({
            "session_id":    s.session_id,
            "line_range":    list(s.line_range),
            "label":         s.label,
            "anomaly_score": anomaly_scores.get(s.session_id),
            "event_sequence": event_seq,
            "raw_lines":     s.raw_lines[:50],   # cap to avoid huge files
        })

    output_payload = {
        "dataset":            dataset,
        "csv_path":           str(csv_path),
        "total_events":       len(events),
        "total_sessions":     len(sessions),
        "anomalous_sessions": len(anomalous),
        "anomaly_rate_pct":   round(len(anomalous) / len(sessions) * 100, 2)
                              if sessions else 0.0,
        "vocabulary_size":    len(vocabulary),
        "weighting":          weighting,
        "window_size":        window_size if method == "sliding_window" else None,
        "step_size":          step_size   if method == "sliding_window" else None,
        "processing_time_sec": round(elapsed, 2),
        "evaluation":         evaluation,
        "sessions":           session_records,
    }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)
    logger.info("Saved anomalies JSON: %s", output_json)

    # ── Console summary ──────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("MODULE 2 - DONE-WHEN CHECKS")
    print("=" * 70)

    print(f"\nDataset          : {dataset}")
    print(f"Sessionization   : {method}")
    if method == "sliding_window":
        print(f"Window / Step    : {window_size} / {step_size}")

    print(f"\nEvents loaded    : {len(events):>10,}")
    print(f"Sessions created : {len(sessions):>10,}")
    print(f"Vocabulary size  : {len(vocabulary):>10,}")
    print(f"Anomalous        : {len(anomalous):>10,}  "
          f"({output_payload['anomaly_rate_pct']:.2f}%)")
    print(f"Processing time  : {elapsed:>10.1f}s")

    if evaluation:
        print(f"\nEvaluation (vs ground truth):")
        print(f"  Precision : {evaluation.get('precision', 0):.4f}")
        print(f"  Recall    : {evaluation.get('recall', 0):.4f}")
        print(f"  F1        : {evaluation.get('f1_score', 0):.4f}")
        print(f"  Accuracy  : {evaluation.get('accuracy', 0):.4f}")
        print(f"  TP={evaluation.get('true_positives',0)}  "
              f"FP={evaluation.get('false_positives',0)}  "
              f"FN={evaluation.get('false_negatives',0)}  "
              f"TN={evaluation.get('true_negatives',0)}")

    print(f"\nOutput files:")
    print(f"  {vocab_path}")
    print(f"  {model_path}")
    print(f"  {output_json}")

    print(f"\nTop 5 anomalous sessions:")
    for rec in session_records[:5]:
        score_str = f"{rec['anomaly_score']:.4f}" if rec["anomaly_score"] else "n/a"
        label_str = rec["label"] or "unlabeled"
        print(f"  {rec['session_id'][:40]:<42} score={score_str}  label={label_str}")
        print(f"    events: {' '.join(rec['event_sequence'][:10])}")

    print("=" * 70)

    return {
        "output_json":        str(output_json),
        "vocab_path":         str(vocab_path),
        "model_path":         str(model_path),
        "dataset":            dataset,
        "total_events":       len(events),
        "total_sessions":     len(sessions),
        "anomalous_sessions": len(anomalous),
        "anomaly_rate_pct":   output_payload["anomaly_rate_pct"],
        "vocabulary_size":    len(vocabulary),
        "evaluation":         evaluation,
        "sessions":           sessions,          # in-memory, for pipeline chaining
        "anomalous":          anomalous,
        "vocabulary":         vocabulary,
        "gate":               gate,
        "processing_time":    round(elapsed, 2),
    }


def _build_arg_parser():
    ap = argparse.ArgumentParser(
        description="Module 2: Session Grouping & Isolation Forest -> anomaly JSON",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/module2_session_anomaly.py data/processed/HDFS_structured.csv --dataset hdfs
  python src/module2_session_anomaly.py data/processed/HDFS_structured.csv --dataset hdfs --label-path data/raw/anomaly_label.csv
  python src/module2_session_anomaly.py data/processed/BGL_structured.csv  --dataset bgl
  python src/module2_session_anomaly.py data/processed/Thunderbird_structured.csv --dataset thunderbird
  python src/module2_session_anomaly.py data/processed/BGL_structured.csv  --dataset bgl --window-size 20 --step-size 10
        """,
    )
    ap.add_argument("csv_path", help="Path to Module 1 structured CSV")
    ap.add_argument(
        "--dataset", required=True,
        choices=["hdfs", "bgl", "thunderbird"],
        help="Log format",
    )
    ap.add_argument("--label-path",   default=None,
                    help="HDFS anomaly_label.csv (BlockId,Label)")
    ap.add_argument("--window-size",  type=int, default=20,
                    help="Sliding window size (BGL/Thunderbird, default 20)")
    ap.add_argument("--step-size",    type=int, default=10,
                    help="Sliding window step (default 10)")
    ap.add_argument("--contamination", type=float, default=0.03,
                    help="IF contamination for unlabeled mode (default 0.03)")
    ap.add_argument("--n-estimators", type=int, default=200,
                    help="Number of IF trees (default 200)")
    ap.add_argument("--max-sessions", type=int, default=None,
                    help="Cap sessions processed (for quick tests)")
    ap.add_argument("--output-json",  default=None, help="Custom output JSON path")
    ap.add_argument("--vocab-path",   default=None, help="Custom vocabulary JSON path")
    ap.add_argument("--model-path",   default=None, help="Custom IF model path")
    ap.add_argument("--no-train",     action="store_true",
                    help="Load existing IF model instead of training")
    ap.add_argument("--weighting",    default=None, choices=["count", "tfidf"],
                    help="Vector weighting: 'count' (default) or 'tfidf'. "
                         "Defaults to per-dataset setting (tfidf for bgl).")
    return ap


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    run_module2(
        csv_path=args.csv_path,
        dataset=args.dataset,
        label_path=args.label_path,
        window_size=args.window_size,
        step_size=args.step_size,
        contamination=args.contamination,
        n_estimators=args.n_estimators,
        max_sessions=args.max_sessions,
        output_json=args.output_json,
        vocab_path=args.vocab_path,
        model_path=args.model_path,
        train=not args.no_train,
        weighting=args.weighting,
    )
