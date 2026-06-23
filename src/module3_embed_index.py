"""
Module 3 - Session Embedding & FAISS Indexing
==============================================

Third step of the LogSense pipeline. Reads anomalous sessions detected by
Module 2, embeds them using sentence-transformers (768-dim), and stores the
embeddings in a FAISS vector index for retrieval by the RAG stage (Module 4).

Also supports manual anomaly ingestion: users can supply their own log files
for any dataset (HDFS, BGL, Thunderbird) to add to the FAISS index without
running the full Module 1→2 pipeline.

Input sources:
    Primary  -- data/processed/<dataset>_anomalies.json  (Module 2 output)
    Manual   -- .json  : same anomalies.json format (Module 2 compatible)
                .txt / .log : raw log lines, one per line
                .csv   : Module 1 structured CSV (re-sessionized on load)

Files produced:
    models/faiss_index/index.faiss       FAISS flat-L2 index
    models/faiss_index/metadata.pkl      Per-vector metadata
    data/processed/<dataset>_embedded.json  Run summary

How to run:
    # Primary — embed Module 2 output
    python src/module3_embed_index.py data/processed/HDFS_anomalies.json --dataset hdfs
    python src/module3_embed_index.py data/processed/bgl_anomalies.json --dataset bgl

    # Manual — add your own anomaly files to the index (--append keeps existing vectors)
    python src/module3_embed_index.py data/processed/HDFS_anomalies.json --dataset hdfs --manual data/raw/my_logs.txt
    python src/module3_embed_index.py --manual-only data/raw/known_anomalies.json --dataset hdfs --append
    python src/module3_embed_index.py --manual-only data/raw/bgl_incidents.txt --dataset bgl --append
    python src/module3_embed_index.py --manual-only data/processed/HDFS_structured.csv --dataset hdfs --append

    # Lighter model (faster, no internet)
    python src/module3_embed_index.py data/processed/HDFS_anomalies.json --dataset hdfs --model all-MiniLM-L6-v2
"""

import sys
import json
import time
import types
import logging
import argparse
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = str(PROJECT_ROOT / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from embedder import SessionEmbedder
from vector_store import FAISSVectorStore
from sessionizer import load_events_from_csv, Sessionizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR  = PROJECT_ROOT / "models"
INDEX_DIR  = MODEL_DIR / "faiss_index"

# Sessionization method used when re-grouping a manual CSV
_SESSION_METHOD = {
    "hdfs":        "block_id",
    "bgl":         "node",
    "thunderbird": "sliding_window",
    "default":     "sliding_window",
}


# ---------------------------------------------------------------------------
# Session factory helpers
# ---------------------------------------------------------------------------

def _make_session(session_id, raw_lines, events,
                  label=None, anomaly_score=None, line_range=None):
    """Return a duck-typed session object compatible with SessionEmbedder."""
    return types.SimpleNamespace(
        session_id=session_id,
        raw_lines=raw_lines or [],
        events=events or [],
        label=label,
        anomaly_score=anomaly_score,
        line_range=tuple(line_range) if line_range else (0, len(raw_lines or [])),
    )


def _event_dicts_from_ids(event_sequence):
    """Convert ['E1', 'E5', ...] to minimal event dicts for the embedder."""
    return [{"event_template": eid, "level": "UNKNOWN"}
            for eid in (event_sequence or [])]


def _build_metadata(session) -> dict:
    """Build the metadata dict stored alongside each FAISS vector."""
    events = getattr(session, "events", []) or []
    event_seq = [
        e.get("event_template", "") if isinstance(e, dict) else str(e)
        for e in events
    ]
    return {
        "session_id":     session.session_id,
        "raw_lines":      (session.raw_lines or [])[:100],
        "label":          getattr(session, "label", None),
        "anomaly_score":  getattr(session, "anomaly_score", None),
        "line_range":     list(session.line_range)
                          if getattr(session, "line_range", None) else [0, 0],
        "event_sequence": event_seq,
    }


# ---------------------------------------------------------------------------
# Session loaders
# ---------------------------------------------------------------------------

def load_sessions_from_anomaly_json(json_path: str) -> tuple:
    """
    Load anomalous sessions from a Module 2 anomalies JSON file.

    Returns:
        (list of session objects, payload metadata dict)
    """
    with open(json_path, encoding="utf-8") as f:
        payload = json.load(f)

    sessions = []
    for rec in payload.get("sessions", []):
        sessions.append(_make_session(
            session_id=rec["session_id"],
            raw_lines=rec.get("raw_lines", []),
            events=_event_dicts_from_ids(rec.get("event_sequence", [])),
            label=rec.get("label"),
            anomaly_score=rec.get("anomaly_score"),
            line_range=rec.get("line_range"),
        ))

    logger.info("Loaded %d sessions from anomaly JSON: %s", len(sessions), json_path)
    return sessions, payload


def load_manual_sessions(file_path: str, dataset: str,
                         session_id_prefix: str = "manual") -> list:
    """
    Load user-provided anomaly data into session objects for any dataset.

    Supported formats (auto-detected by file extension):

      .json  — anomalies.json format (same structure as Module 2 output).
               Each record in "sessions" becomes one session object.

      .txt / .log — plain text file, one log line per line.
               Lines are grouped into windows of 50 (or one session if fewer).
               All sessions are labeled "Anomaly" (user is asserting they are).

      .csv   — Module 1 structured CSV.
               Events are re-sessionized using the dataset's grouping method
               (block_id for HDFS, node for BGL, sliding_window for Thunderbird).
               All resulting sessions are labeled "Anomaly".

    Args:
        file_path         : Path to the manual anomaly file.
        dataset           : "hdfs", "bgl", or "thunderbird".
        session_id_prefix : Prefix prepended to all generated session IDs.

    Returns:
        List of session objects ready for embedding.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Manual anomaly file not found: {file_path}")

    ext = file_path.suffix.lower()

    # ── JSON: same structure as Module 2 anomalies.json ──────────────────
    if ext == ".json":
        sessions, _ = load_sessions_from_anomaly_json(str(file_path))
        for s in sessions:
            if not s.session_id.startswith(session_id_prefix):
                s.session_id = f"{session_id_prefix}_{s.session_id}"
            if s.label is None:
                s.label = "Anomaly"
        logger.info("Manual JSON: loaded %d sessions from %s", len(sessions), file_path.name)
        return sessions

    # ── Plain text / raw log ──────────────────────────────────────────────
    elif ext in (".txt", ".log"):
        with open(file_path, encoding="utf-8", errors="replace") as f:
            lines = [ln.rstrip("\n") for ln in f if ln.strip()]

        if not lines:
            logger.warning("Manual text file is empty: %s", file_path)
            return []

        WINDOW = 50
        sessions = []
        if len(lines) <= WINDOW:
            sessions.append(_make_session(
                session_id=f"{session_id_prefix}_0",
                raw_lines=lines,
                events=[{"event_template": ln[:120], "level": "UNKNOWN"} for ln in lines],
                label="Anomaly",
                line_range=(1, len(lines)),
            ))
        else:
            for i in range(0, len(lines), WINDOW):
                chunk = lines[i: i + WINDOW]
                sessions.append(_make_session(
                    session_id=f"{session_id_prefix}_{i}",
                    raw_lines=chunk,
                    events=[{"event_template": ln[:120], "level": "UNKNOWN"} for ln in chunk],
                    label="Anomaly",
                    line_range=(i + 1, i + len(chunk)),
                ))
        logger.info("Manual text: created %d sessions from %d lines in %s",
                    len(sessions), len(lines), file_path.name)
        return sessions

    # ── Module 1 structured CSV ───────────────────────────────────────────
    elif ext == ".csv":
        method = _SESSION_METHOD.get(dataset, "sliding_window")
        events  = load_events_from_csv(str(file_path), dataset)
        sessionizer = Sessionizer(method=method)
        raw_sessions = sessionizer.create_sessions(iter(events))
        sessions = []
        for s in raw_sessions:
            s.label      = "Anomaly"
            s.session_id = f"{session_id_prefix}_{s.session_id}"
            s.anomaly_score = None
            sessions.append(s)
        logger.info("Manual CSV: created %d sessions via '%s' from %s",
                    len(sessions), method, file_path.name)
        return sessions

    else:
        raise ValueError(
            f"Unsupported manual file format: {ext!r}. "
            "Use .json (anomalies format), .txt/.log (raw lines), or .csv (Module 1 CSV)."
        )


# ---------------------------------------------------------------------------
# Main Module 3 runner
# ---------------------------------------------------------------------------

def run_module3(
    anomaly_json: str = None,
    sessions: list = None,
    dataset: str = "hdfs",
    manual_files: list = None,
    embedding_model: str = "all-mpnet-base-v2",
    embedding_mode: str = "hybrid",
    index_type: str = "flat",
    append: bool = False,
    index_dir: str = None,
    output_json: str = None,
    on_progress: callable = None,
    embedder=None,
) -> dict:
    """
    Run Module 3: embed anomalous sessions and store in FAISS index.

    Args:
        anomaly_json    : Path to Module 2's anomalies JSON (standalone CLI path).
        sessions        : In-memory Session objects (pipeline chaining path).
                          One of anomaly_json or sessions must be provided.
        dataset         : "hdfs", "bgl", or "thunderbird".
        manual_files    : Optional list of manual anomaly file paths to include.
                          Supported: .json, .txt/.log, .csv (auto-detected).
        embedding_model : Sentence-transformer model name.
        embedding_mode  : "hybrid" | "template" | "raw" — text prep for embedder.
        index_type      : "flat" (exact) or "ivf" (approximate, faster at scale).
        append          : If True, add to an existing FAISS index.
                          If False (default), overwrite — rebuild from scratch.
        index_dir       : Override default models/faiss_index/ path.
        output_json     : Override default data/processed/<dataset>_embedded.json.

    Returns:
        Dict with paths, counts, model info, timing, and live embedder/store
        objects for pipeline chaining into Module 4.
    """
    if anomaly_json is None and sessions is None and not manual_files:
        raise ValueError(
            "Provide at least one of: anomaly_json, sessions, or manual_files."
        )

    index_dir   = Path(index_dir)  if index_dir   else INDEX_DIR
    output_json = Path(output_json) if output_json else \
                  OUTPUT_DIR / f"{dataset}_embedded.json"

    index_dir.parent.mkdir(parents=True, exist_ok=True)
    index_dir.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Module 3: Session Embedding & FAISS Indexing")
    logger.info("Dataset         : %s", dataset)
    logger.info("Embedding model : %s", embedding_model)
    logger.info("Embedding mode  : %s", embedding_mode)
    logger.info("Append mode     : %s", append)

    t0 = time.time()

    # ── Step 1: Collect all sessions to embed ──────────────────────────────
    all_sessions = []

    # 1a. Primary source: anomaly JSON or in-memory sessions
    primary_payload = {}
    if sessions is not None:
        all_sessions.extend(sessions)
        logger.info("Pipeline chaining: %d in-memory sessions", len(sessions))
    elif anomaly_json is not None:
        loaded, primary_payload = load_sessions_from_anomaly_json(anomaly_json)
        all_sessions.extend(loaded)

    # 1b. Manual files (optional, any format)
    manual_counts = {}
    for mf in (manual_files or []):
        manual = load_manual_sessions(mf, dataset)
        manual_counts[mf] = len(manual)
        all_sessions.extend(manual)
        logger.info("Manual file '%s': %d sessions added", Path(mf).name, len(manual))

    # ── Filter: only embed confirmed anomalies ─────────────────────────────
    # Sessions labelled "Normal" are Isolation Forest false positives — they
    # must not pollute the reference index.  Unlabelled sessions (manual
    # files, pipeline output without ground truth) are kept because the
    # caller is asserting they are anomalous.
    before = len(all_sessions)
    all_sessions = [
        s for s in all_sessions
        if getattr(s, "label", None) != "Normal"
    ]
    dropped = before - len(all_sessions)
    if dropped:
        logger.info(
            "Filtered out %d Normal-labelled sessions (IF false positives) — "
            "%d remain for indexing", dropped, len(all_sessions)
        )

    if not all_sessions:
        logger.warning("No anomalous sessions to embed — index will not be updated")
        return {
            "output_json":        str(output_json),
            "index_path":         str(index_dir / "index.faiss"),
            "metadata_path":      str(index_dir / "metadata.pkl"),
            "dataset":            dataset,
            "sessions_embedded":  0,
            "index_size":         0,
            "embedding_model":    embedding_model,
            "embedding_dim":      None,
            "embedding_mode":     embedding_mode,
            "processing_time":    0.0,
        }

    logger.info("Total anomalous sessions to embed: %d", len(all_sessions))

    # ── Step 2: Initialize embedder ───────────────────────────────────────
    embedder = SessionEmbedder(model_name=embedding_model)

    # ── Step 3: Initialize / load FAISS store ─────────────────────────────
    store = FAISSVectorStore(
        dimension=embedder.dimension,
        index_type=index_type,
        index_path=str(index_dir),
    )

    if append and (index_dir / "index.faiss").exists():
        store.load()
        logger.info("Appending to existing index (%d vectors)", store.size())
    elif not append:
        store.reset()
        logger.info("Fresh index — previous vectors discarded")

    # ── Step 4: Embed sessions ─────────────────────────────────────────────
    logger.info("Embedding %d sessions (mode=%s)...", len(all_sessions), embedding_mode)
    embeddings = embedder.embed_batch(all_sessions, mode=embedding_mode)

    # ── Step 5: Build metadata and add to index ────────────────────────────
    metadata_list = [_build_metadata(s) for s in all_sessions]
    store.add(embeddings, metadata_list)

    # ── Step 6: Save index ─────────────────────────────────────────────────
    store.save(str(index_dir))

    elapsed = time.time() - t0

    # ── Step 7: Write summary JSON ─────────────────────────────────────────
    output_payload = {
        "dataset":            dataset,
        "embedding_model":    embedder.model_name,
        "embedding_dim":      embedder.dimension,
        "embedding_mode":     embedding_mode,
        "index_type":         index_type,
        "append":             append,
        "sessions_embedded":  len(all_sessions),
        "index_size":         store.size(),
        "processing_time_sec": round(elapsed, 2),
        "primary_source":     anomaly_json or "in-memory",
        "manual_files":       manual_counts,
        "index_path":         str(index_dir / "index.faiss"),
        "metadata_path":      str(index_dir / "metadata.pkl"),
        "sample_sessions": [
            {
                "session_id":    s.session_id,
                "label":         getattr(s, "label", None),
                "anomaly_score": getattr(s, "anomaly_score", None),
                "n_lines":       len(s.raw_lines),
            }
            for s in all_sessions[:5]
        ],
    }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)
    logger.info("Summary saved: %s", output_json)

    # ── Console summary ────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("MODULE 3 - DONE-WHEN CHECKS")
    print("=" * 70)
    print(f"\nDataset          : {dataset}")
    print(f"Embedding model  : {embedder.model_name}")
    print(f"Embedding dim    : {embedder.dimension}")
    print(f"Embedding mode   : {embedding_mode}")
    print(f"Index type       : {index_type}")
    print(f"Append mode      : {append}")
    print(f"\nSessions embedded: {len(all_sessions):>10,}")
    if manual_counts:
        print(f"  (incl. manual) : {sum(manual_counts.values()):>10,}")
    print(f"Vectors in index : {store.size():>10,}")
    print(f"Processing time  : {elapsed:>10.1f}s")
    print(f"\nOutput files:")
    print(f"  {index_dir / 'index.faiss'}")
    print(f"  {index_dir / 'metadata.pkl'}")
    print(f"  {output_json}")
    print(f"\nSample indexed sessions:")
    for rec in output_payload["sample_sessions"]:
        score = f"{rec['anomaly_score']:.4f}" if rec["anomaly_score"] is not None else "n/a"
        print(f"  {rec['session_id'][:45]:<47} label={rec['label'] or 'unlabeled'}  "
              f"score={score}  lines={rec['n_lines']}")
    print("=" * 70)

    return {
        "output_json":        str(output_json),
        "index_path":         str(index_dir / "index.faiss"),
        "metadata_path":      str(index_dir / "metadata.pkl"),
        "dataset":            dataset,
        "sessions_embedded":  len(all_sessions),
        "index_size":         store.size(),
        "embedding_model":    embedder.model_name,
        "embedding_dim":      embedder.dimension,
        "embedding_mode":     embedding_mode,
        "processing_time":    round(elapsed, 2),
        "embedder":           embedder,    # passed to Module 4
        "vector_store":       store,       # passed to Module 4
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser():
    ap = argparse.ArgumentParser(
        description="Module 3: Session Embedding & FAISS Indexing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Primary — embed Module 2 output
  python src/module3_embed_index.py data/processed/HDFS_anomalies.json --dataset hdfs
  python src/module3_embed_index.py data/processed/bgl_anomalies.json --dataset bgl

  # Combine Module 2 output + a manual file in one run
  python src/module3_embed_index.py data/processed/HDFS_anomalies.json --dataset hdfs \\
      --manual data/raw/expert_anomalies.txt

  # Manual-only mode (append to existing index)
  python src/module3_embed_index.py --manual-only data/raw/known_anomalies.json \\
      --dataset hdfs --append
  python src/module3_embed_index.py --manual-only data/raw/bgl_incidents.txt \\
      --dataset bgl --append
  python src/module3_embed_index.py --manual-only data/processed/HDFS_structured.csv \\
      --dataset hdfs --append

  # Use a lighter model (no internet download needed if cached)
  python src/module3_embed_index.py data/processed/HDFS_anomalies.json --dataset hdfs \\
      --model all-MiniLM-L6-v2
        """,
    )

    # Input: primary source is optional when --manual-only is used
    ap.add_argument(
        "anomaly_json", nargs="?", default=None,
        help="Path to Module 2 anomalies JSON (optional when --manual-only is set)",
    )
    ap.add_argument(
        "--dataset", required=True, choices=["hdfs", "bgl", "thunderbird"],
        help="Dataset type (determines sessionization for CSV manual files)",
    )

    # Manual anomaly inputs
    ap.add_argument(
        "--manual", metavar="FILE", action="append", default=[],
        dest="manual_files",
        help="Manual anomaly file to include (.json/.txt/.log/.csv). "
             "Can be repeated to add multiple files.",
    )
    ap.add_argument(
        "--manual-only", metavar="FILE", default=None,
        help="Run in manual-only mode (no anomaly_json required). "
             "The file is embedded and appended/written to the index.",
    )

    # Embedding options
    ap.add_argument(
        "--model", default="all-mpnet-base-v2",
        help="Sentence-transformer model name (default: all-mpnet-base-v2)",
    )
    ap.add_argument(
        "--mode", default="hybrid", choices=["hybrid", "template", "raw"],
        help="Embedding text preparation mode (default: hybrid)",
    )
    ap.add_argument(
        "--index-type", default="flat", choices=["flat", "ivf"],
        help="FAISS index type (default: flat)",
    )

    # Index management
    ap.add_argument(
        "--append", action="store_true",
        help="Append to existing FAISS index (default: overwrite)",
    )
    ap.add_argument(
        "--index-dir", default=None,
        help="Override default models/faiss_index/ path",
    )
    ap.add_argument(
        "--output-json", default=None,
        help="Override default data/processed/<dataset>_embedded.json",
    )

    return ap


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()

    # --manual-only adds its file to the manual list; anomaly_json stays None
    manual_files = list(args.manual_files)
    if args.manual_only:
        manual_files.append(args.manual_only)

    if args.anomaly_json is None and not manual_files:
        _build_arg_parser().error(
            "Provide anomaly_json positional argument and/or --manual-only / --manual FILE."
        )

    run_module3(
        anomaly_json=args.anomaly_json,
        dataset=args.dataset,
        manual_files=manual_files or None,
        embedding_model=args.model,
        embedding_mode=args.mode,
        index_type=args.index_type,
        append=args.append,
        index_dir=args.index_dir,
        output_json=args.output_json,
    )
