"""
Module 4 - RAG Root Cause Analysis
===================================

Fourth step of the LogSense pipeline. Reads anomalous sessions from Module 2
output (or a FAISS index built by Module 3), retrieves the most similar
historical anomalies, and calls an LLM to produce structured root cause
analysis for each session.

LLM provider auto-detection order:
  1. ANTHROPIC_API_KEY  → Claude  (claude-sonnet-4-6)
  2. OPENAI_API_KEY     → OpenAI  (gpt-4o-mini)
  3. Neither            → offline (retrieval + prompt only)

Input:
    data/processed/<dataset>_anomalies.json   (Module 2 output)
    models/faiss_index/                        (Module 3 output)

Output:
    data/processed/<dataset>_rag_results.json  (per-session analysis)

How to run:
    # Auto-detect provider (needs ANTHROPIC_API_KEY or OPENAI_API_KEY in .env)
    python src/module4_rag_analysis.py data/processed/HDFS_anomalies.json --dataset hdfs

    # Offline mode — no LLM, just retrieval + prompt (no API key needed)
    python src/module4_rag_analysis.py data/processed/HDFS_anomalies.json --dataset hdfs --offline

    # Force Claude with a specific model
    python src/module4_rag_analysis.py data/processed/bgl_anomalies.json --dataset bgl \\
        --llm claude --llm-model claude-sonnet-4-6

    # Analyse only the 5 most anomalous sessions
    python src/module4_rag_analysis.py data/processed/HDFS_anomalies.json --dataset hdfs \\
        --max-sessions 5 --top-k 3
"""

import sys
import json
import time
import types
import logging
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = str(PROJECT_ROOT / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from embedder import SessionEmbedder
from vector_store import FAISSVectorStore
from rag_pipeline import RAGPipeline, _detect_provider

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR  = PROJECT_ROOT / "models"
INDEX_DIR  = MODEL_DIR / "faiss_index"

_DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# ---------------------------------------------------------------------------
# Session loader (reuses module3 helper pattern)
# ---------------------------------------------------------------------------

def _make_session(session_id, raw_lines, events, label=None,
                  anomaly_score=None, line_range=None):
    return types.SimpleNamespace(
        session_id   = session_id,
        raw_lines    = raw_lines or [],
        events       = events or [],
        label        = label,
        anomaly_score= anomaly_score,
        line_range   = tuple(line_range) if line_range else (0, len(raw_lines or [])),
    )


def _event_dicts_from_ids(event_sequence):
    return [{"event_template": eid, "level": "UNKNOWN"}
            for eid in (event_sequence or [])]


def load_anomalous_sessions(json_path: str) -> tuple:
    """
    Load only Anomaly-labelled sessions from a Module 2 anomalies JSON.
    Sorts by anomaly_score ascending (most anomalous first).

    Returns:
        (list of session objects, payload dict)
    """
    with open(json_path, encoding="utf-8") as f:
        payload = json.load(f)

    sessions = []
    skipped  = 0
    for rec in payload.get("sessions", []):
        if rec.get("label") == "Normal":
            skipped += 1
            continue
        sessions.append(_make_session(
            session_id   = rec["session_id"],
            raw_lines    = rec.get("raw_lines", []),
            events       = _event_dicts_from_ids(rec.get("event_sequence", [])),
            label        = rec.get("label"),
            anomaly_score= rec.get("anomaly_score"),
            line_range   = rec.get("line_range"),
        ))

    # Most anomalous first (lowest / most negative score)
    sessions.sort(key=lambda s: s.anomaly_score if s.anomaly_score is not None else 0)

    logger.info("Loaded %d anomalous sessions (%d Normal skipped) from %s",
                len(sessions), skipped, json_path)
    return sessions, payload


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_module4(
    anomaly_json: str,
    dataset: str = "hdfs",
    index_dir: str = None,
    embedding_model: str = _DEFAULT_EMBEDDING_MODEL,
    llm_provider: str = "auto",
    llm_model: str = None,
    top_k: int = 3,
    max_sessions: int = 10,
    output_json: str = None,
    offline: bool = False,
    sessions: list = None,
) -> dict:
    """
    Run Module 4: RAG root cause analysis on anomalous sessions.

    Args:
        anomaly_json   : Path to Module 2 anomalies JSON.
        dataset        : "hdfs" | "bgl" | "thunderbird".
        index_dir      : FAISS index directory (default: models/faiss_index/).
        embedding_model: Must match the model used in Module 3.
        llm_provider   : "auto" | "claude" | "openai" | "offline".
        llm_model      : Override default model name.
        top_k          : Similar sessions to retrieve per query.
        max_sessions   : Cap on LLM calls (most anomalous sessions prioritised).
        output_json    : Override default output path.
        offline        : Skip LLM entirely — retrieval + prompt only.
        sessions       : In-memory sessions (pipeline chaining, skips JSON load).

    Returns:
        Dict with results list, counts, timing, and live rag_pipeline object.
    """
    index_dir   = Path(index_dir) if index_dir else INDEX_DIR
    output_json = Path(output_json) if output_json else \
                  OUTPUT_DIR / f"{dataset}_rag_results.json"
    output_json.parent.mkdir(parents=True, exist_ok=True)

    effective_provider = "offline" if offline else llm_provider

    logger.info("Module 4: RAG Root Cause Analysis")
    logger.info("Dataset        : %s", dataset)
    logger.info("LLM provider   : %s", effective_provider)
    logger.info("Top-K retrieval: %d", top_k)
    logger.info("Max sessions   : %d", max_sessions)

    t0 = time.time()

    # ── Step 1: Load sessions ──────────────────────────────────────────────
    if sessions is not None:
        # Pipeline chaining — filter Normal in-place
        all_sessions = [s for s in sessions if getattr(s, "label", None) != "Normal"]
        logger.info("Pipeline chaining: %d anomalous sessions", len(all_sessions))
        payload = {}
    else:
        all_sessions, payload = load_anomalous_sessions(anomaly_json)

    sessions_to_analyse = all_sessions[:max_sessions]
    logger.info("Sessions to analyse: %d (of %d total anomalous)",
                len(sessions_to_analyse), len(all_sessions))

    # ── Step 2: Load embedder + FAISS index ───────────────────────────────
    embedder = SessionEmbedder(model_name=embedding_model)

    store = FAISSVectorStore(dimension=embedder.dimension)
    store.load(str(index_dir))
    logger.info("FAISS index loaded: %d vectors", store.size())

    # ── Step 3: Build RAG pipeline ─────────────────────────────────────────
    rag = RAGPipeline(
        embedder     = embedder,
        vector_store = store,
        dataset      = dataset,
        llm_provider = effective_provider,
        model        = llm_model,
    )

    # ── Step 4: Analyse sessions ───────────────────────────────────────────
    if offline or rag.provider == "offline":
        logger.info("Offline mode — running retrieval + prompt generation only")
        results = [rag.analyze_offline(s, top_k=top_k) for s in sessions_to_analyse]
    else:
        results = rag.analyze_batch(sessions_to_analyse, top_k=top_k)

    elapsed = round(time.time() - t0, 2)

    # ── Step 5: Write output JSON ──────────────────────────────────────────
    batch_tokens = getattr(rag, "last_batch_tokens", {})
    output_payload = {
        "dataset":          dataset,
        "llm_provider":     rag.provider,
        "llm_model":        rag.model,
        "embedding_model":  embedder.model_name,
        "top_k":            top_k,
        "total_anomalous":  len(all_sessions),
        "sessions_analysed": len(results),
        "processing_time_sec": elapsed,
        "offline_mode":     offline or rag.provider == "offline",
        "token_usage_total": batch_tokens,
        "results":          results,
    }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2, default=str)
    logger.info("Results saved: %s", output_json)

    # ── Console summary ────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("MODULE 4 - RAG ROOT CAUSE ANALYSIS")
    print("=" * 70)
    print(f"Dataset          : {dataset}")
    print(f"LLM provider     : {rag.provider}")
    print(f"LLM model        : {rag.model or 'n/a (offline)'}")
    print(f"Embedding model  : {embedder.model_name}")
    print(f"Top-K retrieval  : {top_k}")
    print(f"Sessions analysed: {len(results):>6,}")
    print(f"Processing time  : {elapsed:>6.1f}s")
    if batch_tokens:
        avg_in  = batch_tokens["input"]  // max(len(results), 1)
        avg_out = batch_tokens["output"] // max(len(results), 1)
        print(f"Tokens  (total)  : {batch_tokens['total']:>8,}  "
              f"(in={batch_tokens['input']:,}  out={batch_tokens['output']:,})")
        print(f"Tokens  (per session): in~{avg_in:,}  out~{avg_out:,}  "
              f"total~{avg_in+avg_out:,}")
    print(f"\nOutput: {output_json}")

    if not (offline or rag.provider == "offline"):
        print("\nSample results:")
        for r in results[:3]:
            if "error" in r:
                print(f"  {r['session_id'][:45]:<47} ERROR: {r['error'][:60]}")
            else:
                sev  = r.get("severity", "?")
                conf = r.get("confidence", 0)
                rc   = r.get("root_cause", "?")[:60]
                print(f"  {r['session_id'][:45]:<47} [{sev}] conf={conf:.2f}  {rc}")
    else:
        print("\nOffline mode — prompts generated, no LLM analysis performed.")
        print("Pass the 'prompt' field from the output JSON to any LLM.")

    print("=" * 70)

    return {
        "output_json":       str(output_json),
        "dataset":           dataset,
        "sessions_analysed": len(results),
        "total_anomalous":   len(all_sessions),
        "llm_provider":      rag.provider,
        "llm_model":         rag.model,
        "processing_time":   elapsed,
        "results":           results,
        "rag_pipeline":      rag,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser():
    ap = argparse.ArgumentParser(
        description="Module 4: RAG Root Cause Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect LLM (needs API key in .env or environment)
  python src/module4_rag_analysis.py data/processed/HDFS_anomalies.json --dataset hdfs

  # Offline mode — no API key needed, generates prompts only
  python src/module4_rag_analysis.py data/processed/HDFS_anomalies.json --dataset hdfs --offline

  # BGL with Claude explicitly
  python src/module4_rag_analysis.py data/processed/bgl_anomalies.json --dataset bgl \\
      --llm claude --llm-model claude-sonnet-4-6 --max-sessions 20

  # OpenAI on top 5 most anomalous sessions
  python src/module4_rag_analysis.py data/processed/HDFS_anomalies.json --dataset hdfs \\
      --llm openai --max-sessions 5 --top-k 5
        """,
    )
    ap.add_argument("anomaly_json", help="Path to Module 2 anomalies JSON")
    ap.add_argument("--dataset", required=True,
                    choices=["hdfs", "bgl", "thunderbird"],
                    help="Dataset type")
    ap.add_argument("--llm", default="auto",
                    choices=["auto", "claude", "openai", "offline"],
                    dest="llm_provider",
                    help="LLM provider (default: auto-detect from env vars)")
    ap.add_argument("--llm-model", default=None,
                    help="Override default LLM model name")
    ap.add_argument("--embedding-model", default=_DEFAULT_EMBEDDING_MODEL,
                    help=f"Sentence-transformer model (default: {_DEFAULT_EMBEDDING_MODEL})")
    ap.add_argument("--index-dir", default=None,
                    help="FAISS index directory (default: models/faiss_index/)")
    ap.add_argument("--top-k", type=int, default=3,
                    help="Similar sessions to retrieve per query (default: 3)")
    ap.add_argument("--max-sessions", type=int, default=10,
                    help="Max sessions to analyse with LLM (default: 10)")
    ap.add_argument("--offline", action="store_true",
                    help="Skip LLM — retrieval + prompt generation only")
    ap.add_argument("--output-json", default=None,
                    help="Override default output JSON path")
    return ap


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    run_module4(
        anomaly_json   = args.anomaly_json,
        dataset        = args.dataset,
        index_dir      = args.index_dir,
        embedding_model= args.embedding_model,
        llm_provider   = args.llm_provider,
        llm_model      = args.llm_model,
        top_k          = args.top_k,
        max_sessions   = args.max_sessions,
        output_json    = args.output_json,
        offline        = args.offline,
    )
