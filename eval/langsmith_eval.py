"""
LangSmith evaluation runner for the LogLense pipeline.

Wraps analyze_log() as a black-box target and runs it against the golden
dataset using LangSmith's ls_evaluate() framework.

Usage (analyst runs after building golden dataset):
    python eval/langsmith_eval.py \\
        --log-file data/raw/HDFS.log \\
        --labels data/raw/anomaly_label.csv \\
        --golden eval/data/golden_dataset.json \\
        --dataset-name "logsense-golden-v1" \\
        --max-sessions 20

Prerequisites:
    export ANTHROPIC_API_KEY=...
    export LANGCHAIN_API_KEY=...
    export LANGCHAIN_TRACING_V2=true
    export LANGCHAIN_PROJECT=logsense-eval
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

# Ensure project src/ is on the path so inference_pipeline can be imported
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from eval.golden_dataset import load_golden_dataset, get_anomalous_ids, get_label_map
from eval.judges import judge_all_dimensions
from eval.metrics import (
    JUDGE_DIMENSIONS,
    SessionEvalResult,
    compute_weighted_score,
    aggregate_llm_scores,
)

logger = logging.getLogger(__name__)


# ── Session log-line lookup ────────────────────────────────────────────────────

def _build_session_line_index(log_filepath: str) -> dict:
    """
    Scan the raw log file and build {block_id: [raw_lines]} for fast lookup.
    Block IDs are extracted via the blk_-?\\d+ pattern.

    This is done once per eval run and cached in memory.
    """
    import re
    index: dict = {}
    pattern = re.compile(r"(blk_-?\d+)")
    with open(log_filepath, "r", errors="replace") as f:
        for line in f:
            line = line.rstrip()
            for blk_id in pattern.findall(line):
                if blk_id not in index:
                    index[blk_id] = []
                index[blk_id].append(line)
    logger.info(f"Line index built: {len(index)} unique block IDs from {log_filepath}")
    return index


# ── LangSmith dataset helpers ─────────────────────────────────────────────────

def _get_or_create_langsmith_dataset(client, name: str, description: str = "") -> object:
    """Return existing LangSmith dataset by name or create a new one."""
    from langsmith.schemas import DataType
    datasets = list(client.list_datasets(dataset_name=name))
    if datasets:
        logger.info(f"Using existing LangSmith dataset: {name} ({datasets[0].id})")
        return datasets[0]
    ds = client.create_dataset(name, description=description, data_type=DataType.kv)
    logger.info(f"Created LangSmith dataset: {name} ({ds.id})")
    return ds


def upload_golden_dataset_to_langsmith(
    golden_dataset: dict,
    log_filepath: str,
    dataset_name: str,
    max_sessions: Optional[int] = None,
) -> str:
    """
    Upload golden dataset records as LangSmith examples.

    Each example has:
      inputs:  {session_id, label, log_lines (first 50)}
      outputs: {label}   (ground truth)

    Returns:
        LangSmith dataset ID.
    """
    from langsmith import Client

    client = Client()
    ds = _get_or_create_langsmith_dataset(
        client,
        name=dataset_name,
        description=(
            f"LogLense golden evaluation dataset. "
            f"Source: {golden_dataset['metadata']['labels_source']}. "
            f"Statistically significant stratified sample — "
            f"{golden_dataset['metadata']['sample_size']} sessions."
        ),
    )

    line_index = _build_session_line_index(log_filepath)

    records = golden_dataset["records"]
    if max_sessions:
        records = records[:max_sessions]

    inputs_list = []
    outputs_list = []
    for record in records:
        sid = record["session_id"]
        label = record["label"]
        log_lines = line_index.get(sid, [])[:50]
        inputs_list.append({
            "session_id": sid,
            "label": label,
            "log_lines": log_lines,
        })
        outputs_list.append({"label": label})

    client.create_examples(
        inputs=inputs_list,
        outputs=outputs_list,
        dataset_id=ds.id,
    )
    logger.info(f"Uploaded {len(inputs_list)} examples to dataset {dataset_name}")
    return str(ds.id)


# ── Target function (wraps analyze_log) ───────────────────────────────────────

def make_target_function(log_filepath: str, labels_path: Optional[str] = None):
    """
    Return a target function suitable for ls_evaluate().

    The target takes {session_id, label, log_lines} and returns the
    RAGPipeline analysis result for that session.  Because analyze_log()
    processes the whole file, we run it once per session with max_analyze=1
    and a custom per-session config.

    For a full-file multi-session run, use run_eval.py directly.
    """
    from inference_pipeline import analyze_log  # noqa: E402 (src/ on sys.path)

    def target(inputs: dict) -> dict:
        session_id = inputs["session_id"]
        try:
            report = analyze_log(
                filepath=log_filepath,
                labels_path=labels_path,
                dataset="hdfs",
                max_analyze=1,
                offline=False,
            )
            # Extract the analysis for this specific session if present
            analyses = report.get("stage6_analysis", {}).get("analyses", [])
            for a in analyses:
                if a.get("session_id") == session_id:
                    return a
            # Fallback: return first analysis if session_id not matched
            if analyses:
                return analyses[0]
            return {"session_id": session_id, "error": "No analysis produced"}
        except Exception as e:
            logger.error(f"Target function error for {session_id}: {e}")
            return {"session_id": session_id, "error": str(e)}

    return target


# ── Evaluator functions (LangSmith evaluator protocol) ────────────────────────
# Each evaluator takes (run, example) and returns {"key": str, "score": int}

def _make_dim_evaluator(dimension: str):
    """Factory: returns a LangSmith-compatible evaluator for one judge dimension."""

    def evaluator(run, example):
        output = run.outputs or {}
        inputs = example.inputs or {}
        log_lines = inputs.get("log_lines", [])

        if output.get("error"):
            return {"key": dimension, "score": 0}

        scores = judge_all_dimensions(
            analysis_result=output,
            raw_log_lines=log_lines,
            session_id=inputs.get("session_id"),
        )
        dim_result = scores.get(dimension, {"score": 0})
        return {"key": dimension, "score": dim_result["score"]}

    evaluator.__name__ = f"eval_{dimension}"
    return evaluator


# Build all six evaluator functions
_DIMENSION_EVALUATORS = [_make_dim_evaluator(dim) for dim in JUDGE_DIMENSIONS]


def composite_score_evaluator(run, example):
    """Compute unweighted composite score (mean of 6 dimensions)."""
    output = run.outputs or {}
    inputs = example.inputs or {}
    log_lines = inputs.get("log_lines", [])

    if output.get("error"):
        return {"key": "composite_score", "score": 0}

    scores = judge_all_dimensions(
        analysis_result=output,
        raw_log_lines=log_lines,
        session_id=inputs.get("session_id"),
    )
    raw_scores = [scores[d]["score"] for d in JUDGE_DIMENSIONS if d in scores]
    avg = sum(raw_scores) / len(raw_scores) if raw_scores else 0.0
    return {"key": "composite_score", "score": round(avg, 4)}


def weighted_composite_evaluator(run, example):
    """Compute weighted composite score (grounding weighted 1.5×)."""
    output = run.outputs or {}
    inputs = example.inputs or {}
    log_lines = inputs.get("log_lines", [])

    if output.get("error"):
        return {"key": "weighted_composite_score", "score": 0}

    scores = judge_all_dimensions(
        analysis_result=output,
        raw_log_lines=log_lines,
        session_id=inputs.get("session_id"),
    )
    weighted = compute_weighted_score(scores)
    return {"key": "weighted_composite_score", "score": weighted}


ALL_EVALUATORS = _DIMENSION_EVALUATORS + [composite_score_evaluator, weighted_composite_evaluator]


# ── Main evaluation runner ────────────────────────────────────────────────────

def run_langsmith_eval(
    dataset_name: str,
    log_filepath: str,
    labels_path: Optional[str] = None,
    experiment_prefix: str = "logsense-eval",
) -> dict:
    """
    Run LangSmith evaluation over the named dataset.

    Args:
        dataset_name:      Name of the LangSmith dataset to evaluate against.
        log_filepath:      Path to the raw HDFS log file.
        labels_path:       Optional path to anomaly_label.csv.
        experiment_prefix: LangSmith experiment run name prefix.

    Returns:
        Aggregated evaluation results dict.
    """
    from langsmith.evaluation import evaluate as ls_evaluate

    target = make_target_function(log_filepath, labels_path)

    results = ls_evaluate(
        target,
        data=dataset_name,
        evaluators=ALL_EVALUATORS,
        experiment_prefix=experiment_prefix,
        metadata={
            "log_file": log_filepath,
            "labels_file": labels_path,
            "judge_model": os.getenv("EVAL_JUDGE_MODEL", "claude-haiku-4-5-20251001"),
        },
    )

    logger.info(f"LangSmith evaluation complete — experiment: {experiment_prefix}")
    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    ap = argparse.ArgumentParser(description="Run LangSmith evaluation for LogLense")
    ap.add_argument("--log-file", required=True, help="Path to HDFS.log")
    ap.add_argument("--labels", default=None, help="Path to anomaly_label.csv")
    ap.add_argument("--golden", default="eval/data/golden_dataset.json",
                    help="Path to golden_dataset.json")
    ap.add_argument("--dataset-name", default="logsense-golden-v1",
                    help="LangSmith dataset name")
    ap.add_argument("--max-sessions", type=int, default=None,
                    help="Limit sessions uploaded (dev mode)")
    ap.add_argument("--upload-only", action="store_true",
                    help="Upload dataset to LangSmith but skip evaluation run")
    ap.add_argument("--experiment-prefix", default="logsense-eval")
    args = ap.parse_args()

    golden = load_golden_dataset(args.golden)
    print(f"Golden dataset loaded: {golden['metadata']['sample_size']} sessions")

    print(f"Uploading to LangSmith dataset: {args.dataset_name} ...")
    ds_id = upload_golden_dataset_to_langsmith(
        golden_dataset=golden,
        log_filepath=args.log_file,
        dataset_name=args.dataset_name,
        max_sessions=args.max_sessions,
    )
    print(f"Dataset ID: {ds_id}")

    if not args.upload_only:
        print("Running evaluation ...")
        results = run_langsmith_eval(
            dataset_name=args.dataset_name,
            log_filepath=args.log_file,
            labels_path=args.labels,
            experiment_prefix=args.experiment_prefix,
        )
        print("Evaluation complete. Check LangSmith for results.")
