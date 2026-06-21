"""
LogLense Evaluation Pipeline — CLI entry point.

Subcommands:
    build-dataset   Build and save the golden dataset from anomaly_label.csv.
    gate            Evaluate anomaly gate (precision/recall/F1) against golden labels.
    llm             Run LLM-as-judge evaluation on a set of analysis outputs.
    full            Run gate + llm end-to-end and write a combined report.

Usage examples:
    # Step 1: build golden dataset (one-time, requires downloaded HDFS data)
    python eval/run_eval.py build-dataset \\
        --labels data/raw/anomaly_label.csv \\
        --output eval/data/golden_dataset.json

    # Step 2a: gate-only evaluation (no API key needed)
    python eval/run_eval.py gate \\
        --log-file data/raw/HDFS.log \\
        --golden eval/data/golden_dataset.json

    # Step 2b: LLM evaluation on pre-existing analysis results JSON
    python eval/run_eval.py llm \\
        --analyses eval/data/analyses.json \\
        --log-file data/raw/HDFS.log \\
        --golden eval/data/golden_dataset.json

    # Step 3: full end-to-end (runs analysis + gate + llm judges)
    python eval/run_eval.py full \\
        --log-file data/raw/HDFS.log \\
        --labels data/raw/anomaly_label.csv \\
        --golden eval/data/golden_dataset.json \\
        --output eval/data/eval_report.json \\
        --max-sessions 20
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from eval.golden_dataset import (
    create_stratified_sample,
    save_golden_dataset,
    load_golden_dataset,
    get_label_map,
    get_anomalous_ids,
)
from eval.metrics import (
    compute_gate_metrics,
    JUDGE_DIMENSIONS,
    SessionEvalResult,
    compute_weighted_score,
    aggregate_llm_scores,
)
from eval.judges import judge_all_dimensions

logger = logging.getLogger(__name__)


# ── Shared utilities ──────────────────────────────────────────────────────────

def _build_line_index(log_filepath: str) -> dict:
    """Return {block_id: [raw_lines]} for fast per-session log-line lookup."""
    index: dict = {}
    pattern = re.compile(r"(blk_-?\d+)")
    with open(log_filepath, "r", errors="replace") as f:
        for line in f:
            line = line.rstrip()
            for blk_id in pattern.findall(line):
                if blk_id not in index:
                    index[blk_id] = []
                index[blk_id].append(line)
    logger.info(f"Line index: {len(index)} block IDs from {log_filepath}")
    return index


def _print_gate_report(metrics: dict) -> None:
    print(f"\n{'='*56}")
    print("Gate Evaluation Results")
    print(f"{'='*56}")
    print(f"  Precision:        {metrics['precision']:.4f}")
    print(f"  Recall:           {metrics['recall']:.4f}")
    print(f"  F1 Score:         {metrics['f1_score']:.4f}  (target: ≥0.90)")
    print(f"  Accuracy:         {metrics['accuracy']:.4f}")
    print(f"  TP/FP/FN/TN:     {metrics['true_positives']}/{metrics['false_positives']}/"
          f"{metrics['false_negatives']}/{metrics['true_negatives']}")
    print(f"  Sessions eval'd:  {metrics['total_evaluated']}")
    print(f"  Coverage:         {metrics['coverage']:.2%}")
    print(f"  {metrics.get('note', '')}")


def _print_llm_report(agg: dict) -> None:
    print(f"\n{'='*56}")
    print("LLM Judge Evaluation Results")
    print(f"{'='*56}")
    for dim in JUDGE_DIMENSIONS:
        if dim in agg:
            d = agg[dim]
            print(f"  {dim:<30} {d['pass_rate']:.2%}  ({d['pass_count']}/{d['n']})")
    if "composite_unweighted" in agg:
        print(f"  {'composite (unweighted)':<30} {agg['composite_unweighted']['mean']:.2%}")
    if "composite_weighted" in agg:
        print(f"  {'composite (weighted)':<30} {agg['composite_weighted']['mean']:.2%}")
        print(f"    Note: {agg['composite_weighted']['note']}")


# ── Subcommand: build-dataset ─────────────────────────────────────────────────

def cmd_build_dataset(args) -> None:
    dataset = create_stratified_sample(
        labels_path=args.labels,
        n_anomaly=args.n_anomaly,
        normal_multiplier=args.normal_multiplier,
        seed=args.seed,
    )
    save_golden_dataset(dataset, args.output)
    meta = dataset["metadata"]
    print(f"\nGolden dataset saved to {args.output}")
    print(f"  Population: {meta['population_total']:,} | Sample: {meta['sample_size']} "
          f"({meta['sample_anomaly_count']} anomaly + {meta['sample_normal_count']} normal)")
    print(f"  Stat sig (95%/±5%): {meta['is_significant_at_95_5']} "
          f"(min n={meta['min_n_95pct_ci_5pct_margin']})")


# ── Subcommand: gate ──────────────────────────────────────────────────────────

def cmd_gate(args) -> None:
    from pipeline import LogSensePipeline  # noqa: E402

    golden = load_golden_dataset(args.golden)
    ground_truth = get_label_map(golden)
    logger.info(f"Loaded {len(ground_truth)} ground-truth labels")

    pipeline = LogSensePipeline(config={"dataset_type": "hdfs"})
    pipeline.stage1_parse(args.log_file)
    pipeline.stage2_extract_templates()
    pipeline.stage3_sessionize()
    pipeline.stage4_vectorize()
    pipeline.stage5_train_and_predict()

    sessions = pipeline.sessions or []
    predictions = {}
    for session in sessions:
        label = getattr(session, "label", None)
        if label:
            predictions[session.session_id] = label
        # Fallback: use anomaly_score threshold if label not set
        elif hasattr(session, "anomaly_score"):
            score = session.anomaly_score
            if score is not None:
                predictions[session.session_id] = "Anomaly" if score < 0 else "Normal"

    metrics = compute_gate_metrics(predictions, ground_truth)
    _print_gate_report(metrics)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump({"gate_metrics": metrics}, f, indent=2)
        print(f"\nGate report saved to {args.output}")


# ── Subcommand: llm ───────────────────────────────────────────────────────────

def cmd_llm(args) -> None:
    golden = load_golden_dataset(args.golden)
    ground_truth = get_label_map(golden)
    line_index = _build_line_index(args.log_file)

    with open(args.analyses) as f:
        analyses = json.load(f)
    if isinstance(analyses, dict):
        analyses = analyses.get("analyses", [analyses])

    if args.max_sessions:
        analyses = analyses[:args.max_sessions]

    results: list[SessionEvalResult] = []
    for i, analysis in enumerate(analyses, 1):
        sid = analysis.get("session_id", f"unknown_{i}")
        label = ground_truth.get(sid, "Unknown")
        log_lines = line_index.get(sid, [])
        logger.info(f"Judging {i}/{len(analyses)}: {sid} (label={label})")

        scores = judge_all_dimensions(
            analysis_result=analysis,
            raw_log_lines=log_lines,
            session_id=sid,
        )
        raw_scores = [scores[d]["score"] for d in JUDGE_DIMENSIONS if d in scores]
        composite = sum(raw_scores) / len(raw_scores) if raw_scores else 0.0
        weighted = compute_weighted_score(scores)

        results.append(SessionEvalResult(
            session_id=sid,
            label=label,
            llm_scores=scores,
            llm_composite_score=round(composite, 4),
            llm_weighted_score=weighted,
            latency_sec=analysis.get("latency_sec"),
        ))

    agg = aggregate_llm_scores(results)
    _print_llm_report(agg)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        out = {
            "llm_aggregate": agg,
            "session_results": [r.to_dict() for r in results],
        }
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nLLM eval report saved to {args.output}")


# ── Subcommand: full ──────────────────────────────────────────────────────────

def cmd_full(args) -> None:
    from inference_pipeline import analyze_log  # noqa: E402

    started = time.time()
    golden = load_golden_dataset(args.golden)
    ground_truth = get_label_map(golden)
    anomaly_ids = get_anomalous_ids(golden)
    line_index = _build_line_index(args.log_file)

    # --- Run full pipeline ---
    target_ids = anomaly_ids[:args.max_sessions] if args.max_sessions else anomaly_ids
    logger.info(f"Running inference on {len(target_ids)} anomalous sessions...")

    report = analyze_log(
        filepath=args.log_file,
        labels_path=args.labels,
        dataset="hdfs",
        max_analyze=len(target_ids),
        offline=False,
    )
    analyses = report.get("stage6_analysis", {}).get("analyses", [])
    all_sessions = report.get("pipeline_results", {}).get("sessions", [])

    # --- Gate metrics ---
    gate_predictions = {}
    for session in all_sessions:
        label = getattr(session, "label", None)
        if label:
            gate_predictions[session.session_id] = label
    gate_metrics = compute_gate_metrics(gate_predictions, ground_truth)

    # --- LLM judge scores ---
    eval_results: list[SessionEvalResult] = []
    for analysis in analyses:
        sid = analysis.get("session_id")
        if not sid:
            continue
        label = ground_truth.get(sid, "Unknown")
        log_lines = line_index.get(sid, [])

        scores = judge_all_dimensions(
            analysis_result=analysis,
            raw_log_lines=log_lines,
            session_id=sid,
        )
        raw_scores = [scores[d]["score"] for d in JUDGE_DIMENSIONS if d in scores]
        composite = sum(raw_scores) / len(raw_scores) if raw_scores else 0.0
        weighted = compute_weighted_score(scores)

        eval_results.append(SessionEvalResult(
            session_id=sid,
            label=label,
            gate_prediction=gate_predictions.get(sid),
            llm_scores=scores,
            llm_composite_score=round(composite, 4),
            llm_weighted_score=weighted,
            latency_sec=analysis.get("latency_sec"),
        ))

    llm_aggregate = aggregate_llm_scores(eval_results)

    _print_gate_report(gate_metrics)
    _print_llm_report(llm_aggregate)

    combined_report = {
        "eval_metadata": {
            "log_file": args.log_file,
            "labels_file": args.labels,
            "golden_dataset": args.golden,
            "golden_size": golden["metadata"]["sample_size"],
            "sessions_analyzed": len(analyses),
            "total_latency_sec": round(time.time() - started, 3),
        },
        "gate_metrics": gate_metrics,
        "llm_aggregate": llm_aggregate,
        "session_results": [r.to_dict() for r in eval_results],
    }

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(combined_report, f, indent=2)
        print(f"\nFull eval report saved to {args.output}")


# ── Argument parser ───────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="LogLense Evaluation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # build-dataset
    p_build = sub.add_parser("build-dataset", help="Build golden dataset from labels file")
    p_build.add_argument("--labels", required=True)
    p_build.add_argument("--output", default="eval/data/golden_dataset.json")
    p_build.add_argument("--n-anomaly", type=int, default=100)
    p_build.add_argument("--normal-multiplier", type=float, default=3.0)
    p_build.add_argument("--seed", type=int, default=42)

    # gate
    p_gate = sub.add_parser("gate", help="Evaluate anomaly gate (no API key needed)")
    p_gate.add_argument("--log-file", required=True)
    p_gate.add_argument("--golden", default="eval/data/golden_dataset.json")
    p_gate.add_argument("--output", default=None)

    # llm
    p_llm = sub.add_parser("llm", help="LLM judge evaluation on pre-computed analyses")
    p_llm.add_argument("--analyses", required=True, help="Path to analyses JSON file")
    p_llm.add_argument("--log-file", required=True)
    p_llm.add_argument("--golden", default="eval/data/golden_dataset.json")
    p_llm.add_argument("--max-sessions", type=int, default=None)
    p_llm.add_argument("--output", default=None)

    # full
    p_full = sub.add_parser("full", help="End-to-end gate + LLM evaluation")
    p_full.add_argument("--log-file", required=True)
    p_full.add_argument("--labels", default=None)
    p_full.add_argument("--golden", default="eval/data/golden_dataset.json")
    p_full.add_argument("--max-sessions", type=int, default=20)
    p_full.add_argument("--output", default="eval/data/eval_report.json")

    return parser


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = build_parser()
    args = parser.parse_args()

    if args.command == "build-dataset":
        cmd_build_dataset(args)
    elif args.command == "gate":
        cmd_gate(args)
    elif args.command == "llm":
        cmd_llm(args)
    elif args.command == "full":
        cmd_full(args)
