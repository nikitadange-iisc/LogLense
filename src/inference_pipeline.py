"""
Module 4: LLM Explanation & End-to-End Inference.

This module exposes the main inference entry point:
    analyze_log(filepath) -> report
"""

import argparse
import json
import logging
import os
import time

from pipeline import LogSensePipeline

logger = logging.getLogger(__name__)


def analyze_log(filepath: str, labels_path: str = None, dataset: str = "hdfs",
                max_analyze: int = 5, max_lines: int = None,
                offline: bool = False, output_path: str = None,
                config: dict = None) -> dict:
    """
    Run LogSense inference and return a root-cause analysis report.

    Args:
        filepath: Path to the raw log file.
        labels_path: Optional path to ground-truth labels.
        dataset: Dataset type: hdfs, bgl, or thunderbird.
        max_analyze: Maximum anomalous sessions to explain.
        max_lines: Optional line limit for development runs.
        offline: If True, return prompts without calling Anthropic.
        output_path: Optional JSON output path.
        config: Optional pipeline config overrides.

    Returns:
        Dict containing pipeline summary, per-session analyses, and timing.
    """
    started = time.time()
    merged_config = {
        "dataset_type": dataset,
        "llm_provider": os.getenv("LLM_PROVIDER", "anthropic"),
        "llm_model": os.getenv("LLM_MODEL", "claude-haiku-4-5"),
    }
    if config:
        merged_config.update(config)

    pipeline = LogSensePipeline(config=merged_config)
    results = pipeline.run_full_pipeline(
        input_path=filepath,
        label_path=labels_path,
        max_lines=max_lines,
        train_model=True,
        offline_llm=offline,
        max_analyze=max_analyze,
    )

    report = {
        "report_type": "logsense_root_cause_analysis",
        "source_file": filepath,
        "dataset": dataset,
        "offline": offline,
        "summary": results.get("summary", {}),
        "stage6_analysis": results.get("stage6_analysis", {}),
        "total_latency_sec": round(time.time() - started, 3),
        "pipeline_results": results,
    }

    if output_path:
        pipeline.save_results(report, output_path)

    return report


def main():
    arg_parser = argparse.ArgumentParser(
        description="LogSense Module 4: LLM Explanation & End-to-End Inference"
    )
    arg_parser.add_argument("input_file", help="Path to raw log file")
    arg_parser.add_argument("-l", "--labels", default=None,
                            help="Path to ground-truth labels file")
    arg_parser.add_argument("-d", "--dataset",
                            choices=["hdfs", "bgl", "thunderbird"],
                            default="hdfs", help="Dataset type")
    arg_parser.add_argument("-n", "--max-lines", type=int, default=None,
                            help="Max lines to process")
    arg_parser.add_argument("--max-analyze", type=int, default=5,
                            help="Max anomalous sessions to explain")
    arg_parser.add_argument("--offline", action="store_true",
                            help="Generate prompts without calling Anthropic")
    arg_parser.add_argument("-o", "--output", default=None,
                            help="Output report JSON path")
    arg_parser.add_argument("--llm-model",
                            default=os.getenv("LLM_MODEL", "claude-haiku-4-5"),
                            help="Anthropic model name")
    arg_parser.add_argument("-v", "--verbose", action="store_true",
                            help="Enable verbose logging")
    args = arg_parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    report = analyze_log(
        filepath=args.input_file,
        labels_path=args.labels,
        dataset=args.dataset,
        max_analyze=args.max_analyze,
        max_lines=args.max_lines,
        offline=args.offline,
        output_path=args.output,
        config={
            "llm_provider": "anthropic",
            "llm_model": args.llm_model,
        },
    )

    print(json.dumps(report.get("summary", {}), indent=2))
    if args.output:
        logger.info(f"Inference report saved to {args.output}")


if __name__ == "__main__":
    main()
