"""
Module 1 - Log Ingestion & Drain Parsing
=========================================

First step of the LogSense pipeline. Reads a raw log file, removes
consecutive duplicate lines, parses every line with Drain3, and writes
a structured CSV plus a saved Drain model.

Supported datasets (pass via --dataset):
    hdfs        -- Hadoop Distributed File System logs
    bgl         -- BlueGene/L supercomputer logs
    thunderbird -- Thunderbird supercomputer logs

If --dataset is omitted the dataset is inferred from the filename
(e.g. "BGL.log" -> bgl). If inference also fails, a minimal "default"
schema is used and a warning is printed -- the pipeline still runs.

Everything runs in ONE streaming pass so it works on files of any size
without loading them into memory.

Files produced:
    data/processed/<stem>_structured.csv    one row per log line
    models/drain_state/drain_templates.pkl  the Drain model (re-loadable)
    models/drain_state/drain_templates.json template list (human-readable)

How to run:
    python src/module1_ingest_parse.py data/raw/HDFS.log --dataset hdfs
    python src/module1_ingest_parse.py data/raw/BGL.log  --dataset bgl
    python src/module1_ingest_parse.py data/raw/Thunderbird.log --dataset thunderbird
    python src/module1_ingest_parse.py data/raw/HDFS.log --max-lines 100000
"""

import sys
import csv
import json
import time
import random
import logging
import argparse
import pickle
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = str(PROJECT_ROOT / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from ingestion import read_log_stream
from log_parser import LogParser

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR  = PROJECT_ROOT / "models" / "drain_state"

# CSV column schemas per dataset.
# Column names are Title-Cased; the row builder lowercases them to look up
# the matching key in the parsed-line dict returned by LogParser.parse_line().
# "LineId", "EventId", "EventTemplate", "ParameterList" are special — they are
# filled by the pipeline, not from the parsed header.
DATASET_CSV_COLUMNS = {
    "hdfs": [
        "LineId", "Date", "Time", "Pid", "Level", "Component",
        "Content", "EventId", "EventTemplate", "ParameterList",
    ],
    "bgl": [
        "LineId", "Label", "Timestamp", "Date", "Node", "Time", "NodeRepeat", "Type",
        "Component", "Level", "Content", "EventId", "EventTemplate", "ParameterList",
    ],
    "thunderbird": [
        "LineId", "Label", "Id", "Date", "Admin", "Time", "AdminAddr",
        "Content", "EventId", "EventTemplate", "ParameterList",
    ],
    # Minimal fallback used when the dataset cannot be identified.
    "default": [
        "LineId", "Content", "EventId", "EventTemplate", "ParameterList",
    ],
}

# Columns that are filled by the pipeline, not from the parsed header dict.
_PIPELINE_COLS = {"lineid", "eventid", "eventtemplate", "parameterlist"}


def _infer_dataset(input_path: Path):
    """
    Try to guess the dataset from the filename.
    Returns "hdfs", "bgl", "thunderbird", or None.
    """
    name = input_path.name.lower()
    for ds in ("hdfs", "bgl", "thunderbird"):
        if ds in name:
            return ds
    return None


def _build_row(csv_columns, kept_lines, event_id, parsed):
    """Build one CSV row dict from a parsed-line result."""
    row = {}
    for col in csv_columns:
        key = col.lower()
        if key == "lineid":
            row[col] = kept_lines
        elif key == "eventid":
            row[col] = event_id
        elif key == "eventtemplate":
            row[col] = parsed["event_template"]
        elif key == "parameterlist":
            row[col] = str(parsed["extracted_variables"])
        else:
            row[col] = parsed.get(key, "")
    return row


def run_module1(
    input_path: str,
    dataset: str = None,
    max_lines: int = None,
    output_csv: str = None,
    output_pkl: str = None,
    skip_dedup: bool = False,
    on_progress: callable = None,
    should_cancel: callable = None,
):
    """
    Run all of Module 1 on a single log file.

    Args:
        input_path: Path to the raw log file (HDFS, BGL, Thunderbird, …).
        dataset:    Log format -- "hdfs", "bgl", or "thunderbird".
                    Inferred from the filename when None; falls back to
                    "default" (minimal schema) if inference also fails.
        max_lines:  Stop after this many kept lines (for quick tests).
        output_csv: Custom CSV output path (auto-named if None).
        output_pkl: Custom Drain pickle path (auto-named if None).
        skip_dedup: Set True if the input is already deduplicated.

    Returns:
        Dict with output paths, summary numbers, and the resolved dataset.
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Resolve dataset
    if dataset is None:
        dataset = _infer_dataset(input_path)
        if dataset:
            logger.info("Dataset inferred from filename: %s", dataset)
        else:
            logger.warning(
                "Could not infer dataset from filename '%s' -- using 'default' "
                "fallback schema. Pass --dataset hdfs|bgl|thunderbird for the "
                "correct column layout.",
                input_path.name,
            )
            dataset = "default"

    csv_columns = DATASET_CSV_COLUMNS.get(dataset, DATASET_CSV_COLUMNS["default"])

    stem = input_path.stem
    csv_path = Path(output_csv) if output_csv else OUTPUT_DIR / f"{stem}_structured.csv"
    pkl_path = Path(output_pkl) if output_pkl else MODEL_DIR / "drain_templates.pkl"
    json_path = pkl_path.with_suffix(".json")

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    pkl_path.parent.mkdir(parents=True, exist_ok=True)

    size_mb = input_path.stat().st_size / (1024 * 1024)
    logger.info("Module 1: Log Ingestion & Drain Parsing")
    logger.info("Dataset : %s", dataset)
    logger.info("Input   : %s (%.1f MB)", input_path, size_mb)
    logger.info("Output  : %s", csv_path)
    logger.info("Columns : %s", csv_columns)
    if max_lines:
        logger.info("Limit   : %s lines", f"{max_lines:,}")

    parser = LogParser(dataset=dataset, persist_state=True, state_dir=str(MODEL_DIR))

    total_lines  = 0
    kept_lines   = 0
    duplicates   = 0
    prev_line    = None
    event_id_counts = Counter()

    head_rows      = []
    reservoir      = []
    seen_after_head = 0
    rng = random.Random(42)

    t0 = time.time()
    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()

        for raw_line in read_log_stream(str(input_path)):
            total_lines += 1

            if not skip_dedup and raw_line == prev_line:
                duplicates += 1
                continue
            prev_line = raw_line
            kept_lines += 1

            parsed   = parser.parse_line(raw_line, line_number=kept_lines)
            event_id = f"E{parsed['event_template_id']}"
            event_id_counts[event_id] += 1

            row = _build_row(csv_columns, kept_lines, event_id, parsed)
            writer.writerow(row)

            if kept_lines <= 5:
                head_rows.append(row)
            else:
                seen_after_head += 1
                if len(reservoir) < 5:
                    reservoir.append(row)
                else:
                    j = rng.randint(0, seen_after_head - 1)
                    if j < 5:
                        reservoir[j] = row

            if kept_lines % 500_000 == 0:
                rate = kept_lines / (time.time() - t0)
                logger.info(
                    "  %s lines | %s templates | %s lines/sec",
                    f"{kept_lines:>10,}", parser.get_template_count(),
                    f"{rate:,.0f}",
                )

            if max_lines and kept_lines >= max_lines:
                logger.info("  Reached --max-lines limit (%s)", f"{max_lines:,}")
                break

    elapsed = time.time() - t0
    spot_check_rows = head_rows + reservoir
    n_unique_events = len(event_id_counts)

    logger.info("Saving Drain state...")
    with open(pkl_path, "wb") as f:
        pickle.dump(parser.template_miner, f, protocol=pickle.HIGHEST_PROTOCOL)

    templates = parser.get_templates()
    summary = {
        "dataset": dataset,
        "input_file": str(input_path),
        "total_lines_scanned": total_lines,
        "deduplicated_lines": kept_lines,
        "duplicates_removed": duplicates,
        "unique_event_templates": parser.get_template_count(),
        "processing_time_sec": round(elapsed, 2),
        "lines_per_second": round(kept_lines / elapsed, 1) if elapsed > 0 else 0,
        "csv_columns": csv_columns,
        "templates": [
            {
                "EventId": f"E{t['cluster_id']}",
                "template": t["template"],
                "occurrences": t["size"],
            }
            for t in sorted(templates, key=lambda x: x["cluster_id"])
        ],
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved: %s, %s", pkl_path.name, json_path.name)

    dedup_pct = (duplicates / total_lines * 100) if total_lines else 0
    rate      = kept_lines / elapsed if elapsed > 0 else 0

    print()
    print("=" * 70)
    print("MODULE 1 - DONE-WHEN CHECKS")
    print("=" * 70)

    print(f"\nDataset : {dataset}")
    print(f"Columns : {csv_columns}")

    print("\nOutput files:")
    print("  ", csv_path)
    print("  ", pkl_path)
    print("  ", json_path)

    print("\nStatistics:")
    print(f"  Total lines read   : {total_lines:>10,}")
    print(f"  Kept after dedup   : {kept_lines:>10,}")
    print(f"  Duplicates removed : {duplicates:>10,}  ({dedup_pct:.2f}%)")
    print(f"  Processing time    : {elapsed:>10.1f}s")
    print(f"  Throughput         : {rate:>10,.0f} lines/sec")

    print(f"\nUnique EventIds: {n_unique_events}")
    if 20 <= n_unique_events <= 50:
        print("  -> PASS (count is small and stable)")
    else:
        print("  -> CHECK (template granularity may need tuning)")

    print(f"\nAll {parser.get_template_count()} templates:")
    for t in sorted(templates, key=lambda x: x["cluster_id"]):
        print(f"  E{t['cluster_id']:<3} (n={t['size']:>8,})  {t['template']}")

    print("\n10 spot-checked lines (content -> template):")
    for i, row in enumerate(spot_check_rows[:10], start=1):
        content  = str(row.get("Content", ""))[:80]
        template = str(row.get("EventTemplate", ""))[:80]
        print(f"  [{i:>2}] line {row.get('LineId', '?'):>8}  "
              f"{row.get('EventId', '?'):>4}  {row.get('Level', '?'):<5}  {content}")
        print(f"        template: {template}")

    print("=" * 70)

    return {
        "csv_path":         str(csv_path),
        "pkl_path":         str(pkl_path),
        "json_path":        str(json_path),
        "dataset":          dataset,
        "csv_columns":      csv_columns,
        "total_lines":      total_lines,
        "dedup_lines":      kept_lines,
        "duplicates_removed": duplicates,
        "unique_event_ids": n_unique_events,
        "template_count":   parser.get_template_count(),
        "processing_time":  round(elapsed, 2),
        "templates":        templates,
    }


def _build_arg_parser():
    ap = argparse.ArgumentParser(
        description="Module 1: Log Ingestion & Drain Parsing -> structured CSV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/module1_ingest_parse.py data/raw/HDFS.log --dataset hdfs
  python src/module1_ingest_parse.py data/raw/BGL.log  --dataset bgl
  python src/module1_ingest_parse.py data/raw/Thunderbird.log --dataset thunderbird
  python src/module1_ingest_parse.py data/raw/HDFS.log --max-lines 100000
        """,
    )
    ap.add_argument("input_file", help="Path to the raw log file")
    ap.add_argument(
        "--dataset", default=None,
        choices=["hdfs", "bgl", "thunderbird"],
        help="Log format: hdfs | bgl | thunderbird  (inferred from filename if omitted)",
    )
    ap.add_argument("--max-lines",  type=int, default=None,
                    help="Only process the first N kept lines")
    ap.add_argument("--output-csv", default=None, help="Custom CSV output path")
    ap.add_argument("--output-pkl", default=None, help="Custom Drain pickle path")
    ap.add_argument("--skip-dedup", action="store_true",
                    help="Skip dedup (input is already deduplicated)")
    return ap


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    run_module1(
        input_path=args.input_file,
        dataset=args.dataset,
        max_lines=args.max_lines,
        output_csv=args.output_csv,
        output_pkl=args.output_pkl,
        skip_dedup=args.skip_dedup,
    )
