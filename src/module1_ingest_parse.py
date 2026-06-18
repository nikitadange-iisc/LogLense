"""
Module 1 — Log Ingestion & Drain Parsing
=========================================

End-to-end runner for Module 1 of the LogSense pipeline.

What it does
------------
1. Streaming ingestion of HDFS.log (no full-file load).
2. Consecutive-duplicate removal.
3. Drain3 parsing with HDFS-specific masks (block ID, IP, numeric IDs).
4. Outputs:
   - ``HDFS.log_structured.csv``  — one row per log line with columns:
       LineId, Date, Time, Pid, Level, Component, Content,
       EventId, EventTemplate, ParameterList
   - ``drain_templates.pkl``      — persisted Drain3 TemplateMiner state
   - ``drain_templates.json``     — human-readable template summary
5. Prints done-when-check: ``nunique()`` on EventId, 10 spot-checked lines.

Usage
-----
    # Full HDFS.log (11.17M lines):
    python src/module1_ingest_parse.py data/raw/HDFS.log

    # 1% sample for dev:
    python src/module1_ingest_parse.py data/raw/HDFS_sample_1pct.log

    # Limit to first N lines:
    python src/module1_ingest_parse.py data/raw/HDFS.log --max-lines 100000

    # Use the tiny sample already in repo:
    python src/module1_ingest_parse.py data/raw/sample_hdfs.log
"""

import os
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

import numpy as np

# ── Project paths ───────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ingestion import read_log_stream, stream_deduplicated, deduplicate_stream
from log_parser import LogParser

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── Output directories ─────────────────────────────────────────────────
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models" / "drain_state"


def run_module1(
    input_path: str,
    max_lines: int = None,
    output_csv: str = None,
    output_pkl: str = None,
    skip_dedup: bool = False,
):
    """
    Run Module 1: Ingestion + Drain Parsing.

    Parameters
    ----------
    input_path : str
        Path to raw HDFS log file.
    max_lines : int, optional
        Limit processing to this many lines (for dev/testing).
    output_csv : str, optional
        Custom path for the structured CSV output.
    output_pkl : str, optional
        Custom path for the Drain state pickle.
    skip_dedup : bool
        If True, skip deduplication (file already deduped).

    Returns
    -------
    dict
        Summary statistics and file paths.
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    stem = input_path.stem  # e.g. "HDFS" or "HDFS_sample_1pct"
    csv_path = Path(output_csv) if output_csv else OUTPUT_DIR / f"{stem}_structured.csv"
    pkl_path = Path(output_pkl) if output_pkl else MODEL_DIR / "drain_templates.pkl"
    json_path = pkl_path.with_suffix(".json")

    file_size_mb = input_path.stat().st_size / (1024 * 1024)
    logger.info(f"═══ Module 1: Log Ingestion & Drain Parsing ═══")
    logger.info(f"Input:  {input_path}  ({file_size_mb:.1f} MB)")
    logger.info(f"Output: {csv_path}")
    logger.info(f"State:  {pkl_path}")
    if max_lines:
        logger.info(f"Limit:  {max_lines:,} lines")

    t0 = time.time()

    # ── Stage 1: Streaming ingestion + dedup ────────────────────────────
    logger.info("─── Stage 1: Streaming Ingestion + Deduplication ───")
    dedup_stats = {"total_lines": 0, "deduplicated_lines": 0, "duplicates_removed": 0}

    # ── Stage 2: Drain parsing → structured CSV ─────────────────────────
    logger.info("─── Stage 2: Drain Parsing (HDFS masks) ───")
    parser = LogParser(dataset="hdfs", persist_state=True, state_dir=str(MODEL_DIR))

    # CSV columns matching LogPai / LogHub convention
    csv_columns = [
        "LineId", "Date", "Time", "Pid", "Level", "Component",
        "Content", "EventId", "EventTemplate", "ParameterList",
    ]

    total_lines = 0
    dedup_lines = 0
    prev_line = None
    event_id_counter = Counter()

    # We store spot-check rows for the done-when verification
    spot_check_rows = []
    spot_check_indices = set()

    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()

        for raw_line in read_log_stream(str(input_path)):
            total_lines += 1

            # ── Dedup (consecutive) ─────────────────────────────────────
            if not skip_dedup and raw_line == prev_line:
                dedup_stats["duplicates_removed"] += 1
                continue
            prev_line = raw_line
            dedup_lines += 1

            # ── Parse with Drain ─────────────────────────────────────────
            parsed = parser.parse_line(raw_line, line_number=dedup_lines)

            # Build CSV row
            # Re-extract header fields for CSV (already parsed inside LogParser)
            header = parser._parse_header(raw_line)
            row = {
                "LineId": dedup_lines,
                "Date": header.get("date", ""),
                "Time": header.get("time", ""),
                "Pid": header.get("pid", ""),
                "Level": parsed["level"],
                "Component": parsed["component"],
                "Content": header.get("content", raw_line),
                "EventId": f"E{parsed['event_template_id']}",
                "EventTemplate": parsed["event_template"],
                "ParameterList": str(parsed["extracted_variables"]),
            }
            writer.writerow(row)
            event_id_counter[row["EventId"]] += 1

            # Collect spot-check candidates (first 5 + 5 random later)
            if dedup_lines <= 5:
                spot_check_rows.append(row)
            elif dedup_lines == 100:
                # Pre-select random indices for spot-checking
                # (we'll grab lines near these indices)
                pass

            # Progress logging
            if dedup_lines % 500_000 == 0:
                elapsed = time.time() - t0
                rate = dedup_lines / elapsed
                templates = parser.get_template_count()
                logger.info(
                    f"  {dedup_lines:>10,} lines | "
                    f"{templates:>3} templates | "
                    f"{rate:,.0f} lines/sec | "
                    f"{elapsed:.1f}s elapsed"
                )

            if max_lines and dedup_lines >= max_lines:
                logger.info(f"  Reached --max-lines limit ({max_lines:,})")
                break

    dedup_stats["total_lines"] = total_lines
    dedup_stats["deduplicated_lines"] = dedup_lines

    elapsed = time.time() - t0

    # ── Collect additional spot-check rows (tail of file) ───────────────
    # Re-read last few rows from CSV for spot-checking
    try:
        import pandas as pd
        df = pd.read_csv(csv_path, dtype=str)
        n_rows = len(df)
        if n_rows > 10:
            # 5 from head, 5 random from rest
            random.seed(42)
            tail_indices = sorted(random.sample(range(5, n_rows), min(5, n_rows - 5)))
            for idx in tail_indices:
                spot_check_rows.append(df.iloc[idx].to_dict())
        elif n_rows > 5:
            for idx in range(5, n_rows):
                spot_check_rows.append(df.iloc[idx].to_dict())
        n_unique_events = df["EventId"].nunique()
    except Exception:
        n_unique_events = len(event_id_counter)

    # ── Save Drain state ────────────────────────────────────────────────
    logger.info("─── Saving Drain state ───")
    # Save pickle
    with open(pkl_path, "wb") as f:
        pickle.dump(parser.template_miner, f)
    logger.info(f"  Pickle: {pkl_path}")

    # Save JSON summary
    templates = parser.get_templates()
    summary = {
        "dataset": "hdfs",
        "input_file": str(input_path),
        "total_lines_scanned": total_lines,
        "deduplicated_lines": dedup_lines,
        "duplicates_removed": dedup_stats["duplicates_removed"],
        "unique_event_templates": parser.get_template_count(),
        "processing_time_sec": round(elapsed, 2),
        "lines_per_second": round(dedup_lines / elapsed, 1) if elapsed > 0 else 0,
        "templates": [
            {
                "EventId": f"E{t['cluster_id']}",
                "template": t["template"],
                "occurrences": t["size"],
            }
            for t in sorted(templates, key=lambda x: x["cluster_id"])
        ],
    }
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"  JSON:   {json_path}")

    # ── Done-when-check output ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  MODULE 1 — DONE-WHEN CHECKS")
    print("=" * 70)

    print(f"\n📁 Outputs:")
    print(f"   {csv_path}")
    print(f"   {pkl_path}")
    print(f"   {json_path}")

    print(f"\n📊 Statistics:")
    print(f"   Total lines scanned:     {total_lines:>12,}")
    print(f"   After dedup:             {dedup_lines:>12,}")
    print(f"   Duplicates removed:      {dedup_stats['duplicates_removed']:>12,}")
    dedup_pct = (dedup_stats['duplicates_removed'] / total_lines * 100) if total_lines else 0
    print(f"   Dedup reduction:         {dedup_pct:>11.2f}%")
    print(f"   Processing time:         {elapsed:>11.1f}s")
    rate = dedup_lines / elapsed if elapsed > 0 else 0
    print(f"   Throughput:              {rate:>10,.0f} lines/sec")

    print(f"\n✅ nunique() on EventId:     {n_unique_events}")
    target_range = "25–30"
    if 20 <= n_unique_events <= 50:
        print(f"   → PASS (target: ~{target_range}, got {n_unique_events} — within acceptable range)")
    else:
        print(f"   → NOTE (target: ~{target_range}, got {n_unique_events} — review template granularity)")

    print(f"\n🔍 All discovered templates ({parser.get_template_count()}):")
    for t in sorted(templates, key=lambda x: x["cluster_id"]):
        print(f"   E{t['cluster_id']:>3}  (n={t['size']:>8,})  {t['template']}")

    print(f"\n🔎 10 spot-checked lines:")
    for i, row in enumerate(spot_check_rows[:10]):
        print(f"   [{i+1:>2}] Line {row.get('LineId', '?'):>8} | "
              f"{row.get('EventId', '?'):>5} | "
              f"{row.get('Level', '?'):<5} | "
              f"{row.get('Content', '')[:80]}")
        print(f"        → Template: {row.get('EventTemplate', '')[:80]}")

    print("\n" + "=" * 70)

    return {
        "csv_path": str(csv_path),
        "pkl_path": str(pkl_path),
        "json_path": str(json_path),
        "total_lines": total_lines,
        "dedup_lines": dedup_lines,
        "duplicates_removed": dedup_stats["duplicates_removed"],
        "unique_event_ids": n_unique_events,
        "template_count": parser.get_template_count(),
        "processing_time": round(elapsed, 2),
        "templates": templates,
    }


# ── CLI ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Module 1: Log Ingestion & Drain Parsing → structured CSV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/module1_ingest_parse.py data/raw/HDFS.log
  python src/module1_ingest_parse.py data/raw/HDFS_sample_1pct.log
  python src/module1_ingest_parse.py data/raw/sample_hdfs.log --max-lines 20
        """,
    )
    ap.add_argument("input_file", help="Path to raw HDFS log file")
    ap.add_argument("--max-lines", type=int, default=None,
                    help="Limit to N lines (for dev/testing)")
    ap.add_argument("--output-csv", default=None,
                    help="Custom output CSV path")
    ap.add_argument("--output-pkl", default=None,
                    help="Custom Drain state pickle path")
    ap.add_argument("--skip-dedup", action="store_true",
                    help="Skip deduplication (already deduped input)")
    args = ap.parse_args()

    result = run_module1(
        input_path=args.input_file,
        max_lines=args.max_lines,
        output_csv=args.output_csv,
        output_pkl=args.output_pkl,
        skip_dedup=args.skip_dedup,
    )

