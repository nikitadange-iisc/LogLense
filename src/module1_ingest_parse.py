"""
Module 1 - Log Ingestion & Drain Parsing
=========================================

This script is the first step of the LogSense project. It takes a raw HDFS
log file and turns it into a clean, structured table that the later modules
(session grouping, anomaly detection, ...) can use.

Everything runs in ONE streaming pass over the file, so it also works on the
full 11-million-line HDFS log without running out of memory.

Steps:
    1. Read the log file line by line (we never load the whole file at once).
    2. Drop a line if it is exactly the same as the line right before it
       (consecutive-duplicate removal).
    3. Parse each line with Drain3. Drain finds the repeating "template" of
       the line and pulls out the variable bits (block IDs, IPs, numbers).
    4. Write a structured CSV and save the learned Drain templates so we
       don't have to learn them again next time.

Files produced:
    data/processed/<name>_structured.csv     one row per log line
    models/drain_state/drain_templates.pkl    the Drain model (re-loadable)
    models/drain_state/drain_templates.json   the template list (easy to read)

How to run:
    python src/module1_ingest_parse.py data/raw/sample_hdfs.log
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

# Make sure we can import the other Module 1 files (ingestion, log_parser)
# no matter which folder we launch this script from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ingestion import read_log_stream
from log_parser import LogParser

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Where the outputs go.
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models" / "drain_state"

# Columns of the structured CSV (same names LogHub / LogPai use).
CSV_COLUMNS = [
    "LineId", "Date", "Time", "Pid", "Level", "Component",
    "Content", "EventId", "EventTemplate", "ParameterList",
]


def run_module1(
    input_path: str,
    max_lines: int = None,
    output_csv: str = None,
    output_pkl: str = None,
    skip_dedup: bool = False,
):
    """
    Run all of Module 1 on a single log file.

    Args:
        input_path: Path to the raw HDFS log file.
        max_lines:  Stop after this many kept lines (handy for quick tests).
        output_csv: Where to write the structured CSV (auto-named if None).
        output_pkl: Where to save the Drain model (auto-named if None).
        skip_dedup: Set True if the file is already deduplicated.

    Returns:
        A small dict with the output paths and summary numbers.

    Note:
        We do dedup + parsing + writing all in one pass while reading the
        file, so the memory use stays tiny even for very large logs.
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    stem = input_path.stem  # e.g. "HDFS" or "sample_hdfs"
    csv_path = Path(output_csv) if output_csv else OUTPUT_DIR / f"{stem}_structured.csv"
    pkl_path = Path(output_pkl) if output_pkl else MODEL_DIR / "drain_templates.pkl"
    json_path = pkl_path.with_suffix(".json")

    size_mb = input_path.stat().st_size / (1024 * 1024)
    logger.info("Module 1: Log Ingestion & Drain Parsing")
    logger.info("Input : %s (%.1f MB)", input_path, size_mb)
    logger.info("Output: %s", csv_path)
    if max_lines:
        logger.info("Limit : %s lines", f"{max_lines:,}")

    # The Drain parser learns the templates as we feed it lines.
    parser = LogParser(dataset="hdfs", persist_state=True, state_dir=str(MODEL_DIR))

    # Counters we update while streaming through the file.
    total_lines = 0          # every line we read
    kept_lines = 0           # lines left after removing duplicates
    duplicates = 0           # consecutive duplicates we dropped
    prev_line = None
    event_id_counts = Counter()   # how many lines landed in each EventId

    # We keep a few example rows so we can eyeball that the templates look
    # right. The first 5 lines, plus 5 random lines from the rest chosen
    # with "reservoir sampling" (a simple way to pick random items in a
    # single pass without storing the whole file).
    head_rows = []
    reservoir = []
    seen_after_head = 0
    random.seed(42)

    t0 = time.time()
    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=CSV_COLUMNS)
        writer.writeheader()

        for raw_line in read_log_stream(str(input_path)):
            total_lines += 1

            # Step 1: skip a line that is identical to the previous one.
            if not skip_dedup and raw_line == prev_line:
                duplicates += 1
                continue
            prev_line = raw_line
            kept_lines += 1

            # Step 2: parse the line with Drain. parse_line() also gives us
            # back the header fields (date/time/pid/content), so we don't
            # have to parse the header a second time just to fill the CSV.
            parsed = parser.parse_line(raw_line, line_number=kept_lines)
            event_id = f"E{parsed['event_template_id']}"
            event_id_counts[event_id] += 1

            # Step 3: write one structured row to the CSV.
            row = {
                "LineId": kept_lines,
                "Date": parsed.get("date", ""),
                "Time": parsed.get("time", ""),
                "Pid": parsed.get("pid", ""),
                "Level": parsed["level"],
                "Component": parsed["component"],
                "Content": parsed.get("content", raw_line),
                "EventId": event_id,
                "EventTemplate": parsed["event_template"],
                "ParameterList": str(parsed["extracted_variables"]),
            }
            writer.writerow(row)

            # Step 4: remember some rows for the spot-check at the end.
            if kept_lines <= 5:
                head_rows.append(row)
            else:
                seen_after_head += 1
                if len(reservoir) < 5:
                    reservoir.append(row)
                else:
                    j = random.randint(0, seen_after_head - 1)
                    if j < 5:
                        reservoir[j] = row

            # Print progress now and then on big files.
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
    n_unique_events = len(event_id_counts)   # number of distinct EventIds

    # ---- Save the Drain model so later runs can reuse the templates ----
    logger.info("Saving Drain state...")
    with open(pkl_path, "wb") as f:
        pickle.dump(parser.template_miner, f)

    templates = parser.get_templates()
    summary = {
        "dataset": "hdfs",
        "input_file": str(input_path),
        "total_lines_scanned": total_lines,
        "deduplicated_lines": kept_lines,
        "duplicates_removed": duplicates,
        "unique_event_templates": parser.get_template_count(),
        "processing_time_sec": round(elapsed, 2),
        "lines_per_second": round(kept_lines / elapsed, 1) if elapsed > 0 else 0,
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
    logger.info("Saved: %s, %s", pkl_path.name, json_path.name)

    # ---- Print the "done-when" checks so we can verify by eye ----
    dedup_pct = (duplicates / total_lines * 100) if total_lines else 0
    rate = kept_lines / elapsed if elapsed > 0 else 0

    print()
    print("=" * 70)
    print("MODULE 1 - DONE-WHEN CHECKS")
    print("=" * 70)

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

    print(f"\nUnique EventIds: {n_unique_events}  (target ~25-30)")
    if 20 <= n_unique_events <= 50:
        print("  -> PASS (count is small and stable)")
    else:
        print("  -> CHECK (template granularity may need tuning)")

    print(f"\nAll {parser.get_template_count()} templates:")
    for t in sorted(templates, key=lambda x: x["cluster_id"]):
        print(f"  E{t['cluster_id']:<3} (n={t['size']:>8,})  {t['template']}")

    print("\n10 spot-checked lines (raw line -> template):")
    for i, row in enumerate(spot_check_rows[:10], start=1):
        content = str(row.get("Content", ""))[:80]
        template = str(row.get("EventTemplate", ""))[:80]
        print(f"  [{i:>2}] line {row.get('LineId', '?'):>8}  "
              f"{row.get('EventId', '?'):>4}  {row.get('Level', '?'):<5}  {content}")
        print(f"        template: {template}")

    print("=" * 70)

    return {
        "csv_path": str(csv_path),
        "pkl_path": str(pkl_path),
        "json_path": str(json_path),
        "total_lines": total_lines,
        "dedup_lines": kept_lines,
        "duplicates_removed": duplicates,
        "unique_event_ids": n_unique_events,
        "template_count": parser.get_template_count(),
        "processing_time": round(elapsed, 2),
        "templates": templates,
    }


def _build_arg_parser():
    """Set up the command-line options."""
    ap = argparse.ArgumentParser(
        description="Module 1: Log Ingestion & Drain Parsing -> structured CSV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/module1_ingest_parse.py data/raw/sample_hdfs.log
  python src/module1_ingest_parse.py data/raw/HDFS_sample_1pct.log
  python src/module1_ingest_parse.py data/raw/HDFS.log --max-lines 100000
        """,
    )
    ap.add_argument("input_file", help="Path to the raw HDFS log file")
    ap.add_argument("--max-lines", type=int, default=None,
                    help="Only process the first N (kept) lines")
    ap.add_argument("--output-csv", default=None, help="Custom CSV output path")
    ap.add_argument("--output-pkl", default=None, help="Custom Drain pickle path")
    ap.add_argument("--skip-dedup", action="store_true",
                    help="Skip dedup (input is already deduplicated)")
    return ap


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    run_module1(
        input_path=args.input_file,
        max_lines=args.max_lines,
        output_csv=args.output_csv,
        output_pkl=args.output_pkl,
        skip_dedup=args.skip_dedup,
    )
