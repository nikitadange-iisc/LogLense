"""
Stage 1: Streaming Ingestion & Deduplication

Reads log files line-by-line in a streaming fashion (no full in-memory load),
removes duplicate consecutive log lines, and outputs a cleaned log stream.
"""

import os
import logging
import argparse
from pathlib import Path

logger = logging.getLogger(__name__)


def read_log_stream(filepath: str):
    """
    Generator-based file reader that yields lines one at a time.
    Handles large files without loading them entirely into memory.

    Args:
        filepath: Path to the log file.

    Yields:
        Stripped log lines.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Log file not found: {filepath}")

    logger.info(f"Opening log file: {filepath}")
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            stripped = line.rstrip("\n\r")
            if stripped:  # skip blank lines
                yield stripped


def stream_deduplicated(filepath: str):
    """
    Generator that yields (line_number, line) tuples of deduplicated lines.
    Removes consecutive duplicate lines. Suitable for downstream pipeline
    consumption without writing to disk.

    Args:
        filepath: Path to the log file.

    Yields:
        Tuple of (original_line_number, line_content).
    """
    prev_line = None
    total = 0
    deduped = 0
    duplicates = 0

    for line in read_log_stream(filepath):
        total += 1
        if line == prev_line:
            duplicates += 1
            continue
        prev_line = line
        deduped += 1
        yield (total, line)

    logger.info(
        f"Stream dedup stats — total: {total}, "
        f"deduplicated: {deduped}, duplicates removed: {duplicates}"
    )


def deduplicate_stream(input_path: str, output_path: str = None) -> dict:
    """
    Process a log file and write deduplicated lines to an output file.

    Args:
        input_path: Path to the raw log file.
        output_path: Path for the deduplicated output. If None, auto-generated
                     in a 'processed' sibling directory.

    Returns:
        dict with keys: total_lines, deduplicated_lines, duplicates_removed,
                        output_path.
    """
    input_path = Path(input_path)

    if output_path is None:
        processed_dir = input_path.parent.parent / "processed"
        processed_dir.mkdir(parents=True, exist_ok=True)
        output_path = processed_dir / f"{input_path.stem}_dedup{input_path.suffix}"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    prev_line = None
    total = 0
    deduped = 0
    duplicates = 0

    logger.info(f"Deduplicating: {input_path} -> {output_path}")

    with open(output_path, "w", encoding="utf-8") as out_f:
        for line in read_log_stream(str(input_path)):
            total += 1
            if line == prev_line:
                duplicates += 1
                continue
            prev_line = line
            deduped += 1
            out_f.write(line + "\n")

            if total % 1_000_000 == 0:
                logger.info(f"  Processed {total:,} lines...")

    stats = {
        "total_lines": total,
        "deduplicated_lines": deduped,
        "duplicates_removed": duplicates,
        "output_path": str(output_path),
    }

    logger.info(
        f"Deduplication complete — total: {total:,}, "
        f"kept: {deduped:,}, removed: {duplicates:,} "
        f"({(duplicates / total * 100) if total else 0:.1f}% reduction)"
    )

    return stats


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="Stage 1: Log Ingestion & Deduplication")
    parser.add_argument("input_file", help="Path to raw log file")
    parser.add_argument("-o", "--output", help="Output path (optional)", default=None)
    args = parser.parse_args()

    stats = deduplicate_stream(args.input_file, args.output)
    print(f"\nDeduplication Results:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

