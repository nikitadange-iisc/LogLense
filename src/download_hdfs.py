"""
Download the HDFS_v1 log dataset from Zenodo (LogHub mirror).

The full HDFS.log is ~1.55 GB (11.17 M lines).  This script downloads it
once into data/raw/ and optionally creates a 1% random sample for fast
development iterations.

Usage:
    python src/download_hdfs.py                  # download full + 1% sample
    python src/download_hdfs.py --sample-pct 5   # download full + 5% sample
    python src/download_hdfs.py --sample-only     # just re-sample (no download)
"""

import os
import sys
import random
import logging
import argparse
import urllib.request
import zipfile
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"

# Zenodo link for HDFS_v1 (maintained by LogPai / LogHub)
HDFS_ZENODO_URL = "https://zenodo.org/records/8196385/files/HDFS_v1.zip?download=1"
ZIP_PATH = RAW_DIR / "HDFS_v1.zip"
HDFS_LOG_PATH = RAW_DIR / "HDFS.log"
HDFS_LABELS_PATH = RAW_DIR / "anomaly_label.csv"


def download_hdfs(force: bool = False):
    """Download the HDFS_v1.zip from Zenodo and extract HDFS.log."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if HDFS_LOG_PATH.exists() and not force:
        size_mb = HDFS_LOG_PATH.stat().st_size / (1024 * 1024)
        logger.info(f"HDFS.log already exists ({size_mb:.0f} MB) — skipping download. Use --force to re-download.")
        return

    logger.info(f"Downloading HDFS_v1.zip from Zenodo … (this may take a few minutes)")
    logger.info(f"URL: {HDFS_ZENODO_URL}")

    def _progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        pct = downloaded / total_size * 100 if total_size > 0 else 0
        mb = downloaded / (1024 * 1024)
        sys.stdout.write(f"\r  Downloaded {mb:.1f} MB ({pct:.1f}%)")
        sys.stdout.flush()

    urllib.request.urlretrieve(HDFS_ZENODO_URL, str(ZIP_PATH), reporthook=_progress)
    print()  # newline after progress bar
    logger.info(f"Download complete: {ZIP_PATH}")

    # Extract
    logger.info("Extracting HDFS.log from zip …")
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        # List contents to find the log file
        names = zf.namelist()
        logger.info(f"Zip contents: {names}")
        for name in names:
            if name.endswith("HDFS.log"):
                # Extract to data/raw/
                with zf.open(name) as src, open(HDFS_LOG_PATH, "wb") as dst:
                    import shutil
                    shutil.copyfileobj(src, dst)
                logger.info(f"Extracted: {HDFS_LOG_PATH}")
            elif "anomaly_label" in name.lower() or name.endswith(".csv"):
                with zf.open(name) as src, open(RAW_DIR / Path(name).name, "wb") as dst:
                    import shutil
                    shutil.copyfileobj(src, dst)
                logger.info(f"Extracted: {RAW_DIR / Path(name).name}")

    # Optionally remove zip to save space
    # ZIP_PATH.unlink()

    size_mb = HDFS_LOG_PATH.stat().st_size / (1024 * 1024)
    logger.info(f"HDFS.log ready: {size_mb:.0f} MB")


def count_lines(filepath: Path) -> int:
    """Fast line count without loading whole file."""
    count = 0
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        for _ in f:
            count += 1
    return count


def create_sample(sample_pct: float = 1.0, seed: int = 42):
    """
    Create a random sample of HDFS.log for development.

    We use reservoir-style sampling — read line by line, keep each line
    with probability sample_pct/100.  This preserves the original line
    distribution without loading the whole file.
    """
    if not HDFS_LOG_PATH.exists():
        logger.error(f"HDFS.log not found at {HDFS_LOG_PATH}. Run download first.")
        return

    sample_path = RAW_DIR / f"HDFS_sample_{sample_pct:.0f}pct.log"

    random.seed(seed)
    threshold = sample_pct / 100.0
    total = 0
    kept = 0

    logger.info(f"Creating {sample_pct}% sample of HDFS.log → {sample_path}")
    with open(HDFS_LOG_PATH, "r", encoding="utf-8", errors="replace") as fin, \
         open(sample_path, "w", encoding="utf-8") as fout:
        for line in fin:
            total += 1
            if random.random() < threshold:
                fout.write(line)
                kept += 1
            if total % 1_000_000 == 0:
                logger.info(f"  Scanned {total:,} lines, kept {kept:,}")

    logger.info(
        f"Sample complete — {kept:,} / {total:,} lines "
        f"({kept/total*100:.2f}%) → {sample_path}"
    )
    return sample_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Download HDFS_v1 dataset from Zenodo")
    ap.add_argument("--force", action="store_true", help="Force re-download")
    ap.add_argument("--sample-pct", type=float, default=1.0, help="Sample percentage (default: 1%%)")
    ap.add_argument("--sample-only", action="store_true", help="Skip download, only create sample")
    ap.add_argument("--count", action="store_true", help="Just count lines in HDFS.log")
    args = ap.parse_args()

    if args.count:
        n = count_lines(HDFS_LOG_PATH)
        print(f"HDFS.log has {n:,} lines")
        sys.exit(0)

    if not args.sample_only:
        download_hdfs(force=args.force)

    create_sample(sample_pct=args.sample_pct)

