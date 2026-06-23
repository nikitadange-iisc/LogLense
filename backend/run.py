"""
Start the LogSense backend.

Usage:
    cd backend
    python run.py
    python run.py --port 8080
    python run.py --no-reload
"""
import os
import sys
import argparse
from pathlib import Path

# Set BEFORE torch/transformers are imported — prevents OpenMP and tokenizer
# threads from blocking Ctrl+C on Windows.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

sys.path.insert(0, str(Path(__file__).parent))

import uvicorn


def main():
    ap = argparse.ArgumentParser(description="Run the LogSense backend")
    ap.add_argument("--host",     default="0.0.0.0")
    ap.add_argument("--port",     type=int, default=8000)
    ap.add_argument("--no-reload", action="store_true", help="Disable auto-reload")
    args = ap.parse_args()

    uvicorn.run(
        "main:app",
        host=args.host,
        port=args.port,
        reload=not args.no_reload,
        # Force-exit this many seconds after Ctrl+C even if background
        # threads (torch OpenMP, pipeline worker) haven't finished.
        timeout_graceful_shutdown=3,
    )


if __name__ == "__main__":
    main()
