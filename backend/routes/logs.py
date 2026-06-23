"""
Paginated raw log viewer.

GET /api/logs?page=0
  Returns lines from the original uploaded file (with full timestamps).
  Falls back to the structured CSV Content column when no raw file exists.
"""
import csv
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.concurrency import run_in_threadpool

from state import app_state

logger = logging.getLogger(__name__)
router = APIRouter()

PER_PAGE = 200
_RAW_SUFFIXES = {".log", ".txt"}


def _count_lines_sync(path: Path) -> int:
    """Fast newline-byte count — runs in threadpool."""
    count = 0
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            count += chunk.count(b"\n")
    return max(count, 1)


def _read_page_sync(path: Path, use_raw: bool, offset_start: int, offset_end: int):
    """
    Read one page of lines.  Stops after offset_end+1 matches — never
    scans the whole file.  Runs in a threadpool.
    """
    lines = []
    idx   = 0

    if use_raw:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for raw_line in f:
                stripped = raw_line.rstrip("\n\r").replace("\x00", "")
                if not stripped:
                    continue
                if idx >= offset_end:
                    return lines, True
                if idx >= offset_start:
                    lines.append({"line_number": idx + 1, "content": stripped})
                idx += 1
    else:
        with open(path, newline="", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if idx >= offset_end:
                    return lines, True
                if idx >= offset_start:
                    lines.append({
                        "line_number": idx + 1,
                        "content":     row.get("Content", ""),
                    })
                idx += 1

    return lines, False


@router.get("/logs")
async def get_logs(page: int = Query(0, ge=0)):
    # Pick the best available source
    raw_path = Path(app_state.raw_log_path) if app_state.raw_log_path else None
    csv_path = Path(app_state.csv_path)     if app_state.csv_path     else None

    use_raw   = bool(raw_path and raw_path.exists() and raw_path.suffix.lower() in _RAW_SUFFIXES)
    read_path = raw_path if use_raw else csv_path

    if not read_path or not read_path.exists():
        raise HTTPException(status_code=400, detail="No log file loaded.")

    # Count total lines lazily and cache so subsequent pages are instant
    if app_state.total_log_lines == 0:
        total = await run_in_threadpool(_count_lines_sync, read_path)
        app_state.total_log_lines = total

    offset_start = page * PER_PAGE
    offset_end   = offset_start + PER_PAGE

    try:
        lines, has_next = await run_in_threadpool(
            _read_page_sync, read_path, use_raw, offset_start, offset_end
        )
    except Exception as exc:
        logger.error("Log read failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        "lines":       lines,
        "page":        page,
        "per_page":    PER_PAGE,
        "has_next":    has_next,
        "total_lines": app_state.total_log_lines,
    }
