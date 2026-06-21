"""
Module 2a - Session Grouping & Vectorization
=============================================

Groups parsed log events (read from Module 1 CSV) into sessions:
  - HDFS        : group by Block ID extracted from ParameterList / Content
  - BGL         : group by Node column (concentrates fault bursts per node)
  - Thunderbird : sliding window of window_size events, step step_size

Represents each session as a fixed-length event-count vector over the
global template vocabulary, then optionally attaches ground-truth labels.

Label strategies:
  - HDFS        : load from anomaly_label.csv (BlockId, Label)
  - BGL / TB    : derive from the 'Label' column already in the Module 1 CSV
                  (any event with label != '-' marks the session as Anomaly)
"""

import ast
import csv
import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class Session:
    """Represents a grouped log session."""
    session_id: str
    events: list = field(default_factory=list)
    line_range: tuple = (0, 0)
    raw_lines: list = field(default_factory=list)
    vector: Optional[np.ndarray] = None
    label: Optional[str] = None   # "Normal" | "Anomaly" | None


# ---------------------------------------------------------------------------
# CSV → event-dict loader (Module 1 output → Module 2 input)
# ---------------------------------------------------------------------------

def load_events_from_csv(csv_path: str, dataset: str) -> list:
    """
    Read a Module 1 structured CSV and return a list of event dicts
    compatible with Sessionizer methods.

    Each event dict contains:
        event_template_id : int   (parsed from "E5" -> 5)
        event_template    : str
        extracted_variables: list (ast.literal_eval of ParameterList)
        line_number       : int
        content           : str   (the log message, used as raw_line proxy)
        raw_line          : str   (alias of content, for embed / block-ID fallback)
        level             : str
        label             : str   (BGL / Thunderbird only; '-' for normal)
        ... other dataset-specific columns lowercased
    """
    events = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            event_id_str = row.get("EventId", "E0")
            try:
                template_id = int(event_id_str.lstrip("E"))
            except ValueError:
                template_id = 0

            param_str = row.get("ParameterList", "[]")
            try:
                variables = ast.literal_eval(param_str)
                if not isinstance(variables, list):
                    variables = []
            except Exception:
                variables = []

            content = row.get("Content", "")
            event = {
                "event_template_id":  template_id,
                "event_template":     row.get("EventTemplate", ""),
                "extracted_variables": variables,
                "line_number":        int(row.get("LineId", 0)),
                "content":            content,
                "raw_line":           content,   # best proxy for embedding & block-ID regex
                "level":              row.get("Level", ""),
                # carry the anomaly label for BGL / Thunderbird
                "label":              row.get("Label", "-"),
            }

            # Pass through any extra dataset columns (lowercased)
            for col, val in row.items():
                key = col.lower()
                if key not in event:
                    event[key] = val

            events.append(event)

    logger.info("Loaded %d events from %s (dataset=%s)", len(events), csv_path, dataset)
    return events


# ---------------------------------------------------------------------------
# Sessionizer
# ---------------------------------------------------------------------------

class Sessionizer:
    """Groups parsed log events into sessions and vectorizes them."""

    def __init__(self, method: str = "block_id", window_size: int = 20,
                 step_size: int = 10, weighting: str = "count"):
        """
        Args:
            method      : "block_id" | "node" | "sliding_window"
            window_size : Events per sliding window (default 20 per paper).
            step_size   : Stride for sliding window (default 10 = 50% overlap).
            weighting   : "count" (raw event counts, default) or "tfidf"
                          (TF-IDF weights — upweights rare fault templates).
        """
        if weighting not in ("count", "tfidf"):
            raise ValueError(f"weighting must be 'count' or 'tfidf', got {weighting!r}")
        self.method      = method
        self.window_size = window_size
        self.step_size   = step_size
        self.weighting   = weighting
        logger.info("Sessionizer initialized (method=%s, window=%d, step=%d, weighting=%s)",
                    method, window_size, step_size, weighting)

    # ── Block-ID helpers (HDFS) ─────────────────────────────────────────

    def extract_block_id(self, event: dict) -> Optional[str]:
        """Extract HDFS block ID from extracted_variables or raw_line."""
        for var in event.get("extracted_variables", []):
            if isinstance(var, str) and var.startswith("blk_"):
                return var
        match = re.search(r"(blk_-?\d+)", event.get("raw_line", ""))
        if match:
            return match.group(1)
        return None

    def group_by_block_id(self, events) -> dict:
        groups = defaultdict(list)
        skipped = 0
        for event in events:
            bid = self.extract_block_id(event)
            if bid:
                groups[bid].append(event)
            else:
                skipped += 1
        if skipped:
            logger.warning("%d events had no block ID and were skipped", skipped)
        logger.info("Grouped events into %d block-ID sessions", len(groups))
        return dict(groups)

    # ── Node helper (BGL) ──────────────────────────────────────────────

    def group_by_node(self, events) -> dict:
        """Group events by the Node column — keeps all fault events for a
        node together so the anomaly signal is not diluted across windows."""
        groups = defaultdict(list)
        skipped = 0
        for event in events:
            node = event.get("node", "").strip()
            if not node:
                skipped += 1
                continue
            groups[node].append(event)
        if skipped:
            logger.warning("%d events had no Node field and were skipped", skipped)
        logger.info("Grouped events into %d node sessions", len(groups))
        return dict(groups)

    # ── Sliding-window helper (Thunderbird) ─────────────────────────────

    def group_by_sliding_window(self, events) -> list:
        all_events = list(events)
        sessions = []
        n = len(all_events)
        for i in range(0, max(1, n - self.window_size + 1), self.step_size):
            window = all_events[i: i + self.window_size]
            if not window:
                continue
            line_numbers = [e.get("line_number", 0) for e in window if e.get("line_number")]
            line_range = (
                min(line_numbers) if line_numbers else i,
                max(line_numbers) if line_numbers else i + len(window),
            )
            sessions.append({
                "session_id": f"window_{i}_{i + len(window)}",
                "events":     window,
                "line_range": line_range,
            })
        logger.info("Created %d sliding-window sessions (window=%d, step=%d)",
                    len(sessions), self.window_size, self.step_size)
        return sessions

    # ── Session creation ────────────────────────────────────────────────

    def create_sessions(self, events) -> list:
        """
        Create Session objects using the configured method.

        Args:
            events: Iterable of event dicts (from load_events_from_csv or parser).

        Returns:
            List of Session objects (no vectors yet).
        """
        sessions = []
        if self.method == "block_id":
            groups = self.group_by_block_id(events)
            for bid, evts in groups.items():
                line_nums = [e.get("line_number", 0) for e in evts if e.get("line_number")]
                sessions.append(Session(
                    session_id=bid,
                    events=evts,
                    line_range=(min(line_nums, default=0), max(line_nums, default=0)),
                    raw_lines=[e["raw_line"] for e in evts],
                ))
        elif self.method == "node":
            groups = self.group_by_node(events)
            for node, evts in groups.items():
                line_nums = [e.get("line_number", 0) for e in evts if e.get("line_number")]
                sessions.append(Session(
                    session_id=node,
                    events=evts,
                    line_range=(min(line_nums, default=0), max(line_nums, default=0)),
                    raw_lines=[e["raw_line"] for e in evts],
                ))
        elif self.method == "sliding_window":
            for wg in self.group_by_sliding_window(events):
                sessions.append(Session(
                    session_id=wg["session_id"],
                    events=wg["events"],
                    line_range=wg["line_range"],
                    raw_lines=[e["raw_line"] for e in wg["events"]],
                ))
        else:
            raise ValueError(f"Unknown sessionization method: {self.method}")

        logger.info("Created %d sessions via '%s'", len(sessions), self.method)
        return sessions

    # ── Vectorization ───────────────────────────────────────────────────

    def build_template_vocabulary(self, sessions: list) -> dict:
        """Build vocabulary: template_id (int) -> vector index."""
        template_ids = set()
        for s in sessions:
            for e in s.events:
                template_ids.add(e["event_template_id"])
        vocab = {tid: idx for idx, tid in enumerate(sorted(template_ids))}
        logger.info("Built vocabulary: %d unique templates", len(vocab))
        return vocab

    def build_idf(self, sessions: list, vocabulary: dict) -> np.ndarray:
        """
        Compute smooth IDF weights across all sessions.
        IDF(t) = log((1 + N) / (1 + df(t))) + 1  [sklearn smooth formula]
        Rare templates (fault events) get high weight; frequent ones (E1) get low weight.
        """
        N = len(sessions)
        df = np.zeros(len(vocabulary), dtype=np.float32)
        for s in sessions:
            seen_in_session = set()
            for e in s.events:
                idx = vocabulary.get(e["event_template_id"])
                if idx is not None and idx not in seen_in_session:
                    df[idx] += 1
                    seen_in_session.add(idx)
        idf = np.log((1.0 + N) / (1.0 + df)) + 1.0
        logger.info("IDF computed over %d sessions — min=%.3f max=%.3f",
                    N, float(idf.min()), float(idf.max()))
        return idf

    def vectorize_session(self, events: list, vocabulary: dict,
                          idf: np.ndarray = None) -> np.ndarray:
        """
        Build a vector for one session.
        - weighting='count' (idf=None): raw event counts.
        - weighting='tfidf' (idf provided): binary presence × IDF.
          Uses binary (0/1) instead of raw counts before multiplying by IDF.
          This removes session-length bias — a node with 4865 E1 events and a
          node with 1 E1 event produce the same E1 contribution. The key signal
          for short sessions (BGL avg 4 events/node) is WHICH templates appeared,
          not how many times, so binary representation is more discriminative.
        """
        vec = np.zeros(len(vocabulary), dtype=np.float32)
        for e in events:
            idx = vocabulary.get(e["event_template_id"])
            if idx is not None:
                vec[idx] += 1
        if idf is not None:
            # Binary presence × IDF: 1 if template appeared, 0 otherwise
            vec = (vec > 0).astype(np.float32) * idf
        return vec

    def vectorize_all(self, sessions: list, vocabulary: dict = None):
        """
        Vectorize all sessions using the configured weighting scheme.

        Returns:
            (sessions_with_vectors, vocabulary)
        """
        if vocabulary is None:
            vocabulary = self.build_template_vocabulary(sessions)

        idf = self.build_idf(sessions, vocabulary) if self.weighting == "tfidf" else None

        for s in sessions:
            s.vector = self.vectorize_session(s.events, vocabulary, idf=idf)
        logger.info("Vectorized %d sessions (dim=%d, weighting=%s)",
                    len(sessions), len(vocabulary), self.weighting)
        return sessions, vocabulary

    # ── Vocabulary persistence ──────────────────────────────────────────

    def save_vocabulary(self, vocabulary: dict, path: str,
                        idf: np.ndarray = None) -> None:
        """Save event-template vocabulary (and optional IDF weights) to JSON."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "event_id_to_index": {str(k): v for k, v in vocabulary.items()},
            "template_count":    len(vocabulary),
            "weighting":         self.weighting,
        }
        if idf is not None:
            payload["idf_weights"] = idf.tolist()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        logger.info("Vocabulary saved: %d templates -> %s", len(vocabulary), path)

    def load_vocabulary(self, path: str):
        """
        Load vocabulary from JSON.
        Returns:
            (vocabulary dict, idf np.ndarray or None)
        """
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        raw = data.get("event_id_to_index", data)
        vocab = {int(k): v for k, v in raw.items()}
        idf = np.array(data["idf_weights"], dtype=np.float32) if "idf_weights" in data else None
        logger.info("Vocabulary loaded: %d templates from %s (weighting=%s)",
                    len(vocab), path, data.get("weighting", "count"))
        return vocab, idf

    # ── Label assignment ────────────────────────────────────────────────

    def load_labels(self, label_path: str, sessions: list) -> list:
        """
        Attach ground-truth labels to HDFS sessions from anomaly_label.csv.
        Format: BlockId,Label  (header row, then one block per line).

        For BGL / Thunderbird use assign_labels_from_events() instead —
        the labels are already embedded in each event's 'label' field.
        """
        labels = {}
        with open(label_path, encoding="utf-8") as f:
            next(f)  # skip header
            for line in f:
                parts = line.strip().split(",")
                if len(parts) >= 2:
                    labels[parts[0].strip()] = parts[1].strip()

        matched = 0
        for s in sessions:
            if s.session_id in labels:
                s.label = labels[s.session_id]
                matched += 1

        logger.info("Loaded labels for %d/%d sessions (%d in file)",
                    matched, len(sessions), len(labels))
        return sessions

    def assign_labels_from_events(self, sessions: list) -> list:
        """
        Derive session labels from event-level 'label' fields (BGL / Thunderbird).
        A session is 'Anomaly' if ANY constituent event has label != '-'.
        """
        for s in sessions:
            event_labels = [e.get("label", "-") for e in s.events]
            s.label = "Anomaly" if any(lbl != "-" for lbl in event_labels) else "Normal"
        anomalous = sum(1 for s in sessions if s.label == "Anomaly")
        logger.info("Labels derived from events: %d Anomaly / %d Normal",
                    anomalous, len(sessions) - anomalous)
        return sessions
