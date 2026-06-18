"""
Stage 3: Session Grouping & Vectorization

Groups parsed log events into sessions:
  - HDFS: group by Block ID
  - BGL/Thunderbird: group by sliding window of N events
Represents each session as a fixed-length event count vector.
"""

import re
import logging
import argparse
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Session:
    """Represents a grouped log session."""
    session_id: str
    events: list = field(default_factory=list)
    line_range: tuple = (0, 0)
    raw_lines: list = field(default_factory=list)
    vector: Optional[np.ndarray] = None
    label: Optional[str] = None  # Ground-truth label: "Normal" or "Anomaly"


class Sessionizer:
    """Groups parsed log events into sessions and vectorizes them."""

    def __init__(self, method: str = "block_id", window_size: int = 50,
                 step_size: int = 25):
        """
        Args:
            method: "block_id" for HDFS, "sliding_window" for BGL/Thunderbird.
            window_size: Number of events per sliding window.
            step_size: Step size for sliding window.
        """
        self.method = method
        self.window_size = window_size
        self.step_size = step_size
        logger.info(f"Sessionizer initialized (method={method})")

    def extract_block_id(self, parsed_event: dict) -> str:
        """
        Extract HDFS block ID from a parsed event.

        Args:
            parsed_event: Dict from parser stage with extracted_variables and raw_line.

        Returns:
            Block ID string or None if not found.
        """
        # Check extracted variables first
        for var in parsed_event.get("extracted_variables", []):
            if isinstance(var, str) and var.startswith("blk_"):
                return var

        # Fallback: regex on raw line
        match = re.search(r"(blk_-?\d+)", parsed_event.get("raw_line", ""))
        if match:
            return match.group(1)

        return None

    def group_by_block_id(self, parsed_events) -> dict:
        """
        Group parsed events by HDFS Block ID.

        Args:
            parsed_events: Iterable of parsed event dicts.

        Returns:
            Dict mapping block_id -> list of parsed events.
        """
        groups = defaultdict(list)
        no_block_count = 0

        for event in parsed_events:
            block_id = self.extract_block_id(event)
            if block_id:
                groups[block_id].append(event)
            else:
                no_block_count += 1

        if no_block_count:
            logger.warning(f"{no_block_count} events had no block ID and were skipped")

        logger.info(f"Grouped events into {len(groups)} block-ID sessions")
        return dict(groups)

    def group_by_sliding_window(self, parsed_events) -> list:
        """
        Group parsed events using a sliding window approach.

        Args:
            parsed_events: Iterable of parsed event dicts.

        Returns:
            List of session dicts with session_id, events, line_range.
        """
        # Materialize the events for windowing
        all_events = list(parsed_events)
        sessions = []

        for i in range(0, max(1, len(all_events) - self.window_size + 1), self.step_size):
            window = all_events[i:i + self.window_size]
            if not window:
                continue

            line_numbers = [e.get("line_number", 0) for e in window if e.get("line_number")]
            line_range = (min(line_numbers) if line_numbers else i,
                          max(line_numbers) if line_numbers else i + len(window))

            sessions.append({
                "session_id": f"window_{i}_{i + len(window)}",
                "events": window,
                "line_range": line_range,
            })

        logger.info(f"Created {len(sessions)} sliding-window sessions "
                     f"(window={self.window_size}, step={self.step_size})")
        return sessions

    def build_template_vocabulary(self, sessions: list) -> dict:
        """
        Build a global vocabulary of unique event template IDs.

        Args:
            sessions: List of Session objects.

        Returns:
            Dict mapping template_id -> vector index.
        """
        template_ids = set()
        for session in sessions:
            for event in session.events:
                template_ids.add(event["event_template_id"])

        vocabulary = {tid: idx for idx, tid in enumerate(sorted(template_ids))}
        logger.info(f"Built template vocabulary with {len(vocabulary)} unique templates")
        return vocabulary

    def vectorize_session(self, session_events: list, vocabulary: dict) -> np.ndarray:
        """
        Convert a session's events into a fixed-length count vector.

        Args:
            session_events: List of parsed event dicts.
            vocabulary: Dict mapping template_id -> vector index.

        Returns:
            Numpy array of event counts.
        """
        vector = np.zeros(len(vocabulary), dtype=np.float32)
        for event in session_events:
            tid = event["event_template_id"]
            if tid in vocabulary:
                vector[vocabulary[tid]] += 1
        return vector

    def create_sessions(self, parsed_events) -> list:
        """
        Create Session objects from parsed events using the configured method.

        Args:
            parsed_events: Iterable of parsed event dicts.

        Returns:
            List of Session objects.
        """
        sessions = []

        if self.method == "block_id":
            groups = self.group_by_block_id(parsed_events)
            for block_id, events in groups.items():
                line_numbers = [e.get("line_number", 0) for e in events if e.get("line_number")]
                line_range = (min(line_numbers) if line_numbers else 0,
                              max(line_numbers) if line_numbers else 0)
                raw_lines = [e["raw_line"] for e in events]

                sessions.append(Session(
                    session_id=block_id,
                    events=events,
                    line_range=line_range,
                    raw_lines=raw_lines,
                ))
        elif self.method == "sliding_window":
            window_groups = self.group_by_sliding_window(parsed_events)
            for wg in window_groups:
                raw_lines = [e["raw_line"] for e in wg["events"]]
                sessions.append(Session(
                    session_id=wg["session_id"],
                    events=wg["events"],
                    line_range=wg["line_range"],
                    raw_lines=raw_lines,
                ))
        else:
            raise ValueError(f"Unknown sessionization method: {self.method}")

        logger.info(f"Created {len(sessions)} sessions using '{self.method}' method")
        return sessions

    def vectorize_all(self, sessions: list, vocabulary: dict = None):
        """
        Vectorize all sessions.

        Args:
            sessions: List of Session objects.
            vocabulary: Optional pre-built vocabulary. If None, built from sessions.

        Returns:
            Tuple of (sessions_with_vectors, vocabulary).
        """
        if vocabulary is None:
            vocabulary = self.build_template_vocabulary(sessions)

        for session in sessions:
            session.vector = self.vectorize_session(session.events, vocabulary)

        logger.info(f"Vectorized {len(sessions)} sessions (vector dim={len(vocabulary)})")
        return sessions, vocabulary

    def load_labels(self, label_path: str, sessions: list) -> list:
        """
        Load ground-truth anomaly labels and attach to sessions.
        Supports HDFS anomaly_label.csv format (BlockId, Label).

        Args:
            label_path: Path to label file.
            sessions: List of Session objects.

        Returns:
            Updated sessions with labels.
        """
        labels = {}
        with open(label_path, "r") as f:
            header = f.readline()  # Skip header
            for line in f:
                parts = line.strip().split(",")
                if len(parts) >= 2:
                    block_id = parts[0].strip()
                    label = parts[1].strip()
                    labels[block_id] = label

        labeled_count = 0
        for session in sessions:
            if session.session_id in labels:
                session.label = labels[session.session_id]
                labeled_count += 1

        logger.info(f"Loaded labels for {labeled_count}/{len(sessions)} sessions "
                     f"({len(labels)} labels in file)")
        return sessions


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    arg_parser = argparse.ArgumentParser(description="Stage 3: Session Grouping & Vectorization")
    arg_parser.add_argument("input_file", help="Path to log file")
    arg_parser.add_argument("-m", "--method", default="block_id",
                            choices=["block_id", "sliding_window"])
    arg_parser.add_argument("-n", "--max-lines", type=int, default=1000)
    args = arg_parser.parse_args()

    from ingestion import stream_deduplicated
    from log_parser import LogParser

    parser = LogParser()
    parsed = list(parser.parse_stream(stream_deduplicated(args.input_file)))
    if args.max_lines:
        parsed = parsed[:args.max_lines]

    sessionizer = Sessionizer(method=args.method)
    sessions = sessionizer.create_sessions(iter(parsed))
    sessions, vocab = sessionizer.vectorize_all(sessions)

    print(f"\nSessions created: {len(sessions)}")
    print(f"Vocabulary size: {len(vocab)}")
    if sessions:
        print(f"Sample session: {sessions[0].session_id}, "
              f"vector shape: {sessions[0].vector.shape}, "
              f"lines: {len(sessions[0].raw_lines)}")

