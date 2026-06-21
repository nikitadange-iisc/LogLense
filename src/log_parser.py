"""
Stage 2: Log Parsing with Drain Algorithm

Parses each log line into an event template using the Drain3 library,
extracting dynamic variables (block IDs, IP addresses, timestamps, etc.).
Reduces millions of raw lines to a small set of unique event templates.

Improvements over v1:
  - Dataset-aware header preprocessing (HDFS, BGL, Thunderbird)
  - Proper variable extraction using Drain3 wildcard alignment
  - Log severity/level extraction as first-class signal
  - Full Drain3 TemplateMiner state persistence (pickle-based)
  - Configurable dataset profiles with tuned Drain parameters
"""

import os
import re
import logging
import argparse
import json
import pickle
from pathlib import Path
from typing import Optional

from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig
from drain3.masking import MaskingInstruction

logger = logging.getLogger(__name__)

# ── Dataset-specific header regex patterns ─────────────────────────────
# Each pattern captures structured header fields + free-text content,
# so Drain only parses the *content* portion — not timestamps, PIDs, etc.

HEADER_PATTERNS = {
    "hdfs": re.compile(
        r"^(?P<date>\d{6})\s+"
        r"(?P<time>\d{6})\s+"
        r"(?P<pid>\d+)\s+"
        r"(?P<level>\w+)\s+"
        r"(?P<component>\S+):\s+"
        r"(?P<content>.*)$"
    ),
    "bgl": re.compile(
        r"^(?P<label>-|\w+)\s+"
        r"(?P<timestamp>\d+)\s+"
        r"(?P<date>\d{4}\.\d{2}\.\d{2})\s+"
        r"(?P<node>\S+)\s+"
        r"(?P<time>\S+)\s+"
        r"(?P<noderepeat>\S+)\s+"   # reporting node — same ID as node, not a number
        r"(?P<type>\S+)\s+"
        r"(?P<component>\S+)\s+"
        r"(?P<level>\S+)\s+"
        r"(?P<content>.*)$"
    ),
    "thunderbird": re.compile(
        r"^(?P<label>-|\w+)\s+"
        r"(?P<id>\d+)\s+"
        r"(?P<date>\S+)\s+"
        r"(?P<admin>\S+)\s+"
        r"(?P<time>\S+)\s+"
        r"(?P<adminaddr>\S+)\s+"
        r"(?P<content>.*)$"
    ),
}

# Severity mapping for normalisation
SEVERITY_MAP = {
    "TRACE": 0, "DEBUG": 1, "INFO": 2, "NOTICE": 2,
    "WARN": 3, "WARNING": 3,
    "ERROR": 4, "ERR": 4,
    "CRITICAL": 5, "CRIT": 5, "FATAL": 5, "EMERG": 5,
}

# ── Dataset-specific Drain hyper-parameter profiles ────────────────────
# sim_th  : similarity threshold — higher = stricter template merging
# depth   : parse-tree depth — controls prefix-based routing
# max_children : max child nodes per internal tree node
DRAIN_PROFILES = {
    "hdfs":        {"sim_th": 0.5, "depth": 4, "max_children": 100},
    "bgl":         {"sim_th": 0.5, "depth": 4, "max_children": 120},
    "thunderbird": {"sim_th": 0.4, "depth": 3, "max_children": 120},
    "default":     {"sim_th": 0.4, "depth": 4, "max_children": 100},
}


class LogParser:
    """
    A log parser built on top of Drain3.

    The idea is simple: every log line is made of two parts.
      - a fixed "template"  e.g. "Received block <*> of size <*>"
      - some variable bits  e.g. the block id, the size, an IP address

    Drain groups lines that share the same template together and gives each
    template an id. Before handing a line to Drain we do a bit of clean-up
    so the templates come out nice:

      1. Split off the header (date, time, pid, level, component) so Drain
         only sees the actual message. Otherwise timestamps and pids would
         end up inside the templates.
      2. Keep the log level (INFO / WARN / ERROR ...) and turn it into a
         small number, which later stages can use to weight anomalies.
      3. Pull out the variable bits by lining up the line against Drain's
         <*> wildcards.
      4. Save the whole Drain model with pickle so we don't have to learn
         the templates again next time.
      5. Use slightly different Drain settings per dataset (hdfs/bgl/...).
    """

    def __init__(self, dataset: str = "hdfs", config_path: str = None,
                 persist_state: bool = True,
                 state_dir: str = "models/drain_state"):
        """
        Args:
            dataset: One of "hdfs", "bgl", "thunderbird", or "default".
            config_path: Optional path to drain3 ini file (unused if None).
            persist_state: Whether to persist template miner state.
            state_dir: Directory for saving/loading state files.
        """
        self.dataset = dataset.lower()
        self.persist_state = persist_state
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)

        # Header regex for this dataset (None = use fallback)
        self.header_re = HEADER_PATTERNS.get(self.dataset)

        # ── Drain3 configuration ───────────────────────────────────────
        profile = DRAIN_PROFILES.get(self.dataset, DRAIN_PROFILES["default"])
        self.config = TemplateMinerConfig()
        self.config.drain_sim_th = profile["sim_th"]
        self.config.drain_depth = profile["depth"]
        self.config.drain_max_children = profile["max_children"]
        self.config.profiling_enabled = False

        # Masking instructions — applied *before* Drain clustering.
        # Order matters: more-specific patterns first to avoid partial hits.
        masking_instructions = [
            MaskingInstruction(r"blk_-?\d+", "BLOCK_ID"),
            MaskingInstruction(r"\d+\.\d+\.\d+\.\d+:\d+", "IP_PORT"),
            MaskingInstruction(r"\d+\.\d+\.\d+\.\d+", "IP_ADDR"),
            MaskingInstruction(r"\b0x[0-9a-fA-F]+\b", "HEX"),
            MaskingInstruction(r"(?<= )/[\w./-]+", "PATH"),
            MaskingInstruction(r"\b\d{6,}\b", "LONG_NUM"),
            MaskingInstruction(r"(?<=[ =:,(])\d{1,5}(?=[ ,.):\]]|$)", "NUM"),
        ]
        for mi in masking_instructions:
            self.config.masking_instructions.append(mi)

        self.template_miner = TemplateMiner(config=self.config)

        # Running statistics
        self._stats = {
            "total_lines": 0,
            "header_parsed": 0,
            "header_failed": 0,
        }

        logger.info(
            "LogParser initialised — dataset=%s  sim_th=%.2f  depth=%d  "
            "max_children=%d",
            self.dataset, profile["sim_th"], profile["depth"],
            profile["max_children"],
        )

    # ── Header preprocessing ───────────────────────────────────────────

    def _parse_header(self, line: str) -> dict:
        """
        Split a raw log line into structured header fields + content.

        Returns a dict that always contains at least ``content`` and
        ``level``.  If the dataset-specific regex matches, all named
        groups are included.
        """
        if self.header_re:
            m = self.header_re.match(line)
            if m:
                self._stats["header_parsed"] += 1
                groups = m.groupdict()
                groups["level"] = groups.get("level", "").upper()
                return groups

        # Fallback — try generic level extraction
        self._stats["header_failed"] += 1
        level_match = re.search(
            r"\b(TRACE|DEBUG|INFO|NOTICE|WARN(?:ING)?|ERR(?:OR)?|"
            r"CRIT(?:ICAL)?|FATAL|EMERG)\b",
            line, re.IGNORECASE,
        )
        return {
            "content": line,
            "level": level_match.group(0).upper() if level_match else "UNKNOWN",
        }

    # ── Core parsing ───────────────────────────────────────────────────

    def parse_line(self, line: str, line_number: int = None) -> dict:
        """
        Parse a single log line into an event template.

        Pipeline per line:
          1. Strip structured header -> extract level, component, content.
          2. Feed *content only* into Drain3 for template mining.
          3. Extract variables via wildcard (<*>) alignment.

        Returns:
            A dict describing the line. The main keys are:
              event_template_id   - the template number Drain gave the line
              event_template      - the template text, e.g. "Received <*>"
              extracted_variables - the variable bits taken out of the line
              raw_line, line_number - the original line and its position
              level, severity_score - log level (INFO/WARN/...) and a number
              component           - the logging component (if present)
              date, time, pid, content - the other header fields, passed
                  through so callers don't have to parse the header again
        """
        self._stats["total_lines"] += 1

        header = self._parse_header(line)
        content = header.get("content", line)

        # Drain3 mines only the free-text content
        result = self.template_miner.add_log_message(content)
        cluster_id = result["cluster_id"]
        template = result["template_mined"]

        # Variable extraction via template alignment
        variables = self._extract_variables_from_template(content, template)

        # Ensure block IDs are always captured (critical for HDFS sessions)
        for bid in re.findall(r"blk_-?\d+", line):
            if bid not in variables:
                variables.append(bid)

        level = header.get("level", "UNKNOWN")

        return {
            "event_template_id": cluster_id,
            "event_template": template,
            "extracted_variables": variables,
            "raw_line": line,
            "line_number": line_number,
            "severity_score": SEVERITY_MAP.get(level, -1),
            **header,        # all dataset-specific fields (date, time, pid, node, adminaddr, …)
            "level": level,  # ensure always set even when header fallback fires
            "content": content,  # override with extracted free-text, not the full raw line
        }

    # ── Variable extraction ────────────────────────────────────────────

    def _extract_variables_from_template(
        self, content: str, template: str
    ) -> list:
        """
        Align content tokens against the Drain template.  Every <*>
        slot in the template maps to a concrete token in the content.

        Falls back to regex-based extraction when token counts differ
        (can happen with multi-word wildcards or quoting).
        """
        variables = []
        try:
            content_tokens = content.split()
            template_tokens = template.split()

            if len(content_tokens) == len(template_tokens):
                for ct, tt in zip(content_tokens, template_tokens):
                    if re.search(r"<[^>]+>", tt):
                        variables.append(ct)
            else:
                variables = self._regex_extract(content)
        except Exception:
            variables = self._regex_extract(content)

        return variables

    @staticmethod
    def _regex_extract(text: str) -> list:
        """Fallback variable extraction using common patterns."""
        variables = []
        variables.extend(re.findall(r"blk_-?\d+", text))
        variables.extend(re.findall(r"\d+\.\d+\.\d+\.\d+(?::\d+)?", text))
        variables.extend(re.findall(r"(?<= )/[\w./-]+", text))
        return variables

    # ── Stream parsing ─────────────────────────────────────────────────

    def parse_stream(self, line_stream):
        """
        Parse a stream of log lines (generator-friendly).

        Args:
            line_stream: Iterable of ``(line_number, line)`` tuples or
                         bare line strings.

        Yields:
            Parsed event dicts.
        """
        count = 0
        for item in line_stream:
            if isinstance(item, tuple):
                line_number, line = item
            else:
                line = item
                line_number = None

            parsed = self.parse_line(line, line_number)
            count += 1

            if count % 100_000 == 0:
                logger.info(
                    "Parsed %s lines — %d templates discovered",
                    f"{count:,}", self.get_template_count(),
                )

            yield parsed

        logger.info(
            "Parsing complete — %s lines, %d unique templates  "
            "(header OK: %d, header fallback: %d)",
            f"{count:,}", self.get_template_count(),
            self._stats["header_parsed"], self._stats["header_failed"],
        )

    # ── Template introspection ─────────────────────────────────────────

    def get_template_count(self) -> int:
        """Return the number of unique event templates discovered."""
        return len(self.template_miner.drain.clusters)

    def get_templates(self) -> list:
        """Return all discovered templates with IDs and occurrence counts."""
        return [
            {
                "cluster_id": c.cluster_id,
                "template": c.get_template(),
                "size": c.size,
            }
            for c in self.template_miner.drain.clusters
        ]

    # ── State persistence ──────────────────────────────────────────────

    def save_state(self, path: str = None):
        """
        Persist the full TemplateMiner object via pickle so templates
        survive across runs without re-parsing.
        """
        if path is None:
            path = self.state_dir / f"drain_{self.dataset}.pkl"
        else:
            path = Path(path)

        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump(self.template_miner, f)

        # Human-readable summary alongside the pickle
        summary_path = path.with_suffix(".json")
        with open(summary_path, "w") as f:
            json.dump(
                {
                    "dataset": self.dataset,
                    "template_count": self.get_template_count(),
                    "templates": self.get_templates(),
                    "stats": self._stats,
                },
                f,
                indent=2,
            )

        logger.info(
            "Parser state saved to %s (%d templates)",
            path, self.get_template_count(),
        )

    def load_state(self, path: str = None) -> bool:
        """
        Reload a previously persisted TemplateMiner.

        Returns True on success, False if the file does not exist.
        """
        if path is None:
            path = self.state_dir / f"drain_{self.dataset}.pkl"
        else:
            path = Path(path)

        if not path.exists():
            logger.warning("No state file found at %s", path)
            return False

        with open(path, "rb") as f:
            self.template_miner = pickle.load(f)

        logger.info(
            "Parser state loaded from %s (%d templates)",
            path, self.get_template_count(),
        )
        return True

    def get_stats(self) -> dict:
        """Return parsing statistics."""
        return {**self._stats, "template_count": self.get_template_count()}


# ── CLI entry-point ────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    ap = argparse.ArgumentParser(description="Stage 2: Drain Log Parser")
    ap.add_argument("input_file", help="Path to (deduplicated) log file")
    ap.add_argument(
        "-d", "--dataset", default="hdfs",
        choices=["hdfs", "bgl", "thunderbird", "default"],
    )
    ap.add_argument("-n", "--max-lines", type=int, default=None)
    ap.add_argument("-o", "--output", default=None,
                    help="Output CSV path (default: data/processed/<stem>_structured.csv)")
    args = ap.parse_args()

    from ingestion import stream_deduplicated  # noqa: E402
    import pandas as pd

    parser = LogParser(dataset=args.dataset)

    # Parse and collect all rows
    rows = []
    count = 0
    for parsed in parser.parse_stream(stream_deduplicated(args.input_file)):
        count += 1
        rows.append({
            "LineId": count,
            "Content": parsed["raw_line"],
            "EventId": parsed["event_template_id"],
            "EventTemplate": parsed["event_template"],
            "Level": parsed["level"],
        })
        if count <= 5:
            print(
                f"[{parsed['level']}] Template #{parsed['event_template_id']}: "
                f"{parsed['event_template']}"
            )
            if parsed["extracted_variables"]:
                print(f"       vars: {parsed['extracted_variables']}")
        if args.max_lines and count >= args.max_lines:
            break

    print(f"\nTotal lines parsed: {count}")
    print(f"Unique templates:  {parser.get_template_count()}")
    print(f"Stats: {parser.get_stats()}")
    print("\nAll templates:")
    for t in parser.get_templates():
        print(f"  [{t['cluster_id']}] (size={t['size']}): {t['template']}")

    parser.save_state()

    # ── Save structured CSV ───────────────────────────────────────────
    if args.output:
        csv_path = Path(args.output)
    else:
        input_stem = Path(args.input_file).stem
        csv_path = Path("../data/processed") / f"{input_stem}_structured.csv"

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print(f"\nStructured CSV saved to: {csv_path}")