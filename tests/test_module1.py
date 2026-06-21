"""
Module 1 — Comprehensive test suite.

Covers every feature of Module 1:
  1. Streaming ingestion  (read_log_stream, blank-line skipping)
  2. Consecutive-duplicate removal
  3. HDFS header parsing  (date / time / pid / level / component / content)
  4. BGL header parsing   (timestamp / date / node / time / nodenum / type / component / level)
  5. Thunderbird header parsing (id / date / admin / time / adminaddr)
  6. Default fallback parsing  (unknown format — no crash, content extracted)
  7. Variable extraction fix   (<BLOCK_ID> / <IP_PORT> / <IP_ADDR> / <NUM> all captured)
  8. Dataset inference from filename
  9. Dynamic CSV schema  (correct columns per dataset)
 10. End-to-end run_module1() for each dataset (CSV columns, pkl, JSON summary)
 11. Graceful fallback for unknown log format
 12. Drain state persistence (pkl saves and reloads correctly)

Run with:
    python -m pytest tests/test_module1.py -v
    python tests/test_module1.py
"""

import csv
import json
import os
import pickle
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

# Ensure src/ is on the path regardless of where pytest is invoked from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ingestion import read_log_stream
from log_parser import LogParser
from module1_ingest_parse import (
    DATASET_CSV_COLUMNS,
    _infer_dataset,
    run_module1,
)

# ---------------------------------------------------------------------------
# Sample log lines for each dataset (deliberately short so tests are fast)
# ---------------------------------------------------------------------------

HDFS_LINES = """\
081109 203518 143 INFO dfs.DataNode$DataXceiver: Receiving block blk_-1608999687919862906 src: /10.250.19.102:54106 dest: /10.250.19.102:50010
081109 203519 145 INFO dfs.DataNode$PacketResponder: PacketResponder 1 for block blk_-1608999687919862906 terminating
081109 203519 145 INFO dfs.DataNode$PacketResponder: Received block blk_-1608999687919862906 of size 91178 from /10.250.10.6
081109 203519 145 INFO dfs.DataNode$PacketResponder: Received block blk_-1608999687919862906 of size 91178 from /10.250.10.6
081109 203521 147 INFO dfs.FSNamesystem: BLOCK* NameSystem.addStoredBlock: blockMap updated: 10.250.14.224:50010 is added to blk_-1608999687919862906 size 91178
081109 203521 19 INFO dfs.DataNode: 10.250.14.224:50010 Starting thread to transfer block blk_-1608999687919862906 to 10.251.215.16:50010
"""

BGL_LINES = """\
- 1117838570 2005.06.03 R02-M1-N0-C:J12-U11 2005-06-03-15.42.50.363779 R02-M1-N0-C:J12-U11 RAS KERNEL INFO instruction cache parity error corrected
- 1117838570 2005.06.03 R02-M1-N0-C:J12-U11 2005-06-03-15.42.50.527847 R02-M1-N0-C:J12-U11 RAS KERNEL INFO instruction cache parity error corrected
- 1117838570 2005.06.03 R02-M1-N0-C:J12-U11 2005-06-03-15.42.50.675872 R02-M1-N0-C:J12-U11 RAS KERNEL INFO instruction cache parity error corrected
FATAL 1117838576 2005.06.03 R02-M1-N0-C:J12-U11 2005-06-03-15.42.56.073872 R02-M1-N0-C:J12-U11 RAS KERNEL FATAL kernel panic: fatal exception in kernel
"""

THUNDERBIRD_LINES = """\
- 1131484800 2005.11.09 tbird-admin1 2005-11-09-11.00.00.000000 tbird-admin1 kernel: INFO normal operation completed
- 1131484801 2005.11.09 tbird-admin2 2005-11-09-11.00.01.000000 tbird-admin2 sshd: INFO accepted connection from 10.0.0.1
ALERT 1131484802 2005.11.09 tbird-admin3 2005-11-09-11.00.02.000000 tbird-admin3 kernel: ALERT memory fault detected at address 0xdeadbeef
- 1131484803 2005.11.09 tbird-admin1 2005-11-09-11.00.03.000000 tbird-admin1 kernel: INFO normal operation completed
"""

UNKNOWN_LINES = """\
[2024-01-15 10:23:45] ERROR Something bad happened: connection refused
[2024-01-15 10:23:46] INFO  Connection established successfully
[2024-01-15 10:23:47] WARN  Retrying connection attempt number 3
"""


def _write_log(directory, filename, content):
    """Write content to a temp log file and return its Path."""
    path = Path(directory) / filename
    path.write_text(content, encoding="utf-8")
    return path


# ===========================================================================
# 1. Streaming Ingestion
# ===========================================================================

class TestStreamIngestion(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_read_log_stream_yields_lines(self):
        path = _write_log(self.tmpdir, "hdfs.log", HDFS_LINES)
        lines = list(read_log_stream(str(path)))
        self.assertGreater(len(lines), 0)
        self.assertIsInstance(lines[0], str)

    def test_read_log_stream_skips_blank_lines(self):
        content = "line one\n\n\nline two\n"
        path = _write_log(self.tmpdir, "test.log", content)
        lines = list(read_log_stream(str(path)))
        self.assertEqual(len(lines), 2)

    def test_read_log_stream_strips_newlines(self):
        path = _write_log(self.tmpdir, "test.log", "hello world\n")
        lines = list(read_log_stream(str(path)))
        self.assertEqual(lines[0], "hello world")

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            list(read_log_stream(str(Path(self.tmpdir) / "nonexistent.log")))


# ===========================================================================
# 2. Consecutive-Duplicate Removal (via run_module1 skip_dedup=False)
# ===========================================================================

class TestDeduplication(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_consecutive_duplicates_removed(self):
        # HDFS_LINES has one consecutive duplicate (line 3 == line 4)
        log_path = _write_log(self.tmpdir, "HDFS.log", HDFS_LINES)
        csv_path = Path(self.tmpdir) / "out.csv"
        pkl_path = Path(self.tmpdir) / "drain.pkl"

        result = run_module1(
            str(log_path), dataset="hdfs",
            output_csv=str(csv_path), output_pkl=str(pkl_path),
        )
        self.assertEqual(result["duplicates_removed"], 1)
        self.assertEqual(result["total_lines"], result["dedup_lines"] + 1)

    def test_skip_dedup_keeps_all_lines(self):
        log_path = _write_log(self.tmpdir, "HDFS.log", HDFS_LINES)
        csv_path = Path(self.tmpdir) / "out.csv"
        pkl_path = Path(self.tmpdir) / "drain.pkl"

        result = run_module1(
            str(log_path), dataset="hdfs", skip_dedup=True,
            output_csv=str(csv_path), output_pkl=str(pkl_path),
        )
        self.assertEqual(result["duplicates_removed"], 0)
        self.assertEqual(result["total_lines"], result["dedup_lines"])


# ===========================================================================
# 3. HDFS Header Parsing
# ===========================================================================

class TestHDFSParsing(unittest.TestCase):

    def setUp(self):
        self.parser = LogParser(dataset="hdfs", persist_state=False)
        self.line = (
            "081109 203518 143 INFO dfs.DataNode$DataXceiver: "
            "Receiving block blk_-1608999687919862906 src: /10.250.19.102:54106 "
            "dest: /10.250.19.102:50010"
        )
        self.parsed = self.parser.parse_line(self.line, line_number=1)

    def test_date_extracted(self):
        self.assertEqual(self.parsed["date"], "081109")

    def test_time_extracted(self):
        self.assertEqual(self.parsed["time"], "203518")

    def test_pid_extracted(self):
        self.assertEqual(self.parsed["pid"], "143")

    def test_level_extracted(self):
        self.assertEqual(self.parsed["level"], "INFO")

    def test_component_extracted(self):
        self.assertIn("DataXceiver", self.parsed["component"])

    def test_content_is_message_only(self):
        # Content must not contain the header prefix
        self.assertNotIn("081109", self.parsed["content"])
        self.assertIn("Receiving block", self.parsed["content"])

    def test_event_template_assigned(self):
        self.assertIsNotNone(self.parsed["event_template"])
        self.assertGreater(len(self.parsed["event_template"]), 0)

    def test_severity_score_info(self):
        self.assertEqual(self.parsed["severity_score"], 2)


# ===========================================================================
# 4. BGL Header Parsing
# ===========================================================================

class TestBGLParsing(unittest.TestCase):

    def setUp(self):
        self.parser = LogParser(dataset="bgl", persist_state=False)
        self.line = (
            "- 1117838570 2005.06.03 R02-M1-N0-C:J12-U11 "
            "2005-06-03-15.42.50.363779 R02-M1-N0-C:J12-U11 RAS KERNEL INFO "
            "instruction cache parity error corrected"
        )
        self.parsed = self.parser.parse_line(self.line, line_number=1)

    def test_timestamp_extracted(self):
        self.assertEqual(self.parsed["timestamp"], "1117838570")

    def test_date_extracted(self):
        self.assertEqual(self.parsed["date"], "2005.06.03")

    def test_node_extracted(self):
        self.assertEqual(self.parsed["node"], "R02-M1-N0-C:J12-U11")

    def test_component_extracted(self):
        self.assertEqual(self.parsed["component"], "KERNEL")

    def test_level_extracted(self):
        self.assertEqual(self.parsed["level"], "INFO")

    def test_noderepeat_extracted(self):
        self.assertEqual(self.parsed["noderepeat"], "R02-M1-N0-C:J12-U11")

    def test_content_is_message_only(self):
        self.assertIn("instruction cache parity error corrected", self.parsed["content"])
        self.assertNotIn("1117838570", self.parsed["content"])
        self.assertNotIn("R02-M1-N0-C:J12-U11", self.parsed["content"])

    def test_label_normal_extracted(self):
        self.assertEqual(self.parsed["label"], "-")

    def test_label_fatal_extracted(self):
        fatal_line = (
            "FATAL 1117838576 2005.06.03 R02-M1-N0-C:J12-U11 "
            "2005-06-03-15.42.56.073872 R02-M1-N0-C:J12-U11 RAS KERNEL FATAL "
            "kernel panic: fatal exception"
        )
        p = self.parser.parse_line(fatal_line)
        self.assertEqual(p["label"], "FATAL")

    def test_fatal_level_severity(self):
        fatal_line = (
            "FATAL 1117838576 2005.06.03 R02-M1-N0-C:J12-U11 "
            "2005-06-03-15.42.56.073872 R02-M1-N0-C:J12-U11 RAS KERNEL FATAL "
            "kernel panic: fatal exception"
        )
        p = self.parser.parse_line(fatal_line)
        self.assertEqual(p["level"], "FATAL")
        self.assertEqual(p["severity_score"], 5)


# ===========================================================================
# 5. Thunderbird Header Parsing
# ===========================================================================

class TestThunderbirdParsing(unittest.TestCase):

    def setUp(self):
        self.parser = LogParser(dataset="thunderbird", persist_state=False)
        self.line = (
            "- 1131484800 2005.11.09 tbird-admin1 "
            "2005-11-09-11.00.00.000000 tbird-admin1 "
            "kernel: INFO normal operation completed"
        )
        self.parsed = self.parser.parse_line(self.line, line_number=1)

    def test_id_extracted(self):
        self.assertEqual(self.parsed["id"], "1131484800")

    def test_date_extracted(self):
        self.assertEqual(self.parsed["date"], "2005.11.09")

    def test_admin_extracted(self):
        self.assertEqual(self.parsed["admin"], "tbird-admin1")

    def test_adminaddr_extracted(self):
        self.assertEqual(self.parsed["adminaddr"], "tbird-admin1")

    def test_label_normal_extracted(self):
        self.assertEqual(self.parsed["label"], "-")

    def test_label_alert_extracted(self):
        alert_line = (
            "ALERT 1131484802 2005.11.09 tbird-admin3 "
            "2005-11-09-11.00.02.000000 tbird-admin3 "
            "kernel: ALERT memory fault detected at address 0xdeadbeef"
        )
        p = self.parser.parse_line(alert_line)
        self.assertEqual(p["label"], "ALERT")

    def test_content_is_message_only(self):
        self.assertIn("normal operation", self.parsed["content"])
        self.assertNotIn("1131484800", self.parsed["content"])


# ===========================================================================
# 6. Default / Unknown Format Fallback
# ===========================================================================

class TestDefaultFallback(unittest.TestCase):

    def setUp(self):
        self.parser = LogParser(dataset="default", persist_state=False)

    def test_unknown_format_does_not_crash(self):
        line = "[2024-01-15 10:23:45] ERROR Something bad happened"
        parsed = self.parser.parse_line(line)
        self.assertIn("event_template", parsed)
        self.assertIn("content", parsed)

    def test_content_falls_back_to_full_line(self):
        line = "[2024-01-15 10:23:45] ERROR Something bad happened"
        parsed = self.parser.parse_line(line)
        # Fallback: content == the full raw line
        self.assertEqual(parsed["content"], line)


# ===========================================================================
# 7. Variable Extraction Fix
# ===========================================================================

class TestVariableExtraction(unittest.TestCase):

    def setUp(self):
        self.parser = LogParser(dataset="hdfs", persist_state=False)

    def _vars(self, line):
        return self.parser.parse_line(line)["extracted_variables"]

    def test_block_id_captured(self):
        line = (
            "081109 203518 143 INFO dfs.DataNode$DataXceiver: "
            "Receiving block blk_-1608999687919862906 src: /10.250.19.102:54106 "
            "dest: /10.250.19.102:50010"
        )
        block_ids = [v for v in self._vars(line) if v.startswith("blk_")]
        self.assertGreater(len(block_ids), 0, "Block ID must be in ParameterList")

    def test_ip_port_captured(self):
        line = (
            "081109 203518 143 INFO dfs.DataNode$DataXceiver: "
            "Receiving block blk_-1608999687919862906 src: /10.250.19.102:54106 "
            "dest: /10.250.19.102:50010"
        )
        # After several parses the template will use <IP_PORT> or <*> — either
        # way the IP values must appear in extracted_variables.
        # Parse twice to let the template stabilise.
        self.parser.parse_line(line)
        variables = self._vars(line)
        ip_vals = [v for v in variables if "10.250" in v or "10.251" in v]
        self.assertGreater(len(ip_vals), 0, "IP addresses must be in ParameterList")

    def test_size_number_captured(self):
        # Feed a few lines so Drain builds a template for this pattern.
        for _ in range(3):
            self.parser.parse_line(
                "081109 203519 145 INFO dfs.DataNode$PacketResponder: "
                "Received block blk_-1608999687919862906 of size 91178 from /10.250.10.6"
            )
        variables = self._vars(
            "081109 203519 145 INFO dfs.DataNode$PacketResponder: "
            "Received block blk_7503483334202473044 of size 233217 from /10.251.215.16"
        )
        nums = [v for v in variables if v.isdigit() or v.lstrip("-").isdigit()]
        self.assertGreater(len(nums), 0, "Numeric size must be in ParameterList")

    def test_no_raw_block_id_in_template(self):
        line = (
            "081109 203518 143 INFO dfs.DataNode$DataXceiver: "
            "Receiving block blk_-1608999687919862906 src: /10.250.19.102:54106 "
            "dest: /10.250.19.102:50010"
        )
        parsed = self.parser.parse_line(line)
        self.assertNotIn("blk_-1608999687919862906", parsed["event_template"],
                         "Raw block ID must not leak into the event template")


# ===========================================================================
# 8. Dataset Inference from Filename
# ===========================================================================

class TestDatasetInference(unittest.TestCase):

    def _infer(self, name):
        return _infer_dataset(Path(name))

    def test_infer_hdfs_lowercase(self):
        self.assertEqual(self._infer("hdfs.log"), "hdfs")

    def test_infer_hdfs_uppercase(self):
        self.assertEqual(self._infer("HDFS.log"), "hdfs")

    def test_infer_hdfs_in_path(self):
        self.assertEqual(self._infer("data/raw/HDFS_sample_1pct.log"), "hdfs")

    def test_infer_bgl(self):
        self.assertEqual(self._infer("BGL.log"), "bgl")

    def test_infer_thunderbird(self):
        self.assertEqual(self._infer("Thunderbird.log"), "thunderbird")

    def test_infer_unknown_returns_none(self):
        self.assertIsNone(self._infer("apache_access.log"))

    def test_infer_unknown_returns_none_generic(self):
        self.assertIsNone(self._infer("myapp_2024.log"))


# ===========================================================================
# 9. Dynamic CSV Schema
# ===========================================================================

class TestDynamicSchema(unittest.TestCase):

    def test_hdfs_schema(self):
        cols = DATASET_CSV_COLUMNS["hdfs"]
        for expected in ["LineId", "Date", "Time", "Pid", "Level",
                         "Component", "Content", "EventId",
                         "EventTemplate", "ParameterList"]:
            self.assertIn(expected, cols)

    def test_bgl_schema(self):
        cols = DATASET_CSV_COLUMNS["bgl"]
        for expected in ["LineId", "Label", "Timestamp", "Date", "Node", "Time",
                         "NodeRepeat", "Type", "Component", "Level", "Content",
                         "EventId", "EventTemplate", "ParameterList"]:
            self.assertIn(expected, cols)
        self.assertNotIn("Pid", cols)
        self.assertNotIn("NodeNum", cols)

    def test_thunderbird_schema(self):
        cols = DATASET_CSV_COLUMNS["thunderbird"]
        for expected in ["LineId", "Label", "Id", "Date", "Admin", "Time",
                         "AdminAddr", "Content", "EventId",
                         "EventTemplate", "ParameterList"]:
            self.assertIn(expected, cols)
        self.assertNotIn("Pid", cols)
        self.assertNotIn("Component", cols)

    def test_default_schema_is_minimal(self):
        cols = DATASET_CSV_COLUMNS["default"]
        for expected in ["LineId", "Content", "EventId",
                         "EventTemplate", "ParameterList"]:
            self.assertIn(expected, cols)
        self.assertNotIn("Pid", cols)
        self.assertNotIn("Date", cols)

    def test_all_schemas_start_with_lineid(self):
        for ds, cols in DATASET_CSV_COLUMNS.items():
            self.assertEqual(cols[0], "LineId", f"{ds} schema must start with LineId")

    def test_all_schemas_end_with_parameterlist(self):
        for ds, cols in DATASET_CSV_COLUMNS.items():
            self.assertEqual(cols[-1], "ParameterList",
                             f"{ds} schema must end with ParameterList")


# ===========================================================================
# 10. End-to-End run_module1() per dataset
# ===========================================================================

class TestRunModule1EndToEnd(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def _run(self, filename, content, dataset):
        log_path = _write_log(self.tmpdir, filename, content)
        csv_path = Path(self.tmpdir) / "out.csv"
        pkl_path = Path(self.tmpdir) / "drain.pkl"
        result = run_module1(
            str(log_path), dataset=dataset,
            output_csv=str(csv_path),
            output_pkl=str(pkl_path),
        )
        return result, csv_path, pkl_path

    # -- HDFS --

    def test_hdfs_csv_columns(self):
        result, csv_path, _ = self._run("HDFS.log", HDFS_LINES, "hdfs")
        with open(csv_path, encoding="utf-8") as f:
            header = next(csv.reader(f))
        self.assertEqual(header, DATASET_CSV_COLUMNS["hdfs"])

    def test_hdfs_result_dataset_field(self):
        result, _, _ = self._run("HDFS.log", HDFS_LINES, "hdfs")
        self.assertEqual(result["dataset"], "hdfs")

    def test_hdfs_csv_has_rows(self):
        result, csv_path, _ = self._run("HDFS.log", HDFS_LINES, "hdfs")
        self.assertGreater(result["dedup_lines"], 0)

    def test_hdfs_csv_date_column_filled(self):
        _, csv_path, _ = self._run("HDFS.log", HDFS_LINES, "hdfs")
        with open(csv_path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.assertTrue(all(r["Date"] for r in rows), "Date must be filled for HDFS")

    def test_hdfs_csv_pid_column_filled(self):
        _, csv_path, _ = self._run("HDFS.log", HDFS_LINES, "hdfs")
        with open(csv_path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.assertTrue(all(r["Pid"] for r in rows), "Pid must be filled for HDFS")

    # -- BGL --

    def test_bgl_csv_columns(self):
        result, csv_path, _ = self._run("BGL.log", BGL_LINES, "bgl")
        with open(csv_path, encoding="utf-8") as f:
            header = next(csv.reader(f))
        self.assertEqual(header, DATASET_CSV_COLUMNS["bgl"])

    def test_bgl_result_dataset_field(self):
        result, _, _ = self._run("BGL.log", BGL_LINES, "bgl")
        self.assertEqual(result["dataset"], "bgl")

    def test_bgl_csv_node_column_filled(self):
        _, csv_path, _ = self._run("BGL.log", BGL_LINES, "bgl")
        with open(csv_path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.assertTrue(all(r["Node"] for r in rows), "Node must be filled for BGL")

    def test_bgl_csv_label_column_filled(self):
        _, csv_path, _ = self._run("BGL.log", BGL_LINES, "bgl")
        with open(csv_path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.assertIn("Label", rows[0], "Label column must exist in BGL CSV")
        self.assertTrue(all(r["Label"] for r in rows), "Label must not be empty")

    def test_bgl_csv_fatal_label_preserved(self):
        _, csv_path, _ = self._run("BGL.log", BGL_LINES, "bgl")
        with open(csv_path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        labels = {r["Label"] for r in rows}
        self.assertIn("FATAL", labels, "FATAL label from BGL_LINES must appear in CSV")
        self.assertIn("-", labels, "Normal '-' label must appear in CSV")

    def test_bgl_csv_no_pid_column(self):
        _, csv_path, _ = self._run("BGL.log", BGL_LINES, "bgl")
        with open(csv_path, encoding="utf-8") as f:
            header = next(csv.reader(f))
        self.assertNotIn("Pid", header)

    # -- Thunderbird --

    def test_thunderbird_csv_columns(self):
        result, csv_path, _ = self._run("Thunderbird.log", THUNDERBIRD_LINES, "thunderbird")
        with open(csv_path, encoding="utf-8") as f:
            header = next(csv.reader(f))
        self.assertEqual(header, DATASET_CSV_COLUMNS["thunderbird"])

    def test_thunderbird_result_dataset_field(self):
        result, _, _ = self._run("Thunderbird.log", THUNDERBIRD_LINES, "thunderbird")
        self.assertEqual(result["dataset"], "thunderbird")

    def test_thunderbird_csv_admin_column_filled(self):
        _, csv_path, _ = self._run("Thunderbird.log", THUNDERBIRD_LINES, "thunderbird")
        with open(csv_path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.assertTrue(all(r["Admin"] for r in rows), "Admin must be filled for Thunderbird")

    def test_thunderbird_csv_label_column_filled(self):
        _, csv_path, _ = self._run("Thunderbird.log", THUNDERBIRD_LINES, "thunderbird")
        with open(csv_path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.assertIn("Label", rows[0], "Label column must exist in Thunderbird CSV")
        self.assertTrue(all(r["Label"] for r in rows), "Label must not be empty")

    def test_thunderbird_csv_alert_label_preserved(self):
        _, csv_path, _ = self._run("Thunderbird.log", THUNDERBIRD_LINES, "thunderbird")
        with open(csv_path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        labels = {r["Label"] for r in rows}
        self.assertIn("ALERT", labels, "ALERT label from THUNDERBIRD_LINES must appear in CSV")
        self.assertIn("-", labels, "Normal '-' label must appear in CSV")

    def test_thunderbird_csv_no_pid_column(self):
        _, csv_path, _ = self._run("Thunderbird.log", THUNDERBIRD_LINES, "thunderbird")
        with open(csv_path, encoding="utf-8") as f:
            header = next(csv.reader(f))
        self.assertNotIn("Pid", header)

    # -- Common checks across all datasets --

    def test_event_id_always_present(self):
        for ds, content, fname in [
            ("hdfs",        HDFS_LINES,        "HDFS.log"),
            ("bgl",         BGL_LINES,         "BGL.log"),
            ("thunderbird", THUNDERBIRD_LINES,  "Thunderbird.log"),
        ]:
            with self.subTest(dataset=ds):
                _, csv_path, _ = self._run(fname, content, ds)
                with open(csv_path, encoding="utf-8") as f:
                    rows = list(csv.DictReader(f))
                self.assertTrue(all(r["EventId"].startswith("E") for r in rows))

    def test_event_template_never_empty(self):
        for ds, content, fname in [
            ("hdfs",        HDFS_LINES,        "HDFS.log"),
            ("bgl",         BGL_LINES,         "BGL.log"),
            ("thunderbird", THUNDERBIRD_LINES,  "Thunderbird.log"),
        ]:
            with self.subTest(dataset=ds):
                _, csv_path, _ = self._run(fname, content, ds)
                with open(csv_path, encoding="utf-8") as f:
                    rows = list(csv.DictReader(f))
                self.assertTrue(all(r["EventTemplate"] for r in rows))

    def test_lineid_sequential(self):
        _, csv_path, _ = self._run("HDFS.log", HDFS_LINES, "hdfs")
        with open(csv_path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        ids = [int(r["LineId"]) for r in rows]
        self.assertEqual(ids, list(range(1, len(ids) + 1)))


# ===========================================================================
# 11. Graceful Fallback for Unknown Format
# ===========================================================================

class TestGracefulFallback(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_unknown_format_does_not_crash(self):
        log_path = _write_log(self.tmpdir, "myapp.log", UNKNOWN_LINES)
        csv_path = Path(self.tmpdir) / "out.csv"
        pkl_path = Path(self.tmpdir) / "drain.pkl"
        # Should complete without raising
        result = run_module1(
            str(log_path),
            output_csv=str(csv_path),
            output_pkl=str(pkl_path),
        )
        self.assertEqual(result["dataset"], "default")
        self.assertGreater(result["dedup_lines"], 0)

    def test_unknown_format_uses_minimal_schema(self):
        log_path = _write_log(self.tmpdir, "myapp.log", UNKNOWN_LINES)
        csv_path = Path(self.tmpdir) / "out.csv"
        pkl_path = Path(self.tmpdir) / "drain.pkl"
        run_module1(str(log_path), output_csv=str(csv_path), output_pkl=str(pkl_path))
        with open(csv_path, encoding="utf-8") as f:
            header = next(csv.reader(f))
        self.assertEqual(header, DATASET_CSV_COLUMNS["default"])

    def test_filename_inference_triggers_correct_dataset(self):
        # File named HDFS.log → no --dataset needed
        log_path = _write_log(self.tmpdir, "HDFS.log", HDFS_LINES)
        csv_path = Path(self.tmpdir) / "out.csv"
        pkl_path = Path(self.tmpdir) / "drain.pkl"
        result = run_module1(
            str(log_path),
            output_csv=str(csv_path),
            output_pkl=str(pkl_path),
        )
        self.assertEqual(result["dataset"], "hdfs")


# ===========================================================================
# 12. Drain State Persistence
# ===========================================================================

class TestDrainStatePersistence(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_pkl_is_created(self):
        log_path = _write_log(self.tmpdir, "HDFS.log", HDFS_LINES)
        csv_path = Path(self.tmpdir) / "out.csv"
        pkl_path = Path(self.tmpdir) / "drain.pkl"
        run_module1(str(log_path), dataset="hdfs",
                    output_csv=str(csv_path), output_pkl=str(pkl_path))
        self.assertTrue(pkl_path.exists())
        self.assertGreater(pkl_path.stat().st_size, 0)

    def test_pkl_reloads_and_parses(self):
        log_path = _write_log(self.tmpdir, "HDFS.log", HDFS_LINES)
        csv_path = Path(self.tmpdir) / "out.csv"
        pkl_path = Path(self.tmpdir) / "drain.pkl"
        run_module1(str(log_path), dataset="hdfs",
                    output_csv=str(csv_path), output_pkl=str(pkl_path))

        with open(pkl_path, "rb") as f:
            tm = pickle.load(f)

        result = tm.add_log_message(
            "Receiving block BLOCK_ID src: /IP_PORT dest: /IP_PORT"
        )
        self.assertIn("cluster_id", result)
        self.assertGreater(result["cluster_id"], 0)

    def test_json_summary_created(self):
        log_path = _write_log(self.tmpdir, "HDFS.log", HDFS_LINES)
        csv_path = Path(self.tmpdir) / "out.csv"
        pkl_path = Path(self.tmpdir) / "drain.pkl"
        run_module1(str(log_path), dataset="hdfs",
                    output_csv=str(csv_path), output_pkl=str(pkl_path))

        json_path = pkl_path.with_suffix(".json")
        self.assertTrue(json_path.exists())
        with open(json_path, encoding="utf-8") as f:
            summary = json.load(f)

        self.assertEqual(summary["dataset"], "hdfs")
        self.assertIn("templates", summary)
        self.assertIn("csv_columns", summary)
        self.assertIn("deduplicated_lines", summary)
        self.assertGreater(summary["unique_event_templates"], 0)

    def test_pkl_template_count_matches_json(self):
        log_path = _write_log(self.tmpdir, "HDFS.log", HDFS_LINES)
        csv_path = Path(self.tmpdir) / "out.csv"
        pkl_path = Path(self.tmpdir) / "drain.pkl"
        run_module1(str(log_path), dataset="hdfs",
                    output_csv=str(csv_path), output_pkl=str(pkl_path))

        json_path = pkl_path.with_suffix(".json")
        with open(json_path, encoding="utf-8") as f:
            summary = json.load(f)
        with open(pkl_path, "rb") as f:
            tm = pickle.load(f)

        self.assertEqual(
            summary["unique_event_templates"],
            len(tm.drain.clusters),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
