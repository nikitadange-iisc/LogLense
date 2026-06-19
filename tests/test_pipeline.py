"""
Tests for LogSense pipeline stages.
"""

import os
import sys
import json
import tempfile
import shutil
import unittest
from pathlib import Path

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ingestion import read_log_stream, stream_deduplicated, deduplicate_stream
from log_parser import LogParser
from sessionizer import Sessionizer, Session
from anomaly_gate import AnomalyGate
from module1_ingest_parse import run_module1, CSV_COLUMNS


SAMPLE_HDFS_LOGS = """081109 203518 148 INFO dfs.DataNode$PacketResponder: PacketResponder 1 for block blk_38865049064139660 terminating
081109 203518 148 INFO dfs.DataNode$PacketResponder: PacketResponder 1 for block blk_38865049064139660 terminating
081109 203519 149 INFO dfs.DataNode$DataXceiver: Receiving block blk_-1608999687919862906 src: /10.250.19.102:54106 dest: /10.250.19.102:50010
081109 203519 150 INFO dfs.DataNode$DataXceiver: Receiving block blk_-1608999687919862906 src: /10.250.19.102:54106 dest: /10.250.19.102:50010
081109 203520 151 INFO dfs.FSNamesystem: BLOCK* NameSystem.allocateBlock: /mnt/hadoop/mapred/system/job_200811092030_0001/job.jar. blk_7503483334202473044
081109 203521 152 INFO dfs.FSNamesystem: BLOCK* NameSystem.addStoredBlock: blockMap updated: 10.250.10.6:50010 is added to blk_7503483334202473044 size 267507
081109 203522 153 WARN dfs.FSNamesystem: BLOCK* NameSystem.addStoredBlock: Redundant addStoredBlock request received for blk_7503483334202473044
081109 203523 154 INFO dfs.DataNode$PacketResponder: PacketResponder 0 for block blk_7503483334202473044 terminating
081109 203524 155 INFO dfs.DataNode$DataXceiver: Receiving block blk_-6670958622368987959 src: /10.250.14.224:42420 dest: /10.250.14.224:50010
081109 203525 156 INFO dfs.DataNode$PacketResponder: Received block blk_-6670958622368987959 of size 67108864 from /10.250.14.224
"""


class TestIngestion(unittest.TestCase):
    """Tests for Stage 1: Ingestion & Deduplication."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.sample_file = os.path.join(self.temp_dir, "sample.log")
        with open(self.sample_file, "w") as f:
            f.write(SAMPLE_HDFS_LOGS)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_read_log_stream(self):
        lines = list(read_log_stream(self.sample_file))
        self.assertTrue(len(lines) > 0)
        self.assertIsInstance(lines[0], str)

    def test_stream_deduplicated_removes_duplicates(self):
        results = list(stream_deduplicated(self.sample_file))
        lines = [line for _, line in results]
        # Should have fewer lines than raw (duplicates removed)
        raw_lines = list(read_log_stream(self.sample_file))
        self.assertLess(len(lines), len(raw_lines))

    def test_deduplicate_stream_writes_output(self):
        output_path = os.path.join(self.temp_dir, "dedup.log")
        stats = deduplicate_stream(self.sample_file, output_path)
        self.assertTrue(os.path.exists(output_path))
        self.assertGreater(stats["duplicates_removed"], 0)
        self.assertEqual(stats["total_lines"],
                         stats["deduplicated_lines"] + stats["duplicates_removed"])


class TestParser(unittest.TestCase):
    """Tests for Stage 2: Drain Log Parser."""

    def setUp(self):
        self.parser = LogParser(dataset="hdfs", persist_state=False)
        self.sample_lines = [l.strip() for l in SAMPLE_HDFS_LOGS.strip().split("\n") if l.strip()]

    def test_parse_line(self):
        result = self.parser.parse_line(self.sample_lines[0], line_number=1)
        self.assertIn("event_template_id", result)
        self.assertIn("event_template", result)
        self.assertIn("raw_line", result)
        self.assertIn("level", result)
        self.assertIn("severity_score", result)
        self.assertEqual(result["line_number"], 1)

    def test_parse_line_extracts_level(self):
        result = self.parser.parse_line(self.sample_lines[0], line_number=1)
        self.assertEqual(result["level"], "INFO")
        self.assertEqual(result["severity_score"], 2)

    def test_parse_multiple_lines_creates_templates(self):
        for i, line in enumerate(self.sample_lines):
            self.parser.parse_line(line, i)
        self.assertGreater(self.parser.get_template_count(), 0)

    def test_get_templates(self):
        for i, line in enumerate(self.sample_lines):
            self.parser.parse_line(line, i)
        templates = self.parser.get_templates()
        self.assertIsInstance(templates, list)
        self.assertGreater(len(templates), 0)

    def test_variable_extraction_finds_block_ids(self):
        result = self.parser.parse_line(self.sample_lines[0], line_number=1)
        block_ids = [v for v in result["extracted_variables"] if v.startswith("blk_")]
        self.assertGreater(len(block_ids), 0)


class TestSessionizer(unittest.TestCase):
    """Tests for Stage 3: Session Grouping & Vectorization."""

    def setUp(self):
        self.parser = LogParser(dataset="hdfs", persist_state=False)
        self.sample_lines = [l.strip() for l in SAMPLE_HDFS_LOGS.strip().split("\n") if l.strip()]
        self.parsed_events = [
            self.parser.parse_line(line, i)
            for i, line in enumerate(self.sample_lines)
        ]

    def test_block_id_extraction(self):
        sessionizer = Sessionizer(method="block_id")
        block_id = sessionizer.extract_block_id(self.parsed_events[0])
        self.assertIsNotNone(block_id)
        self.assertTrue(block_id.startswith("blk_"))

    def test_create_sessions_block_id(self):
        sessionizer = Sessionizer(method="block_id")
        sessions = sessionizer.create_sessions(iter(self.parsed_events))
        self.assertGreater(len(sessions), 0)
        self.assertIsInstance(sessions[0], Session)

    def test_vectorize_sessions(self):
        sessionizer = Sessionizer(method="block_id")
        sessions = sessionizer.create_sessions(iter(self.parsed_events))
        sessions, vocab = sessionizer.vectorize_all(sessions)
        self.assertIsNotNone(sessions[0].vector)
        self.assertEqual(len(sessions[0].vector), len(vocab))

    def test_sliding_window_sessions(self):
        sessionizer = Sessionizer(method="sliding_window", window_size=3, step_size=2)
        sessions = sessionizer.create_sessions(iter(self.parsed_events))
        self.assertGreater(len(sessions), 0)


class TestAnomalyGate(unittest.TestCase):
    """Tests for Stage 4: Isolation Forest Anomaly Gate."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.model_path = os.path.join(self.temp_dir, "test_model.joblib")

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_train_and_predict(self):
        np.random.seed(42)
        normal_data = np.random.randn(50, 5)

        gate = AnomalyGate(model_path=self.model_path)
        gate.train(normal_data)

        predictions = gate.predict(normal_data)
        self.assertEqual(len(predictions), 50)
        self.assertTrue(all(p in [-1, 1] for p in predictions))

    def test_save_and_load(self):
        np.random.seed(42)
        data = np.random.randn(50, 5)

        gate = AnomalyGate(model_path=self.model_path)
        gate.train(data)
        gate.save_model()

        gate2 = AnomalyGate(model_path=self.model_path)
        gate2.load_model()
        predictions = gate2.predict(data)
        self.assertEqual(len(predictions), 50)

    def test_filter_anomalous(self):
        np.random.seed(42)
        normal_vectors = np.random.randn(50, 5)
        anomaly_vectors = np.random.randn(5, 5) + 10

        gate = AnomalyGate(model_path=self.model_path)
        gate.train(normal_vectors)

        sessions = []
        for i, v in enumerate(np.vstack([normal_vectors, anomaly_vectors])):
            s = Session(session_id=f"s_{i}", raw_lines=[f"line {i}"])
            s.vector = v
            sessions.append(s)

        anomalous = gate.filter_anomalous(sessions)
        self.assertGreater(len(anomalous), 0)
        self.assertLess(len(anomalous), len(sessions))


class TestModule1Runner(unittest.TestCase):
    """Tests for the Module 1 end-to-end runner (run_module1)."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.input_file = os.path.join(self.temp_dir, "sample.log")
        with open(self.input_file, "w", encoding="utf-8") as f:
            f.write(SAMPLE_HDFS_LOGS)
        self.csv_path = os.path.join(self.temp_dir, "out_structured.csv")
        self.pkl_path = os.path.join(self.temp_dir, "drain.pkl")

        # SAMPLE_HDFS_LOGS has 10 lines; the first two are identical, so
        # one consecutive duplicate is removed -> 9 kept lines.
        self.raw_lines = [l for l in SAMPLE_HDFS_LOGS.strip().split("\n") if l.strip()]

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_run_module1_outputs_and_summary(self):
        result = run_module1(
            input_path=self.input_file,
            output_csv=self.csv_path,
            output_pkl=self.pkl_path,
        )

        # Output files should all exist.
        json_path = os.path.splitext(self.pkl_path)[0] + ".json"
        self.assertTrue(os.path.exists(self.csv_path))
        self.assertTrue(os.path.exists(self.pkl_path))
        self.assertTrue(os.path.exists(json_path))

        # Dedup counters: 10 read, 1 duplicate removed, 9 kept.
        self.assertEqual(result["total_lines"], len(self.raw_lines))
        self.assertEqual(result["duplicates_removed"], 1)
        self.assertEqual(result["dedup_lines"], len(self.raw_lines) - 1)

        # CSV: header matches the expected columns and has one row per kept line.
        import csv as _csv
        with open(self.csv_path, newline="", encoding="utf-8") as f:
            reader = _csv.reader(f)
            header = next(reader)
            rows = list(reader)
        self.assertEqual(header, CSV_COLUMNS)
        self.assertEqual(len(rows), result["dedup_lines"])

        # JSON summary should contain the documented keys.
        with open(json_path, encoding="utf-8") as f:
            summary = json.load(f)
        for key in ("dataset", "input_file", "total_lines_scanned",
                    "deduplicated_lines", "duplicates_removed",
                    "unique_event_templates", "processing_time_sec",
                    "lines_per_second", "templates"):
            self.assertIn(key, summary)

        # Every kept line maps to a template, so the number of distinct
        # EventIds equals the number of templates Drain discovered.
        self.assertEqual(result["unique_event_ids"], result["template_count"])
        self.assertGreater(result["template_count"], 0)


if __name__ == "__main__":
    unittest.main()

