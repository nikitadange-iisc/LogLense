"""
Module 2 — Comprehensive test suite.

Covers every component of Module 2:
  1.  CSV → event loading        (load_events_from_csv)
  2.  Block-ID sessionization    (HDFS group_by_block_id / create_sessions)
  3.  Sliding-window sessionization (BGL / Thunderbird)
  4.  Vectorization              (build_template_vocabulary, vectorize_session, vectorize_all)
  5.  Vocabulary persistence     (save_vocabulary, load_vocabulary)
  6.  HDFS label loading         (load_labels from anomaly_label.csv)
  7.  BGL/TB label derivation    (assign_labels_from_events)
  8.  AnomalyGate training       (train with/without labels, save=True/False)
  9.  AnomalyGate persistence    (save_model / load_model)
  10. AnomalyGate inference      (predict, score, filter_anomalous, get_gate_statistics)
  11. AnomalyGate evaluation     (evaluate with ground-truth labels)
  12. run_module2() end-to-end   (HDFS, BGL, Thunderbird, edge cases)

Run with:
    python -m pytest tests/test_module2.py -v
    python tests/test_module2.py
"""

import csv
import json
import pickle
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sessionizer import Sessionizer, Session, load_events_from_csv
from anomaly_gate import AnomalyGate
from module2_session_anomaly import run_module2

# ---------------------------------------------------------------------------
# Column layouts (mirrors module1_ingest_parse.DATASET_CSV_COLUMNS)
# ---------------------------------------------------------------------------
_HDFS_COLS = [
    "LineId", "Date", "Time", "Pid", "Level", "Component",
    "Content", "EventId", "EventTemplate", "ParameterList",
]
_BGL_COLS = [
    "LineId", "Label", "Timestamp", "Date", "Node", "Time",
    "NodeRepeat", "Type", "Component", "Level",
    "Content", "EventId", "EventTemplate", "ParameterList",
]
_TB_COLS = [
    "LineId", "Label", "Id", "Date", "Admin", "Time", "AdminAddr",
    "Content", "EventId", "EventTemplate", "ParameterList",
]


# ---------------------------------------------------------------------------
# Helpers — write sample CSVs
# ---------------------------------------------------------------------------

def _write_csv(directory: Path, filename: str, cols: list, rows: list) -> Path:
    path = directory / filename
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    return path


def _hdfs_row(line_id, block_id, eid, content=None, level="INFO"):
    content = content or f"Receiving block {block_id} src: /10.0.0.1:5001"
    params  = str([block_id, "/10.0.0.1:5001"])
    return {
        "LineId": line_id, "Date": "081109", "Time": "203518",
        "Pid": str(line_id), "Level": level,
        "Component": "dfs.DataNode", "Content": content,
        "EventId": eid, "EventTemplate": "Receiving block <BLOCK_ID>",
        "ParameterList": params,
    }


def _bgl_row(line_id, eid, label="-", level="INFO", content=None):
    content = content or f"instruction cache parity error {line_id}"
    return {
        "LineId": line_id, "Label": label,
        "Timestamp": str(1117838570 + line_id),
        "Date": "2005.06.03", "Node": "R02-M1-N0",
        "Time": f"2005-06-03-15.42.{line_id:02d}.000",
        "NodeRepeat": "R02-M1-N0", "Type": "RAS", "Component": "KERNEL",
        "Level": level, "Content": content,
        "EventId": eid, "EventTemplate": "instruction cache parity error <NUM>",
        "ParameterList": f'["{line_id}"]',
    }


def _tb_row(line_id, eid, label="-", level="INFO", content=None):
    content = content or f"kernel normal operation {line_id}"
    return {
        "LineId": line_id, "Label": label,
        "Id": str(1131484800 + line_id),
        "Date": "2005.11.09", "Admin": "tbird-admin1",
        "Time": f"2005-11-09-11.00.{line_id:02d}.000",
        "AdminAddr": "tbird-admin1", "Content": content,
        "EventId": eid, "EventTemplate": "kernel <*> operation <NUM>",
        "ParameterList": f'["{line_id}"]',
    }


def _make_hdfs_csv(directory, blocks=None):
    """
    blocks: list of (block_id, event_ids, label)
    Default: 3 blocks, mix of Normal and Anomaly.
    """
    if blocks is None:
        blocks = [
            ("blk_1001", ["E1", "E2", "E3", "E1"], "Normal"),
            ("blk_1002", ["E1", "E4", "E4", "E4"], "Anomaly"),
            ("blk_1003", ["E1", "E2", "E3", "E3"], "Normal"),
        ]
    rows = []
    lid  = 1
    for bid, eids, _ in blocks:
        for eid in eids:
            rows.append(_hdfs_row(lid, bid, eid))
            lid += 1
    return _write_csv(directory, "hdfs.csv", _HDFS_COLS, rows), blocks


def _make_label_csv(directory, blocks):
    """Write anomaly_label.csv for HDFS blocks."""
    path = directory / "anomaly_label.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        f.write("BlockId,Label\n")
        for bid, _, lbl in blocks:
            f.write(f"{bid},{lbl}\n")
    return path


def _make_bgl_csv(directory, n=30, fatal_positions=None):
    """n events, FATAL at fatal_positions (default: [19])."""
    if fatal_positions is None:
        fatal_positions = {19}
    rows = []
    for i in range(1, n + 1):
        label = "FATAL" if i in fatal_positions else "-"
        level = "FATAL" if label == "FATAL" else "INFO"
        eid   = f"E{(i % 3) + 1}"
        rows.append(_bgl_row(i, eid, label=label, level=level))
    return _write_csv(directory, "bgl.csv", _BGL_COLS, rows)


def _make_tb_csv(directory, n=30, alert_positions=None):
    if alert_positions is None:
        alert_positions = {14}
    rows = []
    for i in range(1, n + 1):
        label = "ALERT" if i in alert_positions else "-"
        eid   = f"E{(i % 2) + 1}"
        rows.append(_tb_row(i, eid, label=label))
    return _write_csv(directory, "tb.csv", _TB_COLS, rows)


# ---------------------------------------------------------------------------
# Helpers — AnomalyGate with pre-trained model
# ---------------------------------------------------------------------------

def _make_trained_gate(tmpdir, n_normal=40, n_dim=4, contamination=0.1,
                       n_estimators=50):
    """Train a gate on tight-cluster normal data; return (gate, normal_vecs)."""
    rng         = np.random.default_rng(42)
    normal_vecs = rng.random((n_normal, n_dim))   # all near [0.5]*n_dim
    mp   = str(Path(tmpdir) / "gate.joblib")
    gate = AnomalyGate(model_path=mp, contamination=contamination,
                       n_estimators=n_estimators, random_state=42)
    gate.train(normal_vecs, save=False)
    return gate, normal_vecs


# ===========================================================================
# 1. CSV → event loading
# ===========================================================================

class TestLoadEventsFromCSV(unittest.TestCase):

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_hdfs_event_count(self):
        csv_path, blocks = _make_hdfs_csv(self.tmpdir)
        events = load_events_from_csv(str(csv_path), "hdfs")
        expected = sum(len(eids) for _, eids, _ in blocks)
        self.assertEqual(len(events), expected)

    def test_event_id_parsed_to_int(self):
        csv_path, _ = _make_hdfs_csv(self.tmpdir)
        events = load_events_from_csv(str(csv_path), "hdfs")
        for e in events:
            self.assertIsInstance(e["event_template_id"], int)
            self.assertGreater(e["event_template_id"], 0)

    def test_parameter_list_parsed_to_list(self):
        csv_path, _ = _make_hdfs_csv(self.tmpdir)
        events = load_events_from_csv(str(csv_path), "hdfs")
        for e in events:
            self.assertIsInstance(e["extracted_variables"], list)

    def test_block_id_in_extracted_variables(self):
        csv_path, blocks = _make_hdfs_csv(self.tmpdir)
        events = load_events_from_csv(str(csv_path), "hdfs")
        block_ids_found = {
            v for e in events for v in e["extracted_variables"]
            if isinstance(v, str) and v.startswith("blk_")
        }
        expected_blocks = {bid for bid, _, _ in blocks}
        self.assertEqual(block_ids_found, expected_blocks)

    def test_raw_line_equals_content(self):
        csv_path, _ = _make_hdfs_csv(self.tmpdir)
        events = load_events_from_csv(str(csv_path), "hdfs")
        for e in events:
            self.assertEqual(e["raw_line"], e["content"])

    def test_line_number_from_lineid(self):
        csv_path, _ = _make_hdfs_csv(self.tmpdir)
        events = load_events_from_csv(str(csv_path), "hdfs")
        line_nums = [e["line_number"] for e in events]
        self.assertEqual(line_nums, list(range(1, len(events) + 1)))

    def test_bgl_label_field_loaded(self):
        csv_path = _make_bgl_csv(self.tmpdir, n=5, fatal_positions={3})
        events = load_events_from_csv(str(csv_path), "bgl")
        self.assertEqual(events[2]["label"], "FATAL")
        self.assertEqual(events[0]["label"], "-")

    def test_thunderbird_label_field_loaded(self):
        csv_path = _make_tb_csv(self.tmpdir, n=5, alert_positions={2})
        events = load_events_from_csv(str(csv_path), "thunderbird")
        self.assertEqual(events[1]["label"], "ALERT")
        self.assertEqual(events[0]["label"], "-")

    def test_malformed_parameter_list_falls_back(self):
        rows = [_hdfs_row(1, "blk_9999", "E1")]
        rows[0]["ParameterList"] = "NOT_A_LIST"
        path = _write_csv(self.tmpdir, "bad.csv", _HDFS_COLS, rows)
        events = load_events_from_csv(str(path), "hdfs")
        self.assertEqual(events[0]["extracted_variables"], [])

    def test_extra_columns_carried_through(self):
        csv_path, _ = _make_hdfs_csv(self.tmpdir)
        events = load_events_from_csv(str(csv_path), "hdfs")
        self.assertIn("component", events[0])
        self.assertEqual(events[0]["component"], "dfs.DataNode")


# ===========================================================================
# 2. Block-ID Sessionization (HDFS)
# ===========================================================================

class TestBlockIDSessionization(unittest.TestCase):

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.csv_path, self.blocks = _make_hdfs_csv(self.tmpdir)
        self.events = load_events_from_csv(str(self.csv_path), "hdfs")
        self.sz = Sessionizer(method="block_id")

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_session_count_equals_block_count(self):
        sessions = self.sz.create_sessions(iter(self.events))
        self.assertEqual(len(sessions), len(self.blocks))

    def test_session_ids_are_block_ids(self):
        sessions = self.sz.create_sessions(iter(self.events))
        expected = {bid for bid, _, _ in self.blocks}
        actual   = {s.session_id for s in sessions}
        self.assertEqual(actual, expected)

    def test_events_grouped_correctly(self):
        sessions = self.sz.create_sessions(iter(self.events))
        sid_map  = {s.session_id: s for s in sessions}
        # blk_1001 has 4 events
        self.assertEqual(len(sid_map["blk_1001"].events), 4)

    def test_raw_lines_populated(self):
        sessions = self.sz.create_sessions(iter(self.events))
        for s in sessions:
            self.assertEqual(len(s.raw_lines), len(s.events))
            self.assertTrue(all(isinstance(r, str) for r in s.raw_lines))

    def test_line_range_is_tuple(self):
        sessions = self.sz.create_sessions(iter(self.events))
        for s in sessions:
            self.assertEqual(len(s.line_range), 2)
            self.assertLessEqual(s.line_range[0], s.line_range[1])

    def test_events_with_no_block_id_skipped(self):
        rows = [_hdfs_row(1, "blk_1001", "E1")]
        rows.append({
            "LineId": 2, "Date": "081109", "Time": "203518",
            "Pid": "2", "Level": "INFO", "Component": "dfs",
            "Content": "some unrelated log line",
            "EventId": "E2", "EventTemplate": "some unrelated log",
            "ParameterList": "[]",
        })
        path   = _write_csv(self.tmpdir, "mixed.csv", _HDFS_COLS, rows)
        events = load_events_from_csv(str(path), "hdfs")
        sz     = Sessionizer(method="block_id")
        sessions = sz.create_sessions(iter(events))
        # Only blk_1001 should produce a session
        self.assertEqual(len(sessions), 1)
        self.assertEqual(sessions[0].session_id, "blk_1001")

    def test_extract_block_id_from_raw_line_fallback(self):
        """Block ID found via regex on raw_line when ParameterList is empty."""
        rows = [_hdfs_row(1, "blk_9999", "E1")]
        rows[0]["ParameterList"] = "[]"   # empty — no blk_ in extracted_variables
        rows[0]["Content"] = "Received block blk_9999 of size 91178"
        path   = _write_csv(self.tmpdir, "fallback.csv", _HDFS_COLS, rows)
        events = load_events_from_csv(str(path), "fallback")
        sz     = Sessionizer(method="block_id")
        sessions = sz.create_sessions(iter(events))
        self.assertEqual(len(sessions), 1)
        self.assertEqual(sessions[0].session_id, "blk_9999")


# ===========================================================================
# 3. Sliding-Window Sessionization (BGL / Thunderbird)
# ===========================================================================

class TestSlidingWindowSessionization(unittest.TestCase):

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def _sessions(self, n, window_size, step_size, fatal_positions=None):
        csv_path = _make_bgl_csv(self.tmpdir, n=n,
                                  fatal_positions=fatal_positions or set())
        events = load_events_from_csv(str(csv_path), "bgl")
        sz = Sessionizer(method="sliding_window",
                         window_size=window_size, step_size=step_size)
        return sz.create_sessions(iter(events)), events

    def test_window_count_formula(self):
        # 30 events, window=10, step=5 → range(0,21,5) → 5 windows
        sessions, _ = self._sessions(30, 10, 5)
        self.assertEqual(len(sessions), 5)

    def test_window_size_respected(self):
        sessions, _ = self._sessions(30, 10, 5)
        for s in sessions:
            self.assertEqual(len(s.events), 10)

    def test_session_id_format(self):
        sessions, _ = self._sessions(20, 10, 10)
        for s in sessions:
            self.assertTrue(s.session_id.startswith("window_"))

    def test_line_range_monotonic(self):
        sessions, _ = self._sessions(30, 10, 5)
        for s in sessions:
            self.assertLessEqual(s.line_range[0], s.line_range[1])

    def test_raw_lines_length_matches_events(self):
        sessions, _ = self._sessions(20, 10, 5)
        for s in sessions:
            self.assertEqual(len(s.raw_lines), len(s.events))

    def test_partial_last_window_excluded(self):
        # 25 events, window=10, step=10 → range(0,16,10) → [0,10] → 2 full windows
        sessions, _ = self._sessions(25, 10, 10)
        self.assertEqual(len(sessions), 2)

    def test_default_window_is_20(self):
        sz = Sessionizer(method="sliding_window")
        self.assertEqual(sz.window_size, 20)

    def test_default_step_is_10(self):
        sz = Sessionizer(method="sliding_window")
        self.assertEqual(sz.step_size, 10)

    def test_thunderbird_sliding_window(self):
        csv_path = _make_tb_csv(self.tmpdir, n=30)
        events   = load_events_from_csv(str(csv_path), "thunderbird")
        sz       = Sessionizer(method="sliding_window", window_size=10, step_size=5)
        sessions = sz.create_sessions(iter(events))
        self.assertEqual(len(sessions), 5)

    def test_invalid_method_raises(self):
        sz = Sessionizer(method="unknown_method")
        with self.assertRaises(ValueError):
            sz.create_sessions(iter([]))


# ===========================================================================
# 4. Vectorization
# ===========================================================================

class TestVectorization(unittest.TestCase):

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        # 3 blocks: blk_1001 has E1,E2; blk_1002 has E1,E3; blk_1003 has E2,E2
        blocks = [
            ("blk_1001", ["E1", "E2"], "Normal"),
            ("blk_1002", ["E1", "E3"], "Normal"),
            ("blk_1003", ["E2", "E2"], "Normal"),
        ]
        self.csv_path, _ = _make_hdfs_csv(self.tmpdir, blocks=blocks)
        events = load_events_from_csv(str(self.csv_path), "hdfs")
        sz = Sessionizer(method="block_id")
        self.sessions = sz.create_sessions(iter(events))
        self.sz = sz

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_vocabulary_size(self):
        vocab = self.sz.build_template_vocabulary(self.sessions)
        self.assertEqual(len(vocab), 3)   # E1, E2, E3 → template IDs 1,2,3

    def test_vocabulary_keys_are_ints(self):
        vocab = self.sz.build_template_vocabulary(self.sessions)
        for k in vocab:
            self.assertIsInstance(k, int)

    def test_vocabulary_values_are_sequential_indices(self):
        vocab = self.sz.build_template_vocabulary(self.sessions)
        indices = sorted(vocab.values())
        self.assertEqual(indices, list(range(len(vocab))))

    def test_vector_dimension_matches_vocabulary(self):
        sessions, vocab = self.sz.vectorize_all(self.sessions)
        for s in sessions:
            self.assertEqual(s.vector.shape[0], len(vocab))

    def test_vector_counts_correct(self):
        sessions, vocab = self.sz.vectorize_all(self.sessions)
        sid_map = {s.session_id: s for s in sessions}
        # blk_1003 has E2,E2 → index of template 2 should be 2.0
        idx_of_2 = vocab[2]
        self.assertEqual(sid_map["blk_1003"].vector[idx_of_2], 2.0)

    def test_vector_dtype_float32(self):
        sessions, _ = self.sz.vectorize_all(self.sessions)
        for s in sessions:
            self.assertEqual(s.vector.dtype, np.float32)

    def test_vectorize_all_returns_same_sessions(self):
        sessions_out, vocab = self.sz.vectorize_all(self.sessions)
        self.assertIs(sessions_out, self.sessions)

    def test_prebuilt_vocabulary_accepted(self):
        vocab = {1: 0, 2: 1, 3: 2, 99: 3}   # extra unseen template 99
        sessions, returned_vocab = self.sz.vectorize_all(self.sessions, vocabulary=vocab)
        self.assertIs(returned_vocab, vocab)
        for s in sessions:
            self.assertEqual(s.vector.shape[0], 4)   # uses provided dim


# ===========================================================================
# 5. Vocabulary Persistence
# ===========================================================================

class TestVocabularyPersistence(unittest.TestCase):

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.sz     = Sessionizer()

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_save_creates_file(self):
        vocab = {1: 0, 2: 1, 3: 2}
        path  = str(self.tmpdir / "vocab.json")
        self.sz.save_vocabulary(vocab, path)
        self.assertTrue(Path(path).exists())

    def test_saved_file_is_valid_json(self):
        vocab = {1: 0, 2: 1}
        path  = str(self.tmpdir / "vocab.json")
        self.sz.save_vocabulary(vocab, path)
        with open(path) as f:
            data = json.load(f)
        self.assertIn("event_id_to_index", data)
        self.assertIn("template_count", data)

    def test_template_count_in_json(self):
        vocab = {1: 0, 2: 1, 5: 2}
        path  = str(self.tmpdir / "vocab.json")
        self.sz.save_vocabulary(vocab, path)
        with open(path) as f:
            data = json.load(f)
        self.assertEqual(data["template_count"], 3)

    def test_round_trip_exact(self):
        vocab = {1: 0, 3: 1, 7: 2}
        path  = str(self.tmpdir / "vocab.json")
        self.sz.save_vocabulary(vocab, path)
        loaded, _ = self.sz.load_vocabulary(path)
        self.assertEqual(vocab, loaded)

    def test_loaded_keys_are_ints(self):
        vocab = {4: 0, 9: 1}
        path  = str(self.tmpdir / "vocab.json")
        self.sz.save_vocabulary(vocab, path)
        loaded, _ = self.sz.load_vocabulary(path)
        for k in loaded:
            self.assertIsInstance(k, int)


# ===========================================================================
# 6. HDFS Label Loading
# ===========================================================================

class TestHDFSLabelLoading(unittest.TestCase):

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.csv_path, self.blocks = _make_hdfs_csv(self.tmpdir)
        events        = load_events_from_csv(str(self.csv_path), "hdfs")
        sz            = Sessionizer(method="block_id")
        self.sessions = sz.create_sessions(iter(events))
        self.sz       = sz
        self.label_csv = _make_label_csv(self.tmpdir, self.blocks)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_all_sessions_labeled(self):
        sessions = self.sz.load_labels(str(self.label_csv), self.sessions)
        for s in sessions:
            self.assertIsNotNone(s.label)

    def test_normal_label_correct(self):
        sessions = self.sz.load_labels(str(self.label_csv), self.sessions)
        sid_map  = {s.session_id: s for s in sessions}
        self.assertEqual(sid_map["blk_1001"].label, "Normal")
        self.assertEqual(sid_map["blk_1003"].label, "Normal")

    def test_anomaly_label_correct(self):
        sessions = self.sz.load_labels(str(self.label_csv), self.sessions)
        sid_map  = {s.session_id: s for s in sessions}
        self.assertEqual(sid_map["blk_1002"].label, "Anomaly")

    def test_unmatched_session_stays_unlabeled(self):
        # Add extra session that has no entry in label CSV
        extra = Session(session_id="blk_9999", events=[], label=None)
        sessions = self.sz.load_labels(str(self.label_csv),
                                        self.sessions + [extra])
        unmatched = [s for s in sessions if s.session_id == "blk_9999"]
        self.assertEqual(unmatched[0].label, None)


# ===========================================================================
# 7. BGL / Thunderbird Label Derivation
# ===========================================================================

class TestAssignLabelsFromEvents(unittest.TestCase):

    def setUp(self):
        self.sz = Sessionizer()

    def _make_sessions(self, event_label_lists):
        """event_label_lists: list of lists of label strings per session."""
        sessions = []
        for i, labels in enumerate(event_label_lists):
            events = [{"label": lbl} for lbl in labels]
            sessions.append(Session(session_id=f"w{i}", events=events))
        return sessions

    def test_all_normal_events_gives_normal_session(self):
        sessions = self._make_sessions([["-", "-", "-"]])
        sessions = self.sz.assign_labels_from_events(sessions)
        self.assertEqual(sessions[0].label, "Normal")

    def test_one_fatal_gives_anomaly_session(self):
        sessions = self._make_sessions([["-", "-", "FATAL"]])
        sessions = self.sz.assign_labels_from_events(sessions)
        self.assertEqual(sessions[0].label, "Anomaly")

    def test_all_fatal_gives_anomaly_session(self):
        sessions = self._make_sessions([["FATAL", "FATAL"]])
        sessions = self.sz.assign_labels_from_events(sessions)
        self.assertEqual(sessions[0].label, "Anomaly")

    def test_alert_gives_anomaly_session(self):
        sessions = self._make_sessions([["-", "ALERT", "-"]])
        sessions = self.sz.assign_labels_from_events(sessions)
        self.assertEqual(sessions[0].label, "Anomaly")

    def test_multiple_sessions_labeled_independently(self):
        sessions = self._make_sessions([
            ["-", "-"],       # Normal
            ["-", "FATAL"],   # Anomaly
            ["-", "-"],       # Normal
        ])
        sessions = self.sz.assign_labels_from_events(sessions)
        self.assertEqual(sessions[0].label, "Normal")
        self.assertEqual(sessions[1].label, "Anomaly")
        self.assertEqual(sessions[2].label, "Normal")

    def test_empty_events_session_is_normal(self):
        sessions = [Session(session_id="w0", events=[])]
        sessions = self.sz.assign_labels_from_events(sessions)
        self.assertEqual(sessions[0].label, "Normal")

    def test_from_bgl_csv_end_to_end(self):
        tmpdir = Path(tempfile.mkdtemp())
        try:
            csv_path = _make_bgl_csv(tmpdir, n=30, fatal_positions={15})
            events   = load_events_from_csv(str(csv_path), "bgl")
            sz       = Sessionizer(method="sliding_window", window_size=10, step_size=10)
            sessions = sz.create_sessions(iter(events))
            sessions = sz.assign_labels_from_events(sessions)
            labels   = [s.label for s in sessions]
            self.assertIn("Anomaly", labels)
            self.assertIn("Normal",  labels)
        finally:
            shutil.rmtree(tmpdir)


# ===========================================================================
# 8. AnomalyGate — Training
# ===========================================================================

class TestAnomalyGateTraining(unittest.TestCase):

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.mp     = str(self.tmpdir / "gate.joblib")
        rng = np.random.default_rng(42)
        self.normal_vecs = rng.random((30, 5)).astype(np.float32)
        self.all_vecs    = np.vstack([
            self.normal_vecs,
            rng.random((3, 5)).astype(np.float32) + 10,  # outliers
        ])

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_train_normal_only_fits_model(self):
        gate = AnomalyGate(model_path=self.mp, contamination=0.03, n_estimators=50)
        gate.train(self.normal_vecs, save=False)
        self.assertIsNotNone(gate.model)

    def test_train_all_vectors_fits_model(self):
        gate = AnomalyGate(model_path=self.mp, contamination=0.1, n_estimators=50)
        gate.train(self.normal_vecs, all_vectors=self.all_vecs, save=False)
        self.assertIsNotNone(gate.model)

    def test_save_false_does_not_write_file(self):
        gate = AnomalyGate(model_path=self.mp, n_estimators=50)
        gate.train(self.normal_vecs, save=False)
        self.assertFalse(Path(self.mp).exists())

    def test_save_true_writes_file(self):
        gate = AnomalyGate(model_path=self.mp, n_estimators=50)
        gate.train(self.normal_vecs, save=True)
        self.assertTrue(Path(self.mp).exists())
        self.assertGreater(Path(self.mp).stat().st_size, 0)

    def test_default_contamination_is_0_03(self):
        gate = AnomalyGate(model_path=self.mp)
        self.assertEqual(gate.contamination, 0.03)

    def test_predict_before_train_raises(self):
        gate = AnomalyGate(model_path=self.mp)
        with self.assertRaises(RuntimeError):
            gate.predict(self.normal_vecs)

    def test_score_before_train_raises(self):
        gate = AnomalyGate(model_path=self.mp)
        with self.assertRaises(RuntimeError):
            gate.score(self.normal_vecs)


# ===========================================================================
# 9. AnomalyGate — Persistence
# ===========================================================================

class TestAnomalyGatePersistence(unittest.TestCase):

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.mp     = str(self.tmpdir / "gate.joblib")
        rng = np.random.default_rng(0)
        self.vecs = rng.random((20, 4)).astype(np.float32)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_save_and_load_round_trip(self):
        gate = AnomalyGate(model_path=self.mp, n_estimators=50)
        gate.train(self.vecs, save=True)
        preds_before = gate.predict(self.vecs)

        gate2 = AnomalyGate(model_path=self.mp, n_estimators=50)
        gate2.load_model()
        preds_after = gate2.predict(self.vecs)

        np.testing.assert_array_equal(preds_before, preds_after)

    def test_load_missing_file_raises(self):
        gate = AnomalyGate(model_path=str(self.tmpdir / "nonexistent.joblib"))
        with self.assertRaises(FileNotFoundError):
            gate.load_model()

    def test_custom_save_path(self):
        custom = str(self.tmpdir / "sub" / "custom.joblib")
        gate   = AnomalyGate(model_path=self.mp, n_estimators=50)
        gate.train(self.vecs, save=False)
        gate.save_model(custom)
        self.assertTrue(Path(custom).exists())


# ===========================================================================
# 10. AnomalyGate — Inference
# ===========================================================================

class TestAnomalyGateInference(unittest.TestCase):

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        rng = np.random.default_rng(7)
        # Tight cluster of normal vectors near zero
        normal = rng.random((50, 4)).astype(np.float32) * 0.1
        self.gate, _ = _make_trained_gate(str(self.tmpdir), n_normal=50,
                                           n_dim=4, contamination=0.05,
                                           n_estimators=100)
        self.normal_vec  = np.array([[0.05, 0.05, 0.05, 0.05]], dtype=np.float32)
        self.outlier_vec = np.array([[100., 100., 100., 100.]], dtype=np.float32)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_predict_returns_plus_or_minus_one(self):
        vecs  = np.vstack([self.normal_vec] * 5 + [self.outlier_vec])
        preds = self.gate.predict(vecs)
        self.assertTrue(set(preds).issubset({1, -1}))

    def test_predict_output_shape(self):
        vecs  = np.vstack([self.normal_vec] * 3)
        preds = self.gate.predict(vecs)
        self.assertEqual(preds.shape, (3,))

    def test_score_returns_floats(self):
        vecs   = np.vstack([self.normal_vec] * 3 + [self.outlier_vec])
        scores = self.gate.score(vecs)
        self.assertEqual(scores.dtype.kind, "f")

    def test_extreme_outlier_flagged(self):
        preds = self.gate.predict(self.outlier_vec)
        self.assertEqual(preds[0], -1, "Extreme outlier must be flagged as anomaly")

    def test_filter_anomalous_empty_list(self):
        result = self.gate.filter_anomalous([])
        self.assertEqual(result, [])

    def test_filter_anomalous_contains_outlier_session(self):
        s_normal  = [Session(session_id=f"n{i}", events=[],
                             vector=np.array([0.05]*4, dtype=np.float32))
                     for i in range(10)]
        s_outlier = Session(session_id="outlier", events=[],
                            vector=np.array([100.]*4, dtype=np.float32))
        anomalous = self.gate.filter_anomalous(s_normal + [s_outlier])
        self.assertIn(s_outlier, anomalous)

    def test_filter_anomalous_excludes_normal_sessions(self):
        # Use [0.5]*4 — the center of the uniform [0,1] training distribution —
        # so these sessions are in the highest-density region and won't be flagged.
        s_normal = [Session(session_id=f"n{i}", events=[],
                            vector=np.array([0.5]*4, dtype=np.float32))
                    for i in range(10)]
        s_outlier = Session(session_id="outlier", events=[],
                            vector=np.array([100.]*4, dtype=np.float32))
        anomalous = self.gate.filter_anomalous(s_normal + [s_outlier])
        # Outlier must be flagged; normal sessions at distribution center must not be.
        self.assertIn(s_outlier, anomalous)
        non_outlier_flagged = [s for s in anomalous if s.session_id != "outlier"]
        self.assertEqual(non_outlier_flagged, [],
                         "Center-of-distribution sessions should not be flagged")

    def test_get_gate_statistics_keys(self):
        sessions = [Session(session_id=f"s{i}", events=[],
                            vector=np.array([0.05]*4, dtype=np.float32))
                    for i in range(5)]
        stats = self.gate.get_gate_statistics(sessions)
        for key in ("total_sessions", "normal_count",
                    "anomalous_count", "anomaly_percentage"):
            self.assertIn(key, stats)

    def test_get_gate_statistics_totals_consistent(self):
        sessions = [Session(session_id=f"s{i}", events=[],
                            vector=np.array([0.05]*4, dtype=np.float32))
                    for i in range(8)]
        stats = self.gate.get_gate_statistics(sessions)
        self.assertEqual(stats["normal_count"] + stats["anomalous_count"],
                         stats["total_sessions"])

    def test_get_gate_statistics_empty(self):
        stats = self.gate.get_gate_statistics([])
        self.assertEqual(stats["total_sessions"], 0)
        self.assertEqual(stats["anomaly_percentage"], 0.0)


# ===========================================================================
# 11. AnomalyGate — Evaluation
# ===========================================================================

class TestAnomalyGateEvaluation(unittest.TestCase):

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        # Train on tight cluster of normal vectors
        rng          = np.random.default_rng(0)
        normal_vecs  = rng.random((50, 4)).astype(np.float32) * 0.05
        self.gate, _ = _make_trained_gate(str(self.tmpdir), n_normal=50,
                                           n_dim=4, contamination=0.05,
                                           n_estimators=100)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def _labeled_sessions(self, n_normal=8, n_anomaly=2):
        normal = [Session(session_id=f"n{i}", events=[], label="Normal",
                          vector=np.array([0.05]*4, dtype=np.float32))
                  for i in range(n_normal)]
        anomalous = [Session(session_id=f"a{i}", events=[], label="Anomaly",
                             vector=np.array([100.]*4, dtype=np.float32))
                     for i in range(n_anomaly)]
        return normal + anomalous

    def test_evaluate_returns_dict_with_required_keys(self):
        sessions = self._labeled_sessions()
        metrics  = self.gate.evaluate(sessions)
        for key in ("precision", "recall", "f1_score", "accuracy",
                    "true_positives", "false_positives",
                    "false_negatives", "true_negatives", "total_labeled"):
            self.assertIn(key, metrics)

    def test_evaluate_total_labeled_count(self):
        sessions = self._labeled_sessions(n_normal=8, n_anomaly=2)
        metrics  = self.gate.evaluate(sessions)
        self.assertEqual(metrics["total_labeled"], 10)

    def test_evaluate_confusion_matrix_adds_up(self):
        sessions = self._labeled_sessions()
        metrics  = self.gate.evaluate(sessions)
        total = (metrics["true_positives"]  + metrics["false_positives"] +
                 metrics["false_negatives"] + metrics["true_negatives"])
        self.assertEqual(total, metrics["total_labeled"])

    def test_evaluate_perfect_detection(self):
        """Extreme outlier should always be detected → TP > 0, recall > 0."""
        sessions = self._labeled_sessions(n_normal=10, n_anomaly=2)
        metrics  = self.gate.evaluate(sessions)
        self.assertGreater(metrics["true_positives"], 0)
        self.assertGreater(metrics["recall"], 0.0)

    def test_evaluate_no_labels_returns_empty(self):
        sessions = [Session(session_id="x", events=[],
                            vector=np.array([0.05]*4, dtype=np.float32))]
        metrics  = self.gate.evaluate(sessions)
        self.assertEqual(metrics, {})

    def test_evaluate_metrics_in_valid_range(self):
        sessions = self._labeled_sessions()
        metrics  = self.gate.evaluate(sessions)
        for key in ("precision", "recall", "f1_score", "accuracy"):
            self.assertGreaterEqual(metrics[key], 0.0)
            self.assertLessEqual(metrics[key],    1.0)


# ===========================================================================
# 12. run_module2() — End-to-End
# ===========================================================================

class TestRunModule2EndToEnd(unittest.TestCase):

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def _run(self, csv_path, dataset, **kwargs):
        defaults = dict(
            output_json=str(self.tmpdir / "anomalies.json"),
            vocab_path =str(self.tmpdir / "vocab.json"),
            model_path =str(self.tmpdir / "gate.joblib"),
        )
        defaults.update(kwargs)
        return run_module2(csv_path=str(csv_path), dataset=dataset, **defaults)

    # -- HDFS --

    def test_hdfs_output_files_created(self):
        csv_path, blocks = _make_hdfs_csv(self.tmpdir)
        self._run(csv_path, "hdfs")
        self.assertTrue((self.tmpdir / "anomalies.json").exists())
        self.assertTrue((self.tmpdir / "vocab.json").exists())
        self.assertTrue((self.tmpdir / "gate.joblib").exists())

    def test_hdfs_result_keys(self):
        csv_path, _ = _make_hdfs_csv(self.tmpdir)
        result = self._run(csv_path, "hdfs")
        for key in ("total_events", "total_sessions", "anomalous_sessions",
                    "vocabulary_size", "output_json", "vocab_path",
                    "model_path", "dataset", "evaluation", "sessions",
                    "anomalous", "vocabulary", "gate", "processing_time"):
            self.assertIn(key, result)

    def test_hdfs_total_events_count(self):
        csv_path, blocks = _make_hdfs_csv(self.tmpdir)
        result = self._run(csv_path, "hdfs")
        expected = sum(len(eids) for _, eids, _ in blocks)
        self.assertEqual(result["total_events"], expected)

    def test_hdfs_total_sessions_equals_block_count(self):
        csv_path, blocks = _make_hdfs_csv(self.tmpdir)
        result = self._run(csv_path, "hdfs")
        self.assertEqual(result["total_sessions"], len(blocks))

    def test_hdfs_vocabulary_size_correct(self):
        blocks = [
            ("blk_1001", ["E1", "E2"], "Normal"),
            ("blk_1002", ["E3"],       "Anomaly"),
        ]
        csv_path, _ = _make_hdfs_csv(self.tmpdir, blocks=blocks)
        result = self._run(csv_path, "hdfs")
        self.assertEqual(result["vocabulary_size"], 3)   # E1, E2, E3

    def test_hdfs_with_label_path_evaluation_populated(self):
        csv_path, blocks = _make_hdfs_csv(self.tmpdir)
        label_csv = _make_label_csv(self.tmpdir, blocks)
        result = self._run(csv_path, "hdfs", label_path=str(label_csv))
        self.assertIn("precision", result["evaluation"])
        self.assertIn("recall",    result["evaluation"])
        self.assertIn("f1_score",  result["evaluation"])

    def test_hdfs_anomalies_json_structure(self):
        csv_path, _ = _make_hdfs_csv(self.tmpdir)
        self._run(csv_path, "hdfs")
        with open(self.tmpdir / "anomalies.json") as f:
            data = json.load(f)
        for key in ("dataset", "total_events", "total_sessions",
                    "anomalous_sessions", "vocabulary_size", "sessions"):
            self.assertIn(key, data)

    # -- BGL --

    def test_bgl_output_files_created(self):
        csv_path = _make_bgl_csv(self.tmpdir, n=30, fatal_positions={15})
        self._run(csv_path, "bgl", window_size=10, step_size=5)
        self.assertTrue((self.tmpdir / "anomalies.json").exists())
        self.assertTrue((self.tmpdir / "vocab.json").exists())

    def test_bgl_evaluation_populated_without_label_file(self):
        """BGL labels are derived from events — no separate label file needed."""
        csv_path = _make_bgl_csv(self.tmpdir, n=30, fatal_positions={15})
        result   = self._run(csv_path, "bgl", window_size=10, step_size=5)
        self.assertNotEqual(result["evaluation"], {})
        self.assertIn("precision", result["evaluation"])

    def test_bgl_session_count(self):
        # 30 events, all with the same node "R02-M1-N0" → 1 node-based session
        csv_path = _make_bgl_csv(self.tmpdir, n=30)
        result   = self._run(csv_path, "bgl", window_size=10, step_size=5)
        self.assertEqual(result["total_sessions"], 1)

    def test_bgl_dataset_in_result(self):
        csv_path = _make_bgl_csv(self.tmpdir, n=30)
        result   = self._run(csv_path, "bgl", window_size=10, step_size=5)
        self.assertEqual(result["dataset"], "bgl")

    # -- Thunderbird --

    def test_thunderbird_output_created(self):
        csv_path = _make_tb_csv(self.tmpdir, n=30, alert_positions={14})
        self._run(csv_path, "thunderbird", window_size=10, step_size=5)
        self.assertTrue((self.tmpdir / "anomalies.json").exists())

    def test_thunderbird_evaluation_populated(self):
        csv_path = _make_tb_csv(self.tmpdir, n=30, alert_positions={14})
        result   = self._run(csv_path, "thunderbird", window_size=10, step_size=5)
        self.assertNotEqual(result["evaluation"], {})

    # -- Edge cases --

    def test_max_sessions_caps_count(self):
        csv_path, blocks = _make_hdfs_csv(self.tmpdir)
        result = self._run(csv_path, "hdfs", max_sessions=1)
        self.assertEqual(result["total_sessions"], 1)

    def test_missing_csv_raises(self):
        with self.assertRaises(FileNotFoundError):
            self._run(self.tmpdir / "nonexistent.csv", "hdfs")

    def test_train_false_loads_existing_model(self):
        """Second run with train=False must load the model saved in the first run."""
        csv_path, _ = _make_hdfs_csv(self.tmpdir)
        self._run(csv_path, "hdfs")                        # train=True (default)
        result2 = self._run(csv_path, "hdfs", train=False) # load only
        self.assertIsNotNone(result2["gate"].model)

    def test_vocab_json_round_trip(self):
        csv_path, _ = _make_hdfs_csv(self.tmpdir)
        result = self._run(csv_path, "hdfs")
        with open(self.tmpdir / "vocab.json") as f:
            data = json.load(f)
        self.assertEqual(data["template_count"], result["vocabulary_size"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
