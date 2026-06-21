"""
Tests for Module 3 — Session Embedding & FAISS Indexing
========================================================
Covers: JSON loader, manual file loaders, FAISSVectorStore.reset(),
SessionEmbedder compatibility with minimal event dicts,
and end-to-end run_module3() including manual ingestion.
All tests use TF-IDF fallback (no sentence-transformers download needed).
"""

import json
import pickle
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

# ── resolve src/ on the path ─────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from module3_embed_index import (
    _make_session,
    _event_dicts_from_ids,
    _build_metadata,
    load_sessions_from_anomaly_json,
    load_manual_sessions,
    run_module3,
)
from embedder import SessionEmbedder
from vector_store import FAISSVectorStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_anomaly_json(path: Path, n: int = 4, dataset: str = "hdfs") -> dict:
    """Write a minimal anomalies.json and return the payload."""
    sessions = []
    for i in range(n):
        label = "Anomaly"
        sessions.append({
            "session_id":    f"blk_{1000 + i}",
            "line_range":    [i * 10 + 1, i * 10 + 10],
            "label":         label,
            "anomaly_score": -(0.05 + i * 0.01),
            "event_sequence": ["E1", "E2", "E3"],
            "raw_lines":     [f"Log line {j} for session {i}" for j in range(5)],
        })
    payload = {
        "dataset":            dataset,
        "total_sessions":     100,
        "anomalous_sessions": n,
        "sessions":           sessions,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _make_tfidf_embedder(dim: int = 64) -> SessionEmbedder:
    """Return a SessionEmbedder forced into TF-IDF fallback mode (no network)."""
    with patch("builtins.__import__", side_effect=ImportError):
        try:
            emb = SessionEmbedder.__new__(SessionEmbedder)
        except Exception:
            pass
    emb = SessionEmbedder.__new__(SessionEmbedder)
    emb._use_tfidf  = True
    emb.model_name  = "tfidf-svd-64"
    emb.dimension   = dim
    emb.model       = None
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    emb._tfidf   = TfidfVectorizer(max_features=500, sublinear_tf=True)
    emb._svd     = TruncatedSVD(n_components=dim, random_state=42)
    emb._fitted  = False
    return emb


# ---------------------------------------------------------------------------
# 1. load_sessions_from_anomaly_json
# ---------------------------------------------------------------------------

class TestLoadSessionsFromJson(unittest.TestCase):

    def setUp(self):
        import tempfile
        self.tmp = Path(tempfile.mkdtemp())

    def test_correct_count(self):
        path = self.tmp / "test.json"
        _write_anomaly_json(path, n=4)
        sessions, _ = load_sessions_from_anomaly_json(str(path))
        self.assertEqual(len(sessions), 4)

    def test_fields_populated(self):
        path = self.tmp / "test.json"
        _write_anomaly_json(path, n=2)
        sessions, payload = load_sessions_from_anomaly_json(str(path))
        s = sessions[0]
        self.assertEqual(s.session_id, "blk_1000")
        self.assertEqual(s.label, "Anomaly")
        self.assertIsNotNone(s.anomaly_score)
        self.assertEqual(len(s.raw_lines), 5)
        self.assertEqual(s.line_range, (1, 10))

    def test_events_are_dicts(self):
        path = self.tmp / "test.json"
        _write_anomaly_json(path, n=1)
        sessions, _ = load_sessions_from_anomaly_json(str(path))
        for ev in sessions[0].events:
            self.assertIsInstance(ev, dict)
            self.assertIn("event_template", ev)
            self.assertIn("level", ev)

    def test_empty_sessions_list(self):
        path = self.tmp / "empty.json"
        path.write_text(json.dumps({"sessions": []}), encoding="utf-8")
        sessions, _ = load_sessions_from_anomaly_json(str(path))
        self.assertEqual(sessions, [])

    def test_null_anomaly_score(self):
        path = self.tmp / "null.json"
        path.write_text(json.dumps({"sessions": [{
            "session_id": "blk_x", "line_range": [1, 5],
            "label": "Anomaly", "anomaly_score": None,
            "event_sequence": ["E1"], "raw_lines": ["line 1"],
        }]}), encoding="utf-8")
        sessions, _ = load_sessions_from_anomaly_json(str(path))
        self.assertIsNone(sessions[0].anomaly_score)

    def test_payload_metadata_returned(self):
        path = self.tmp / "meta.json"
        _write_anomaly_json(path, n=3, dataset="bgl")
        _, payload = load_sessions_from_anomaly_json(str(path))
        self.assertEqual(payload["dataset"], "bgl")
        self.assertEqual(payload["total_sessions"], 100)


# ---------------------------------------------------------------------------
# 2. load_manual_sessions
# ---------------------------------------------------------------------------

class TestLoadManualSessions(unittest.TestCase):

    def setUp(self):
        import tempfile
        self.tmp = Path(tempfile.mkdtemp())

    # ── JSON format ──────────────────────────────────────────────────────
    def test_manual_json_loads(self):
        path = self.tmp / "manual.json"
        _write_anomaly_json(path, n=3)
        sessions = load_manual_sessions(str(path), "hdfs", "manual")
        self.assertEqual(len(sessions), 3)

    def test_manual_json_prefixes_ids(self):
        path = self.tmp / "manual.json"
        _write_anomaly_json(path, n=2)
        sessions = load_manual_sessions(str(path), "hdfs", "manual")
        for s in sessions:
            self.assertTrue(s.session_id.startswith("manual_"))

    def test_manual_json_sets_anomaly_label(self):
        path = self.tmp / "manual.json"
        _write_anomaly_json(path, n=2)
        sessions = load_manual_sessions(str(path), "hdfs", "manual")
        for s in sessions:
            self.assertIsNotNone(s.label)

    # ── Plain text format ────────────────────────────────────────────────
    def test_manual_txt_small_file_one_session(self):
        path = self.tmp / "small.txt"
        path.write_text("\n".join([f"Log line {i}" for i in range(10)]),
                        encoding="utf-8")
        sessions = load_manual_sessions(str(path), "hdfs")
        self.assertEqual(len(sessions), 1)
        self.assertEqual(len(sessions[0].raw_lines), 10)

    def test_manual_txt_large_file_windowed(self):
        path = self.tmp / "large.txt"
        path.write_text("\n".join([f"Log line {i}" for i in range(130)]),
                        encoding="utf-8")
        sessions = load_manual_sessions(str(path), "hdfs")
        # 130 lines / 50-window = 3 sessions
        self.assertEqual(len(sessions), 3)

    def test_manual_txt_empty_file(self):
        path = self.tmp / "empty.txt"
        path.write_text("", encoding="utf-8")
        sessions = load_manual_sessions(str(path), "hdfs")
        self.assertEqual(sessions, [])

    def test_manual_txt_label_is_anomaly(self):
        path = self.tmp / "lines.txt"
        path.write_text("Error in disk\nKernel panic\n", encoding="utf-8")
        sessions = load_manual_sessions(str(path), "hdfs")
        self.assertEqual(sessions[0].label, "Anomaly")

    def test_manual_log_extension_works(self):
        path = self.tmp / "kern.log"
        path.write_text("kernel: fatal error\n", encoding="utf-8")
        sessions = load_manual_sessions(str(path), "bgl")
        self.assertEqual(len(sessions), 1)

    # ── Unsupported format ───────────────────────────────────────────────
    def test_unsupported_extension_raises(self):
        path = self.tmp / "data.parquet"
        path.write_bytes(b"dummy")
        with self.assertRaises(ValueError):
            load_manual_sessions(str(path), "hdfs")

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_manual_sessions(str(self.tmp / "nonexistent.txt"), "hdfs")


# ---------------------------------------------------------------------------
# 3. FAISSVectorStore.reset()
# ---------------------------------------------------------------------------

class TestFAISSVectorStoreReset(unittest.TestCase):

    def setUp(self):
        self.dim = 16

    def _store(self):
        return FAISSVectorStore(dimension=self.dim, index_type="flat",
                                index_path="/tmp/test_faiss")

    def test_reset_clears_vectors(self):
        store = self._store()
        vecs = np.random.rand(5, self.dim).astype(np.float32)
        store.add(vecs, [{"id": i} for i in range(5)])
        self.assertEqual(store.size(), 5)
        store.reset()
        self.assertEqual(store.size(), 0)

    def test_reset_clears_metadata(self):
        store = self._store()
        vecs = np.random.rand(3, self.dim).astype(np.float32)
        store.add(vecs, [{"id": i} for i in range(3)])
        store.reset()
        self.assertEqual(len(store.metadata), 0)

    def test_reset_then_add(self):
        store = self._store()
        vecs = np.random.rand(4, self.dim).astype(np.float32)
        store.add(vecs, [{"id": i} for i in range(4)])
        store.reset()
        new_vecs = np.random.rand(2, self.dim).astype(np.float32)
        store.add(new_vecs, [{"id": 10}, {"id": 11}])
        self.assertEqual(store.size(), 2)

    def test_dimension_preserved_after_reset(self):
        store = self._store()
        store.reset()
        self.assertEqual(store.dimension, self.dim)


# ---------------------------------------------------------------------------
# 4. SessionEmbedder with minimal event dicts (from JSON loader)
# ---------------------------------------------------------------------------

class TestEmbedderWithJsonSessions(unittest.TestCase):

    def setUp(self):
        self.embedder = _make_tfidf_embedder(dim=32)

    def _session(self, raw_lines, event_ids):
        return _make_session(
            session_id="test_blk",
            raw_lines=raw_lines,
            events=_event_dicts_from_ids(event_ids),
            label="Anomaly",
            anomaly_score=-0.05,
        )

    def test_embed_session_no_error(self):
        s = self._session(["Error in block", "Connection lost"], ["E1", "E5"])
        vec = self.embedder.embed_session(s, mode="hybrid")
        self.assertEqual(vec.shape[0], 32)

    def test_embed_batch_shape(self):
        sessions = [
            self._session([f"line {j}" for j in range(5)], ["E1", "E2"])
            for _ in range(6)
        ]
        embeddings = self.embedder.embed_batch(sessions, mode="hybrid")
        self.assertEqual(embeddings.shape, (6, 32))

    def test_template_mode_no_error(self):
        s = self._session(["raw line"], ["E3", "E4", "E3"])
        vec = self.embedder.embed_session(s, mode="template")
        self.assertIsInstance(vec, np.ndarray)

    def test_raw_mode_no_error(self):
        s = self._session(["raw log line one", "raw log line two"], ["E1"])
        vec = self.embedder.embed_session(s, mode="raw")
        self.assertIsInstance(vec, np.ndarray)

    def test_empty_raw_lines_no_crash(self):
        s = _make_session("empty_blk", [], _event_dicts_from_ids(["E1"]))
        vec = self.embedder.embed_session(s, mode="hybrid")
        self.assertIsInstance(vec, np.ndarray)

    def test_truncation_of_long_session(self):
        long_lines = [f"Very long log line number {i} with extra detail" * 3
                      for i in range(200)]
        s = self._session(long_lines, ["E1"] * 200)
        text = SessionEmbedder._prepare_session_text(s, mode="hybrid", max_chars=500)
        self.assertLessEqual(len(text), 600)  # some slack for truncation marker


# ---------------------------------------------------------------------------
# 5. _build_metadata
# ---------------------------------------------------------------------------

class TestBuildMetadata(unittest.TestCase):

    def test_all_keys_present(self):
        s = _make_session("blk_99", ["line a", "line b"],
                          _event_dicts_from_ids(["E1", "E2"]),
                          label="Anomaly", anomaly_score=-0.07,
                          line_range=(10, 20))
        meta = _build_metadata(s)
        for key in ("session_id", "raw_lines", "label", "anomaly_score",
                    "line_range", "event_sequence"):
            self.assertIn(key, meta)

    def test_raw_lines_capped_at_100(self):
        s = _make_session("blk_x", [f"line {i}" for i in range(200)], [])
        meta = _build_metadata(s)
        self.assertLessEqual(len(meta["raw_lines"]), 100)

    def test_event_sequence_from_dicts(self):
        s = _make_session("blk_y", ["l1"],
                          _event_dicts_from_ids(["E3", "E7", "E3"]))
        meta = _build_metadata(s)
        self.assertEqual(meta["event_sequence"], ["E3", "E7", "E3"])

    def test_none_score_handled(self):
        s = _make_session("blk_z", ["line"], [], anomaly_score=None)
        meta = _build_metadata(s)
        self.assertIsNone(meta["anomaly_score"])


# ---------------------------------------------------------------------------
# 6. run_module3 — end-to-end
# ---------------------------------------------------------------------------

class TestRunModule3EndToEnd(unittest.TestCase):

    def setUp(self):
        import tempfile
        self.tmp      = Path(tempfile.mkdtemp())
        self.json_path = self.tmp / "HDFS_anomalies.json"
        self.index_dir = self.tmp / "faiss_index"
        self.out_json  = self.tmp / "hdfs_embedded.json"
        _write_anomaly_json(self.json_path, n=5)

    def _run(self, **kwargs):
        defaults = dict(
            anomaly_json=str(self.json_path),
            dataset="hdfs",
            embedding_model="all-MiniLM-L6-v2",
            index_dir=str(self.index_dir),
            output_json=str(self.out_json),
        )
        defaults.update(kwargs)
        return run_module3(**defaults)

    def test_output_files_created(self):
        self._run()
        self.assertTrue((self.index_dir / "index.faiss").exists())
        self.assertTrue((self.index_dir / "metadata.pkl").exists())
        self.assertTrue(self.out_json.exists())

    def test_index_size_matches_sessions(self):
        result = self._run()
        self.assertEqual(result["index_size"], 5)

    def test_return_dict_keys(self):
        result = self._run()
        for key in ("output_json", "index_path", "metadata_path", "dataset",
                    "sessions_embedded", "index_size", "embedding_model",
                    "embedding_dim", "embedding_mode", "processing_time",
                    "embedder", "vector_store"):
            self.assertIn(key, result)

    def test_embedder_and_store_returned(self):
        result = self._run()
        self.assertIsInstance(result["embedder"], SessionEmbedder)
        self.assertIsInstance(result["vector_store"], FAISSVectorStore)

    def test_summary_json_content(self):
        self._run()
        data = json.loads(self.out_json.read_text(encoding="utf-8"))
        self.assertEqual(data["dataset"], "hdfs")
        self.assertEqual(data["sessions_embedded"], 5)
        self.assertEqual(data["index_size"], 5)

    def test_metadata_loadable(self):
        self._run()
        with open(self.index_dir / "metadata.pkl", "rb") as f:
            meta = pickle.load(f)
        self.assertEqual(len(meta), 5)
        self.assertIn("session_id", meta[0])
        self.assertIn("raw_lines",  meta[0])

    def test_append_mode_accumulates(self):
        self._run(append=False)
        self._run(append=True)
        result = self._run(append=True)
        self.assertEqual(result["index_size"], 15)  # 5 * 3 runs

    def test_overwrite_mode_resets(self):
        self._run(append=False)
        result = self._run(append=False)
        self.assertEqual(result["index_size"], 5)

    def test_no_input_raises(self):
        with self.assertRaises(ValueError):
            run_module3(
                dataset="hdfs",
                index_dir=str(self.index_dir),
                output_json=str(self.out_json),
            )

    def test_in_memory_sessions_path(self):
        sessions = [
            _make_session(f"blk_{i}", [f"line {j}" for j in range(3)],
                          _event_dicts_from_ids(["E1", "E2"]),
                          label="Anomaly", anomaly_score=-0.05)
            for i in range(3)
        ]
        result = run_module3(
            sessions=sessions,
            dataset="hdfs",
            embedding_model="all-MiniLM-L6-v2",
            index_dir=str(self.index_dir),
            output_json=str(self.out_json),
        )
        self.assertEqual(result["index_size"], 3)


# ---------------------------------------------------------------------------
# 7. Manual anomaly ingestion via run_module3
# ---------------------------------------------------------------------------

class TestRunModule3ManualIngestion(unittest.TestCase):

    def setUp(self):
        import tempfile
        self.tmp      = Path(tempfile.mkdtemp())
        self.index_dir = self.tmp / "faiss_index"
        self.out_json  = self.tmp / "hdfs_embedded.json"

    def _run(self, **kwargs):
        defaults = dict(
            dataset="hdfs",
            embedding_model="all-MiniLM-L6-v2",
            index_dir=str(self.index_dir),
            output_json=str(self.out_json),
        )
        defaults.update(kwargs)
        return run_module3(**defaults)

    def test_manual_txt_indexed(self):
        txt = self.tmp / "incidents.txt"
        txt.write_text("\n".join([f"Error: disk fault on node {i}" for i in range(10)]),
                       encoding="utf-8")
        result = self._run(manual_files=[str(txt)])
        self.assertEqual(result["index_size"], 1)

    def test_manual_json_indexed(self):
        j = self.tmp / "manual.json"
        _write_anomaly_json(j, n=3)
        result = self._run(manual_files=[str(j)])
        self.assertEqual(result["index_size"], 3)

    def test_manual_plus_primary_combined(self):
        primary = self.tmp / "HDFS_anomalies.json"
        manual  = self.tmp / "manual.json"
        _write_anomaly_json(primary, n=4)
        _write_anomaly_json(manual, n=2)
        result = self._run(anomaly_json=str(primary), manual_files=[str(manual)])
        self.assertEqual(result["index_size"], 6)

    def test_multiple_manual_files(self):
        txt1 = self.tmp / "a.txt"
        txt2 = self.tmp / "b.txt"
        txt1.write_text("Error alpha\n", encoding="utf-8")
        txt2.write_text("Error beta\n", encoding="utf-8")
        result = self._run(manual_files=[str(txt1), str(txt2)])
        self.assertEqual(result["index_size"], 2)

    def test_manual_append_to_existing(self):
        primary = self.tmp / "HDFS_anomalies.json"
        manual  = self.tmp / "extra.txt"
        _write_anomaly_json(primary, n=3)
        manual.write_text("New anomaly log line\n", encoding="utf-8")
        self._run(anomaly_json=str(primary), append=False)
        result = self._run(manual_files=[str(manual)], append=True)
        self.assertEqual(result["index_size"], 4)


# ---------------------------------------------------------------------------
# 8. CLI argument parser
# ---------------------------------------------------------------------------

class TestModule3ArgParser(unittest.TestCase):

    def _parse(self, args):
        from module3_embed_index import _build_arg_parser
        return _build_arg_parser().parse_args(args)

    def test_dataset_required(self):
        from module3_embed_index import _build_arg_parser
        import io, contextlib
        with self.assertRaises(SystemExit):
            with contextlib.redirect_stderr(io.StringIO()):
                _build_arg_parser().parse_args(["some.json"])

    def test_defaults(self):
        args = self._parse(["some.json", "--dataset", "hdfs"])
        self.assertEqual(args.model, "all-mpnet-base-v2")
        self.assertEqual(args.mode, "hybrid")
        self.assertEqual(args.index_type, "flat")
        self.assertFalse(args.append)
        self.assertIsNone(args.manual_only)
        self.assertEqual(args.manual_files, [])

    def test_append_flag(self):
        args = self._parse(["some.json", "--dataset", "bgl", "--append"])
        self.assertTrue(args.append)

    def test_multiple_manual_flags(self):
        args = self._parse([
            "some.json", "--dataset", "hdfs",
            "--manual", "file1.txt", "--manual", "file2.json"
        ])
        self.assertEqual(args.manual_files, ["file1.txt", "file2.json"])

    def test_manual_only_flag(self):
        args = self._parse([
            "--dataset", "bgl", "--manual-only", "incidents.txt"
        ])
        self.assertIsNone(args.anomaly_json)
        self.assertEqual(args.manual_only, "incidents.txt")


if __name__ == "__main__":
    unittest.main(verbosity=2)
