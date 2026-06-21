"""
Tests for Module 4 — RAG Root Cause Analysis
=============================================
Covers: prompt building, offline retrieval, JSON parsing, batch analysis,
provider auto-detection, CLI parser, and run_module4 end-to-end (offline).
All tests run offline — no LLM API key required.
"""

import json
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from module4_rag_analysis import (
    _make_session,
    _event_dicts_from_ids,
    load_anomalous_sessions,
    run_module4,
    _build_arg_parser,
)
from rag_pipeline import RAGPipeline, _detect_provider, _system_prompt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _session(sid="blk_1234", n_lines=5, event_ids=None, label="Anomaly", score=-0.05):
    return _make_session(
        session_id    = sid,
        raw_lines     = [f"log line {i}" for i in range(n_lines)],
        events        = _event_dicts_from_ids(event_ids or ["E1", "E2"]),
        label         = label,
        anomaly_score = score,
        line_range    = (1, n_lines),
    )


def _mock_rag(dataset="hdfs") -> RAGPipeline:
    """Return a RAGPipeline with mocked embedder and store (offline provider)."""
    embedder = MagicMock()
    embedder.embed_session.return_value = np.zeros(384, dtype=np.float32)
    embedder.model_name = "test-model"

    store = MagicMock()
    store.size.return_value = 5
    store.search.return_value = [
        (
            {
                "session_id": "blk_9999",
                "raw_lines": ["historical line 1", "historical line 2"],
                "label": "Anomaly",
                "anomaly_score": -0.08,
                "event_sequence": ["E1", "E5"],
                "line_range": [10, 20],
            },
            0.95,
        )
    ]

    with patch.dict("os.environ", {}, clear=False):
        rag = RAGPipeline(
            embedder     = embedder,
            vector_store = store,
            dataset      = dataset,
            llm_provider = "offline",
        )
    return rag


def _write_anomaly_json(path: Path, n: int = 4, dataset: str = "hdfs"):
    sessions = []
    for i in range(n):
        label = "Anomaly" if i % 2 == 0 else "Normal"
        sessions.append({
            "session_id":    f"blk_{1000 + i}",
            "line_range":    [i * 10 + 1, i * 10 + 10],
            "label":         label,
            "anomaly_score": -(0.05 + i * 0.01),
            "event_sequence": ["E1", "E2"],
            "raw_lines":     [f"line {j}" for j in range(3)],
        })
    payload = {
        "dataset": dataset,
        "total_sessions": 100,
        "anomalous_sessions": n,
        "sessions": sessions,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


# ---------------------------------------------------------------------------
# 1. _detect_provider
# ---------------------------------------------------------------------------

class TestDetectProvider(unittest.TestCase):

    def test_detects_anthropic(self):
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-ant-test", "OPENAI_API_KEY": ""}):
            self.assertEqual(_detect_provider(), "claude")

    def test_detects_openai_when_no_anthropic(self):
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "", "OPENAI_API_KEY": "sk-openai"}):
            self.assertEqual(_detect_provider(), "openai")

    def test_offline_when_no_keys(self):
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(_detect_provider(), "offline")


# ---------------------------------------------------------------------------
# 2. Dataset-aware system prompt
# ---------------------------------------------------------------------------

class TestSystemPrompt(unittest.TestCase):

    def test_hdfs_prompt_mentions_datanode(self):
        p = _system_prompt("hdfs")
        self.assertIn("DataNode", p)
        self.assertIn("Block ID", p)

    def test_bgl_prompt_mentions_machine_check(self):
        p = _system_prompt("bgl")
        self.assertIn("machine check", p)
        self.assertIn("FATAL", p)

    def test_thunderbird_prompt_returned(self):
        p = _system_prompt("thunderbird")
        self.assertIn("Thunderbird", p)

    def test_unknown_dataset_returns_base(self):
        p = _system_prompt("unknown_ds")
        self.assertIn("root cause", p)

    def test_case_insensitive(self):
        self.assertEqual(_system_prompt("HDFS"), _system_prompt("hdfs"))


# ---------------------------------------------------------------------------
# 3. RAGPipeline.build_prompt
# ---------------------------------------------------------------------------

class TestBuildPrompt(unittest.TestCase):

    def setUp(self):
        self.rag = _mock_rag("hdfs")
        self.s   = _session(score=-0.07, event_ids=["E1", "E5", "E22"])

    def _retrieved(self):
        return self.rag.vector_store.search(None, top_k=1)

    def test_contains_session_id(self):
        p = self.rag.build_prompt(self.s, self._retrieved())
        self.assertIn("blk_1234", p)

    def test_contains_anomaly_score(self):
        p = self.rag.build_prompt(self.s, self._retrieved())
        self.assertIn("-0.0700", p)

    def test_contains_event_sequence(self):
        p = self.rag.build_prompt(self.s, self._retrieved())
        self.assertIn("E1", p)
        self.assertIn("E22", p)

    def test_contains_raw_lines(self):
        p = self.rag.build_prompt(self.s, self._retrieved())
        self.assertIn("log line 0", p)

    def test_contains_retrieved_session_id(self):
        p = self.rag.build_prompt(self.s, self._retrieved())
        self.assertIn("blk_9999", p)

    def test_contains_retrieved_distance(self):
        p = self.rag.build_prompt(self.s, self._retrieved())
        self.assertIn("0.9500", p)

    def test_contains_retrieved_raw_lines(self):
        p = self.rag.build_prompt(self.s, self._retrieved())
        self.assertIn("historical line 1", p)

    def test_no_retrieved_no_historical_section(self):
        p = self.rag.build_prompt(self.s, [])
        self.assertNotIn("SIMILAR HISTORICAL", p)

    def test_contains_task_section(self):
        p = self.rag.build_prompt(self.s, [])
        self.assertIn("TASK", p)

    def test_dataset_label_in_prompt(self):
        p = self.rag.build_prompt(self.s, [])
        self.assertIn("HDFS", p)


# ---------------------------------------------------------------------------
# 4. RAGPipeline.retrieve_similar
# ---------------------------------------------------------------------------

class TestRetrieveSimilar(unittest.TestCase):

    def setUp(self):
        self.rag = _mock_rag()
        self.s   = _session()

    def test_returns_list(self):
        results = self.rag.retrieve_similar(self.s, top_k=1)
        self.assertIsInstance(results, list)

    def test_embed_called_with_hybrid_mode(self):
        self.rag.retrieve_similar(self.s, top_k=1)
        self.rag.embedder.embed_session.assert_called_once_with(self.s, mode="hybrid")

    def test_store_search_called_with_top_k(self):
        self.rag.retrieve_similar(self.s, top_k=2)
        self.rag.vector_store.search.assert_called_once()
        _, kwargs = self.rag.vector_store.search.call_args
        self.assertEqual(kwargs.get("top_k", None) or
                         self.rag.vector_store.search.call_args[0][1], 2)

    def test_result_has_metadata_and_distance(self):
        results = self.rag.retrieve_similar(self.s, top_k=1)
        meta, dist = results[0]
        self.assertIn("session_id", meta)
        self.assertIsInstance(dist, float)


# ---------------------------------------------------------------------------
# 5. RAGPipeline.analyze_offline
# ---------------------------------------------------------------------------

class TestAnalyzeOffline(unittest.TestCase):

    def setUp(self):
        self.rag = _mock_rag()
        self.s   = _session()

    def test_returns_dict(self):
        result = self.rag.analyze_offline(self.s)
        self.assertIsInstance(result, dict)

    def test_has_required_keys(self):
        result = self.rag.analyze_offline(self.s)
        for key in ("session_id", "retrieved_examples", "prompt", "llm_provider"):
            self.assertIn(key, result)

    def test_provider_is_offline(self):
        result = self.rag.analyze_offline(self.s)
        self.assertEqual(result["llm_provider"], "offline")

    def test_llm_model_is_none(self):
        result = self.rag.analyze_offline(self.s)
        self.assertIsNone(result["llm_model"])

    def test_retrieved_examples_list(self):
        result = self.rag.analyze_offline(self.s, top_k=1)
        self.assertIsInstance(result["retrieved_examples"], list)
        self.assertEqual(len(result["retrieved_examples"]), 1)

    def test_retrieved_example_has_distance(self):
        result = self.rag.analyze_offline(self.s, top_k=1)
        ex = result["retrieved_examples"][0]
        self.assertIn("distance", ex)
        self.assertIn("session_id", ex)

    def test_prompt_is_string(self):
        result = self.rag.analyze_offline(self.s)
        self.assertIsInstance(result["prompt"], str)
        self.assertGreater(len(result["prompt"]), 50)

    def test_session_id_in_result(self):
        result = self.rag.analyze_offline(self.s)
        self.assertEqual(result["session_id"], "blk_1234")


# ---------------------------------------------------------------------------
# 6. RAGPipeline.analyze — JSON parse + error handling
# ---------------------------------------------------------------------------

class TestAnalyzeJsonParsing(unittest.TestCase):

    def _rag_with_llm(self, llm_response: str) -> RAGPipeline:
        rag = _mock_rag()
        rag.provider = "claude"
        rag._client  = MagicMock()
        rag._call_llm = MagicMock(return_value=llm_response)
        return rag

    def test_valid_json_parsed(self):
        payload = json.dumps({
            "root_cause": "block replication failure",
            "confidence": 0.9,
            "severity":   "high",
            "explanation": "pipeline broke",
            "failure_trace": [],
            "recommended_action": "restart DataNode",
            "affected_line_range": [1, 5],
        })
        rag    = self._rag_with_llm(payload)
        result = rag.analyze(_session(), top_k=1)
        self.assertEqual(result["root_cause"], "block replication failure")
        self.assertEqual(result["confidence"], 0.9)

    def test_json_in_markdown_fence_parsed(self):
        payload = "```json\n" + json.dumps({"root_cause": "disk error", "confidence": 0.7,
                  "severity": "medium", "explanation": "x", "failure_trace": [],
                  "recommended_action": "y", "affected_line_range": [1, 2]}) + "\n```"
        rag    = self._rag_with_llm(payload)
        result = rag.analyze(_session(), top_k=1)
        self.assertEqual(result["root_cause"], "disk error")

    def test_invalid_json_falls_back(self):
        rag    = self._rag_with_llm("This is plain text, not JSON.")
        result = rag.analyze(_session(), top_k=1)
        self.assertIn("root_cause", result)
        self.assertEqual(result["confidence"], 0.5)

    def test_result_always_has_session_id(self):
        rag    = self._rag_with_llm("{}")
        result = rag.analyze(_session(sid="blk_XYZ"), top_k=1)
        self.assertEqual(result["session_id"], "blk_XYZ")

    def test_result_has_llm_provider(self):
        rag    = self._rag_with_llm("{}")
        result = rag.analyze(_session(), top_k=1)
        self.assertIn("llm_provider", result)

    def test_result_has_retrieved_count(self):
        rag    = self._rag_with_llm(json.dumps({"root_cause": "x", "confidence": 0.5,
                  "severity": "low", "explanation": "", "failure_trace": [],
                  "recommended_action": "", "affected_line_range": [0, 0]}))
        result = rag.analyze(_session(), top_k=1)
        self.assertIn("retrieved_examples_count", result)


# ---------------------------------------------------------------------------
# 7. RAGPipeline.analyze_batch
# ---------------------------------------------------------------------------

class TestAnalyzeBatch(unittest.TestCase):

    def setUp(self):
        self.rag = _mock_rag()

    def test_offline_batch_returns_list(self):
        sessions = [_session(f"blk_{i}") for i in range(3)]
        # analyze_batch on offline rag uses analyze() which raises RuntimeError
        # — use analyze_offline via analyze_batch indirectly by switching provider
        results = [self.rag.analyze_offline(s) for s in sessions]
        self.assertEqual(len(results), 3)

    def test_batch_error_isolation(self):
        rag = _mock_rag()
        rag.provider = "claude"
        rag._call_llm = MagicMock(side_effect=[
            json.dumps({"root_cause": "ok", "confidence": 0.8, "severity": "low",
                        "explanation": "", "failure_trace": [], "recommended_action": "",
                        "affected_line_range": [0, 0]}),
            Exception("API timeout"),
            json.dumps({"root_cause": "ok2", "confidence": 0.7, "severity": "medium",
                        "explanation": "", "failure_trace": [], "recommended_action": "",
                        "affected_line_range": [0, 0]}),
        ])
        sessions = [_session(f"blk_{i}") for i in range(3)]
        results  = rag.analyze_batch(sessions, top_k=1)
        self.assertEqual(len(results), 3)
        self.assertIn("error", results[1])
        self.assertNotIn("error", results[0])
        self.assertNotIn("error", results[2])


# ---------------------------------------------------------------------------
# 8. load_anomalous_sessions
# ---------------------------------------------------------------------------

class TestLoadAnomalousSessions(unittest.TestCase):

    def setUp(self):
        import tempfile
        self.tmp = Path(tempfile.mkdtemp())

    def test_filters_normal_sessions(self):
        path = self.tmp / "test.json"
        _write_anomaly_json(path, n=4)
        sessions, _ = load_anomalous_sessions(str(path))
        for s in sessions:
            self.assertNotEqual(s.label, "Normal")

    def test_only_anomaly_sessions_returned(self):
        path = self.tmp / "test.json"
        _write_anomaly_json(path, n=4)
        sessions, _ = load_anomalous_sessions(str(path))
        # n=4 → 2 Anomaly (i=0,2), 2 Normal (i=1,3)
        self.assertEqual(len(sessions), 2)

    def test_sorted_most_anomalous_first(self):
        path = self.tmp / "test.json"
        _write_anomaly_json(path, n=6)
        sessions, _ = load_anomalous_sessions(str(path))
        scores = [s.anomaly_score for s in sessions if s.anomaly_score is not None]
        self.assertEqual(scores, sorted(scores))

    def test_payload_returned(self):
        path = self.tmp / "test.json"
        _write_anomaly_json(path, n=2, dataset="bgl")
        _, payload = load_anomalous_sessions(str(path))
        self.assertEqual(payload["dataset"], "bgl")

    def test_events_are_dicts(self):
        path = self.tmp / "test.json"
        _write_anomaly_json(path, n=2)
        sessions, _ = load_anomalous_sessions(str(path))
        for ev in sessions[0].events:
            self.assertIsInstance(ev, dict)
            self.assertIn("event_template", ev)


# ---------------------------------------------------------------------------
# 9. run_module4 — end-to-end offline
# ---------------------------------------------------------------------------

class TestRunModule4Offline(unittest.TestCase):

    def setUp(self):
        import tempfile
        self.tmp       = Path(tempfile.mkdtemp())
        self.json_path = self.tmp / "HDFS_anomalies.json"
        self.index_dir = self.tmp / "faiss_index"
        self.out_json  = self.tmp / "hdfs_rag_results.json"
        _write_anomaly_json(self.json_path, n=6)

        # Build a tiny real FAISS index so run_module4 can load it
        import faiss, pickle
        self.index_dir.mkdir(parents=True, exist_ok=True)
        dim = 384
        idx = faiss.IndexFlatL2(dim)
        vecs = np.random.rand(2, dim).astype(np.float32)
        idx.add(vecs)
        faiss.write_index(idx, str(self.index_dir / "index.faiss"))
        meta = [
            {"session_id": "blk_9001", "raw_lines": ["line a"], "label": "Anomaly",
             "anomaly_score": -0.09, "event_sequence": ["E1"], "line_range": [1, 1]},
            {"session_id": "blk_9002", "raw_lines": ["line b"], "label": "Anomaly",
             "anomaly_score": -0.07, "event_sequence": ["E2"], "line_range": [2, 2]},
        ]
        with open(self.index_dir / "metadata.pkl", "wb") as f:
            pickle.dump(meta, f)

    def _run(self, **kwargs):
        defaults = dict(
            anomaly_json    = str(self.json_path),
            dataset         = "hdfs",
            index_dir       = str(self.index_dir),
            embedding_model = "all-MiniLM-L6-v2",
            offline         = True,
            max_sessions    = 5,
            output_json     = str(self.out_json),
        )
        defaults.update(kwargs)
        return run_module4(**defaults)

    def test_output_json_created(self):
        self._run()
        self.assertTrue(self.out_json.exists())

    def test_only_anomaly_sessions_analysed(self):
        result = self._run()
        for r in result["results"]:
            sid = r.get("session_id", "")
            # session_ids from _write_anomaly_json: blk_1000(A), blk_1001(N), blk_1002(A)...
            # Normal sessions should not appear
            self.assertNotIn("blk_1001", sid)
            self.assertNotIn("blk_1003", sid)
            self.assertNotIn("blk_1005", sid)

    def test_results_count_capped_by_max_sessions(self):
        result = self._run(max_sessions=2)
        self.assertLessEqual(len(result["results"]), 2)

    def test_provider_is_offline(self):
        result = self._run()
        self.assertEqual(result["llm_provider"], "offline")

    def test_each_result_has_prompt(self):
        result = self._run()
        for r in result["results"]:
            self.assertIn("prompt", r)

    def test_each_result_has_retrieved_examples(self):
        result = self._run()
        for r in result["results"]:
            self.assertIn("retrieved_examples", r)

    def test_output_json_content(self):
        self._run()
        data = json.loads(self.out_json.read_text(encoding="utf-8"))
        self.assertEqual(data["dataset"], "hdfs")
        self.assertTrue(data["offline_mode"])
        self.assertIn("results", data)

    def test_return_dict_keys(self):
        result = self._run()
        for key in ("output_json", "dataset", "sessions_analysed",
                    "total_anomalous", "llm_provider", "results", "rag_pipeline"):
            self.assertIn(key, result)

    def test_rag_pipeline_returned(self):
        result = self._run()
        self.assertIsInstance(result["rag_pipeline"], RAGPipeline)

    def test_in_memory_sessions_path(self):
        sessions = [_session(f"blk_{i}", score=-(0.05 + i * 0.01)) for i in range(4)]
        result = run_module4(
            anomaly_json    = None,
            sessions        = sessions,
            dataset         = "hdfs",
            index_dir       = str(self.index_dir),
            embedding_model = "all-MiniLM-L6-v2",
            offline         = True,
            max_sessions    = 4,
            output_json     = str(self.out_json),
        )
        self.assertEqual(result["sessions_analysed"], 4)


# ---------------------------------------------------------------------------
# 10. CLI argument parser
# ---------------------------------------------------------------------------

class TestModule4ArgParser(unittest.TestCase):

    def _parse(self, args):
        return _build_arg_parser().parse_args(args)

    def test_dataset_required(self):
        import io, contextlib
        with self.assertRaises(SystemExit):
            with contextlib.redirect_stderr(io.StringIO()):
                _build_arg_parser().parse_args(["some.json"])

    def test_defaults(self):
        args = self._parse(["some.json", "--dataset", "hdfs"])
        self.assertEqual(args.llm_provider, "auto")
        self.assertEqual(args.top_k, 3)
        self.assertEqual(args.max_sessions, 10)
        self.assertFalse(args.offline)
        self.assertIsNone(args.llm_model)

    def test_offline_flag(self):
        args = self._parse(["some.json", "--dataset", "bgl", "--offline"])
        self.assertTrue(args.offline)

    def test_llm_choices(self):
        for provider in ("auto", "claude", "openai", "offline"):
            args = self._parse(["f.json", "--dataset", "hdfs", "--llm", provider])
            self.assertEqual(args.llm_provider, provider)

    def test_max_sessions(self):
        args = self._parse(["f.json", "--dataset", "hdfs", "--max-sessions", "25"])
        self.assertEqual(args.max_sessions, 25)

    def test_top_k(self):
        args = self._parse(["f.json", "--dataset", "hdfs", "--top-k", "5"])
        self.assertEqual(args.top_k, 5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
