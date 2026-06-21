# LogSense: Agentic AI Framework for Root Cause Analysis of Large-Scale System Logs

An end-to-end, multi-stage retrieval-augmented pipeline that compresses log volume before any LLM involvement, enabling precise anomaly detection, root cause identification, and failure trace generation at scale.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Project Structure](#project-structure)
3. [Setup & Installation](#setup--installation)
4. [Data & Datasets](#data--datasets)
5. [Step-by-Step Execution Guide](#step-by-step-execution-guide)
   - [Module 1 — Ingestion & Parsing](#module-1--ingestion--parsing)
   - [Module 2 — Session Anomaly Detection](#module-2--session-anomaly-detection)
   - [Module 3 — Embedding & FAISS Indexing](#module-3--embedding--faiss-indexing)
   - [Module 4 — RAG Root Cause Analysis](#module-4--rag-root-cause-analysis)
   - [Full Pipeline (All Modules)](#full-pipeline-all-modules)
6. [Data Flow: Output → Input Between Modules](#data-flow-output--input-between-modules)
7. [Additional Capabilities & Enhancements](#additional-capabilities--enhancements)
8. [Testing](#testing)
9. [Tech Stack](#tech-stack)
10. [Team](#team)

---

## Architecture Overview

```
Raw Log File  (.log)
       │
       ▼
┌─────────────────────────────────────────────────────────────────┐
│  MODULE 1 — Ingestion & Drain Parsing                           │
│  • Streaming reader (no memory blow-up on 200M-line logs)       │
│  • Consecutive-line deduplication                               │
│  • Drain3 algorithm → event templates (E1, E2, …, EN)          │
│  Output: data/processed/<dataset>_structured.csv                │
└──────────────────────────┬──────────────────────────────────────┘
                           │ structured CSV
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  MODULE 2 — Session Grouping & Anomaly Detection                │
│  • HDFS   → sessions by Block ID                                │
│  • BGL    → sessions by Node ID  (hardware fault bursts)        │
│  • Thunderbird → sliding window                                 │
│  • Vectorize: count (HDFS) or binary×IDF (BGL)                 │
│  • Isolation Forest → anomaly score per session                 │
│  Output: data/processed/<dataset>_anomalies.json                │
└──────────────────────────┬──────────────────────────────────────┘
                           │ anomalies JSON (Anomaly-labelled only)
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  MODULE 3 — Sentence Embedding & FAISS Indexing                 │
│  • Filters out Normal-labelled sessions (IF false positives)    │
│  • Embeds each session with all-MiniLM-L6-v2  (384-dim)        │
│  • Stores embeddings + metadata in FAISS flat-L2 index          │
│  • Supports manual anomaly ingestion (.json/.txt/.log/.csv)     │
│  • Append mode: grow index across multiple runs/datasets        │
│  Output: models/faiss_index/index.faiss                         │
│          models/faiss_index/metadata.pkl                        │
│          data/processed/<dataset>_embedded.json                 │
└──────────────────────────┬──────────────────────────────────────┘
                           │ FAISS index + embedder
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  MODULE 4 — RAG Root Cause Analysis                             │
│  • For each new anomalous session:                              │
│    1. Embed query session                                       │
│    2. Retrieve top-K similar historical anomalies from FAISS    │
│    3. Assemble dataset-aware prompt (HDFS / BGL / Thunderbird)  │
│    4. Call LLM (Claude or OpenAI, auto-detected)                │
│  • Returns structured JSON: root_cause, severity, confidence,   │
│    failure_trace, recommended_action                            │
│  Output: data/processed/<dataset>_rag_results.json              │
└─────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
LogLense-main/
│
├── data/
│   ├── raw/                        # Raw log files (HDFS.log, BGL.log, …)
│   └── processed/                  # All generated outputs
│       ├── HDFS_structured.csv     # Module 1 output
│       ├── bgl_structured.csv
│       ├── HDFS_anomalies.json     # Module 2 output
│       ├── bgl_anomalies.json
│       ├── hdfs_embedded.json      # Module 3 summary
│       ├── bgl_embedded.json
│       ├── hdfs_rag_results.json   # Module 4 output
│       └── bgl_rag_results.json
│
├── models/
│   ├── isolation_forest.joblib     # Trained Isolation Forest model
│   ├── event_columns.json          # Event vocabulary (template IDs)
│   └── faiss_index/
│       ├── index.faiss             # FAISS vector index
│       └── metadata.pkl            # Per-vector metadata (raw lines, labels, scores)
│
├── src/
│   ├── module1_ingest_parse.py     # ★ Standalone Module 1 runner
│   ├── module2_session_anomaly.py  # ★ Standalone Module 2 runner
│   ├── module3_embed_index.py      # ★ Standalone Module 3 runner
│   ├── module4_rag_analysis.py     # ★ Standalone Module 4 runner
│   │
│   ├── ingestion.py                # Streaming reader + deduplicator
│   ├── log_parser.py               # Drain3 wrapper
│   ├── sessionizer.py              # Session grouping, vectorization, IDF weighting
│   ├── anomaly_gate.py             # Isolation Forest wrapper
│   ├── embedder.py                 # SessionEmbedder (sentence-transformers + TF-IDF fallback)
│   ├── vector_store.py             # FAISSVectorStore (add / search / save / load / reset)
│   ├── rag_pipeline.py             # RAGPipeline class (retrieve → prompt → LLM → parse)
│   ├── pipeline.py                 # End-to-end orchestration (all 4 modules)
│   │
│   ├── retriever.py                # Legacy retrieval helper
│   ├── download_hdfs.py            # Data download helper
│   └── download_model.py           # Model download helper
│
├── tests/
│   ├── test_module1.py             # Module 1 test suite
│   ├── test_module2.py             # Module 2 test suite
│   ├── test_module3.py             # Module 3 test suite  (50 tests)
│   ├── test_module4.py             # Module 4 test suite  (59 tests)
│   ├── test_pipeline.py            # End-to-end pipeline tests
│   ├── test_spotcheck.py           # Spot-check tests
│   ├── test_stability.py           # Stability tests
│   ├── validate_module1.py         # Module 1 validation script
│   └── verify_outcomes.py          # Cross-module outcome verification
│
├── all-MiniLM-L6-v2/              # Local sentence-transformer model (offline use)
├── requirements.txt
├── .env.example
└── spec.md
```

---

## Setup & Installation

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `faiss-cpu>=1.8.0` is required for numpy 2.x compatibility.
> If you have an older version installed: `pip install "faiss-cpu>=1.8.0"`

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env`:

```env
# LLM — set one (or both). Module 4 auto-detects: Claude first, then OpenAI.
ANTHROPIC_API_KEY=your-anthropic-api-key-here
OPENAI_API_KEY=your-openai-api-key-here

# Optional overrides
LLM_MODEL=claude-sonnet-4-6        # or gpt-4o-mini
EMBEDDING_MODEL=all-MiniLM-L6-v2
FAISS_TOP_K=3
ANOMALY_CONTAMINATION=0.03
```

### 3. Sentence-transformer model (offline environments)

The embedding model (`all-MiniLM-L6-v2`) is included as a local folder in the project root. No internet access is required for embedding. If you need to download it on another machine:

```bash
python -c "
from huggingface_hub import snapshot_download
snapshot_download('sentence-transformers/all-MiniLM-L6-v2', local_dir='all-MiniLM-L6-v2')
"
```

---

## Data & Datasets

Place raw log files in `data/raw/`. Source: [LogHub](https://github.com/logpai/loghub)

| Dataset | Raw file | Lines | Sessionization method | Label file |
|---|---|---|---|---|
| HDFS | `HDFS.log` | ~11M | Block ID grouping | `anomaly_label.csv` |
| BGL | `BGL.log` | ~4.7M | Node ID grouping | Built-in (alert column) |
| Thunderbird | `Thunderbird.log` | ~211M | Sliding window (50 lines) | Built-in (label column) |

> **HDFS** requires a separate `anomaly_label.csv` that maps block IDs to Anomaly/Normal labels.
> **BGL** and **Thunderbird** extract labels directly from the log's first column (`-` = normal, anything else = anomaly).

---

## Step-by-Step Execution Guide

Run all commands from the **project root** directory (`LogLense-main/`).

---

### Module 1 — Ingestion & Parsing

**What it does:** Reads the raw log file, deduplicates consecutive identical lines, and runs the Drain3 algorithm to extract event templates. Produces a structured CSV with one row per log line.

**Input:** Raw log file (`data/raw/HDFS.log` or `data/raw/BGL.log`)

**Command:**

```bash
# HDFS
python src/module1_ingest_parse.py data/raw/HDFS.log --dataset hdfs

# BGL
python src/module1_ingest_parse.py data/raw/BGL.log --dataset bgl

# Limit lines for testing
python src/module1_ingest_parse.py data/raw/HDFS.log --dataset hdfs --max-lines 100000
```

**Output:** `data/processed/HDFS_structured.csv` (or `bgl_structured.csv`)

```
LineId, Date, Time, Pid, Level, Component, Content, EventId, EventTemplate
1, 081109, 203518, 143, INFO, dfs.DataNode, Receiving block blk_-1608999687919862906..., E1, Receiving block <*> src: <*> dest: <*>
...
```

---

### Module 2 — Session Anomaly Detection

**What it does:** Groups log events into sessions, vectorizes them, trains an Isolation Forest model, and flags anomalous sessions.

**Input:** `data/processed/<dataset>_structured.csv` (Module 1 output)

**Command:**

```bash
# HDFS — with ground-truth labels for evaluation
python src/module2_session_anomaly.py data/processed/HDFS_structured.csv \
    --dataset hdfs \
    --labels data/raw/anomaly_label.csv \
    --contamination 0.03

# BGL — node-based sessionization + binary×IDF weighting (best for BGL)
python src/module2_session_anomaly.py data/processed/bgl_structured.csv \
    --dataset bgl \
    --contamination 0.05

# Thunderbird — sliding window
python src/module2_session_anomaly.py data/processed/thunderbird_structured.csv \
    --dataset thunderbird \
    --contamination 0.05 \
    --window-size 50
```

**Output:** `data/processed/<dataset>_anomalies.json`

```json
{
  "dataset": "hdfs",
  "total_sessions": 7954,
  "anomalous_sessions": 302,
  "sessions": [
    {
      "session_id": "blk_-3102267849859399193",
      "label": "Anomaly",
      "anomaly_score": -0.0471,
      "event_sequence": ["E1", "E5", "E22"],
      "raw_lines": ["Receiving block blk_... ", "..."],
      "line_range": [120, 136]
    },
    ...
  ]
}
```

> `anomaly_score`: More negative = more anomalous (Isolation Forest output, boundary at 0).
> Both `Anomaly` and `Normal`-labelled sessions appear here — Module 3 filters to `Anomaly` only.

---

### Module 3 — Embedding & FAISS Indexing

**What it does:** Loads only the **Anomaly-labelled** sessions from Module 2 output, embeds them using `all-MiniLM-L6-v2` (384-dim), and stores the embeddings in a FAISS flat-L2 vector index for similarity retrieval.

**Input:** `data/processed/<dataset>_anomalies.json` (Module 2 output)

**Command:**

```bash
# HDFS — build fresh index
python src/module3_embed_index.py data/processed/HDFS_anomalies.json \
    --dataset hdfs \
    --model all-MiniLM-L6-v2

# BGL — append to existing HDFS index (multi-dataset index)
python src/module3_embed_index.py data/processed/bgl_anomalies.json \
    --dataset bgl \
    --model all-MiniLM-L6-v2 \
    --append

# Add a manual anomaly file on top of the primary source
python src/module3_embed_index.py data/processed/HDFS_anomalies.json \
    --dataset hdfs \
    --model all-MiniLM-L6-v2 \
    --manual data/raw/expert_known_incidents.txt

# Manual-only mode — no primary JSON needed
python src/module3_embed_index.py \
    --manual-only data/raw/my_incidents.txt \
    --dataset hdfs \
    --model all-MiniLM-L6-v2 \
    --append
```

**Output:**
- `models/faiss_index/index.faiss` — FAISS vector index
- `models/faiss_index/metadata.pkl` — per-vector metadata (session_id, raw_lines, label, anomaly_score, event_sequence, line_range)
- `data/processed/<dataset>_embedded.json` — run summary

**Verify the index:**

```bash
python -c "
import sys, pickle
sys.path.insert(0, 'src')
import faiss
from collections import Counter

idx  = faiss.read_index('models/faiss_index/index.faiss')
meta = pickle.load(open('models/faiss_index/metadata.pkl', 'rb'))
print('Vectors:', idx.ntotal)
print('Labels:', dict(Counter(m['label'] for m in meta)))
"
```

**Manual anomaly file formats supported:**

| Format | Extension | How it's handled |
|---|---|---|
| Anomalies JSON | `.json` | Same structure as Module 2 output |
| Raw log lines | `.txt` / `.log` | Windowed into 50-line sessions, all labelled Anomaly |
| Module 1 CSV | `.csv` | Re-sessionized using dataset's method |

---

### Module 4 — RAG Root Cause Analysis

**What it does:** For each anomalous session, retrieves the top-K most similar historical anomalies from the FAISS index, assembles a dataset-aware prompt, and calls an LLM for structured root cause analysis.

**Input:**
- `data/processed/<dataset>_anomalies.json` (Module 2 output) — sessions to analyse
- `models/faiss_index/` (Module 3 output) — reference knowledge base

**Command:**

```bash
# Auto-detect LLM (reads ANTHROPIC_API_KEY or OPENAI_API_KEY from .env)
python src/module4_rag_analysis.py data/processed/HDFS_anomalies.json \
    --dataset hdfs

# BGL — analyse top 20 most anomalous sessions
python src/module4_rag_analysis.py data/processed/bgl_anomalies.json \
    --dataset bgl \
    --max-sessions 20 \
    --top-k 3

# Offline mode — no API key needed (generates retrieval + prompts only)
python src/module4_rag_analysis.py data/processed/HDFS_anomalies.json \
    --dataset hdfs \
    --offline

# Force Claude
python src/module4_rag_analysis.py data/processed/HDFS_anomalies.json \
    --dataset hdfs \
    --llm claude \
    --llm-model claude-sonnet-4-6

# Force OpenAI
python src/module4_rag_analysis.py data/processed/bgl_anomalies.json \
    --dataset bgl \
    --llm openai \
    --llm-model gpt-4o-mini
```

**Output:** `data/processed/<dataset>_rag_results.json`

```json
{
  "dataset": "hdfs",
  "llm_provider": "claude",
  "llm_model": "claude-sonnet-4-6",
  "sessions_analysed": 10,
  "results": [
    {
      "session_id": "blk_-3102267849859399193",
      "root_cause": "DataNode pipeline write failure due to lost connection",
      "severity": "high",
      "confidence": 0.91,
      "affected_line_range": [2, 7],
      "explanation": "The session shows a block replication pipeline error...",
      "failure_trace": [
        {"line": "Exception in receiveBlock ...", "annotation": "pipeline write failed"}
      ],
      "recommended_action": "Check DataNode disk health and network connectivity",
      "retrieved_examples_count": 3,
      "anomaly_score": -0.0471
    }
  ]
}
```

> Sessions are analysed in **most-anomalous-first** order (most negative score first).
> `--max-sessions` caps LLM API calls — increase it to analyse more sessions.

---

### Full Pipeline (All Modules)

Run all four modules in sequence with a single command:

```bash
# HDFS — full run with LLM analysis
python src/pipeline.py data/raw/HDFS.log \
    --labels data/raw/anomaly_label.csv \
    --dataset hdfs \
    --contamination 0.03 \
    --max-analyze 10

# BGL — offline (no API key)
python src/pipeline.py data/raw/BGL.log \
    --dataset bgl \
    --offline

# Limit to first 50,000 lines for quick testing
python src/pipeline.py data/raw/HDFS.log \
    --dataset hdfs \
    --max-lines 50000 \
    --offline
```

---

## Data Flow: Output → Input Between Modules

```
Module 1 output                     Used by
─────────────────────────────────── ──────────
data/processed/HDFS_structured.csv  Module 2 (--input)
data/processed/bgl_structured.csv   Module 2 (--input)

Module 2 output                     Used by
─────────────────────────────────── ──────────
data/processed/HDFS_anomalies.json  Module 3 (positional arg) + Module 4 (positional arg)
data/processed/bgl_anomalies.json   Module 3 (positional arg) + Module 4 (positional arg)

Module 3 output                     Used by
─────────────────────────────────── ──────────
models/faiss_index/index.faiss      Module 4 (--index-dir)
models/faiss_index/metadata.pkl     Module 4 (--index-dir)

Module 4 output                     Used by
─────────────────────────────────── ──────────
data/processed/<ds>_rag_results.json Final results / downstream dashboards
```

> **Important:** Run modules in order. Module 3 and 4 both read from Module 2's
> `_anomalies.json` — Module 3 to build the knowledge base, Module 4 to query it.

---

## Additional Capabilities & Enhancements

The following improvements were made beyond the original architecture:

### Anomaly Detection Improvements (Module 2)

**Isolation Forest contamination bug fix**
The original code used `contamination='auto'` when labelled-normal sessions were available, which is sklearn's fixed -0.5 threshold (~17% flagged) instead of the user's specified value. Fixed to use the user's `--contamination` value in all training paths.
- HDFS result: flagged sessions dropped from 1,417 (17.8%) to 302 (3.8%), F1 improved from 0.18 → 0.41.

**Node-based sessionization for BGL**
The original BGL implementation used a sliding window (9,999 windows of 50 events). Fault bursts are concentrated on specific nodes, so sliding windows dilute the signal by mixing normal events from other nodes. The new `node` method groups all events from the same Node ID together, concentrating fault events per session.
- BGL F1 improved from 0.015 → 0.061 with this change alone.

**Binary×IDF weighting for BGL**
Standard count vectorization gives huge weight to high-frequency normal events (E1 appearing 4,865 times per node). Binary×IDF weighting (`(count > 0) × IDF`) removes this length bias: a FATAL event appearing once gets the same presence weight as E1, and IDF de-emphasises templates that appear in many sessions.
- BGL F1 improved from 0.061 → 0.493 with binary×IDF.

### Embedding & Indexing Improvements (Module 3)

**Normal-session filter**
The FAISS index is a reference knowledge base of *known* anomalies. The original code indexed all 302 HDFS sessions (175 Normal + 127 Anomaly) and all 1,102 BGL sessions (590 Normal + 512 Anomaly). Normal-labelled sessions are Isolation Forest false positives and corrupt retrieval quality. The filter ensures only confirmed anomalies enter the index.

**Local model path resolution**
`SessionEmbedder` now checks `<project_root>/<model_name>` before attempting a HuggingFace download. The bundled `all-MiniLM-L6-v2/` folder is found automatically — no internet access required.

**sentence-transformers API compatibility**
Fixed `get_sentence_embedding_dimension()` → `get_embedding_dimension()` for sentence-transformers v5.x, with backward-compatible wrapper for older versions.

**TF-IDF dimension stability fix**
The TF-IDF fallback embedder (used when sentence-transformers is unavailable) previously mutated `self.dimension` during fitting, causing FAISS dimension mismatches when the corpus was small. Fixed by saving the target dimension before any SVD truncation and always padding output to the declared dimension.

**FAISS compatibility upgrade**
Upgraded `faiss-cpu` from 1.7.4 → 1.14.3 for numpy 2.x compatibility (`swig_ptr` was rejecting valid float32 arrays on numpy 2.0+).

**Manual anomaly ingestion**
Users can inject their own known anomaly files into the FAISS index without re-running the full pipeline. Three formats are auto-detected by extension:
- `.json` — same structure as Module 2 anomalies output
- `.txt` / `.log` — raw log lines, windowed into 50-line sessions
- `.csv` — Module 1 structured CSV, re-sessionized by dataset method

**Append mode**
`--append` adds new vectors to an existing index without discarding previous ones. This allows building a multi-dataset index incrementally (e.g., run HDFS first, then append BGL).

### RAG Pipeline Improvements (Module 4)

**Auto-detect LLM provider**
Module 4 checks environment variables in order: `ANTHROPIC_API_KEY` → Claude (claude-sonnet-4-6), `OPENAI_API_KEY` → OpenAI (gpt-4o-mini), neither → offline mode. No code changes needed when switching providers.

**Claude (Anthropic) support**
Added full Claude API integration using the Anthropic SDK alongside the existing OpenAI integration. Response parsing handles Claude's tendency to wrap JSON in markdown code fences.

**Dataset-aware system prompts**
The LLM system prompt is tailored to the dataset:
- **HDFS**: explains DataNode/NameNode roles, block replication, pipeline errors
- **BGL**: explains machine check exceptions, RAS events, FATAL hardware faults, ciod I/O daemon
- **Thunderbird**: explains sliding-window context, kernel panic patterns

**Accurate prompt context**
The original `build_prompt` showed `"Known Root Cause: Unknown"` for every retrieved example because the FAISS metadata has no `root_cause` field. Replaced with the actual stored fields: `event_sequence`, `anomaly_score`, `label`, and the first 30 raw log lines. This gives the LLM real context to reason from.

**Most-anomalous-first ordering**
Sessions are sorted by `anomaly_score` ascending (most negative first) before the `--max-sessions` cap is applied, ensuring LLM API budget is spent on the worst anomalies first.

**Hybrid embedding mode consistency**
`retrieve_similar` now explicitly passes `mode="hybrid"` to `embed_session`, matching the mode used when sessions were indexed in Module 3.

---

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run per-module
python -m pytest tests/test_module1.py -v
python -m pytest tests/test_module2.py -v
python -m pytest tests/test_module3.py -v   # 50 tests
python -m pytest tests/test_module4.py -v   # 59 tests

# Run with short traceback on failure
python -m pytest tests/ --tb=short
```

All Module 3 and Module 4 tests run fully offline — no LLM API key or internet access required.

---

## Tech Stack

| Component | Library / Tool | Version |
|---|---|---|
| Log parsing | drain3 | ≥ 0.9.11 |
| Anomaly detection | scikit-learn (IsolationForest) | ≥ 1.3.0 |
| Embedding | sentence-transformers | ≥ 5.0 |
| Embedding model | all-MiniLM-L6-v2 (384-dim) | local folder |
| Vector search | faiss-cpu | ≥ 1.8.0 |
| LLM — Claude | anthropic SDK | ≥ 0.49.0 |
| LLM — OpenAI | openai SDK | ≥ 2.0.0 |
| Data handling | pandas, numpy | ≥ 2.0, ≥ 2.2 |
| Runtime | Python | 3.10+ |

---

## Team

LogSense — Vasanthakumar S, Atreyee Mondal, Dange Nikita Dilip, Sujith Shetty, Dhananjaya B R, Raj Shekhar, Balla Malleswara Rao, Aele Santhosh
