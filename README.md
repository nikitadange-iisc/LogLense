# LogSense: Agentic AI Framework for Root Cause Analysis of Large-Scale System Logs

An end-to-end multi-stage retrieval-augmented agentic pipeline that compresses log volume before LLM involvement, enabling precise anomaly explanation and failure trace identification at scale.

## Architecture

```
Raw Log File
    ↓ Stage 1: Streaming ingestion + deduplication
    ↓ Stage 2: Drain algorithm → event templates
    ↓ Stage 3: Session grouping (Block ID / sliding window) → count vectors
    ↓ Stage 4: Isolation Forest → anomaly scoring & gating
    ↓ Stage 5: Sentence-transformer embedding → FAISS indexing
    ↓ Stage 6: RAG prompt assembly → Anthropic root cause analysis
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your Anthropic API key
```

### 3. Run with Sample Data

```bash
cd src

# Full pipeline (offline mode — no LLM API calls)
python pipeline.py ../data/raw/sample_hdfs.log -l ../data/raw/sample_labels.csv -d hdfs --offline

# Full pipeline with Anthropic analysis
python pipeline.py ../data/raw/sample_hdfs.log -l ../data/raw/sample_labels.csv -d hdfs

# Run individual stages
python ingestion.py ../data/raw/sample_hdfs.log
python parser.py ../data/raw/sample_hdfs.log -n 100
```

### 4. Run Tests

```bash
python -m pytest tests/ -v
```

## Pipeline Stages

| Stage | Module | Description |
|-------|--------|-------------|
| 1 | `ingestion.py` | Streaming file reader with consecutive deduplication |
| 2 | `parser.py` | Drain3-based log parsing into event templates |
| 3 | `sessionizer.py` | Session grouping (Block ID / sliding window) + vectorization |
| 4 | `anomaly_gate.py` | Isolation Forest training & anomaly detection |
| 5 | `embedder.py` / `vector_store.py` | Sentence-transformer embedding + FAISS indexing |
| 6 | `rag_pipeline.py` | RAG prompt assembly + Anthropic analysis |

## CLI Options

```
python pipeline.py <input_file> [options]

Options:
  -l, --labels          Path to ground-truth labels CSV
  -d, --dataset         Dataset type: hdfs, bgl, thunderbird
  -n, --max-lines       Max lines to process (for testing)
  --max-analyze         Max sessions to analyze with LLM (default: 10)
  --offline             Skip LLM calls, generate prompts only
  --no-train            Load existing model instead of training
  --contamination       Isolation Forest contamination (default: 0.1)
  --window-size         Sliding window size for BGL/Thunderbird (default: 50)
  --top-k               Similar examples to retrieve (default: 3)
  -v, --verbose         Enable debug logging
```

## Datasets

Source: [LogHub](https://github.com/logpai/loghub)

| Dataset | Size | Sessionization |
|---------|------|---------------|
| HDFS | ~11M lines | Block ID grouping |
| BGL | ~4.7M lines | Sliding window |
| Thunderbird | ~211M lines | Sliding window |

## Tech Stack

- **Python 3.10+**, **Drain3** (log parsing), **scikit-learn** (Isolation Forest)
- **sentence-transformers** (embeddings), **FAISS** (vector search), **Anthropic** (reasoning)

## Team

LogSense — Vasanthakumar S, Atreyee Mondal, Dange Nikita Dilip, Sujith Shetty, Dhananjaya B R, Raj Shekhar, Balla Malleswara Rao, Aele Santhosh
