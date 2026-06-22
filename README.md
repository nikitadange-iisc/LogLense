# LogSense: Agentic AI Framework for Root Cause Analysis of Large-Scale System Logs

An end-to-end, multi-stage retrieval-augmented pipeline that compresses log volume before any LLM involvement, enabling precise anomaly detection, root cause identification, and failure trace generation at scale.

## Architecture
Raw Log File

↓

Module 1 — Ingestion & Drain Parsing      → data/processed/<dataset>_structured.csv

↓

Module 2 — Session Anomaly Detection      → data/processed/<dataset>_anomalies.json

↓

Module 3 — Sentence Embedding & FAISS     → models/faiss_index/

↓

Module 4 — RAG Root Cause Analysis        → data/processed/<dataset>_rag_results.json
## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Add your ANTHROPIC_API_KEY to .env
```

## Run UI (Single Command)

```bash
uvicorn api_main:app --port 8001 --reload
```

Open browser: **http://localhost:8001**

## Run Pipeline (CLI)

```bash
# Module 1
python src/module1_ingest_parse.py data/raw/HDFS_sample_1pct.log --dataset hdfs

# Module 2
python src/module2_session_anomaly.py data/processed/HDFS_sample_1pct_structured.csv \
    --dataset hdfs --label-path data/raw/anomaly_label.csv --contamination 0.03

# Module 3
python src/module3_embed_index.py data/processed/HDFS_sample_1pct_anomalies.json \
    --dataset hdfs --model all-MiniLM-L6-v2

# Module 4
python src/module4_rag_analysis.py data/processed/HDFS_sample_1pct_anomalies.json \
    --dataset hdfs --max-sessions 5
```

## Run Tests

```bash
python -m pytest tests/ -v
```

**295/295 tests passing**

## Key Results (HDFS Dataset)

| Metric | Value |
|---|---|
| Lines processed | 99,805 |
| Sessions created | 7,940 |
| Anomalies detected | 307 (3.87%) |
| Compression before LLM | 98.2% |
| F1 Score | 0.416 |
| Precision | 0.420 |
| Recall | 0.412 |
| Accuracy | 0.954 |
| LLM Provider | Anthropic Claude Sonnet |

## Tech Stack

| Component | Library | Version |
|---|---|---|
| Log parsing | drain3 | ≥0.9.11 |
| Anomaly detection | scikit-learn IsolationForest | ≥1.3.0 |
| Embedding | sentence-transformers | ≥2.2.2 |
| Embedding model | all-MiniLM-L6-v2 | 384-dim |
| Vector search | faiss-cpu | ≥1.7.4 |
| LLM | anthropic Claude Sonnet | ≥0.75.0 |
| API backend | FastAPI + Uvicorn | ≥0.100.0 |
| Frontend | React 17 (CDN) | No build step |

## Project Structure
LogLense/

├── api_main.py              # FastAPI backend + React UI server

├── ui/index.html            # React frontend (5 tabs)

├── src/

│   ├── module1_ingest_parse.py

│   ├── module2_session_anomaly.py

│   ├── module3_embed_index.py

│   ├── module4_rag_analysis.py

│   ├── pipeline.py

│   ├── anomaly_gate.py

│   ├── sessionizer.py

│   ├── embedder.py

│   ├── vector_store.py

│   └── rag_pipeline.py

├── tests/                   # 295 tests across 5 files

├── data/

│   ├── raw/                 # Input log files

│   └── processed/           # Pipeline outputs

├── models/                  # Trained models (auto-generated)

├── requirements.txt

└── .env.example
## Team

Vasanthakumar S, Atreyee Mondal, Dange Nikita Dilip, Sujith Shetty,
Dhananjaya B R, Raj Shekhar, Balla Malleswara Rao, Aele Santhosh
