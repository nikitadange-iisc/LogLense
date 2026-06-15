# LogLense Spec
# LogSense: An Agentic AI Framework for Root Cause Analysis of Large-Scale System Logs


### Abstract
Modern distributed systems generate log data at large scale, making it hard to analyze and computationally expensive for naive LLM-based analysis. Direct ingestion of raw logs into an LLM exhausts context windows, inflates token cost, and fails to achieve proper reasoning over failure events.

This project proposes a multi-stage retrieval-augmented agentic pipeline that compresses log volume before any LLM involvement, enabling precise anomaly explanation and failure trace identification at scale. The pipeline uses the LogHub dataset (HDFS ~11M lines, BGL ~4.7M lines, Thunderbird ~211M lines).

## Pipeline Overview
## Stage 1: Streaming Ingestion & Deduplication

- Read log file line-by-line (streaming, no full in-memory load).
- Remove duplicate consecutive log lines.
- Output: cleaned log stream written to disk/iterator for downstream stages.

### Implementation Steps
1. Implement a generator-based file reader (`read_log_stream(filepath)`).
2. Track previous line; skip current line if identical to previous.
3. Write deduplicated lines to an intermediate file or yield to next stage.
4. Unit test with a small sample log file containing repeated lines.

## Stage 2: Log Parsing with Drain Algorithm

- Parse each log line into an event template, extracting dynamic variables (block IDs, IP addresses, timestamps, etc.).
- Reduces millions of raw lines to a small set of unique event templates per dataset.

### Implementation Steps
1. Integrate Drain3 library (or implement Drain tree-based parser).
2. Configure regex masks for known variable patterns (IP, block ID, numbers).
3. For each incoming line, output: `(event_template_id, extracted_variables, raw_line, line_number)`.
4. Persist the template miner state (drain config/state file) for reuse across runs.
5. Validate template count is approximately stable (sanity check against expected unique templates per dataset).

## Stage 3: Session Grouping & Vectorization

- Group parsed lines into sessions:
  - HDFS: group by Block ID.
  - BGL/Thunderbird: group by sliding window of N events.
- Represent each session as a fixed-length event count vector (count of each event template ID within the session).

### Implementation Steps
1. Implement Block ID extraction from Drain variable output (HDFS).
2. Implement sliding-window session builder for non-block-based datasets.
3. Build a global event template vocabulary (template_id → vector index).
4. Construct fixed-length count vectors per session (length = number of unique templates).
5. Store session metadata: session_id, line range, raw lines, vector.

## Stage 4: Isolation Forest Anomaly Gate

- Train Isolation Forest on normal session vectors using LogHub ground-truth labels.
- At inference, score each session vector; discard normal sessions, pass only flagged anomalous sessions downstream.

### Implementation Steps
1. Load LogHub ground-truth labels and filter normal-labeled sessions for training.
2. Train `IsolationForest` (scikit-learn) on normal session vectors; save model (joblib/pickle).
3. At inference, run `model.predict()` / `decision_function()` on each session vector.
4. Threshold scores to classify session as normal/anomalous.
5. Log gate statistics (e.g., % sessions discarded) for monitoring.

## Stage 5: Embedding & FAISS Vector Store

- Transform flagged anomalous sessions into embeddings using a sentence-transformer model.
- Store only anomalous session embeddings in a FAISS vector database (normal sessions never indexed).

### Implementation Steps
1. Choose sentence-transformer model (e.g., `all-MiniLM-L6-v2`).
2. Convert each flagged session's text (concatenated raw lines or template sequence) into an embedding.
3. Build/maintain a FAISS index (e.g., `IndexFlatL2` or `IndexIVFFlat`) for anomalous session embeddings.
4. Store mapping: FAISS index position → session metadata (raw lines, line numbers, known root cause if labeled).
5. Persist FAISS index and metadata store to disk.

## Stage 6: Retrieval-Augmented Agentic Reasoning

- At inference, embed the flagged session and retrieve top-K similar historical failures from FAISS.
- Assemble a prompt with the flagged lines + retrieved examples.
- Pass to GPT-4o-mini (or equivalent LLM agent) for root cause identification and line-level failure trace output.

### Implementation Steps
1. Embed incoming flagged session using the same sentence-transformer model.
2. Query FAISS index for top-K (e.g., K=3) nearest neighbors.
3. Build prompt template combining: flagged log lines + top-K retrieved failure examples + instructions for output format.
4. Call GPT-4o-mini (or Claude API) with assembled prompt, optionally via an agent loop for multi-step reasoning (e.g., tool calls to re-query FAISS or fetch additional context).
5. Parse response into structured output: affected line range, root cause description, confidence/explanation.

## Project File Structure
LogLense/
├── data/
│   ├── raw/              # Raw LogHub log files (HDFS, BGL, Thunderbird)
│   └── processed/        # Deduplicated/parsed intermediate outputs
├── src/
│   ├── ingestion.py       # Stage 1: streaming read + dedup
│   ├── parser.py          # Stage 2: Drain-based parsing
│   ├── sessionizer.py      # Stage 3: session grouping + vectorization
│   ├── anomaly_gate.py    # Stage 4: Isolation Forest training/inference
│   ├── embedder.py        # Stage 5: sentence-transformer embeddings
│   ├── vector_store.py    # Stage 5: FAISS index management
│   ├── rag_pipeline.py    # Stage 6: retrieval + LLM/agent prompt assembly
│   └── pipeline.py        # End-to-end orchestration
├── models/
│   ├── isolation_forest.joblib
│   └── faiss_index/
├── tests/
├── .env
├── requirements.txt
└── spec.md
## Milestones

1. **M1**: Streaming ingestion + deduplication module working on sample HDFS logs.
2. **M2**: Drain parser integrated, event templates extracted and validated.
3. **M3**: Session grouping (HDFS by Block ID) + vectorization complete.
4. **M4**: Isolation Forest trained on labeled normal sessions, gating functional.
5. **M5**: Embedding + FAISS index built for flagged anomalous sessions.
6. **M6**: End-to-end RAG pipeline with LLM/agent root-cause output on HDFS dataset.
7. **M7**: Extend pipeline to BGL/Thunderbird with sliding-window sessionization.
8. **M8**: Evaluation against LogHub ground-truth labels (precision/recall on anomaly detection, qualitative review of root-cause explanations).

## Final Submission Requirements (Due: June 24, 8:00 AM)

1. **PDF Report** (ECAI template, 3 pages main text + appendix from page 4 with section-linked supporting details).
2. **Demo link** (YouTube video or hosted web demo, e.g. github.io/vercel).
3. **GitHub repo** with full commit history, remote accessible to evaluator.
4. **MS Forms entry** (individual): summary of personal contribution, tied to commit history.


## Note to Team

This is the initial project specification for LogSense, outlining the proposed pipeline stages, architecture, and implementation plan. Team members are requested to review this spec. Any further modifications, additions, or corrections will be discussed, updated, and pushed accordingly as the project progresses.


## Team

- Team Name: LogSense
- Members:
  - Vasanthakumar S
  - Atreyee Mondal
  - Dange Nikita Dilip
  - Sujith Shetty
  - Dhananjaya B R
  - Raj Shekhar
  - Balla Malleswara Rao
  - Aele Santhosh


## Datasets

Source: LogHub (https://github.com/logpai/loghub)

| Dataset    | Size      | Use Case                          |
|------------|-----------|-----------------------------------|
| HDFS       | ~11M lines | Primary pipeline development (Block ID sessions) |
| BGL        | ~4.7M lines | Sliding-window sessionization, evaluation |
| Thunderbird| ~211M lines | Scale testing, extended evaluation |

Each dataset includes ground-truth anomaly labels used for Isolation Forest training and final evaluation.

