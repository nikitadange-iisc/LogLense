# LogLense Spec
# LogSense: An Agentic AI Framework for Root Cause Analysis of Large-Scale System Logs

## Project Spec

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

