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

