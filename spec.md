# LogLense Spec
# LogSense: An Agentic AI Framework for Root Cause Analysis of Large-Scale System Logs

## Project Spec

### Abstract
Modern distributed systems generate log data at large scale, making it hard to analyze and computationally expensive for naive LLM-based analysis. Direct ingestion of raw logs into an LLM exhausts context windows, inflates token cost, and fails to achieve proper reasoning over failure events.

This project proposes a multi-stage retrieval-augmented agentic pipeline that compresses log volume before any LLM involvement, enabling precise anomaly explanation and failure trace identification at scale. The pipeline uses the LogHub dataset (HDFS ~11M lines, BGL ~4.7M lines, Thunderbird ~211M lines).

