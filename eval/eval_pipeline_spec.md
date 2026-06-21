# LogLense Evaluation Pipeline — Specification

**Team:** LogSense  
**Author:** Dange Nikita Dilip  
**Branch:** `eval_changes`  
**Date:** 2026-06-21  

---

## 1. Purpose

The LogLense inference pipeline (gate + RAG + LLM) produces root-cause analysis reports for anomalous HDFS log sessions.  This evaluation pipeline measures whether those outputs are **correct**, **well-grounded**, and **actionable** — in a way that can be re-run automatically on every model or prompt change.

**Non-goals:** This pipeline does not modify any existing source files. It wraps the existing `analyze_log()` entry point as a black-box target.

---

## 2. Architecture

```
anomaly_label.csv
       │
       ▼
 golden_dataset.py         ← Cochran-formula stratified sample
 (400 sessions)
       │
       ├──────────────────────────────────────────────┐
       ▼                                              ▼
 gate evaluation                            LLM evaluation
 (no API key)                               (ANTHROPIC_API_KEY)
       │                                              │
  AnomalyGate.evaluate()              RAGPipeline.analyze() per session
       │                                              │
  compute_gate_metrics()              judge_all_dimensions()  ← 6 Claude chains
       │                                              │
  precision/recall/F1              6 pass-rates + weighted composite
       │                                              │
       └──────────────────── combined report ─────────┘
                                         │
                              eval/data/eval_report.json
                              LangSmith experiment (optional)
```

---

## 3. Golden Dataset

### 3.1 Population

| Property | Value |
|----------|-------|
| Source | `data/raw/anomaly_label.csv` (downloaded by `download_hdfs.py`) |
| Total sessions | 16,838 |
| Anomalous sessions | 493 (2.93%) |
| Normal sessions | 16,345 (97.07%) |

### 3.2 Sampling Strategy

Stratified, over-sampling anomalies to ensure enough positive examples for LLM judging:

| Parameter | Default | Rationale |
|-----------|---------|-----------|
| `n_anomaly` | 100 | 100% of anomalies is too many API calls; 100 is representative |
| `normal_multiplier` | 3.0 | 1:3 anomaly:normal ratio; matches rough expected test-set distribution |
| Total sample | 400 | Well above all statistical significance thresholds |
| `seed` | 42 | Reproducibility |

### 3.3 Statistical Significance

Cochran formula with finite-population correction:

```
n₀ = Z² × p × (1-p) / e²
n  = n₀ / (1 + (n₀-1) / N)
```

At p=0.0293 (observed anomaly rate), 95% confidence, ±5% margin:

| Threshold | Required n | Our n | Satisfied |
|-----------|-----------|-------|-----------|
| 95% CI / ±5% | 44 | 400 | ✓ |
| 95% CI / ±10% | 11 | 400 | ✓ |

The sample exceeds the minimum at 9× headroom. Anomaly-class coverage: 100/493 = 20.3%.

---

## 4. Evaluation Tiers

### 4.1 Tier 1 — Gate Metrics (Deterministic)

Evaluates the **Isolation Forest anomaly detection gate** against ground-truth labels.

| Metric | Description | Target |
|--------|-------------|--------|
| Precision | TP / (TP + FP) | ≥ 0.90 |
| Recall | TP / (TP + FN) | ≥ 0.90 |
| F1 Score | Harmonic mean | ≥ 0.90 (LogHub HDFS benchmark) |
| Accuracy | (TP + TN) / total | Informational |
| Coverage | Predicted / ground-truth | ≥ 0.95 |

**Implementation:** `eval/metrics.py::compute_gate_metrics()`

### 4.2 Tier 2 — LLM Quality Metrics (Judge-Based)

Six binary (0/1) rubric dimensions, each scored by a separate Claude judge chain.

| Dimension | Weight | Pass Criterion |
|-----------|--------|----------------|
| `root_cause_specificity` | 1.0× | Root cause cites specific block IDs / IPs / patterns from the actual log lines |
| `grounding` | **1.5×** | Every line in `failure_trace` exists verbatim (or near-verbatim) in the session |
| `completeness` | 1.0× | Explanation covers all major anomalous events visible in the session |
| `severity_calibration` | 0.75× | Severity label is proportionate (not inflated or deflated by ≥2 levels) |
| `actionability` | 0.75× | Recommended action names specific resources (block ID, node IP) and concrete steps |
| `retrieval_relevance` | 0.75× | ≥1 retrieved historical example matches the current failure pattern |

**Grounding is weighted 1.5× because hallucination is the most critical failure mode** — an analysis citing invented log lines is worse than an incomplete one.

**Weighted composite:**  
```
score = Σ(dim_score × weight) / Σ(weight)
```

**Pass threshold:** Weighted composite ≥ 0.70 is acceptable; ≥ 0.80 is target.

---

## 5. Judge Chain Design

Each dimension uses a LangChain chain:

```
ChatPromptTemplate.from_template(rubric) 
  | ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0.0, max_tokens=256)
  | JsonOutputParser()
```

All judges return `{"score": 0 | 1, "reason": "one sentence"}`.

Model: `claude-haiku-4-5-20251001` (fast, low-cost, sufficient for binary rubric scoring).  
Temperature: 0.0 (deterministic scoring, no randomness).

---

## 6. LangSmith Integration

### 6.1 Dataset

Golden dataset records are uploaded as LangSmith examples:
- **inputs:** `{session_id, label, log_lines[0:50]}`
- **outputs:** `{label}`

### 6.2 Evaluator Functions

Each of the 6 dimensions + 2 composite scores maps to a LangSmith evaluator function (returns `{"key": dimension, "score": 0|1}`).

### 6.3 Tracing

Set environment variables to enable full LangSmith tracing:

```bash
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=<your-key>
export LANGCHAIN_PROJECT=logsense-eval
```

---

## 7. File Structure

```
eval/
├── __init__.py
├── eval_pipeline_spec.md        ← this file
├── requirements.txt             ← langchain, langchain-anthropic, langsmith
├── golden_dataset.py            ← Cochran sampling + label loading
├── metrics.py                   ← Gate metrics + judge rubrics + aggregation
├── judges.py                    ← 6 LangChain judge chains
├── langsmith_eval.py            ← LangSmith dataset upload + ls_evaluate() runner
├── run_eval.py                  ← CLI: build-dataset | gate | llm | full
└── data/
    ├── .gitkeep
    ├── golden_dataset.json      ← generated by build-dataset (gitignored)
    └── eval_report.json         ← generated by full run (gitignored)
```

---

## 8. Running the Pipeline

### Prerequisites

```bash
# Install eval dependencies (in addition to root requirements.txt)
uv pip install -r eval/requirements.txt

# Set API keys
export ANTHROPIC_API_KEY=<your-key>
export LANGCHAIN_API_KEY=<your-key>       # optional, for LangSmith
export LANGCHAIN_TRACING_V2=true          # optional

# Download HDFS data (if not already present)
cd src && uv run python download_hdfs.py
```

### Step-by-step

```bash
# 1. Build golden dataset
uv run python eval/run_eval.py build-dataset \
    --labels data/raw/anomaly_label.csv \
    --output eval/data/golden_dataset.json

# 2. Gate evaluation (no LLM API key required)
uv run python eval/run_eval.py gate \
    --log-file data/raw/HDFS.log \
    --golden eval/data/golden_dataset.json \
    --output eval/data/gate_report.json

# 3. Full evaluation (gate + LLM judges, ~20 sessions)
uv run python eval/run_eval.py full \
    --log-file data/raw/HDFS.log \
    --labels data/raw/anomaly_label.csv \
    --golden eval/data/golden_dataset.json \
    --max-sessions 20 \
    --output eval/data/eval_report.json

# 4. (Optional) LangSmith — upload dataset and run evaluation
uv run python eval/langsmith_eval.py \
    --log-file data/raw/HDFS.log \
    --labels data/raw/anomaly_label.csv \
    --golden eval/data/golden_dataset.json \
    --dataset-name logsense-golden-v1
```

---

## 9. Output Schema

### Gate report

```json
{
  "gate_metrics": {
    "precision": 0.9712,
    "recall": 0.9345,
    "f1_score": 0.9525,
    "accuracy": 0.9980,
    "true_positives": 93,
    "false_positives": 3,
    "false_negatives": 7,
    "true_negatives": 297,
    "total_evaluated": 400,
    "coverage": 1.0
  }
}
```

### Full eval report

```json
{
  "eval_metadata": { ... },
  "gate_metrics": { ... },
  "llm_aggregate": {
    "root_cause_specificity": {"pass_rate": 0.85, "pass_count": 17, "n": 20, "weight": 1.0},
    "grounding":              {"pass_rate": 0.90, "pass_count": 18, "n": 20, "weight": 1.5},
    "completeness":           {"pass_rate": 0.80, "pass_count": 16, "n": 20, "weight": 1.0},
    "severity_calibration":   {"pass_rate": 0.75, "pass_count": 15, "n": 20, "weight": 0.75},
    "actionability":          {"pass_rate": 0.70, "pass_count": 14, "n": 20, "weight": 0.75},
    "retrieval_relevance":    {"pass_rate": 0.95, "pass_count": 19, "n": 20, "weight": 0.75},
    "composite_weighted": {"mean": 0.8214, "n": 20}
  },
  "session_results": [...]
}
```

---

## 10. Pass/Fail Thresholds

| Metric | Target | Critical Failure |
|--------|--------|-----------------|
| Gate F1 | ≥ 0.90 | < 0.85 |
| Gate Recall | ≥ 0.90 | < 0.80 (missing anomalies) |
| Grounding pass rate | ≥ 0.85 | < 0.70 (hallucination risk) |
| Weighted composite | ≥ 0.80 | < 0.70 |
| Root cause specificity | ≥ 0.80 | < 0.65 |

---

## 11. Design Decisions

| Decision | Rationale |
|----------|-----------|
| Binary (0/1) rubrics not 1–5 scales | Reduces judge variance; pass/fail is actionable |
| Grounding weighted 1.5× | Hallucinated log lines are the highest-risk failure mode |
| Haiku for judging | 6 chains × N sessions → cost matters; haiku is sufficient for binary rubric |
| `temperature=0.0` for judges | Deterministic scores; same input → same score on re-run |
| Stratified over-sample of anomalies | True rate (2.93%) too sparse for useful LLM eval with small N |
| Black-box wrapping of `analyze_log()` | Eval works independently of refactors in `src/` |
| No modification to existing files | Clean merge to main; eval is purely additive |
