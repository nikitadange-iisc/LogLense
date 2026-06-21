"""
Evaluation metrics for the LogLense pipeline.

Two tiers:

  Tier 1 — Gate metrics (deterministic, no LLM):
    Precision, recall, F1, accuracy for the Isolation Forest anomaly gate,
    computed against ground-truth labels from anomaly_label.csv.

  Tier 2 — LLM quality metrics (judge-based):
    Six rubric dimensions scored 0/1 by a Claude judge per analysis output.
    Dimensions: root_cause_specificity, grounding, completeness,
                severity_calibration, actionability, retrieval_relevance.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ── Tier 1: Gate metrics ──────────────────────────────────────────────────────

def compute_gate_metrics(predictions: dict, ground_truth: dict) -> dict:
    """
    Compute binary classification metrics for the anomaly gate.

    Args:
        predictions:  {session_id: "Anomaly" | "Normal"}  — gate outputs.
        ground_truth: {session_id: "Anomaly" | "Normal"}  — label file.

    Returns:
        Dict with precision, recall, f1_score, accuracy, confusion matrix counts,
        and coverage (fraction of ground-truth sessions that were predicted).
    """
    common = set(predictions.keys()) & set(ground_truth.keys())
    if not common:
        return {"error": "No session IDs in common between predictions and ground truth."}

    tp = fp = fn = tn = 0
    for sid in common:
        pred_anom = predictions[sid].lower() == "anomaly"
        gt_anom = ground_truth[sid].lower() == "anomaly"
        if pred_anom and gt_anom:
            tp += 1
        elif pred_anom and not gt_anom:
            fp += 1
        elif not pred_anom and gt_anom:
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(common)
    coverage = len(common) / max(len(ground_truth), 1)

    return {
        "precision":       round(precision, 4),
        "recall":          round(recall, 4),
        "f1_score":        round(f1, 4),
        "accuracy":        round(accuracy, 4),
        "true_positives":  tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives":  tn,
        "total_evaluated": len(common),
        "coverage":        round(coverage, 4),
        "note": (
            f"Evaluated {len(common)}/{len(ground_truth)} ground-truth sessions "
            f"(those present in the log file). "
            f"F1 benchmark: ≥0.90 (LogHub HDFS standard)."
        ),
    }


# ── Tier 2: LLM quality rubrics ───────────────────────────────────────────────

JUDGE_DIMENSIONS = [
    "root_cause_specificity",
    "grounding",
    "completeness",
    "severity_calibration",
    "actionability",
    "retrieval_relevance",
]

# Rubric templates use {placeholder} keys matched by judges.py
JUDGE_RUBRICS = {
    "root_cause_specificity": {
        "weight": 1.0,
        "description": "Root cause is specific, non-generic, and references actual session components.",
        "rubric": """You are evaluating a log root cause analysis output.

TASK: Score whether the root_cause is SPECIFIC (references actual log components, block IDs, or patterns) vs GENERIC (could apply to any log session without evidence).

Session ID: {session_id}
Root cause claimed: {root_cause}
Actual log lines (first 15):
{log_lines}

SCORING:
- Score 1 (PASS): Root cause names specific components, block IDs, IP addresses, or timing patterns visible in the actual log lines above. It could not be written without reading these specific logs.
- Score 0 (FAIL): Root cause is generic ("network issue", "disk error", "replication failure") without grounding in specific log evidence, OR it invents components not present in the log.

Reply ONLY with valid JSON — no explanation outside the JSON:
{{"score": 0, "reason": "one sentence"}}""",
    },

    "grounding": {
        "weight": 1.5,  # higher weight: hallucination is most critical failure mode
        "description": "Every line cited in failure_trace exists verbatim (or near-verbatim) in the session.",
        "rubric": """You are evaluating a log root cause analysis output for hallucination.

TASK: Check whether every log line cited in failure_trace actually appears in the actual session log lines below. A line is grounded if it appears verbatim or with only minor whitespace/truncation differences.

Actual session log lines:
{log_lines}

Failure trace (lines cited by the LLM):
{failure_trace}

SCORING:
- Score 1 (PASS): Every line in the failure_trace exists in the actual log lines (verbatim or near-verbatim). Zero invented lines.
- Score 0 (FAIL): One or more lines in the failure_trace do NOT appear in the actual log lines — the LLM invented or paraphrased log content.

Reply ONLY with valid JSON:
{{"score": 0, "reason": "one sentence; if score=0, quote the first hallucinated line"}}""",
    },

    "completeness": {
        "weight": 1.0,
        "description": "Explanation addresses all major anomalous events visible in the session.",
        "rubric": """You are evaluating a log root cause analysis output for completeness.

TASK: Assess whether the explanation covers the main anomalous patterns visible in the log lines.

Actual session log lines:
{log_lines}

LLM explanation:
{explanation}

SCORING:
- Score 1 (PASS): The explanation mentions the key anomalous patterns visible in the logs (e.g., if multiple DataNodes serve the same block, or if a block is served before being fully written).
- Score 0 (FAIL): The explanation ignores one or more significant anomalous patterns clearly visible in the log lines above.

Reply ONLY with valid JSON:
{{"score": 0, "reason": "one sentence; if score=0, name the pattern that was missed"}}""",
    },

    "severity_calibration": {
        "weight": 0.75,
        "description": "Severity label (critical/high/medium/low) is appropriate given the evidence.",
        "rubric": """You are evaluating a log root cause analysis output for appropriate severity classification.

TASK: Judge whether the assigned severity is proportionate to the evidence in the log lines.

Severity assigned: {severity}
Root cause: {root_cause}
Session log lines:
{log_lines}

SEVERITY GUIDE:
- critical: potential data loss, corruption, or complete service failure with concrete evidence
- high: clear anomaly with service-impacting behaviour (wrong nodes, timing violations)
- medium: anomaly present but impact unclear or recoverable
- low: minor deviation, likely self-healing

SCORING:
- Score 1 (PASS): Severity is proportionate — neither inflated (e.g., "critical" for a minor retry) nor deflated (e.g., "low" for clear data corruption evidence).
- Score 0 (FAIL): Severity is clearly miscalibrated by at least two levels given the evidence.

Reply ONLY with valid JSON:
{{"score": 0, "reason": "one sentence"}}""",
    },

    "actionability": {
        "weight": 0.75,
        "description": "Recommended actions are specific, concrete, and tied to this session.",
        "rubric": """You are evaluating a log root cause analysis output for actionability.

TASK: Score whether the recommended_action provides specific, concrete investigation steps tied to this exact session (not generic advice).

Session ID: {session_id}
Recommended action: {recommended_action}
Root cause: {root_cause}

SCORING:
- Score 1 (PASS): Recommendations name specific resources (block IDs, node IPs, log paths), describe concrete investigation steps, and are clearly tied to this session's failure.
- Score 0 (FAIL): Recommendations are generic ("check the logs", "contact the team", "investigate the cluster") without session-specific guidance.

Reply ONLY with valid JSON:
{{"score": 0, "reason": "one sentence"}}""",
    },

    "retrieval_relevance": {
        "weight": 0.75,
        "description": "Retrieved historical examples are relevant to the current failure pattern.",
        "rubric": """You are evaluating whether retrieved historical examples are relevant to the current anomalous log session.

TASK: Judge whether the retrieved examples show failure patterns similar to the current session.

Current session log lines (first 10):
{log_lines}

Retrieved historical examples:
{retrieved_examples}

SCORING:
- Score 1 (PASS): At least one retrieved example shows a clearly similar failure pattern (same event types, similar HDFS block operations, or same anomaly category).
- Score 0 (FAIL): All retrieved examples are unrelated to the current failure, OR no examples were retrieved (empty retrieval).

Reply ONLY with valid JSON:
{{"score": 0, "reason": "one sentence"}}""",
    },
}


# ── Score aggregation ─────────────────────────────────────────────────────────

@dataclass
class SessionEvalResult:
    """Stores evaluation output for one analyzed session."""
    session_id: str
    label: str
    gate_prediction: Optional[str] = None
    llm_scores: dict = field(default_factory=dict)      # dim -> {score, reason}
    llm_composite_score: Optional[float] = None
    llm_weighted_score: Optional[float] = None
    latency_sec: Optional[float] = None
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "session_id":         self.session_id,
            "label":              self.label,
            "gate_prediction":    self.gate_prediction,
            "llm_scores":         self.llm_scores,
            "llm_composite_score": self.llm_composite_score,
            "llm_weighted_score": self.llm_weighted_score,
            "latency_sec":        self.latency_sec,
            "error":              self.error,
        }


def compute_weighted_score(scores: dict) -> float:
    """Weighted composite score across all judge dimensions."""
    total_weight = 0.0
    weighted_sum = 0.0
    for dim, rubric in JUDGE_RUBRICS.items():
        if dim in scores:
            w = rubric.get("weight", 1.0)
            weighted_sum += scores[dim].get("score", 0) * w
            total_weight += w
    return round(weighted_sum / total_weight, 4) if total_weight > 0 else 0.0


def aggregate_llm_scores(results: list) -> dict:
    """
    Aggregate LLM judge scores across a list of SessionEvalResults.

    Returns per-dimension pass rates + weighted composite average.
    """
    dim_scores = {dim: [] for dim in JUDGE_DIMENSIONS}
    composite_scores = []
    weighted_scores = []

    for result in results:
        if not result.llm_scores:
            continue
        for dim in JUDGE_DIMENSIONS:
            if dim in result.llm_scores:
                dim_scores[dim].append(result.llm_scores[dim].get("score", 0))
        if result.llm_composite_score is not None:
            composite_scores.append(result.llm_composite_score)
        if result.llm_weighted_score is not None:
            weighted_scores.append(result.llm_weighted_score)

    agg = {}
    for dim, scores in dim_scores.items():
        if scores:
            agg[dim] = {
                "pass_rate": round(float(np.mean(scores)), 4),
                "pass_count": int(sum(scores)),
                "n": len(scores),
                "weight": JUDGE_RUBRICS[dim].get("weight", 1.0),
                "description": JUDGE_RUBRICS[dim]["description"],
            }

    if composite_scores:
        agg["composite_unweighted"] = {
            "mean": round(float(np.mean(composite_scores)), 4),
            "n": len(composite_scores),
        }
    if weighted_scores:
        agg["composite_weighted"] = {
            "mean": round(float(np.mean(weighted_scores)), 4),
            "n": len(weighted_scores),
            "note": "Grounding has 1.5× weight (hallucination is highest-risk failure mode).",
        }

    return agg
