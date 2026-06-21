"""
Golden Dataset Builder for LogLense Evaluation Pipeline.

Creates a statistically significant stratified sample of labeled HDFS sessions
from anomaly_label.csv.  Used as ground truth for:
  - Gate evaluation  (precision / recall / F1 on anomaly detection)
  - LLM evaluation   (judge calls scored against known session labels)

Statistical significance:
  We use the Cochran formula with finite-population correction to guarantee
  that the sample is large enough to estimate the anomaly rate at a given
  confidence level and margin of error.

  Default: n_anomaly=100, normal_multiplier=3.0 → 400 total sessions.
  With population=16,838, p=0.0293, 95% CI, ±5% margin → min_n ≈ 44.
  400 >> 44, so the sample is statistically significant.

Usage:
    python eval/golden_dataset.py \\
        --labels data/raw/anomaly_label.csv \\
        --n-anomaly 100 \\
        --output eval/data/golden_dataset.json
"""

import argparse
import json
import logging
import math
import random
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ── Label loading ─────────────────────────────────────────────────────────────

def load_labels(labels_path: str) -> dict:
    """
    Load anomaly_label.csv into a {block_id: label} dict.
    Supports both "Anomaly"/"Normal" and "anomaly"/"normal" casing.
    """
    labels = {}
    with open(labels_path, "r") as f:
        f.readline()  # skip header
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 2:
                block_id = parts[0].strip()
                label = parts[1].strip().capitalize()  # normalise to "Anomaly"/"Normal"
                labels[block_id] = label
    logger.info(f"Loaded {len(labels)} labeled sessions from {labels_path}")
    return labels


# ── Statistical significance ──────────────────────────────────────────────────

def compute_min_sample_size(
    population_size: int,
    p: float = 0.5,
    confidence: float = 0.95,
    margin: float = 0.05,
) -> int:
    """
    Cochran formula with finite-population correction.

    Args:
        population_size: Total labeled sessions available.
        p: Estimated proportion (use 0.5 for worst-case / maximum variance).
        confidence: Target confidence level (0.90 / 0.95 / 0.99).
        margin: Acceptable margin of error (e.g. 0.05 = ±5 percentage points).

    Returns:
        Minimum sample size (ceiling).
    """
    z = {0.90: 1.645, 0.95: 1.960, 0.99: 2.576}.get(confidence, 1.960)
    n0 = (z ** 2 * p * (1.0 - p)) / (margin ** 2)
    # Finite-population correction: n = n0 / (1 + (n0-1)/N)
    n = n0 / (1.0 + (n0 - 1.0) / population_size)
    return math.ceil(n)


# ── Stratified sampling ───────────────────────────────────────────────────────

def create_stratified_sample(
    labels_path: str,
    n_anomaly: int = 100,
    normal_multiplier: float = 3.0,
    seed: int = 42,
) -> dict:
    """
    Stratified sample of labeled HDFS sessions.

    Strategy:
      - Take up to n_anomaly anomalous sessions (all if fewer exist).
      - Take n_anomaly * normal_multiplier normal sessions.
      - This over-samples anomalies relative to the true 2.93% rate, which is
        intentional: LLM judge calls need enough anomalies to be meaningful.

    Args:
        labels_path: Path to anomaly_label.csv.
        n_anomaly: Target number of anomalous sessions (default: 100).
        normal_multiplier: Normal/anomaly ratio in the sample (default: 3.0).
        seed: Random seed for reproducibility.

    Returns:
        Dict with metadata and shuffled records list.
    """
    labels = load_labels(labels_path)

    anomaly_ids = sorted(bid for bid, lbl in labels.items() if lbl == "Anomaly")
    normal_ids = sorted(bid for bid, lbl in labels.items() if lbl == "Normal")

    rng = random.Random(seed)

    n_anom = min(n_anomaly, len(anomaly_ids))
    n_norm = min(int(n_anom * normal_multiplier), len(normal_ids))

    sampled_anomalies = rng.sample(anomaly_ids, n_anom)
    sampled_normals = rng.sample(normal_ids, n_norm)

    total = n_anom + n_norm
    actual_rate = n_anom / total if total > 0 else 0.0

    min_n_95_5 = compute_min_sample_size(
        population_size=len(labels),
        p=len(anomaly_ids) / max(len(labels), 1),
        confidence=0.95,
        margin=0.05,
    )
    min_n_95_10 = compute_min_sample_size(
        population_size=len(labels),
        p=len(anomaly_ids) / max(len(labels), 1),
        confidence=0.95,
        margin=0.10,
    )

    records = (
        [{"session_id": bid, "label": "Anomaly"} for bid in sampled_anomalies]
        + [{"session_id": bid, "label": "Normal"} for bid in sampled_normals]
    )
    rng.shuffle(records)

    return {
        "metadata": {
            "labels_source": str(labels_path),
            "seed": seed,
            "n_anomaly_requested": n_anomaly,
            "normal_multiplier": normal_multiplier,
            # Population stats
            "population_total": len(labels),
            "population_anomaly_count": len(anomaly_ids),
            "population_normal_count": len(normal_ids),
            "population_anomaly_rate": round(len(anomaly_ids) / max(len(labels), 1), 4),
            # Sample stats
            "sample_size": total,
            "sample_anomaly_count": n_anom,
            "sample_normal_count": n_norm,
            "sample_anomaly_rate": round(actual_rate, 4),
            # Statistical significance
            "min_n_95pct_ci_5pct_margin": min_n_95_5,
            "min_n_95pct_ci_10pct_margin": min_n_95_10,
            "is_significant_at_95_5": total >= min_n_95_5,
            "is_significant_at_95_10": total >= min_n_95_10,
            "confidence_note": (
                f"Sample of {total} sessions from population of {len(labels)}. "
                f"95% CI ±5% requires n≥{min_n_95_5} (satisfied: {total >= min_n_95_5}). "
                f"Anomaly class: {n_anom} sessions (population: {len(anomaly_ids)}, "
                f"coverage: {n_anom/max(len(anomaly_ids),1)*100:.1f}%)."
            ),
        },
        "records": records,
    }


# ── Persistence ───────────────────────────────────────────────────────────────

def save_golden_dataset(dataset: dict, output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=2)
    logger.info(
        f"Golden dataset saved → {output_path} "
        f"({dataset['metadata']['sample_size']} records)"
    )


def load_golden_dataset(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


# ── Accessors ─────────────────────────────────────────────────────────────────

def get_label_map(dataset: dict) -> dict:
    """Return {session_id: label} for all records."""
    return {r["session_id"]: r["label"] for r in dataset["records"]}


def get_anomalous_ids(dataset: dict) -> list:
    return [r["session_id"] for r in dataset["records"] if r["label"] == "Anomaly"]


def get_normal_ids(dataset: dict) -> list:
    return [r["session_id"] for r in dataset["records"] if r["label"] == "Normal"]


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    ap = argparse.ArgumentParser(description="Build LogLense golden evaluation dataset")
    ap.add_argument("--labels", required=True, help="Path to anomaly_label.csv")
    ap.add_argument("--n-anomaly", type=int, default=100,
                    help="Target anomalous sessions in sample (default: 100)")
    ap.add_argument("--normal-multiplier", type=float, default=3.0,
                    help="Normal/anomaly ratio in sample (default: 3.0 → 300 normal)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    ap.add_argument("--output", default="eval/data/golden_dataset.json",
                    help="Output JSON path (default: eval/data/golden_dataset.json)")
    args = ap.parse_args()

    dataset = create_stratified_sample(
        labels_path=args.labels,
        n_anomaly=args.n_anomaly,
        normal_multiplier=args.normal_multiplier,
        seed=args.seed,
    )
    save_golden_dataset(dataset, args.output)

    meta = dataset["metadata"]
    print(f"\n{'='*60}")
    print("Golden Dataset Summary")
    print(f"{'='*60}")
    print(f"  Population:            {meta['population_total']:,} sessions")
    print(f"  Population anomalies:  {meta['population_anomaly_count']} ({meta['population_anomaly_rate']*100:.2f}%)")
    print(f"  Sample size:           {meta['sample_size']}")
    print(f"  Sample anomalies:      {meta['sample_anomaly_count']}")
    print(f"  Sample normals:        {meta['sample_normal_count']}")
    print(f"  Anomaly coverage:      {meta['sample_anomaly_count']/meta['population_anomaly_count']*100:.1f}% of all anomalies")
    print(f"  Stat sig (95%/±5%):    {meta['is_significant_at_95_5']} (min n={meta['min_n_95pct_ci_5pct_margin']})")
    print(f"  Stat sig (95%/±10%):   {meta['is_significant_at_95_10']} (min n={meta['min_n_95pct_ci_10pct_margin']})")
    print(f"  Saved to:              {args.output}")
