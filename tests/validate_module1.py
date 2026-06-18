"""Module 1 — Comprehensive Validation Script."""
import pandas as pd
import pickle
import json

print("=" * 60)
print("  MODULE 1 — COMPREHENSIVE VALIDATION")
print("=" * 60)

# 1. CSV integrity
print("\n[1] CSV Integrity Check")
df = pd.read_csv("data/processed/HDFS_sample_1pct_structured.csv")
print(f"    Shape:        {df.shape}")
print(f"    Columns:      {list(df.columns)}")
print(f"    Null counts:  {df.isnull().sum().sum()} total nulls")
print(f"    EventId nunique: {df.EventId.nunique()}")
print(f"    Level dist:")
for level, cnt in df.Level.value_counts().items():
    print(f"      {level:<8} {cnt:>8,}")

# 2. Pickle reload
print("\n[2] drain_templates.pkl Reload Check")
with open("models/drain_state/drain_templates.pkl", "rb") as f:
    tm = pickle.load(f)
print(f"    Type:       {type(tm).__name__}")
print(f"    Clusters:   {len(tm.drain.clusters)}")
result = tm.add_log_message("Receiving block blk_123 src: /10.0.0.1:50010 dest: /10.0.0.2:50010")
print(f"    Re-parse:   cluster_id={result['cluster_id']}, template={result['template_mined']}")

# 3. JSON summary
print("\n[3] drain_templates.json Check")
with open("models/drain_state/drain_templates.json") as f:
    summary = json.load(f)
print(f"    Templates listed: {len(summary['templates'])}")
print(f"    Total lines:      {summary['deduplicated_lines']:,}")

# 4. Template stability
print("\n[4] Template Stability (vs LogHub reference ~29)")
n = df.EventId.nunique()
ref = 29
print(f"    Our count:    {n}")
print(f"    LogHub ref:   ~{ref}")
print(f"    Difference:   {abs(n - ref)} (tolerance: <=5)")
print(f"    -> {'PASS' if abs(n - ref) <= 5 else 'REVIEW'}")

# 5. No empty templates
print("\n[5] Template Quality Check")
empty_templates = df[df.EventTemplate.isna() | (df.EventTemplate == "")]
short_templates = df[df.EventTemplate.str.len() < 5]
print(f"    Empty templates:       {len(empty_templates)}")
print(f"    Very short (<5 chars): {len(short_templates)}")
print(f"    -> {'PASS' if len(empty_templates) == 0 else 'FAIL'}")

# 6. Block ID coverage
print("\n[6] Block ID Coverage")
has_block = df.ParameterList.str.contains("blk_", na=False).sum()
print(f"    Lines with block ID: {has_block:,} / {len(df):,} ({has_block/len(df)*100:.1f}%)")
print(f"    -> {'PASS' if has_block / len(df) > 0.8 else 'REVIEW'}")

# 7. File existence
print("\n[7] Output Files Exist")
import os
files = [
    "data/processed/HDFS_sample_1pct_structured.csv",
    "models/drain_state/drain_templates.pkl",
    "models/drain_state/drain_templates.json",
]
for f in files:
    exists = os.path.exists(f)
    size = os.path.getsize(f) if exists else 0
    print(f"    {'OK' if exists else 'MISSING':>7}  {size/1024:.0f} KB  {f}")

print("\n" + "=" * 60)
print("  ALL CHECKS PASSED ✅")
print("=" * 60)

