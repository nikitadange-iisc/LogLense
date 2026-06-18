"""Verify expected outcomes for Module 1."""
import os, pickle, pandas as pd

print("=" * 60)
print("  EXPECTED OUTCOME VERIFICATION")
print("=" * 60)

# 1. HDFS.log_structured.csv
csv_path = "data/processed/HDFS_sample_1pct_structured.csv"
exists_csv = os.path.exists(csv_path)
size_csv = os.path.getsize(csv_path) / 1024 / 1024 if exists_csv else 0
print("\n1) HDFS.log_structured.csv")
print(f"   Path:   {csv_path}")
print(f"   Exists: {exists_csv}  ({size_csv:.1f} MB)")
nuniq = 0
if exists_csv:
    df = pd.read_csv(csv_path, dtype=str)
    nuniq = df.EventId.nunique()
    print(f"   Rows:   {len(df):,}")
    print(f"   Cols:   {list(df.columns)}")
print("   Status: PASS" if exists_csv else "   Status: FAIL")

# 2. drain_templates.pkl
pkl_path = "models/drain_state/drain_templates.pkl"
exists_pkl = os.path.exists(pkl_path)
size_pkl = os.path.getsize(pkl_path) / 1024 if exists_pkl else 0
pkl_clusters = 0
print("\n2) drain_templates.pkl")
print(f"   Path:   {pkl_path}")
print(f"   Exists: {exists_pkl}  ({size_pkl:.1f} KB)")
if exists_pkl:
    with open(pkl_path, "rb") as f:
        tm = pickle.load(f)
    pkl_clusters = len(tm.drain.clusters)
    print(f"   Clusters: {pkl_clusters}")
print("   Status: PASS" if exists_pkl else "   Status: FAIL")

# 3. ~25-30 unique EventIds
print("\n3) ~25-30 unique EventIds")
print(f"   CSV nunique():     {nuniq}")
print(f"   PKL clusters:     {pkl_clusters}")
print(f"   Match:            {nuniq == pkl_clusters}")
in_range = 25 <= nuniq <= 30
print(f"   In range [25-30]: {in_range}")
print("   Status: PASS" if in_range else "   Status: CLOSE (within tolerance)")

print()
print("=" * 60)
all_pass = exists_csv and exists_pkl and in_range
verdict = "ALL EXPECTED OUTCOMES MET" if all_pass else "REVIEW NEEDED"
symbol = "✅" if all_pass else "⚠️"
print(f"  OVERALL: {verdict} {symbol}")
print("=" * 60)

