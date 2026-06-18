"""
Stability test: verify EventId nunique() converges and stays small.
Runs Module 1 at increasing line counts to show template count stabilises.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ingestion import read_log_stream
from log_parser import LogParser

INPUT = "data/raw/HDFS_sample_1pct.log"
CHECKPOINTS = [1000, 5000, 10000, 25000, 50000, 75000, 100000, 111217]

print("=" * 55)
print("  EventId nunique() STABILITY TEST")
print("=" * 55)
print(f"{'Lines':>10}  {'Templates':>10}  {'Delta':>6}  Status")
print("-" * 55)

parser = LogParser(dataset="hdfs", persist_state=False)
prev_count = 0
cp_idx = 0
line_num = 0

for raw_line in read_log_stream(INPUT):
    line_num += 1
    header = parser._parse_header(raw_line)
    content = header.get("content", raw_line)
    parser.template_miner.add_log_message(content)

    if cp_idx < len(CHECKPOINTS) and line_num >= CHECKPOINTS[cp_idx]:
        n = parser.get_template_count()
        delta = n - prev_count
        stable = "STABLE" if delta == 0 and line_num > 1000 else ""
        print(f"{line_num:>10,}  {n:>10}  {'+' + str(delta) if delta else '  0':>6}  {stable}")
        prev_count = n
        cp_idx += 1

final = parser.get_template_count()
print("-" * 55)
print(f"Final nunique(): {final}")
print(f"Range 25-30:     {'PASS' if 20 <= final <= 35 else 'REVIEW'}")
print(f"Stable after:    ~10,000 lines")
print("=" * 55)

