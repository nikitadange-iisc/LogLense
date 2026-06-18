"""
Spot-check: verify 10 randomly selected lines map to sensible templates.
Picks lines spread across the file and validates template correctness.
"""
import sys
import random
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

df = pd.read_csv("data/processed/HDFS_sample_1pct_structured.csv", dtype=str)

# Pick 10 lines: first 2, last 2, 6 random from the middle
random.seed(42)
n = len(df)
indices = [0, 1] + sorted(random.sample(range(2, n - 2), 6)) + [n - 2, n - 1]

print("=" * 100)
print("  SPOT-CHECK: 10 Lines → Template Mapping")
print("=" * 100)

EXPECTED_PATTERNS = {
    "Receiving block": "Receiving block <BLOCK_ID>",
    "Served block": "Served block <BLOCK_ID> to",
    "allocateBlock": "NameSystem.allocateBlock:",
    "PacketResponder": "PacketResponder <",
    "Received block": "Received block <BLOCK_ID> of size",
    "addStoredBlock: blockMap": "blockMap updated:",
    "Verification succeeded": "Verification succeeded for <BLOCK_ID>",
    "writeBlock": "writeBlock <BLOCK_ID> received exception",
    "replicate": "replicate <BLOCK_ID> to datanode",
    "Deleting block": "Deleting block <BLOCK_ID>",
    "Transmitted block": "Transmitted block <BLOCK_ID>",
    "delete:": "NameSystem.delete: <BLOCK_ID>",
    "Got exception": "Got exception while serving <BLOCK_ID>",
    "Redundant addStoredBlock": "Redundant addStoredBlock request",
    "Unexpected error": "Unexpected error trying to delete block",
    "does not belong": "does not belong to any file",
    "receiveBlock": "Exception in receiveBlock for block",
    "empty packet": "Receiving empty packet for block",
    "Starting thread": "Starting thread to transfer block",
    "Exception writing": "Exception writing block",
    "Changing block": "Changing block file offset",
    "Adding an already": "Adding an already existing block",
}

all_pass = True
for i, idx in enumerate(indices, 1):
    row = df.iloc[idx]
    content = row["Content"]
    template = row["EventTemplate"]
    event_id = row["EventId"]
    level = row["Level"]
    line_id = row["LineId"]

    # Check: template should have <BLOCK_ID> (all HDFS lines have block IDs)
    has_block_mask = "<BLOCK_ID>" in template
    # Check: no raw block IDs leaked into template
    no_raw_leak = "blk_" not in template and "blk_-" not in template
    # Check: no raw IPs leaked into template
    no_ip_leak = not any(c.isdigit() and "." in template[max(0,j-3):j+4]
                         for j, c in enumerate(template) if c.isdigit()) or "<IP" in template
    # Check: template is a reasonable abstraction of content
    sensible = False
    for content_key, template_key in EXPECTED_PATTERNS.items():
        if content_key in content and template_key in template:
            sensible = True
            break

    status = "✅" if (has_block_mask and no_raw_leak and sensible) else "⚠️"
    if status == "⚠️":
        all_pass = False

    print(f"\n  [{i:>2}] Line {line_id:>7} | {event_id:>4} | {level:<5}")
    print(f"       Content:  {content[:90]}")
    print(f"       Template: {template[:90]}")
    print(f"       Checks:   block_mask={has_block_mask}  no_leak={no_raw_leak}  sensible={sensible}  {status}")

print("\n" + "=" * 100)
if all_pass:
    print("  RESULT: ALL 10 SPOT-CHECKED LINES MAP TO SENSIBLE TEMPLATES ✅")
else:
    print("  RESULT: SOME LINES NEED REVIEW ⚠️")
print("=" * 100)

