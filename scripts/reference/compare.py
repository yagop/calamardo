"""Compares Calamardo's greedy tokens with the reference's.

Usage: compare.py REFERENCE_JSON CALAMARDO_IDS
Exits with status 1 if the generated tokens differ.
"""
import json
import sys

ref = json.load(open(sys.argv[1]))["generated"]
ours = [int(x) for x in open(sys.argv[2]).read().split()]
diff = next((i for i, (a, b) in enumerate(zip(ref, ours)) if a != b), None)
if diff is None and len(ref) == len(ours):
    print(f"identical: {len(ours)} tokens")
    sys.exit(0)
diff = diff if diff is not None else min(len(ref), len(ours))
print(f"differ at token {diff}: reference {ref[diff:diff + 5]} vs calamardo {ours[diff:diff + 5]}")
sys.exit(1)
