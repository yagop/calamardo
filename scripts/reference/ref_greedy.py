"""Greedy reference for Calamardo: mlx-lm (PR #2086) generate_step.

Usage: ref_greedy.py SNAPSHOT_DIR PROMPT_FILE MAX_TOKENS
PROMPT_FILE holds the full prompt text (chat format already applied), or
token ids separated by spaces if its name ends in .ids.
Prints a JSON line with the prompt ids and the generated ids.
"""
import json
import sys
import time

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import generate_step

snapshot, prompt_file, max_tokens = sys.argv[1], sys.argv[2], int(sys.argv[3])
model, tokenizer = load(snapshot)
text = open(prompt_file).read()
if prompt_file.endswith(".ids"):
    ids = [int(x) for x in text.split()]
else:
    ids = tokenizer.encode(text, add_special_tokens=False)
eos = 200006
out = []
t = time.time()
for tok, _ in generate_step(mx.array(ids), model, max_tokens=max_tokens):
    out.append(int(tok))
    if tok == eos:
        break
print(json.dumps({"prompt": ids, "generated": out, "seconds": time.time() - t}))
