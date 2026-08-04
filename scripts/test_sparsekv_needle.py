#!/usr/bin/env python3
"""Long-context needle-in-a-haystack test for SparseKV swap correctness.

Builds a long prompt with a secret passphrase placed EARLY (outside the
most-recent hot-buffer window), then asks for it. With hot_buffer_size <
context length, the needle is not in the initial hot set, so retrieving it
correctly requires the indexer to select its position and the coordinator to
swap that token's KV in from the CPU cold pool. Correct retrieval => swapped-in
KV is correct.

Usage: python scripts/test_sparsekv_needle.py [PORT] [FILLER_LINES] [DEPTH_FRAC]
"""

import json
import sys
import urllib.request

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 30000
N_LINES = int(sys.argv[2]) if len(sys.argv) > 2 else 2600
DEPTH = float(sys.argv[3]) if len(sys.argv) > 3 else 0.08  # needle near the start
MODEL = "/mnt/models/GLM-5.2-MXFP4"
PASSPHRASE = "aurora-7731-zephyr"

lines = [
    f"Record {i}: the inventory checkpoint for sector {i} completed nominally "
    f"with no anomalies detected during the routine sweep."
    for i in range(N_LINES)
]
needle_at = int(N_LINES * DEPTH)
lines[needle_at] = (
    f"SPECIAL NOTE: The secret passphrase is {PASSPHRASE}. "
    f"Keep this passphrase safe and remember it."
)
haystack = "\n".join(lines)
prompt = (
    "You are reviewing a long log. Read it carefully.\n\n"
    + haystack
    + "\n\nQuestion: What is the secret passphrase mentioned in the SPECIAL NOTE "
    "above? Reply with ONLY the passphrase, nothing else."
)

body = json.dumps(
    {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 512,
    }
).encode()

req = urllib.request.Request(
    f"http://127.0.0.1:{PORT}/v1/chat/completions",
    data=body,
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(req, timeout=600) as r:
    resp = json.load(r)

answer = resp["choices"][0]["message"]["content"].strip()
usage = resp.get("usage", {})
prompt_toks = usage.get("prompt_tokens", "?")
found = PASSPHRASE in answer
print(f"prompt_tokens = {prompt_toks}  (needle at line {needle_at}/{N_LINES})")
print(f"model answer  = {answer!r}")
print(f"RESULT: {'PASS' if found else 'FAIL'} (expected {PASSPHRASE!r})")
sys.exit(0 if found else 1)
