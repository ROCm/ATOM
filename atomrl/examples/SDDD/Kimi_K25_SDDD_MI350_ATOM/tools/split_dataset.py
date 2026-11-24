# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Split kimi-mtp-dataset into Phase 1 (perfectblend) and Phase 2 (mixed)."""

import json
import os

INPUT = "/dev/shm/kimi-mtp-dataset/data/train-00000-of-00001.jsonl"
PHASE1_DIR = "/dev/shm/kimi-mtp-dataset-phase1"
PHASE2_DIR = "/dev/shm/kimi-mtp-dataset-phase2"

PHASE1_SOURCE = "perfectblend"

os.makedirs(PHASE1_DIR, exist_ok=True)
os.makedirs(PHASE2_DIR, exist_ok=True)

p1_count = p2_count = 0
with (
    open(INPUT) as fin,
    open(f"{PHASE1_DIR}/train.jsonl", "w") as f1,
    open(f"{PHASE2_DIR}/train.jsonl", "w") as f2,
):
    for line in fin:
        d = json.loads(line)
        if d.get("source") == PHASE1_SOURCE:
            f1.write(line)
            p1_count += 1
        else:
            f2.write(line)
            p2_count += 1

print(f"Phase 1 (perfectblend): {p1_count} samples -> {PHASE1_DIR}/train.jsonl")
print(f"Phase 2 (mixed):        {p2_count} samples -> {PHASE2_DIR}/train.jsonl")
print(f"Total:                  {p1_count + p2_count}")
