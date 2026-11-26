# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Generate the minimal Qwen-Image SFT smoke dataset.

8 distinct 256x256 RGB PNGs plus a JSONL that cycles through them.

Record count matters: the DiT pipeline forces dyn_bsz=False, so the native
dataloader is a plain StatefulDistributedSampler with batch_size=1 and
drop_last=True. Each rank therefore sees floor(N / dp_size) batches, and the
trainer breaks out of the epoch on StopIteration. To run S optimizer steps on
D ranks you need at least S * D records.

  usage: make_data.py [num_records] [jsonl_path]
"""

import json
import os
import sys

from PIL import Image, ImageDraw

DATA_DIR = os.environ.get("DATA_DIR", "/work/data/qwen_image_smoke")
IMG_DIR = os.path.join(DATA_DIR, "images")
S = 256

N = int(sys.argv[1]) if len(sys.argv) > 1 else 32
JSONL = sys.argv[2] if len(sys.argv) > 2 else os.path.join(DATA_DIR, "train.jsonl")

os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(os.path.dirname(os.path.abspath(JSONL)), exist_ok=True)

SPECS = [
    ("A red square centered on a white background", "white", "rect", (220, 30, 30)),
    ("A blue circle centered on a white background", "white", "ellipse", (30, 60, 220)),
    (
        "A green triangle centered on a white background",
        "white",
        "triangle",
        (30, 180, 60),
    ),
    ("A yellow square centered on a black background", "black", "rect", (240, 220, 40)),
    (
        "A magenta circle centered on a black background",
        "black",
        "ellipse",
        (220, 40, 200),
    ),
    (
        "A cyan triangle centered on a black background",
        "black",
        "triangle",
        (40, 200, 220),
    ),
    ("An orange square centered on a gray background", "gray", "rect", (240, 140, 20)),
    (
        "A purple circle centered on a gray background",
        "gray",
        "ellipse",
        (130, 40, 200),
    ),
]


def draw(shape, bg, color):
    im = Image.new("RGB", (S, S), bg)
    d = ImageDraw.Draw(im)
    a, b = 64, 192
    if shape == "rect":
        d.rectangle([a, a, b, b], fill=color)
    elif shape == "ellipse":
        d.ellipse([a, a, b, b], fill=color)
    else:
        d.polygon([(S // 2, a), (a, b), (b, b)], fill=color)
    return im


paths = []
for i, (text, bg, shape, color) in enumerate(SPECS):
    p = os.path.join(IMG_DIR, f"{i:03d}.png")
    draw(shape, bg, color).save(p, format="PNG")
    paths.append((text, p))

with open(JSONL, "w") as f:
    for n in range(N):
        text, p = paths[n % 8]
        f.write(json.dumps({"text": text, "image_path": p}) + "\n")

# ---- verification ----
with open(JSONL) as jsonl_file:
    recs = [json.loads(line) for line in jsonl_file]
assert len(recs) == N, len(recs)
seen = set()
for r in recs:
    assert set(r.keys()) == {"text", "image_path"}, r.keys()
    assert os.path.isabs(r["image_path"]), r
    assert "image_bytes" not in r
    im = Image.open(r["image_path"])
    assert im.mode == "RGB", (r["image_path"], im.mode)
    assert im.size == (S, S), (r["image_path"], im.size)
    seen.add(r["image_path"])
assert len(seen) == 8, seen
assert os.path.dirname(JSONL) != IMG_DIR

print(f"OK: {len(recs)} jsonl records, {len(seen)} unique 256x256 RGB PNGs")
print("jsonl :", JSONL)
print("images:", IMG_DIR)
for d in (8, 1):
    print(f"  supports up to {len(recs) // d} optimizer steps on {d} rank(s)")
