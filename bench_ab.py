#!/usr/bin/env python3
"""Tiny throughput A/B: concurrent requests, UNIQUE prompts (no prefix-cache
hits → measures raw prefill+decode), fixed output len. Reports decode tok/s,
total tok/s, and per-request latency. Usage: bench_ab.py PORT TAG"""

import json
import sys
import threading
import time
import urllib.request

PORT = sys.argv[1] if len(sys.argv) > 1 else "8100"
TAG = sys.argv[2] if len(sys.argv) > 2 else "run"
BASE = f"http://127.0.0.1:{PORT}/v1/completions"
MODEL = "/shared/data/amd_int/models/DeepSeek-V4-Flash-FP8"
N = 96  # total requests
CONC = 32  # concurrency
OUT = 128  # max_tokens (decode-heavy)
IN_REPEAT = 130  # ~ 900 in-tokens base

results = []
lock = threading.Lock()
sem = threading.Semaphore(CONC)


def one(i):
    with sem:
        # Unique prefix per request → no cross-request prefix-cache hit.
        prompt = (
            f"req{i} unique seed {i*7919}. "
            + "the archival index records that fact again " * IN_REPEAT
        )
        body = json.dumps(
            {"model": MODEL, "prompt": prompt, "max_tokens": OUT, "temperature": 0}
        ).encode()
        t0 = time.time()
        try:
            d = json.loads(
                urllib.request.urlopen(
                    urllib.request.Request(
                        BASE, data=body, headers={"Content-Type": "application/json"}
                    ),
                    timeout=600,
                ).read()
            )
            dt = time.time() - t0
            ct = d["usage"]["completion_tokens"]
            pt = d["usage"]["prompt_tokens"]
            with lock:
                results.append((dt, ct, pt))
        except Exception as e:
            with lock:
                results.append((None, 0, 0))
            print(f"  req{i} ERR {e}", flush=True)


t0 = time.time()
ths = [threading.Thread(target=one, args=(i,)) for i in range(N)]
for t in ths:
    t.start()
for t in ths:
    t.join()
wall = time.time() - t0

ok = [r for r in results if r[0] is not None]
tot_ct = sum(r[1] for r in ok)
tot_pt = sum(r[2] for r in ok)
lat = sorted(r[0] for r in ok)
p50 = lat[len(lat) // 2] if lat else 0
p90 = lat[int(len(lat) * 0.9)] if lat else 0
print(f"[{TAG}] reqs={len(ok)}/{N} conc={CONC} out={OUT} wall={wall:.1f}s")
print(
    f"[{TAG}] decode_tok/s={tot_ct/wall:.1f} total_tok/s={(tot_ct+tot_pt)/wall:.1f} "
    f"prompt_tok={tot_pt} completion_tok={tot_ct}"
)
print(f"[{TAG}] latency p50={p50:.2f}s p90={p90:.2f}s")
