#!/usr/bin/env python3
"""Throughput for one arm of an offload on/off pair, reuse designed not raced.

The companion to `kv_offload_accuracy_arms.py`: that one asks whether the
tier returns the right bytes, this one asks what having it is worth. Run it
on both arms of a time-adjacent pair; the design it belongs to is in
`recipes/atom_vllm/DeepSeek-V4.1-LMCache-perf-preregistration.md`, written
before any number existed.

`l4.sh` asked six distinct prompts and sent twenty-four requests, so four
copies of each prefix were in flight at once. How many of them hit depended
on whether they arrived together or staggered, which is why its graph-mode
numbers were bimodal (2.63 to 7.74 req/s on one configuration). Averaging
over that measures the scheduler's luck.

Here the reuse is a property of the design, not of the arrival order:

  * DOCS distinct documents are primed ONE AT A TIME before the window, so
    each prefix is stored exactly once with nothing racing it.
  * Every request in the measured window carries a UNIQUE tail, so no two
    requests ever contend for the same store, and each one's reuse is
    exactly the document prefix.
  * `ignore_eos` with a fixed `max_tokens` makes the decode work identical
    in every request and every arm -- `l4.sh` let it vary by 8%.
  * The window is long enough that start-up jitter is not the signal.

Reported per window rather than lifetime: tier supply and HBM hit rate are
deltas across the window, because the lifetime values carry the priming.
"""

import json
import sys
import threading
import time
import urllib.request

PORT = sys.argv[1] if len(sys.argv) > 1 else "8021"
LANES = int(sys.argv[2]) if len(sys.argv) > 2 else 24
REQS = int(sys.argv[3]) if len(sys.argv) > 3 else 240
DOCS = int(sys.argv[4]) if len(sys.argv) > 4 else 12
OUT_TOKENS = int(sys.argv[5]) if len(sys.argv) > 5 else 64
# Document size in repetitions of WORDS. ~7.5k tokens: long enough to own
# several 256-token blocks and a 4,096 state rung, short enough that the OFF
# arm -- which has no reuse at all and so prefills every request in full --
# finishes a 240-request window in minutes rather than an hour.
REPS = int(sys.argv[6]) if len(sys.argv) > 6 else 500
# Tier size this arm was launched with, GiB. 0 skips the pre-flight, which is
# what the OFF arm wants.
TIER_GIB = float(sys.argv[7]) if len(sys.argv) > 7 else 0.0
# PAGE bytes per token. DeepSeek-V4.1-Flash: 739,840 B per 256-token block
# over 5 planes (main latent 655,360 + three ratio-2 index planes at 16,896
# + one ratio-1 at 33,792), measured off the registration line, not derived
# from config. Another model needs its own number here.
BYTES_PER_TOK = float(sys.argv[8]) if len(sys.argv) > 8 else 739840 / 256
MODEL = "DeepSeek-V4.1-Flash"
WORDS = "alpha bravo charlie delta echo foxtrot golf hotel india juliet "


def document(d):
    # Distinct content per document, same length for all of them, so the two
    # arms see the same prefill bytes whichever documents they reuse.
    return f"Dossier {d:03d}.\n" + WORDS * REPS


def post(doc_id, tail, timeout=900):
    body = json.dumps(
        {
            "model": MODEL,
            "messages": [
                {"role": "user", "content": document(doc_id) + f"\nQuery {tail}: reply."}
            ],
            "temperature": 0.0,
            "max_tokens": OUT_TOKENS,
            "ignore_eos": True,
            "seed": 0,
        }
    ).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions", body,
        {"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as h:
        d = json.load(h)
    return time.time() - t0, d["usage"]["prompt_tokens"], d["usage"]["completion_tokens"]


def metrics():
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/metrics", timeout=10) as h:
            text = h.read().decode()
    except Exception:
        return {}
    keep = (
        "vllm:external_prefix_cache_queries_total",
        "vllm:external_prefix_cache_hits_total",
        "vllm:prefix_cache_queries_total",
        "vllm:prefix_cache_hits_total",
    )
    out = {}
    for line in text.splitlines():
        for k in keep:
            if line.startswith(k):
                out[k.split(":")[1]] = float(line.rsplit(" ", 1)[1])
    return out


# Prime one document at a time: a store that races another store is a store
# that may be declined, and then the window would measure a cache that was
# never filled rather than one that was.
primed_tokens = 0
for d in range(DOCS):
    _, primed_tokens, _ = post(d, f"prime-{d}")

# Pre-flight, not post-hoc. A supply gate can only invalidate a window after
# it has been paid for; whether the tier can physically hold the working set
# is known before the window starts, so it is checked here. Printed as a
# decomposition rather than a verdict: "insufficient" makes the next person
# re-derive this model to find where theirs differs, whereas the terms let
# them diff term by term.
if TIER_GIB > 0:
    working_set = DOCS * primed_tokens * BYTES_PER_TOK
    tier_bytes = TIER_GIB * (1 << 30)
    print(
        f"PREFLIGHT working_set={working_set / (1 << 30):.2f}GiB "
        f"= {DOCS} docs x {primed_tokens} tok x {BYTES_PER_TOK:.0f} B/tok  "
        f"tier={TIER_GIB:.0f}GiB  headroom={tier_bytes / working_set:.1f}x"
    )
    if working_set > tier_bytes:
        print(
            "PREFLIGHT REFUSED: the documents cannot all be resident, so the "
            "window would measure eviction rather than reuse. Lower DOCS or "
            "REPS, or raise the tier."
        )
        sys.exit(2)

before = metrics()
lat, ptok, ctok, errs = [], 0, 0, []
work = [(i % DOCS, f"w{i}") for i in range(REQS)]
lock = threading.Lock()
cursor = [0]


def worker():
    global ptok, ctok
    while True:
        with lock:
            if cursor[0] >= len(work):
                return
            doc_id, tail = work[cursor[0]]
            cursor[0] += 1
        try:
            dt, p, c = post(doc_id, tail)
        except Exception as exc:
            with lock:
                errs.append(type(exc).__name__)
            continue
        with lock:
            lat.append(dt)
            ptok += p
            ctok += c


t0 = time.time()
threads = [threading.Thread(target=worker) for _ in range(LANES)]
for t in threads:
    t.start()
for t in threads:
    t.join()
wall = time.time() - t0
after = metrics()


def delta(key):
    if key not in before or key not in after:
        return None
    return after[key] - before[key]


eq = delta("external_prefix_cache_queries_total")
eh = delta("external_prefix_cache_hits_total")
lq = delta("prefix_cache_queries_total")
lh = delta("prefix_cache_hits_total")
lat.sort()


def q(p):
    return lat[min(len(lat) - 1, int(len(lat) * p))] if lat else float("nan")


print(
    f"PERF req_s={len(lat) / wall:.4f} wall={wall:.2f} done={len(lat)}/{REQS} "
    f"errors={len(errs)} prompt_tok={ptok} out_tok={ctok} "
    f"p50={q(0.5):.2f} p90={q(0.9):.2f} p99={q(0.99):.2f}"
)
if eq is None:
    print("SUPPLY unavailable")
else:
    n = max(1, len(lat))
    print(
        f"SUPPLY tier={eh:.0f}/{eq:.0f} ({100 * eh / eq if eq else 0:.1f}%) "
        f"hbm={lh:.0f}/{lq:.0f} ({100 * lh / lq if lq else 0:.1f}%)"
    )
    # Per request, against what one prompt actually costs. The percentage
    # alone cannot distinguish "the tier was asked and could not answer" from
    # "the tier was never asked", and those want opposite fixes.
    print(
        f"SUPPLY_PER_REQ asked={eq / n:.0f} served={eh / n:.0f} "
        f"prompt={ptok / n:.0f} tok  (served/prompt={100 * eh / ptok if ptok else 0:.1f}%)"
    )
