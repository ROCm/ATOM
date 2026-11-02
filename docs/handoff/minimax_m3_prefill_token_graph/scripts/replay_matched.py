#!/usr/bin/env python3
"""Matched-shape replay for MiniMax-M3 TP4 + EAGLE3, one profiler window per phase.

Reproduces the two step shapes of the MI455-vs-MI355 comparison in LLM-Booster #201
(profile/results/minimax-m3-mi355-vs-mi455-tp4-eagle3-agentx.md):

  prefill  prefill[bs=1 tok=2517 ctx=105301 ...]
           prewarm a 102,784-token prompt (803 full blocks of 128), then send it again
           with 2,517 fresh tokens appended -> one step with 102,784 cached + 2,517 new.
  decode   decode[bs=3 tok=12 d=3 spec=3 ... sk~661362]
           three distinct 220,454-token prompts, each prewarmed first, then sent together:
           all three re-prefill only their 38-token tail in the same step and enter decode
           together, so the window starts at sk = 3 x 220,454 + a few tokens.

Prompts are random token ids, so no tokenizer is needed and lengths are exact.
The server must run with --enable-prefix-caching, block size 128, and PROFILE_DIR set
(server.sh). Usage (in container):
  python3 replay_matched.py --out /logs/agentic/matched --server-log <server log>
"""
import argparse
import os
import random
import shutil
import sys
import threading
import time
from pathlib import Path

import requests

p = argparse.ArgumentParser()
p.add_argument("--url", default="http://localhost:8890")
p.add_argument("--model", default="/models/MiniMax-M3-MXFP4")
p.add_argument("--out", required=True, help="dir; traces are moved to <out>/{prefill,decode}")
p.add_argument("--profile-dir", required=True, help="server PROFILE_DIR (where ranks write)")
p.add_argument("--server-log", help="server log, to verify the scheduled prefill shapes")
p.add_argument("--prefill-cached", type=int, default=102784)
p.add_argument("--prefill-new", type=int, default=2517)
p.add_argument("--prefill-decode-tokens", type=int, default=1,
               help="max_tokens of the traced prefill request; >1 keeps its bs=1 decode steps in the window")
p.add_argument("--decode-ctx", type=int, default=220454)
p.add_argument("--decode-bs", type=int, default=3)
p.add_argument("--decode-window", type=float, default=5.0, help="seconds of decode to trace")
p.add_argument("--phases", default="prefill,decode")
p.add_argument("--seed", type=int, default=20261026)
p.add_argument("--no-profile", action="store_true", help="server runs without a profiler; only time decode")
a = p.parse_args()

out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
rng = random.Random(a.seed)
# Plain-text vocabulary range of the MiniMax-M3 tokenizer; avoids special/control ids.
LO, HI = 1000, 150000


def log(msg):
    line = f"{time.strftime('%F %T')} {msg}"
    print(line, flush=True)
    with open(out / "replay.log", "a") as f:
        f.write(line + "\n")


def ids(n):
    return [rng.randrange(LO, HI) for _ in range(n)]


def complete(prompt, max_tokens, stream=False, on_token=None, timeout=3600):
    body = {"model": a.model, "prompt": prompt, "max_tokens": max_tokens,
            "temperature": 0, "ignore_eos": True, "stream": stream}
    r = requests.post(f"{a.url}/v1/completions", json=body, stream=stream, timeout=timeout)
    r.raise_for_status()
    if not stream:
        return r.json()
    for line in r.iter_lines():
        if line.startswith(b"data: ") and line != b"data: [DONE]" and on_token:
            on_token()


def profile(cmd):
    if a.no_profile:
        return {}
    r = requests.post(f"{a.url}/{cmd}_profile", timeout=3600)
    r.raise_for_status()
    return r.json()


def collect(phase, t0):
    """Move the trace files written since t0 into <out>/<phase>/rank_N/."""
    dst = out / phase
    n = 0
    for f in Path(a.profile_dir).glob("rank_*/*.json.gz"):
        if f.stat().st_mtime >= t0:
            (dst / f.parent.name).mkdir(parents=True, exist_ok=True)
            shutil.move(str(f), dst / f.parent.name / f.name)
            n += 1
    log(f"{phase}: {n} trace files -> {dst}")


def server_lines(since):
    if not a.server_log:
        return []
    with open(a.server_log, errors="replace") as f:
        f.seek(since)
        return f.read().splitlines()


def log_pos():
    return Path(a.server_log).stat().st_size if a.server_log else 0


log(f"args {vars(a)}")
complete(ids(2048), 16)
log("warm request done")

if "prefill" in a.phases:
    base = ids(a.prefill_cached)
    complete(base, 1)
    log(f"prefill prewarm: {a.prefill_cached} tokens cached")
    pos, t0 = log_pos(), time.time()
    profile("start")
    complete(base + ids(a.prefill_new), a.prefill_decode_tokens)
    profile("stop")
    want = f"1 reqs, {a.prefill_new} new tokens (done: [{a.prefill_cached}]"
    hit = [l for l in server_lines(pos) if "Scheduled prefill batch" in l and want in l]
    log(f"prefill shape check '{want}': {'OK' if hit else 'NOT FOUND'}")
    if a.server_log and not hit:
        for l in server_lines(pos):
            if "Scheduled prefill batch" in l:
                log(f"  saw: {l.strip()[:200]}")
    collect("prefill", t0)

if "decode" in a.phases:
    prompts = [ids(a.decode_ctx) for _ in range(a.decode_bs)]
    for i, pr in enumerate(prompts):
        complete(pr, 1)
        log(f"decode prewarm {i}: {a.decode_ctx} tokens cached")
    first = threading.Barrier(a.decode_bs + 1, timeout=1800)
    counts = [0] * a.decode_bs

    def run(i):
        seen = [False]

        def tok():
            counts[i] += 1
            if not seen[0]:
                seen[0] = True
                first.wait()
        try:
            complete(prompts[i], 32768, stream=True, on_token=tok)
        except Exception as e:  # connection closed when the script exits
            log(f"decode stream {i} ended: {e!r}"[:200])

    for i in range(a.decode_bs):
        threading.Thread(target=run, args=(i,), daemon=True).start()
    pos = log_pos()
    first.wait()
    log(f"all {a.decode_bs} decoding, chunks so far {counts}")

    def rate(label, secs):
        c0, s0 = sum(counts), time.time()
        time.sleep(secs)
        c1, s1 = sum(counts), time.time()
        log(f"decode rate {label}: {(c1 - c0) / (s1 - s0) / a.decode_bs:.2f} chunks/s per stream "
            f"({(s1 - s0) * a.decode_bs / max(c1 - c0, 1) * 1000:.2f} ms/chunk)")

    time.sleep(3)
    rate("before profiler", a.decode_window)
    t0 = time.time()
    profile("start")
    rate("profiler on" if not a.no_profile else "no profiler, 2nd window", a.decode_window)
    profile("stop")
    log(f"decode window {a.decode_window}s done, stream chunks {counts}")
    tails = [l.strip()[:200] for l in server_lines(pos) if "Scheduled prefill batch" in l]
    log("decode re-prefill: " + (" | ".join(tails[:4]) or "n/a"))
    collect("decode", t0)
    # Streams are still open; the driver stops the server right after this script.

log("DONE")
sys.stdout.flush()
os._exit(0)
