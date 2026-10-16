#!/usr/bin/env python3
"""The trigger, not just the shape: a `block*m + 1` prompt sent twice.

The logprob probe ran with prefix caching off, which is what isolates the
cudagraph question -- but it also makes #2501's trigger unreachable. That
fault needs the SECOND send: a block-aligned prefix hit leaves exactly one
token to compute, and a prefill step whose every row is one token is what
faults. With caching off there is no hit and no 1-token tail.

So this arm runs with caching on (and the connector, which V4.1 requires for
caching at all), sends each exact shape twice, and asks only the question a
fault answers: did the engine survive. Output correctness is not compared --
a hit is not bit-reproducible against a cold run, which is what killed two
earlier criteria.

What makes it evidence rather than a liveness check is the hit counter: each
shape reports how many prompt tokens the second send got from cache, so a
shape that never took a hit is reported as NOT REACHED instead of as passing.
"""

import json
import sys
import urllib.request

PORT, OUT = sys.argv[1], sys.argv[2]
MODEL = "DeepSeek-V4.1-Flash"
BLOCK, WINDOW, MAXB = 256, 128, 8192


def post(payload, timeout=600):
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/completions",
        json.dumps(payload).encode(),
        {"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as h:
        return json.load(h)


def completion(prompt):
    return post(
        {
            "model": MODEL,
            "prompt": prompt,
            "max_tokens": 1,
            "temperature": 0.0,
            "seed": 0,
        }
    )


def hits():
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{PORT}/metrics", timeout=10
        ) as h:
            t = h.read().decode()
    except Exception:  # noqa: BLE001 - a probe reports failures, it does not raise
        return None
    out = {}
    for line in t.splitlines():
        for k in (
            "vllm:prefix_cache_hits_total",
            "vllm:external_prefix_cache_hits_total",
        ):
            if line.startswith(k):
                out[k] = float(line.rsplit(" ", 1)[1])
    return out


def prompt_of(n):
    unit = "token "
    lo, hi = 1, n * 2 + 8
    while lo <= hi:
        mid = (lo + hi) // 2
        got = completion(unit * mid)["usage"]["prompt_tokens"]
        if got == n:
            return unit * mid
        if got < n:
            lo = mid + 1
        else:
            hi = mid - 1
    return None


targets = [(f"block_m1_{m}", BLOCK * m + 1) for m in (1, 2, 4, 8, 32)]
targets += [(f"win_m1_{m}", WINDOW * m + 1) for m in (1, 3, 5)]
targets += [("maxb_k1", MAXB + 1), ("block_aligned", BLOCK * 8)]

rows = []
for name, n in targets:
    rec = {"shape": name, "target": n}
    text = prompt_of(n)
    if text is None:
        rec["status"] = "MISSIZED"
        rows.append(rec)
        print(f"{name:16s} {rec['status']}")
        continue
    try:
        r1 = completion(text)
        rec["prompt_tokens"] = r1["usage"]["prompt_tokens"]
        before = hits()
        r2 = completion(text)
        after = hits()
        got_hits = None
        if before and after:
            got_hits = sum(after.get(k, 0) - before.get(k, 0) for k in after)
        rec["second_send_cached_tokens"] = got_hits
        tail = None if got_hits is None else rec["prompt_tokens"] - got_hits
        rec["uncached_tail"] = tail
        if got_hits in (None, 0):
            rec["status"] = "NOT REACHED: the second send took no hit"
        else:
            rec["status"] = "survived"
        print(
            f"{name:16s} tokens={rec['prompt_tokens']:6d} cached={got_hits} "
            f"tail={tail}  {rec['status']}"
        )
    except Exception as exc:  # noqa: BLE001 - a probe records faults as data
        rec["status"] = f"FAULT {type(exc).__name__}: {exc}"
        print(f"{name:16s} {rec['status']}")
        rows.append(rec)
        json.dump(rows, open(OUT, "w"))
        sys.exit(3)
    rows.append(rec)
json.dump(rows, open(OUT, "w"))
reached = [r for r in rows if r.get("status") == "survived"]
print(f"\n{len(reached)}/{len(rows)} shapes actually took a hit on the second send")
print(
    "1-token tails reached:",
    ", ".join(r["shape"] for r in reached if r.get("uncached_tail") == 1) or "NONE",
)
