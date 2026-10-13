#!/usr/bin/env python3
"""Teacher-forced logprobs on exact prompt shapes. Text identity was no use.

Two criteria died before this one, and both were mine:

  * Sampled text. 13 of 17 shapes had the EAGER arm differing from itself
    between two sends, so the oracle was not stable. The control shape --
    block-aligned, implicated by nothing -- also differed, which is how it
    was caught.
  * Logprobs compared across a cache hit. Two identical requests to one
    server with prefix caching on agreed on 1 of 161 prompt positions, max
    deviation 1.71. A prefix-cache hit is not bit-reproducible against a cold
    run; comparing across one measures that, not the shape.

So: prefix caching OFF in both arms, every request a miss, concurrency 1.
The question left is exactly #2502's -- does the graph arm compute a short
prefill differently from eager -- with the cache question removed.

The noise floor is measured here, not assumed: the same prompt is sent twice
to the same arm, both misses, and the deviation between those is the yardstick
the eager-vs-graph deviation is read against.

Shapes are exact. `/v1/completions` applies no chat template, and every shape
is built by tokenizing up to the target and verified against
`usage.prompt_tokens`; a shape whose count does not match its target is
reported as MISSIZED rather than silently measured. The chat endpoint added
~88 tokens to every prompt in the previous attempt, so none of the
`block*m + 1` shapes it claimed to test were ever sent.

Usage: shape_lp.py <port> <arm> <out.json>
"""

import json
import sys
import urllib.request

PORT, ARM, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
MODEL = "DeepSeek-V4.1-Flash"
BLOCK, WINDOW, MAXB = 256, 128, 8192


def post(path, payload, timeout=600):
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}{path}", json.dumps(payload).encode(),
        {"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as h:
        return json.load(h)


def completion(prompt):
    return post("/v1/completions", {
        "model": MODEL, "prompt": prompt, "max_tokens": 1,
        "temperature": 0.0, "prompt_logprobs": 0, "seed": 0,
    })


def logprobs_of(resp):
    lp = resp["choices"][0].get("prompt_logprobs") or []
    return [None if e is None else max(v["logprob"] for v in e.values()) for e in lp]


# Build a prompt of exactly n tokens by growing a word list and measuring.
_CACHE = {}


def prompt_of(n):
    if n in _CACHE:
        return _CACHE[n]
    unit = "token "
    lo, hi = 1, n * 2 + 8
    best = None
    while lo <= hi:
        mid = (lo + hi) // 2
        text = unit * mid
        got = completion(text)["usage"]["prompt_tokens"]
        if got == n:
            best = text
            break
        if got < n:
            lo = mid + 1
        else:
            hi = mid - 1
    _CACHE[n] = best
    return best


def deviation(a, b):
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if not pairs:
        return None
    return max(abs(x - y) for x, y in pairs), sum(abs(x - y) for x, y in pairs) / len(pairs)


targets = [("one_token", 1), ("two_token", 2)]
targets += [(f"block_m1_{m}", BLOCK * m + 1) for m in (1, 2, 4, 8)]
targets += [(f"win_m1_{m}", WINDOW * m + 1) for m in (1, 3, 5) if (WINDOW * m + 1) % BLOCK != 1]
targets += [("maxb_k1", MAXB + 1)]
targets += [(f"tiny_{n}", n) for n in (8, 64, 128)]
targets += [("block_aligned", BLOCK * 8), ("win_aligned", WINDOW * 9)]

out = {"arm": ARM, "shapes": {}}
for name, n in targets:
    rec = {"target": n}
    try:
        text = prompt_of(n)
        if text is None:
            rec["status"] = "MISSIZED: no word count hits this token target"
            out["shapes"][name] = rec
            print(f"{ARM} {name:16s} target={n:6d}  {rec['status']}")
            continue
        r1 = completion(text)
        r2 = completion(text)
        got = r1["usage"]["prompt_tokens"]
        rec["prompt_tokens"] = got
        if got != n:
            rec["status"] = f"MISSIZED: asked {n}, server counted {got}"
            out["shapes"][name] = rec
            print(f"{ARM} {name:16s} {rec['status']}")
            continue
        a, b = logprobs_of(r1), logprobs_of(r2)
        rec["logprobs"] = a
        rec["self_deviation"] = deviation(a, b)
        rec["status"] = "ok"
        sd = rec["self_deviation"]
        print(f"{ARM} {name:16s} tokens={got:6d}  self_max={sd[0]:.3e} self_mean={sd[1]:.3e}")
    except Exception as exc:
        rec["status"] = f"ERROR {type(exc).__name__}: {exc}"
        print(f"{ARM} {name:16s} {rec['status']}")
    out["shapes"][name] = rec

json.dump(out, open(OUT, "w"))
print(f"{ARM}: wrote {OUT}")
