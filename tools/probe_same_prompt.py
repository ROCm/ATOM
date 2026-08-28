#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Send one prompt N times concurrently and report whether the answers agree.

Built for the asymmetric-rapidserve failure "most requests finish but some do
not". Under greedy decoding (temperature 0) the same prompt must produce a
BYTE-IDENTICAL completion every time, no matter which decode rank serves it. So
any divergence localises the fault, and the shape of the divergence names it:

  all identical, all finish_reason=stop   -> decode is healthy at this shape
  all identical, all finish_reason=length -> uniformly wrong (or max_tokens
                                             is simply too small — check the text)
  a MINORITY differ                       -> rank-dependent. With N decode ranks
                                             and least_requests routing, expect
                                             the bad group to be about n/ranks.
  every response differs                  -> not deterministic at all; suspect
                                             batch-dependent state rather than
                                             weights

CAVEAT, measured: this stack is NOT batch-invariant. Baseline (no rapidserve)
already produces several distinct greedy completions for one prompt, so
"responses differ" on its own proves nothing. Always take a baseline with the
same --n and --max-tokens and compare the two, rather than reading a single run.

SPLITTING PREFILL FROM DECODE. Under rapidserve the FIRST generated token is
sampled in the prefill process and shipped to decode (PrefillDone
.sampled_token_id, engine_core.py:910). So:

    --max-tokens 1   exercises prefill only — its weights, its attention, and
                     the KV masking. Decode merely relays the token.
    --max-tokens 32  brings decode's own forward passes in.

If --max-tokens 1 agrees with baseline and longer runs diverge, the fault is
downstream of prefill. If it already disagrees, the fault is in prefill.

Use --save/--compare to diff two servers (you cannot run both at once on the
same GPUs):

    python tools/probe_same_prompt.py --n 8 --max-tokens 1 --save asym.json
    # restart the server without --enable-rapidserve
    python tools/probe_same_prompt.py --n 8 --max-tokens 1 --save base.json
    python tools/probe_same_prompt.py --compare asym.json base.json

Usage:
    python tools/probe_same_prompt.py --n 32
    python tools/probe_same_prompt.py --n 32 --max-tokens 64 --show-text
    python tools/probe_same_prompt.py --n 8 --sequential
"""

import argparse
import collections
import random
import concurrent.futures
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request


def one_request(url: str, payload: dict, timeout: float) -> dict:
    """Return {ok, text, finish_reason, latency, error} for a single call."""
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.load(resp)
    except urllib.error.HTTPError as e:
        return {"ok": False, "error": f"HTTP {e.code}: {e.read()[:200]!r}"}
    except Exception as e:  # timeout, connection reset, ...
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
    dt = time.perf_counter() - t0

    try:
        choice = data["choices"][0]
        text = choice.get("text")
        if text is None:  # chat endpoint
            text = choice.get("message", {}).get("content", "")
        return {
            "ok": True,
            "text": text,
            "finish_reason": choice.get("finish_reason"),
            "latency": dt,
        }
    except (KeyError, IndexError):
        return {"ok": False, "error": f"unexpected response: {str(data)[:200]}"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000/v1/completions")
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-V4-Pro")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument(
        "--prompt-tokens",
        type=int,
        help="ignore --prompt and synthesise one of exactly this many tokens. "
        "Use >2048 to force V4 onto the sparse/absorbed prefill path, where the "
        "KV masking is actually exercised (attention_mla.py:1402).",
    )
    ap.add_argument("--n", type=int, default=32, help="how many copies to send")
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="keep at 0 — determinism is the whole point of this probe",
    )
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument(
        "--sequential",
        action="store_true",
        help="send one at a time instead of concurrently; isolates whether a "
        "divergence needs a real multi-request batch to appear",
    )
    ap.add_argument("--save", metavar="FILE", help="write results as JSON")
    ap.add_argument(
        "--compare",
        nargs=2,
        metavar=("A", "B"),
        help="diff two saved runs instead of sending anything",
    )
    ap.add_argument("--show-text", action="store_true", help="print each variant")
    args = ap.parse_args()

    if args.compare:
        return compare_runs(*args.compare)

    prompt = args.prompt
    if args.prompt_tokens:
        prompt, actual = build_prompt(args.prompt_tokens, args.model)
        print(f"synthesised prompt: {actual} tokens ({len(prompt)} chars)")
        if actual != args.prompt_tokens:
            print(f"  note: asked for {args.prompt_tokens}; retokenised to {actual}")

    payload = {
        "model": args.model,
        "prompt": 1000*args.prompt,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
    }
    mode = "sequentially" if args.sequential else "concurrently"
    print(f"Sending {args.n} identical requests {mode} to {args.url}")
    shown = prompt if len(prompt) < 60 else prompt[:57] + "..."
    print(f"  prompt={shown!r} max_tokens={args.max_tokens} "
          f"temperature={args.temperature}")

    t0 = time.perf_counter()
    if args.sequential:
        results = [one_request(args.url, payload, args.timeout) for _ in range(args.n)]
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.n) as pool:
            results = list(
                pool.map(
                    lambda _: one_request(args.url, payload, args.timeout),
                    range(args.n),
                )
            )
    wall = time.perf_counter() - t0

    failed = [r for r in results if not r["ok"]]
    ok = [r for r in results if r["ok"]]
    print(f"\nwall={wall:.1f}s  ok={len(ok)}  failed={len(failed)}")

    if failed:
        print("\n-- transport failures --")
        for msg, n in collections.Counter(r["error"] for r in failed).most_common():
            print(f"  {n:4d}x {msg}")

    if not ok:
        print("\nno successful responses — nothing to compare")
        return 2

    reasons = collections.Counter(r["finish_reason"] for r in ok)
    print("\n-- finish_reason --")
    for reason, n in reasons.most_common():
        note = "" if reason == "stop" else "   <-- never emitted EOS"
        print(f"  {n:4d}x {reason}{note}")

    lat = sorted(r["latency"] for r in ok)
    print(
        f"\n-- latency --\n  min={lat[0]:.2f}s  median={lat[len(lat) // 2]:.2f}s  "
        f"max={lat[-1]:.2f}s"
    )

    groups = collections.defaultdict(list)
    for i, r in enumerate(ok):
        groups[hashlib.sha1(r["text"].encode()).hexdigest()[:8]].append(i)

    print(f"\n-- distinct completions: {len(groups)} --")
    ranked = sorted(groups.items(), key=lambda kv: -len(kv[1]))
    for digest, idxs in ranked:
        text = ok[idxs[0]]["text"]
        reason = ok[idxs[0]]["finish_reason"]
        preview = text if args.show_text else text[:100].replace("\n", "\\n")
        print(f"\n  [{digest}] {len(idxs):4d}x  finish_reason={reason}")
        print(f"      requests: {idxs[:12]}{' ...' if len(idxs) > 12 else ''}")
        print(f"      {preview!r}")

    if args.save:
        with open(args.save, "w") as f:
            json.dump({"args": vars(args), "results": results}, f, indent=2)
        print(f"\nsaved {len(results)} responses to {args.save}")

    print("\n-- verdict --")
    if len(groups) == 1 and set(reasons) == {"stop"}:
        print("  All identical and all stopped. Decode looks healthy at this shape.")
        print("  Re-run with a larger --n, or --max-tokens, to widen the net.")
        return 0
    if len(groups) == 1:
        print("  All identical but not all stopped. Either max_tokens is simply")
        print("  too small for this prompt, or decode is uniformly wrong — read")
        print("  the text above to tell which.")
        return 1

    minority = sum(len(v) for _, v in ranked[1:])
    print(f"  {len(groups)} distinct answers to one greedy prompt: NOT deterministic.")
    print(f"  Majority {len(ranked[0][1])}, minority {minority}.")
    print("  If the minority is roughly n/num_decode_ranks, suspect one bad rank:")
    print("  re-run with --sequential; if that agrees, the fault is batch- or")
    print("  capture-shape-dependent rather than a bad weight.")
    return 1


def build_prompt(n_tokens: int, model: str) -> tuple[str, int]:
    """A deterministic prompt of (as close as possible to) n_tokens tokens.

    Exact length matters here: the whole point of a long prompt is to cross
    `topk_tokens` so V4 takes the sparse/absorbed prefill path, and "roughly 8k"
    is not a claim you can make from a character count. Tokenises with the real
    tokenizer and reports what it actually got.

    Deliberately varied text rather than one repeated sentence — a highly
    repetitive prompt tokenises unusually densely, so a word-count estimate
    would be off by a lot in exactly the direction that matters.
    """
    rng = random.Random(0)
    vocab = (
        "system memory kernel tensor gradient parallel latency throughput cache "
        "buffer pointer thread process schedule allocate compute matrix vector "
        "attention weight layer model token sequence batch inference training "
        "quantise dispatch pipeline register operand fabric interconnect"
    ).split()
    text = " ".join(rng.choice(vocab) for _ in range(max(n_tokens * 2, 64)))
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    except Exception as exc:
        print(f"warning: no tokenizer ({exc}); falling back to a word estimate")
        words = " ".join(rng.choice(vocab) for _ in range(int(n_tokens * 0.75)))
        return words, -1
    ids = tok(text, add_special_tokens=False)["input_ids"]
    while len(ids) < n_tokens:
        text += " " + " ".join(rng.choice(vocab) for _ in range(n_tokens))
        ids = tok(text, add_special_tokens=False)["input_ids"]
    out = tok.decode(ids[:n_tokens])
    return out, len(tok(out, add_special_tokens=False)["input_ids"])


def compare_runs(path_a: str, path_b: str) -> int:
    """Diff two saved runs. Set-vs-set, because order is not meaningful."""
    with open(path_a) as f:
        a = json.load(f)
    with open(path_b) as f:
        b = json.load(f)
    texts_a = {r["text"] for r in a["results"] if r.get("ok")}
    texts_b = {r["text"] for r in b["results"] if r.get("ok")}
    print(f"{path_a}: {len(texts_a)} distinct / {len(a['results'])} responses")
    print(f"{path_b}: {len(texts_b)} distinct / {len(b['results'])} responses")

    shared = texts_a & texts_b
    print(f"\nshared completions: {len(shared)}")
    only_a = texts_a - texts_b
    only_b = texts_b - texts_a
    for label, only in ((path_a, only_a), (path_b, only_b)):
        if only:
            print(f"\n-- only in {label} ({len(only)}) --")
            for t in list(only)[:5]:
                print(f"   {t[:100]!r}")

    print("\n-- verdict --")
    if not only_a and not only_b:
        print("  Identical completion sets. No divergence at this shape.")
        return 0
    if shared:
        print("  Overlapping but not identical. Given this stack is not")
        print("  batch-invariant, that is weak evidence on its own — widen --n")
        print("  before concluding, and check whether the non-shared answers are")
        print("  merely different phrasings or actually wrong.")
        return 1
    print("  DISJOINT completion sets: the two servers do not agree at all.")
    print("  That is not sampling noise.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
