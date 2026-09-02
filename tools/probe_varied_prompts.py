#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Send MANY DIFFERENT prompts and report whether each one terminates.

Companion to probe_same_prompt.py, which repeats a single prompt. That one
cannot see an expert-routing fault: identical prompts route to identical
experts every time, so an expert-parallel MoE all-to-all is barely exercised.
A workload of varied prompts scatters tokens across every expert and every EP
rank, which is what lm_eval does and what a repeated prompt does not.

So the two probes answer different questions:

    probe_same_prompt.py   are the weights right?   (determinism, one route)
    probe_varied_prompts.py are the ROUTES right?   (many prompts, full fan-out)

THE NUMBER THAT MATTERS is the finish_reason split. A model that answers well
but never emits EOS shows up here as `length` on every request while the text
still looks fine — the exact failure that makes lm_eval hang while a hand-run
prompt looks correct.

Two prompt sets:

  facts   short prompts with a checkable answer, so correctness is decided
          here rather than by eye. Wrong answers point at weights; right
          answers that never stop point at termination.
  varied  synthetic prompts of differing length and content, to spread the
          expert routing as widely as possible.

Usage:
    # is termination working at all, on prompts we can grade?
    python tools/probe_varied_prompts.py --mode facts --max-tokens 32

    # does a broad routing fan-out break it?
    python tools/probe_varied_prompts.py --mode varied --n 64 --max-tokens 64

    # A/B two servers (they cannot run at once on the same GPUs)
    python tools/probe_varied_prompts.py --mode facts --save ep.json
    python tools/probe_varied_prompts.py --compare ep.json tp.json
"""

import argparse
import collections
import concurrent.futures
import json
import random
import sys
import time
import urllib.error
import urllib.request

# Short prompts with a substring we can grade against. Deliberately trivial:
# the point is to separate "the model is wrong" from "the model never stops",
# not to measure capability.
FACTS = [
    ("The capital of France is", "Paris"),
    ("2 + 2 =", "4"),
    ("The largest planet in our solar system is", "Jupiter"),
    ("Water freezes at 0 degrees", "Celsius"),
    ("The chemical symbol for gold is", "Au"),
    ("The author of 'Romeo and Juliet' is", "Shakespeare"),
    ("There are how many days in a leap year? Answer:", "366"),
    ("The square root of 81 is", "9"),
    ("The primary colour made by mixing blue and yellow is", "green"),
    ("The first person to walk on the moon was", "Armstrong"),
    ("10 * 12 =", "120"),
    ("The ocean between Europe and North America is the", "Atlantic"),
]

_TOPICS = (
    "distributed systems memory hierarchies compiler design network protocols "
    "numerical methods graph algorithms operating system schedulers database "
    "indexing cryptographic hashing signal processing control theory robotics "
    "computer vision language models reinforcement learning type systems"
).split()


def varied_prompts(n: int, seed: int = 0) -> list[str]:
    """`n` distinct prompts of differing length and vocabulary.

    Deterministic for a given seed so two runs are comparable. Length is varied
    on purpose: a uniform prompt length gives a uniform token count per rank,
    which is the easy case for an all-to-all.
    """
    rng = random.Random(seed)
    out = []
    for i in range(n):
        topic = rng.choice(_TOPICS)
        filler = " ".join(rng.choice(_TOPICS) for _ in range(rng.randint(5, 120)))
        out.append(
            f"Question {i}: briefly explain {topic}. "
            f"Context keywords: {filler}. Answer:"
        )
    return out


def one_request(url: str, payload: dict, timeout: float) -> dict:
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
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
    dt = time.perf_counter() - t0
    try:
        choice = data["choices"][0]
        text = choice.get("text")
        if text is None:
            text = choice.get("message", {}).get("content", "")
        return {
            "ok": True,
            "text": text,
            "finish_reason": choice.get("finish_reason"),
            "latency": dt,
        }
    except (KeyError, IndexError):
        return {"ok": False, "error": f"unexpected response: {str(data)[:200]}"}


def _divergence(text_b: str, text_a: str, ctx: int = 12) -> tuple:
    """Index of the first differing word, with a little context from each side.

    Word-level rather than character-level: a single wrong token usually
    changes one word, and word indices read directly as "how many tokens in
    did this run go wrong".
    """
    wb, wa = text_b.split(), text_a.split()
    n = min(len(wb), len(wa))
    i = 0
    while i < n and wb[i] == wa[i]:
        i += 1
    return (
        i,
        " ".join(wb[max(0, i - 3) : i + ctx]),
        " ".join(wa[max(0, i - 3) : i + ctx]),
    )


def compare_runs(path_a: str, path_b: str) -> int:
    """Diff two saved runs prompt by prompt."""
    with open(path_a) as f:
        a = json.load(f)
    with open(path_b) as f:
        b = json.load(f)
    ra, rb = a["results"], b["results"]
    if len(ra) != len(rb):
        print(f"different run sizes: {len(ra)} vs {len(rb)}")
        return 2

    same_text = differ = 0
    reason_flips = collections.Counter()
    divergences = []
    for i, (x, y) in enumerate(zip(ra, rb)):
        if not (x.get("ok") and y.get("ok")):
            continue
        if x["text"] == y["text"]:
            same_text += 1
        else:
            differ += 1
            divergences.append((i, _divergence(y["text"], x["text"])))
        if x["finish_reason"] != y["finish_reason"]:
            reason_flips[(y["finish_reason"], x["finish_reason"])] += 1

    print(f"{path_a} vs {path_b}: {same_text} identical, {differ} differing")
    if divergences:
        # Where a run goes wrong localises the fault far better than whether
        # the final text reads correctly. Diverging at word 0 means the very
        # first sampled token differs — the forward pass itself. Diverging
        # hundreds of words in means the two agreed until some length-dependent
        # boundary (sliding-window rollover, a new KV block, a CUDA-graph
        # capture-size bucket) was crossed.
        print("\n-- first divergence (word index into the completion) --")
        for i, (w, ctx_b, ctx_a) in divergences:
            print(f"  prompt {i:3d}: word {w}")
            print(f"      b: ...{ctx_b}")
            print(f"      a: ...{ctx_a}")
        early = sum(1 for _, (w, _, _) in divergences if w == 0)
        print(
            f"\n  {early}/{len(divergences)} diverge at the first word "
            f"(forward-pass difference); {len(divergences) - early} diverge "
            f"later (length-dependent)."
        )
    if reason_flips:
        print("\n-- finish_reason changes (b -> a) --")
        for (fb, fa), n in reason_flips.most_common():
            print(f"  {n:4d}x  {fb} -> {fa}")
    print("\n-- verdict --")
    if differ == 0 and not reason_flips:
        print("  Identical. No divergence on this prompt set.")
        return 0
    if reason_flips:
        print("  Termination behaviour changed between the two runs. That is")
        print("  the signal that matters: a stop->length flip means the model")
        print("  stopped emitting EOS, which hangs lm_eval regardless of how")
        print("  reasonable the text looks.")
    return 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000/v1/completions")
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-V4-Pro")
    ap.add_argument(
        "--mode",
        choices=("facts", "varied", "mixed"),
        default="mixed",
        help="facts: gradeable answers. varied: wide routing fan-out.",
    )
    ap.add_argument("--n", type=int, default=32, help="prompts in varied/mixed")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sequential", action="store_true")
    ap.add_argument("--show-text", action="store_true")
    ap.add_argument("--save", metavar="FILE")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    args = ap.parse_args()

    if args.compare:
        return compare_runs(*args.compare)

    if args.mode == "facts":
        pairs = list(FACTS)
    elif args.mode == "varied":
        pairs = [(p, None) for p in varied_prompts(args.n, args.seed)]
    else:
        pairs = list(FACTS) + [
            (p, None) for p in varied_prompts(args.n, args.seed)
        ]

    print(f"Sending {len(pairs)} DIFFERENT prompts to {args.url}")
    print(f"  mode={args.mode} max_tokens={args.max_tokens} temp={args.temperature}")

    def send(pair):
        prompt, _ = pair
        return one_request(
            args.url,
            {
                "model": args.model,
                "prompt": prompt,
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
            },
            args.timeout,
        )

    t0 = time.perf_counter()
    if args.sequential:
        results = [send(p) for p in pairs]
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(pairs)) as pool:
            results = list(pool.map(send, pairs))
    wall = time.perf_counter() - t0

    ok = [r for r in results if r["ok"]]
    failed = [r for r in results if not r["ok"]]
    print(f"\nwall={wall:.1f}s  ok={len(ok)}  failed={len(failed)}")
    if failed:
        print("\n-- transport failures --")
        for msg, n in collections.Counter(r["error"] for r in failed).most_common():
            print(f"  {n:4d}x {msg}")
    if not ok:
        print("\nno successful responses")
        return 2

    reasons = collections.Counter(r["finish_reason"] for r in ok)
    print("\n-- finish_reason (THE signal) --")
    for reason, n in reasons.most_common():
        note = "" if reason == "stop" else "   <-- never emitted EOS"
        print(f"  {n:4d}x {reason}{note}")

    graded = [(p, exp, r) for (p, exp), r in zip(pairs, results) if exp and r["ok"]]
    if graded:
        right = [g for g in graded if g[1].lower() in g[2]["text"].lower()]
        print(f"\n-- answer check: {len(right)}/{len(graded)} correct --")
        for prompt, exp, r in graded:
            hit = exp.lower() in r["text"].lower()
            mark = "ok  " if hit else "WRONG"
            body = r["text"] if args.show_text else r["text"][:60].replace("\n", "\\n")
            print(f"  [{mark}] want {exp!r:14} finish={r['finish_reason']:<7} {body!r}")

    if args.mode != "facts":
        uniq = len({r["text"] for r in ok})
        print(f"\n-- distinct completions: {uniq}/{len(ok)} --")
        if uniq == 1:
            print("  Every different prompt produced the SAME text — the model")
            print("  is not conditioning on input at all.")

    if args.save:
        with open(args.save, "w") as f:
            json.dump(
                {"args": vars(args), "prompts": [p for p, _ in pairs],
                 "results": results},
                f,
                indent=2,
            )
        print(f"\nsaved {len(results)} responses to {args.save}")

    print("\n-- verdict --")
    if set(reasons) == {"stop"}:
        print("  Every request stopped on its own. Termination works on this")
        print("  prompt set; if lm_eval still hangs, widen --n or use longer")
        print("  prompts to spread the expert routing further.")
        return 0
    if "stop" not in reasons:
        print("  NOTHING stopped on its own — every request ran to max_tokens.")
        print("  EOS is never being sampled. With a large or unset max_tokens")
        print("  that is exactly what makes lm_eval never finish.")
        return 1
    print("  MIXED: some stopped, some did not. Prompt-dependent, which fits a")
    print("  routing fault far better than a bad weight — a wrong weight would")
    print("  affect every prompt equally. Compare the two groups' prompts.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
