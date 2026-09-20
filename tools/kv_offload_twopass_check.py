#!/usr/bin/env python3
"""Two-pass KV-offload correctness check for a vLLM server with a KV connector.

A single pass only ever SAVES, so it proves nothing about restore. This does:

  pass 1  ask N marker questions           -> populates HBM and the LMCache tier
  flood   push F unrelated long prompts    -> evicts the pass-1 prefixes from HBM
  pass 2  ask the SAME N questions again   -> the prefix can now only come back
                                              through the external tier

Each prompt hides a random marker near its start, behind a long filler, and asks
for it back under greedy decoding. That makes the answer depend on the exact
bytes of the offloaded prefix rather than on anything the model could re-derive:
a restore that lands the wrong blocks loses the marker.

What is compared is pass2-vs-pass1 text. Run it on the LMCache-off arm too --
that arm's own pass2-vs-pass1 delta is the noise floor, and it is the only
honest one, because a prefix-cache hit is not bit-reproducible against a cold
run even with no connector attached.
"""

import argparse
import json
import os
import random
import sys
import urllib.request


def post(url, payload, timeout=3600):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def ask(url, model, prompt, max_tokens):
    body = post(
        f"{url}/v1/chat/completions",
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "seed": 0,
        },
    )
    usage = body.get("usage", {}) or {}
    details = usage.get("prompt_tokens_details") or {}
    return (
        body["choices"][0]["message"]["content"],
        int(usage.get("prompt_tokens", 0)),
        int(details.get("cached_tokens", 0) or 0),
    )


def make_prompt(rng, filler_words, marker):
    filler = " ".join(
        rng.choice(("alpha", "bravo", "charlie", "delta", "echo", "foxtrot"))
        for _ in range(filler_words)
    )
    return (
        f"MARKER={marker}\n"
        "Remember the MARKER above. The following log is irrelevant.\n"
        f"{filler}\n"
        "Question: repeat the MARKER exactly, as MARKER=<value>, and nothing else."
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8330")
    ap.add_argument("--model", default="amd/GLM-5.2-MXFP4")
    ap.add_argument("--n", type=int, default=8, help="marker prompts per pass")
    ap.add_argument("--flood", type=int, default=24, help="eviction prompts")
    ap.add_argument("--filler-words", type=int, default=12000)
    # 256, not a couple of dozen: a reasoning model emits a short chain of
    # thought before the answer, and a small budget cuts the marker off -- which
    # scores as a restore failure when it is really a truncation.
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--label", default="arm")
    ap.add_argument(
        "--out-dir",
        default=".",
        help="where to write twopass_<label>.json",
    )
    args = ap.parse_args()

    # Up front, not next to the write: a bad --out-dir should fail in 0 s
    # rather than after both passes and the flood have already run.
    os.makedirs(args.out_dir, exist_ok=True)

    rng = random.Random(args.seed)
    # Markers are drawn from the run's seed, so both arms see the same prompts.
    prompts = []
    for _ in range(args.n):
        marker = f"{rng.randrange(10**7, 10**8)}"
        prompts.append((marker, make_prompt(rng, args.filler_words, marker)))

    def run_pass(name):
        out = []
        for i, (marker, p) in enumerate(prompts):
            text, ptok, cached = ask(args.url, args.model, p, args.max_tokens)
            out.append(
                {
                    "i": i,
                    "marker": marker,
                    "text": text,
                    "prompt_tokens": ptok,
                    "cached_tokens": cached,
                    "recalled": marker in text,
                }
            )
            print(
                f"[{args.label}/{name}] {i} ptok={ptok} cached={cached} "
                f"recalled={marker in text} text={text[:60]!r}",
                flush=True,
            )
        return out

    p1 = run_pass("pass1")

    print(f"[{args.label}] flooding {args.flood} prompts to evict HBM", flush=True)
    for j in range(args.flood):
        _t, ptok, cached = ask(
            args.url,
            args.model,
            make_prompt(rng, args.filler_words, f"flood{j}"),
            8,
        )
        print(f"[{args.label}/flood] {j} ptok={ptok} cached={cached}", flush=True)

    p2 = run_pass("pass2")

    same = sum(1 for a, b in zip(p1, p2) if a["text"] == b["text"])
    rec1 = sum(1 for a in p1 if a["recalled"])
    rec2 = sum(1 for a in p2 if a["recalled"])
    ext2 = sum(b["cached_tokens"] for b in p2)
    summary = {
        "label": args.label,
        "n": args.n,
        "identical_text_pass2_vs_pass1": same,
        "marker_recall_pass1": rec1,
        "marker_recall_pass2": rec2,
        "cached_tokens_pass1": sum(a["cached_tokens"] for a in p1),
        "cached_tokens_pass2": ext2,
        "pass1": p1,
        "pass2": p2,
    }
    print(
        json.dumps(
            {k: v for k, v in summary.items() if k not in ("pass1", "pass2")}, indent=2
        )
    )
    out = os.path.join(args.out_dir, f"twopass_{args.label}.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    return 0 if rec2 == args.n else 1


if __name__ == "__main__":
    sys.exit(main())
