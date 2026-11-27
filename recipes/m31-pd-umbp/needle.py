#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""needle.py <base url> <tokenizer dir>: V1's read-back of prompts HBM evicted into the UMBP tier.

1. N needle prompts, each a haystack of words distinct from every other prompt's, with a needle (a vault's 6-digit
   code) in its first few hundred tokens and the question at its end, answered at temperature 0. HBM now holds them.
2. A flood of distinct prompts, --flood times HBM's KV tokens (atom:kv_cache_blocks_total), so that HBM evicts the
   needle prompts; under save_policy lazy their blocks go to UMBP as HBM is about to evict them.
3. The same N prompts again. They must load from UMBP (atom:lmcache_loaded_tokens grows by most of their tokens, no
   load fails) and every answer must still name the code.
The last line printed is READBACK PASS or READBACK FAIL <why>; the exit status follows it.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import random
import re
import sys
import time
import urllib.request

from transformers import AutoTokenizer

SYLLABLES = [
    "ka",
    "lo",
    "mir",
    "tes",
    "van",
    "dor",
    "pel",
    "su",
    "rin",
    "gat",
    "ob",
    "lee",
    "fra",
    "zun",
    "hok",
    "ten",
    "mi",
    "rau",
    "sel",
    "quo",
    "bra",
    "nim",
    "ost",
    "yel",
    "cro",
    "dav",
    "ish",
    "plo",
    "vex",
    "wun",
]


def words(rng: random.Random, n: int) -> list[str]:
    return [
        "".join(rng.choice(SYLLABLES) for _ in range(rng.randint(2, 3)))
        for _ in range(n)
    ]


def text_of(
    tok, rng: random.Random, n_tokens: int, head: str = "", tail: str = ""
) -> str:
    """Words until the whole text is about n_tokens tokens."""
    budget = n_tokens - len(tok.encode(head + tail, add_special_tokens=False))
    body = []
    while budget > 0:
        chunk = " ".join(words(rng, max(64, budget // 3))) + ". "
        body.append(chunk)
        budget -= len(tok.encode(chunk, add_special_tokens=False))
    return head + "".join(body) + tail


def post(url: str, body: dict, timeout: float = 3600) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def metrics(base: str) -> dict[str, float]:
    with urllib.request.urlopen(base + "/metrics", timeout=60) as response:
        text = response.read().decode()
    out = {}
    for line in text.splitlines():
        m = re.match(r"^(atom:[a-z_]+)(\{[^}]*\})?\s+([0-9.eE+-]+)$", line)
        if m:
            out[m.group(1)] = out.get(m.group(1), 0.0) + float(m.group(3))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("tokenizer")
    ap.add_argument("--prompts", type=int, default=10)
    ap.add_argument("--prompt-tokens", type=int, default=32768)
    ap.add_argument(
        "--flood", type=float, default=2.0, help="flood tokens, in HBM KV tokens"
    )
    ap.add_argument("--flood-prompt-tokens", type=int, default=32768)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--flood-concurrency", type=int, default=0, help="0: --concurrency")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=816)
    args = ap.parse_args()

    base = args.base.rstrip("/")
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    served = json.load(urllib.request.urlopen(base + "/v1/models", timeout=60))["data"][
        0
    ]["id"]
    rng = random.Random(args.seed)
    needles = []
    for i in range(args.prompts):
        name, code = (
            f"V{i:02d}-{rng.randint(100, 999)}",
            f"{rng.randint(100000, 999999)}",
        )
        head = (
            f"Document {i} of the archive. Important: the access code for vault {name} is {code}. "
            "Keep it in mind while reading the filler below.\n\n"
        )
        tail = f"\n\nWhat is the access code for vault {name}? Answer with the 6-digit code only."
        needles.append(
            (
                name,
                code,
                text_of(
                    tok,
                    random.Random(args.seed * 1000 + i),
                    args.prompt_tokens,
                    head,
                    tail,
                ),
            )
        )

    def ask(prompt: str, max_tokens: int) -> str:
        reply = post(
            base + "/v1/chat/completions",
            {
                "model": served,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "max_tokens": max_tokens,
            },
        )
        message = reply["choices"][0]["message"]
        return (
            (message.get("content") or "")
            + "\n"
            + (message.get("reasoning_content") or "")
        )

    def needle_pass(label: str) -> tuple[int, list[str]]:
        t0 = time.time()
        with cf.ThreadPoolExecutor(args.concurrency) as pool:
            answers = list(pool.map(lambda n: ask(n[2], args.max_tokens), needles))
        right = 0
        for (name, code, _), answer in zip(needles, answers):
            ok = code in answer
            right += ok
            print(
                f"  {label} vault {name}: {'ok' if ok else 'WRONG'} ({answer.strip()[-80:]!r})",
                flush=True,
            )
        print(
            f"{label}: {right}/{len(needles)} right in {time.time() - t0:.0f} s",
            flush=True,
        )
        return right, answers

    m0 = metrics(base)
    hbm_tokens = int(m0.get("atom:kv_cache_blocks_total", 0)) * 128
    print(
        f"HBM KV: {hbm_tokens} tokens; {args.prompts} needle prompts of {args.prompt_tokens} tokens",
        flush=True,
    )
    first, _ = needle_pass("pass 1")
    m1 = metrics(base)

    flood_tokens = int(args.flood * hbm_tokens)
    n_flood = max(1, flood_tokens // args.flood_prompt_tokens)
    print(
        f"flood: {n_flood} prompts of {args.flood_prompt_tokens} tokens ({n_flood * args.flood_prompt_tokens} tokens)",
        flush=True,
    )
    t0 = time.time()
    flood_rng = random.Random(args.seed + 1)
    seeds = [flood_rng.randrange(1 << 30) for _ in range(n_flood)]

    def flood_one(seed: int) -> None:
        prompt = text_of(
            tok,
            random.Random(seed),
            args.flood_prompt_tokens,
            f"Filler {seed}. ",
            "\n\nSay OK.",
        )
        ask(prompt, 1)

    with cf.ThreadPoolExecutor(args.flood_concurrency or args.concurrency) as pool:
        list(pool.map(flood_one, seeds))
    m2 = metrics(base)
    print(f"flood done in {time.time() - t0:.0f} s", flush=True)
    # The tier's demotions finish after the flood's last prefill.
    time.sleep(30)
    m2 = metrics(base)

    second, _ = needle_pass("pass 2")
    m3 = metrics(base)

    def delta(a, b, k):
        return int(b.get(k, 0) - a.get(k, 0))

    needle_tokens = args.prompts * args.prompt_tokens
    loaded = delta(m2, m3, "atom:lmcache_loaded_tokens_total")
    failed = delta(m2, m3, "atom:lmcache_load_failures_total")
    for k in (
        "atom:lmcache_saved_tokens_total",
        "atom:lmcache_save_requests_total",
        "atom:lmcache_lazy_evicted_unsaved_blocks_total",
        "atom:lmcache_lazy_demotions_failed_total",
        "atom:lmcache_loaded_tokens_total",
        "atom:lmcache_load_requests_total",
        "atom:lmcache_load_failures_total",
        "atom:prefix_cache_offload_tokens_total",
    ):
        print(
            f"  {k}: pass1 {delta(m0, m1, k)}  flood {delta(m1, m2, k)}  pass2 {delta(m2, m3, k)}",
            flush=True,
        )
    problems = []
    if first < len(needles):
        problems.append(
            f"pass 1 (computed) found {first}/{len(needles)}: the haystack is too hard, not a tier fault"
        )
    if second < len(needles):
        problems.append(f"pass 2 (read back) found {second}/{len(needles)}")
    if loaded < needle_tokens // 2:
        problems.append(
            f"pass 2 loaded {loaded} tokens from the tier, under half the needle prompts' {needle_tokens}"
        )
    if failed:
        problems.append(f"{failed} loads failed in pass 2")
    print(f"pass 2 loaded {loaded} of {needle_tokens} needle tokens from the tier")
    print("READBACK PASS" if not problems else "READBACK FAIL " + "; ".join(problems))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
