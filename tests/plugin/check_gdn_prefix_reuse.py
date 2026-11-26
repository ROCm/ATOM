#!/usr/bin/env python3
"""Acceptance check for GDN prefix caching: a cache hit must not change output.

Not a pytest unit test -- it drives a running SGLang server, hence the `check_`
prefix so collection skips it. Run it against a server started with
`--mamba-radix-cache-strategy extra_buffer --page-size 64`.

The failure mode this guards against is silent. A hybrid GDN model's radix node
carries both KV pages and a recurrent state snapshot; if the snapshot is stale
or was never taken, a prefix hit resumes from the wrong state and the model
keeps generating fluent-but-wrong text. Accuracy benchmarks blur that into
noise, so compare byte-for-byte instead:

  warm: [flush] -> P (populates the tree) -> P+S (hits the P prefix)
  cold: [flush] -> P+S (no prefix to hit), several times

The tree is populated with P alone because extra_buffer only stores a GDN
state at the last 64-token boundary a request reached: populating with P+S1
puts that boundary inside S1, past where P+S2 diverges, and nothing is reused.

Two things make a byte-for-byte verdict trustworthy, and both are enforced here
rather than left to the reader:

* **The warm arm must actually hit.** Without a hit the two arms are the same
  computation and every case passes vacuously. The warm request's
  `cached_tokens` is read back and a zero is reported INCONCLUSIVE.
* **The cold arm is not byte-deterministic at temperature 0**, with or without
  DFLASH: MoE reductions and speculative verify batch shapes both move
  near-ties, so the first tokens can flip between e.g. entering and skipping
  the thinking block. The cold arm therefore runs several times and the warm
  output must equal one of them byte for byte. It is a FAIL only when every
  cold run agrees and the warm run differs; anything else is INCONCLUSIVE --
  the harness failing to measure, not the cache failing.

Usage: check_gdn_prefix_reuse.py [--port 31000] [--prefix-tokens 2048 2080]
                                 [--cold-runs 3]
                                 [--multiturn]
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

PORT = 31000
MODEL = "/model/Qwen3.5-397B-A17B-FP8"

# Constrained enough that a near-tie on formatting cannot flip it; see the
# module docstring on why free-form answers are unusable under DFLASH.
BARE = " Reply with only the numeric value: no words, no punctuation, no markdown."


def post(path, payload=None, port=PORT, timeout=600, retries=30):
    # /flush_cache answers 400 while warmup requests are still in flight.
    if path == "/flush_cache":
        for _ in range(retries):
            try:
                return _post(path, payload, port, timeout)
            except urllib.error.HTTPError as e:
                if e.code != 400:
                    raise
                time.sleep(2)
    return _post(path, payload, port, timeout)


def _post(path, payload, port, timeout):
    data = json.dumps(payload).encode() if payload is not None else b""
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read().decode()
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return body


class Sample:
    """One temperature-0 completion: text, per-token (token id, logprob), hit size."""

    def __init__(self, text, tokens, cached):
        self.text = text
        self.tokens = tokens
        self.cached = cached


def gen(prompt, port, max_new_tokens=32, input_ids=None):
    payload = {
        "sampling_params": {"temperature": 0, "max_new_tokens": max_new_tokens},
        "return_logprob": True,
    }
    if input_ids is not None:
        payload["input_ids"] = input_ids
    else:
        payload["text"] = prompt
    out = post("/generate", payload, port=port)
    meta = out["meta_info"]
    tokens = [(t[1], t[0]) for t in meta.get("output_token_logprobs") or []]
    return Sample(out["text"], tokens, meta.get("cached_tokens"))


def logprob_gap(a, b):
    """Max |logprob| difference over the tokens two samples agree on.

    Compared up to the first differing token: past a near-tie flip the two runs
    condition on different text and their logprobs are no longer comparable.
    Returns None when not even the first token agrees.
    """
    gap = None
    for (tok_a, lp_a), (tok_b, lp_b) in zip(a.tokens, b.tokens):
        if tok_a != tok_b:
            break
        gap = max(gap or 0.0, abs(lp_a - lp_b))
    return gap


def build_prefix(approx_tokens):
    """Deterministic, non-repeating filler so the radix key is a real prefix."""
    words = []
    n = 0
    i = 0
    while n < approx_tokens:
        words.append(f"Item {i} records value {(i * 7919) % 10007}.")
        n += 9
        i += 1
    return " ".join(words)


def verdict(label, warm, colds, logprob_tol):
    """PASS / FAIL / INCONCLUSIVE, printed and returned.

    Text alone is too coarse: a quarter of Qwen3.5's layers are full attention
    over the intact KV cache, so a short retrieval answer often survives a
    corrupted GDN state. The warm run's logprobs must also sit within the
    cold-to-cold spread (floored at ``logprob_tol``) of the nearest cold run.

    ``warm.cached`` is None when the endpoint does not report the hit size
    (then the server log is the only evidence that the warm run hit).
    """
    if warm.cached == 0:
        print(f"[INCONCLUSIVE] {label} -- warm request hit no cached tokens")
        return "inconclusive"

    noise = [
        g
        for i, a in enumerate(colds)
        for b in colds[i + 1 :]
        if (g := logprob_gap(a, b)) is not None
    ]
    bound = max([logprob_tol] + [2 * g for g in noise])
    gaps = [g for c in colds if (g := logprob_gap(warm, c)) is not None]
    warm_gap = min(gaps) if gaps else None
    texts = [c.text for c in colds]
    stats = (
        f"cached_tokens={warm.cached}, warm/cold logprob gap="
        f"{'n/a' if warm_gap is None else f'{warm_gap:.4f}'}, "
        f"cold/cold max={max(noise) if noise else float('nan'):.4f}, bound={bound:.4f}"
    )

    if warm_gap is not None and warm_gap > bound:
        print(f"[FAIL] {label} -- logprobs drift past the cold spread ({stats})")
        print(f"    warm: {warm.text[:160]!r}")
        print(f"    cold: {texts[0][:160]!r}")
        return "fail"
    if warm.text in texts:
        print(
            f"[PASS] {label} (matches {texts.count(warm.text)}/{len(texts)} "
            f"cold runs; {stats})"
        )
        return "pass"
    if len(set(texts)) > 1:
        print(f"[INCONCLUSIVE] {label} -- cold runs disagree and warm matches none")
        for i, text in enumerate(texts):
            print(f"    cold #{i + 1}: {text[:160]!r}")
        print(f"    warm   : {warm.text[:160]!r}")
        return "inconclusive"
    print(f"[FAIL] {label} ({stats})")
    print(f"    warm (cache hit): {warm.text[:160]!r}")
    print(f"    cold (no cache) : {texts[0][:160]!r}")
    return "fail"


def check_prompt_prefix(port, approx, cold_runs, logprob_tol):
    """Reuse a prompt prefix -- the prefill-side snapshot path."""
    prefix = build_prefix(approx)
    suffix = "\n\nWhat is the single largest recorded value above?" + BARE

    post("/flush_cache", {}, port=port)
    gen(prefix, port, max_new_tokens=1)  # populate the tree with P
    warm = gen(prefix + suffix, port)  # should hit the P prefix

    colds = []
    for _ in range(cold_runs):
        post("/flush_cache", {}, port=port)
        colds.append(gen(prefix + suffix, port))

    return verdict(f"prompt prefix ~{approx} tokens", warm, colds, logprob_tol)


def check_multiturn(port, cold_runs, logprob_tol, model, track_interval=256):
    """Reuse *generated* tokens as a prefix -- the decode-side snapshot path.

    The prompt-prefix checks never exercise it: they only reuse the prompt,
    whose state was snapshotted during prefill. A snapshot taken while decoding
    only matters once a request's own output becomes someone else's prefix,
    which is what the second turn of a conversation does.

    Turn 2 is built from token ids -- turn 1's prompt ids, then the ids turn 1
    generated, then the templated follow-up -- because re-rendering the history
    through the chat template re-tokenizes (and may rewrite) the assistant turn,
    leaving only the prompt as a shared prefix. The warm hit must then extend
    past turn 1's prompt, or the decode-side snapshot was never used.

    Turn 1 generates just past a `track_interval` boundary (SGLang's
    `--mamba-track-interval`, 256 by default) so the hit lands a few tokens
    before turn 1's end. GDN's forget gate washes a stale resumed state out over
    a long recomputed tail, which is what makes a long tail pass vacuously.
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    prompt = build_prefix(1024) + (
        "\n\nList every item above with its value, one per line, no commentary."
    )
    ask = "Which of the items you just listed has the largest value?" + BARE

    def templated(messages):
        return tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            # Thinking off, so turn 1 spends its budget on the answer.
            enable_thinking=False,
        )

    turn1_ids = tok(templated([{"role": "user", "content": prompt}]))["input_ids"]
    boundary = (len(turn1_ids) // track_interval + 2) * track_interval
    post("/flush_cache", {}, port=port)
    turn1 = gen(
        None, port, max_new_tokens=boundary + 4 - len(turn1_ids), input_ids=turn1_ids
    )
    generated = [t for t, _ in turn1.tokens]
    # The follow-up exactly as the template renders it after an assistant turn.
    sentinel = "QZXJSENTINEL"
    full = templated(
        [
            {"role": "user", "content": "x"},
            {"role": "assistant", "content": sentinel},
            {"role": "user", "content": ask},
        ]
    )
    tail = full[full.index(sentinel) + len(sentinel) :]
    # Turn 1 may already have emitted the end-of-turn token the tail opens with.
    last = tok.convert_ids_to_tokens(generated[-1]) if generated else None
    if last and tail.startswith(last):
        tail = tail[len(last) :]
    turn2_ids = turn1_ids + generated + tok(tail, add_special_tokens=False)["input_ids"]

    warm = gen(None, port, input_ids=turn2_ids)
    colds = []
    for _ in range(cold_runs):
        post("/flush_cache", {}, port=port)
        colds.append(gen(None, port, input_ids=turn2_ids))

    label = (
        f"multi-turn (turn 1: {len(turn1_ids)} prompt + {len(generated)} "
        f"generated tokens)"
    )
    if warm.cached is not None and warm.cached <= len(turn1_ids):
        print(
            f"[INCONCLUSIVE] {label} -- warm hit {warm.cached} tokens, none past "
            "turn 1's prompt, so no decode-side snapshot was reused"
        )
        return "inconclusive"
    return verdict(label, warm, colds, logprob_tol)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--model", default=MODEL, help="tokenizer for the multi-turn check")
    ap.add_argument(
        "--prefix-tokens",
        type=int,
        nargs="+",
        default=[2048, 2080],
        help="approx prefix lengths; include one that is not a multiple of 64",
    )
    ap.add_argument(
        "--multiturn",
        action="store_true",
        help="also reuse generated tokens as a prefix (decode-side snapshot)",
    )
    ap.add_argument(
        "--cold-runs",
        type=int,
        default=3,
        help="cold repetitions the warm output is compared against",
    )
    ap.add_argument(
        "--logprob-tol",
        type=float,
        # Cold-to-cold gaps up to ~0.057 are routine on Qwen3.5-397B-A17B-FP8.
        default=0.06,
        help="floor of the allowed warm/cold logprob gap",
    )
    args = ap.parse_args()

    results = [
        check_prompt_prefix(args.port, a, args.cold_runs, args.logprob_tol)
        for a in args.prefix_tokens
    ]
    if args.multiturn:
        results.append(
            check_multiturn(args.port, args.cold_runs, args.logprob_tol, args.model)
        )

    fails = results.count("fail")
    unknown = results.count("inconclusive")
    print(
        f"\nRESULT: {results.count('pass')} pass, {fails} fail, "
        f"{unknown} inconclusive"
    )
    return 1 if fails or unknown else 0


if __name__ == "__main__":
    sys.exit(main())
