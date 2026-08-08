#!/usr/bin/env python3
"""Needle-in-a-haystack probe for the SparseKV decode swap path.

Builds long prompts (> hot_buffer) with a unique code embedded at a given depth,
asks the model to retrieve it, and checks the answer. Contexts above the
hot_buffer force the decode node to gather KV from the paged host cold pool via
the sparse top-k swap path, so a wrong/missing answer at long context (but a
correct one at the short control) isolates a swap/top-k retrieval defect.

Run against the mesh, concurrency 1 (sequential) to stay under the SparseKV
request-slot cap and isolate each case.
"""

import argparse
import sys
import time

import requests
from transformers import AutoTokenizer

FILLER = (
    "The city archives record countless mundane details about daily civic life. "
    "Clerks filed reports on weather, road repairs, market prices, and festivals. "
    "None of these ordinary entries carry any special significance whatsoever. "
)


def build_prompt(tok, target_tokens, depth, code, nonce=""):
    needle = f"\n\nIMPORTANT: The special access code for vault {code} is {code}.\n\n"
    # grow filler to target token budget (leave room for needle + question)
    unit = tok(FILLER, add_special_tokens=False)["input_ids"]
    reps = max(1, target_tokens // max(1, len(unit)))
    body = FILLER * reps
    ids = tok(body, add_special_tokens=False)["input_ids"]
    cut = int(len(ids) * depth)
    left = tok.decode(ids[:cut])
    right = tok.decode(ids[cut:])
    hay = nonce + left + needle + right
    q = (
        "\n\nQuestion: What is the special access code mentioned in the text above? "
        "Reply with only the number, nothing else."
    )
    return hay + q


def ask(base_url, model, prompt, max_tokens=1024):
    r = requests.post(
        f"{base_url}/v1/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": max_tokens,
        },
        timeout=600,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/mnt/models/GLM-5.2-MXFP4")
    ap.add_argument("--base-url", default="http://localhost:30000")
    ap.add_argument("--lengths", default="4000,16000,32000,64000")
    ap.add_argument("--depths", default="0.1,0.5,0.9")
    ap.add_argument(
        "--unique-prefix",
        action="store_true",
        help="prepend a per-case nonce so no two prompts share a cacheable "
        "prefix; isolates swap-path defects from prefix-reuse defects",
    )
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    lengths = [int(x) for x in args.lengths.split(",")]
    depths = [float(x) for x in args.depths.split(",")]

    npass = nfail = 0
    print(
        f"{'ctx_tok':>8} {'depth':>6} {'code':>7} {'answer':>20} {'result':>6} {'sec':>6}"
    )
    for L in lengths:
        for d in depths:
            code = 10000 + L // 1000 * 10 + int(d * 9)  # unique per (L,d)
            nonce = f"Case {code} log {time.time_ns()}. " if args.unique_prefix else ""
            prompt = build_prompt(tok, L, d, code, nonce)
            ntok = len(tok(prompt, add_special_tokens=False)["input_ids"])
            t0 = time.time()
            try:
                ans = ask(args.base_url, args.model, prompt)
            except Exception as e:
                ans = f"ERR:{e}"
            dt = time.time() - t0
            ok = str(code) in ans
            swap = "swap" if ntok > 8192 else "ctrl"
            npass += ok
            nfail += not ok
            disp = ans.replace("\n", " ")[-26:]
            print(
                f"{ntok:>8} {d:>6.2f} {code:>7} {disp:>26} "
                f"{'PASS' if ok else 'FAIL':>6} {dt:>6.1f}  [{swap}]"
            )
            sys.stdout.flush()
    print(
        f"\nSUMMARY: {npass} PASS / {nfail} FAIL "
        f"({npass}/{npass+nfail} = {100*npass/(npass+nfail):.0f}%)"
    )


if __name__ == "__main__":
    main()
