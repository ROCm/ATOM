# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Paired per-question comparison of two lm_eval runs (McNemar).

Differencing two exact_match rates discards the pairing: the ~1270 questions
both arms answer identically contribute variance and no information. All the
evidence about a difference between two arms lives in the questions where they
disagree, so count those and test their asymmetry.

lm_eval writes ONE RECORD PER (doc_id, filter) -- 2638 lines for gsm8k's 1319
questions -- with a scalar `exact_match` and the filter name in `filter`.
Keying on doc_id alone silently keeps whichever filter was written last and
makes both filters report identical numbers, which looks entirely plausible.
Hence the composite key, and the assertions below: every one of them exists
because its silent failure yields a confident wrong answer rather than an
error.
"""

import glob
import json
import sys
from math import comb


def load(arm_dir):
    files = sorted(glob.glob(f"{arm_dir}/**/samples_gsm8k_*.jsonl", recursive=True))
    if not files:
        sys.exit(f"FAIL: no sample file under {arm_dir} -- was --log_samples passed?")
    scores = {}
    with open(files[-1]) as fh:
        for line in fh:
            d = json.loads(line)
            key = (d["doc_id"], d["filter"])
            if key in scores:
                sys.exit(f"FAIL: duplicate record for {key} in {files[-1]}")
            scores[key] = float(d["exact_match"])
    return files[-1], scores


def mcnemar_p(b, c):
    """Exact two-sided binomial test on the discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2 * tail)


def main():
    a_dir, b_dir, a_name, b_name = sys.argv[1:5]
    fa, A = load(a_dir)
    fb, B = load(b_dir)
    print(f"{a_name}: {len(A)} (doc,filter) records from {fa}")
    print(f"{b_name}: {len(B)} (doc,filter) records from {fb}")

    if not (set(A) & set(B)):
        sys.exit("FAIL: the two arms share no (doc,filter) keys -- nothing compared")
    if set(A) != set(B):
        sys.exit(
            f"FAIL: key sets differ ({len(set(A) ^ set(B))} unmatched) -- "
            "a partial join would compute b and c over a subset"
        )

    filters = sorted({f for _, f in A})
    print(f"joined on {len(A)} records, no drops; filters: {filters}")

    for filt in filters:
        docs = sorted(d for d, f in A if f == filt)
        if not docs:
            sys.exit(f"FAIL: zero questions for filter {filt}")
        b = c = both = neither = 0
        for d in docs:
            x, y = A[(d, filt)], B[(d, filt)]
            if x > 0.5 and y < 0.5:
                b += 1
            elif x < 0.5 and y > 0.5:
                c += 1
            elif x > 0.5:
                both += 1
            else:
                neither += 1
        n = len(docs)
        ra, rb = (both + b) / n, (both + c) / n
        print(f"\n--- {filt} ({n} questions) ---")
        print(f"  {a_name} {ra:.4f}   {b_name} {rb:.4f}   delta {rb - ra:+.4f}")
        print(f"  both right {both}   both wrong {neither}")
        print(
            f"  b ({a_name} only right) = {b}   c ({b_name} only right) = {c}"
            f"   discordant = {b + c}"
        )
        if b + c == 0:
            print("  the two arms agree on every question")
            continue
        print(f"  McNemar exact two-sided p = {mcnemar_p(b, c):.4f}")


if __name__ == "__main__":
    if len(sys.argv) != 5:
        sys.exit(
            "usage: compare_lm_eval_paired.py <dir_a> <dir_b> <name_a> <name_b>\n"
            "  each dir is an lm_eval --output_path run made with --log_samples"
        )
    main()
