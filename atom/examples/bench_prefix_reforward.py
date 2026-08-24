# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
"""Wall-clock microbenchmark for the V4 prefix-cache B' re-forward cost.

For each prompt length L, we measure prefill wall-clock for:
  - SEED  : first time the prompt is seen -> no hit -> full prefill (forward all L)
  - HIT   : same prompt again -> prefix hit + B' rollback -> forward only the
            (rolled-back blocks + new tail) instead of all L

Comparing HIT vs SEED at several L shows how much B' actually saves (and, at
short L where the rollback eats most of the hit, how little it saves). This is
the data point for deciding whether fix C (per-block SWA, no re-forward) is
worth the BlockManager rewrite.

We use max_tokens=1 so the measured time is dominated by prefill (the part B'
affects), not decode.
"""

import argparse
import time

from atom import SamplingParams
from atom.model_engine.arg_utils import EngineArgs
from transformers import AutoTokenizer

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="Prefix-cache B' re-forward wall-clock benchmark",
)
EngineArgs.add_cli_args(parser)

FILLER = (
    "The mitochondrion is a double-membrane-bound organelle found in most "
    "eukaryotic cells. It generates most of the cell's supply of adenosine "
    "triphosphate, used as a source of chemical energy. Mitochondria have "
    "their own genetic material and machinery to manufacture RNAs and "
    "proteins. The number per cell varies widely across tissues and species. "
)


def main():
    args = parser.parse_args()
    args.cudagraph_capture_sizes = "[1, 2, 4, 8]"
    llm = EngineArgs.from_cli_args(args).create_engine()
    tok = AutoTokenizer.from_pretrained(args.model)

    base_ids = tok(FILLER * 4096, add_special_tokens=False)["input_ids"]
    sp = SamplingParams(temperature=0.0, max_tokens=1)

    # Prompt lengths spanning short (hit ~= window, B' nearly all re-forward)
    # to long (rollback amortized).
    lengths = [256, 512, 1024, 2048, 4096, 8192, 16384]

    print("\n" + "=" * 78)
    print("V4 PREFIX-CACHE B' RE-FORWARD WALL-CLOCK (max_tokens=1, prefill-dominated)")
    print("=" * 78)
    print(
        f"{'L':>7} | {'SEED ms':>9} | {'HIT ms':>9} | {'HIT/SEED':>9} | {'saved %':>8}"
    )
    print("-" * 78)

    for L in lengths:
        # Unique prefix per L so each L's seed is a true no-hit run.
        marker = tok(f"BENCHMARKL{L}MARKER. ", add_special_tokens=False)["input_ids"]
        ids = (marker + base_ids)[:L]
        txt = tok.decode(ids)

        # SEED: first time this exact prompt is seen -> no hit -> full prefill
        # (forwards all L tokens). This is the baseline.
        t_seed_start = time.perf_counter()
        _ = llm.generate([txt], sp)
        t_seed = (time.perf_counter() - t_seed_start) * 1000

        # HIT: same prompt again -> prefix hit + B' rollback -> forwards only
        # (rolled-back blocks + tail), not all L.
        t_hit_start = time.perf_counter()
        _ = llm.generate([txt], sp)
        t_hit = (time.perf_counter() - t_hit_start) * 1000

        ratio = t_hit / t_seed if t_seed > 0 else 0
        saved = 100 * (1 - ratio)
        print(
            f"{L:>7} | {t_seed:>9.1f} | {t_hit:>9.1f} | {ratio:>8.2f}x | {saved:>7.1f}%"
        )

    print("=" * 78)
    print("HIT/SEED -> 1.0 means B' saved nothing (short L); -> 0 means big save.")


if __name__ == "__main__":
    main()
