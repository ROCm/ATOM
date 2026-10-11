# SPDX-License-Identifier: MIT
"""Qwen3.8 all-visible QSA selection on one gfx950 TP rank.

Compares the ordinary scorer, stable top-k and expansion with the one-launch
FlyDSL shortcut. Selection and selection-plus-sparse-attention are timed as
captured graphs with synthetic inputs; neither includes the rest of a layer.
"""

import argparse
import json
import statistics

import torch
import triton.testing

from atom.model_ops.qwen4_exp.ops.qsa import (
    qsa_select_paged_tokens,
    qsa_sparse_paged_gqa,
)


def _time(fn, repeats: int) -> float:
    fn()
    torch.cuda.synchronize()
    return round(
        statistics.median(
            triton.testing.do_bench_cudagraph(fn, rep=50) * 1000 for _ in range(repeats)
        ),
        3,
    )


def run(context: int, rows: int, distinct: bool, repeats: int) -> dict:
    if context % 64 or not 64 <= context <= 2048 or rows < 1 or rows > context:
        raise ValueError("context must be 64..2048 in pages of 64; rows 1..context")
    page, ratio, budget = 64, 4, 2048
    pages = context // page
    requests_count = rows if distinct else 1
    total_pages = requests_count * pages
    torch.manual_seed(2026 + context + rows)
    device = "cuda"
    table = (
        torch.randperm(total_pages, device=device, dtype=torch.int64)
        .to(torch.int32)
        .view(requests_count, pages)
    )
    requests = (
        torch.arange(rows, device=device, dtype=torch.int32)
        if distinct
        else torch.zeros(rows, device=device, dtype=torch.int32)
    )
    positions = (
        torch.full((rows,), context - 1, device=device, dtype=torch.int64)
        if distinct
        else torch.arange(context - rows, context, device=device, dtype=torch.int64)
    )
    lengths = torch.full((requests_count,), context, device=device, dtype=torch.int32)
    index_q = torch.randn(rows, 4, 128, device=device, dtype=torch.bfloat16)
    compressed = torch.randn(
        total_pages, page // ratio, 1, 128, device=device, dtype=torch.bfloat16
    )
    q = torch.randn(rows, 12, 256, device=device, dtype=torch.bfloat16)
    k = torch.randn(total_pages, page, 1, 256, device=device, dtype=torch.bfloat16)
    v = torch.randn_like(k)

    def select(shortcut: bool):
        return qsa_select_paged_tokens(
            index_q,
            compressed,
            table,
            requests,
            positions,
            lengths,
            budget,
            ratio,
            max_seq_len=context,
            allow_shortcut=shortcut,
        )

    def full(shortcut: bool):
        selected = select(shortcut)
        return qsa_sparse_paged_gqa(
            q,
            k,
            v,
            selected,
            table,
            requests,
            num_decode_requests=requests_count,
        )

    if not torch.equal(select(False), select(True)):
        raise AssertionError("the shortcut changed selected token ids")
    if not torch.equal(full(False), full(True)):
        raise AssertionError("the shortcut changed sparse attention")
    native_sel = _time(lambda: select(False), repeats)
    short_sel = _time(lambda: select(True), repeats)
    native_full = _time(lambda: full(False), repeats)
    short_full = _time(lambda: full(True), repeats)
    return {
        "context": context,
        "rows": rows,
        "requests": "distinct" if distinct else "shared",
        "selection_native_us": native_sel,
        "selection_short_us": short_sel,
        "selection_speedup": round(native_sel / short_sel, 3),
        "qsa_native_us": native_full,
        "qsa_short_us": short_full,
        "qsa_speedup": round(native_full / short_full, 3),
        "exact_selected_tokens": True,
        "exact_sparse_attention": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contexts", type=int, nargs="+", default=[512, 2048])
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 2, 8, 32])
    parser.add_argument("--distinct-requests", action="store_true")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    print(
        json.dumps(
            {
                "gpu": torch.cuda.get_device_name(),
                "cases": [
                    run(context, rows, args.distinct_requests, args.repeats)
                    for context in args.contexts
                    for rows in args.rows
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
