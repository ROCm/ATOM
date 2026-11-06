# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Compare FP8 non-DCP decode scoring under two logits budgets.

Run with an ATOM/AITER GPU environment, from the repository root:
    PYTHONPATH=. python tests/model_ops/bench_sparse_indexer_budget.py --output budget.json

This measures one replicated indexer's score/top-k on one GPU, not model TP8
throughput. Each request has separate, randomly initialized cache pages. Both
budgets use identical inputs and must produce identical stable top-k. Timings
are CUDA graph replays, alternating budget order in each sampling round.
Workspace sizing includes the shared prefill plane (8192 batched tokens).
"""

import argparse
import json
import statistics
from pathlib import Path

import torch
from aiter import dtypes, indexer_k_quant_and_cache

from atom.model_ops.indexer import (
    FP8_MQA_LOGITS_TAKES_OUT,
    IndexCache,
    fp8_rows,
    score_topk,
)
from atom.model_ops.sparse_indexer_chunk import LogitsWorkspace, logits_rows
from atom.utils import envs


def _inputs(batch, live, next_n, heads):
    block, dim = 128, 128
    pages = live // block
    cache = torch.empty(batch * pages, block, dim + 4, dtype=torch.uint8, device="cuda")
    # Bound the initialization temporary independently of cache size.
    for first in range(0, batch, 16):
        count = min(16, batch - first) * live
        keys = torch.randn(count, dim, dtype=torch.bfloat16, device="cuda")
        slots = torch.arange(first * live, first * live + count, device="cuda")
        indexer_k_quant_and_cache(keys, cache, slots, dim, "ue8m0", preshuffle=True)
    tables = torch.arange(batch * pages, dtype=torch.int32, device="cuda").view(
        batch, pages
    )
    ends = torch.full((batch,), live, dtype=torch.int32, device="cuda")
    query = torch.randn(batch * next_n, heads, dim, device="cuda").to(dtypes.fp8)
    weights = torch.randn(batch * next_n, heads, device="cuda")
    return query, weights, IndexCache(cache, None, block), ends, tables


def _capture(query, weights, cache, ends, tables, *, width, next_n, budget):
    envs.ATOM_SPARSE_INDEXER_LOGITS_BUDGET_MB = budget
    # Same max(decode, prefill) sizing as AiterMLAMetadataBuilder, DCP1.
    max_rows = max(512 * next_n, 8192)
    workspace = LogitsWorkspace(logits_rows(max_rows, width) * width, "cuda")
    rows = fp8_rows(ends, width, workspace, tables, next_n=next_n)
    out = torch.empty(query.shape[0], 2048, dtype=torch.int32, device="cuda")

    def run():
        score_topk(query, weights, cache, rows, topk=2048, stable=True, out=out)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    graph.replay()
    torch.cuda.synchronize()
    return {
        "graph": graph,
        "out": out,
        "workspace": workspace,
        "workspace_mib": workspace.nbytes / 2**20,
        "bands": len(rows.bands),
        "band_rows": [end - first for first, end in rows.bands],
    }


def _case(batch, live, width, next_n, heads, rounds, repeats):
    torch.manual_seed(3043)
    inputs = _inputs(batch, live, next_n, heads)
    variants = {
        budget: _capture(*inputs, width=width, next_n=next_n, budget=budget)
        for budget in (2048, 1024)
    }
    torch.testing.assert_close(
        variants[1024]["out"], variants[2048]["out"], rtol=0, atol=0
    )
    times = {1024: [], 2048: []}
    for sample in range(rounds):
        for budget in ((1024, 2048) if sample % 2 else (2048, 1024)):
            start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                enable_timing=True
            )
            start.record()
            for _ in range(repeats):
                variants[budget]["graph"].replay()
            stop.record()
            stop.synchronize()
            times[budget].append(start.elapsed_time(stop) / repeats)
    result = {
        "batch": batch,
        "live_keys": live,
        "max_model_len": width,
        "next_n": next_n,
        "heads": heads,
        "topk_equal": True,
        "budgets": {},
    }
    for budget, variant in variants.items():
        result["budgets"][budget] = {
            name: variant[name] for name in ("workspace_mib", "bands", "band_rows")
        } | {"median_ms": statistics.median(times[budget]), "samples_ms": times[budget]}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--rounds", type=int, default=9)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--heads", type=int, default=64)
    args = parser.parse_args()
    if not FP8_MQA_LOGITS_TAKES_OUT:
        parser.error(
            "shared prefill/decode sizing requires AITER fp8_mqa_logits(logits=)"
        )
    cases = [
        # Control: both budgets already fit all decode rows.
        (512, 8192, 32768, 1),
        (512, 8192, 131072, 1),
        (512, 32768, 131072, 4),
        # Long model limit: smaller buffers force additional row bands.
        (32, 32768, 1048576, 1),
        (128, 32768, 1048576, 1),
        (512, 32768, 1048576, 1),
        (512, 32768, 1048576, 4),
        (512, 131072, 1048576, 1),
    ]
    report = {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "rounds": args.rounds,
        "repeats": args.repeats,
        "scope": "single-GPU FP8 decode score/top-k; not model throughput",
        "cases": [],
    }
    for batch, live, width, next_n in cases:
        result = _case(
            batch, live, width, next_n, args.heads, args.rounds, args.repeats
        )
        report["cases"].append(result)
        print(json.dumps(result), flush=True)
        torch.cuda.empty_cache()
        if args.output:
            args.output.write_text(
                json.dumps(report, indent=2) + "\n", encoding="utf-8"
            )


if __name__ == "__main__":
    main()
