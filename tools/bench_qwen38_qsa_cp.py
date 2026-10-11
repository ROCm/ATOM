# SPDX-License-Identifier: MIT
"""Qwen QSA index-only context partition parity and graph timing.

Run with ``python -m torch.distributed.run --nproc_per_node=2`` or ``4``. The caches and
page table are replicated, matching Qwen's TP2 indexer. This benchmarks QSA
selection alone, with synthetic keys and queries; it is not a decoder layer.
"""

from __future__ import annotations

import argparse
import json
import os

import torch
import torch.distributed as dist
from aiter.dist.parallel_state import (
    destroy_model_parallel,
    get_tp_group,
    init_distributed_environment,
    initialize_model_parallel,
)

from atom.distributed.qwen_qsa_cp import warmup_qsa_cp_exchange
from atom.model_ops.qwen4_exp.ops.qsa import (
    qsa_select_paged_tokens,
    qsa_sparse_paged_gqa,
)


def _graph_time(fn, repeats: int) -> tuple[float, torch.Tensor, torch.cuda.CUDAGraph]:
    fn()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = fn()
    graph.replay()
    torch.cuda.synchronize()
    dist.barrier()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / repeats, output, graph


def run(
    context: int,
    rows: int,
    repeats: int,
    rank: int,
    group,
    shared_request: bool,
    with_attention: bool,
    ragged: bool,
    zero_index: bool,
) -> dict:
    if context not in (131072, 262144) or rows < 1:
        raise ValueError("this benchmark targets 128k/256k and positive rows")
    torch.manual_seed(2026 + context)
    device, page, ratio, budget = "cuda", 64, 4, 2048
    pages = context // page
    requests_count = 1 if shared_request else rows
    total_pages = pages * requests_count
    table = (
        torch.randperm(total_pages, dtype=torch.int64, device=device)
        .to(torch.int32)
        .view(requests_count, pages)
    )
    cache = torch.randn(
        total_pages, page // ratio, 1, 128, dtype=torch.bfloat16, device=device
    )
    q = torch.randn(rows, 4, 128, dtype=torch.bfloat16, device=device)
    requests = (
        torch.zeros(rows, dtype=torch.int32, device=device)
        if shared_request
        else torch.arange(rows, dtype=torch.int32, device=device)
    )
    positions = (
        torch.arange(context - rows, context, dtype=torch.int64, device=device)
        if shared_request
        else torch.full((rows,), context - 1, dtype=torch.int64, device=device)
    )
    lengths = torch.full((requests_count,), context, dtype=torch.int32, device=device)
    if zero_index:
        q.zero_()
        cache.zero_()
    if ragged:
        if shared_request:
            raise ValueError("ragged mode requires distinct requests")
        for row in range(rows):
            if row % 4 == 1:
                lengths[row] = context // 2
            elif row % 4 == 2:
                lengths[row] = context // 4 + 2
            elif row % 4 == 3:
                # Padded graph row: it must select no tokens.
                requests[row] = -1
        positions = lengths.long() - 1
        positions[requests < 0] = -1

    def native():
        return qsa_select_paged_tokens(
            q,
            cache,
            table,
            requests,
            positions,
            lengths,
            budget,
            ratio,
            max_seq_len=context,
        )

    def parallel():
        return qsa_select_paged_tokens(
            q,
            cache,
            table,
            requests,
            positions,
            lengths,
            budget,
            ratio,
            max_seq_len=context,
            index_cp=(rank, group.world_size, group),
        )

    reference = native()
    actual = parallel()
    torch.testing.assert_close(actual, reference, atol=0, rtol=0)
    native_us, native_result, native_graph = _graph_time(native, repeats)
    cp_us, cp_result, cp_graph = _graph_time(parallel, repeats)

    # Replaying a captured collective must consume changed queries and cache
    # contents, and the two ranks must continue to select identically.
    torch.manual_seed(418 + context)
    q.copy_(torch.randn_like(q))
    cache.copy_(torch.randn_like(cache))
    native_graph.replay()
    cp_graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(cp_result, native_result, atol=0, rtol=0)
    torch.testing.assert_close(cp_result, native(), atol=0, rtol=0)
    result = {
        "context": context,
        "rows": rows,
        "request_layout": "shared" if shared_request else "distinct",
        "ragged": ragged,
        "zero_index": zero_index,
        "native_select_us": round(native_us, 3),
        "index_cp_select_us": round(cp_us, 3),
        "speedup": round(native_us / cp_us, 3),
        "exact_selected_tokens": True,
        "changed_input_graph_replay": True,
    }
    if with_attention:
        # TP2 has one KV head per rank; TP4 replicates it across two ranks.
        # The sparse GQA kernel sees the local TP shape in both cases.
        query = torch.randn(
            rows, 24 // group.world_size, 256, dtype=torch.bfloat16, device=device
        )
        keys = torch.randn(
            total_pages, page, 1, 256, dtype=torch.bfloat16, device=device
        )
        values = torch.randn_like(keys)

        def native_full():
            selection = native()
            return qsa_sparse_paged_gqa(
                query,
                keys,
                values,
                selection,
                table,
                requests,
                num_decode_requests=requests_count,
            )

        def parallel_full():
            selection = parallel()
            return qsa_sparse_paged_gqa(
                query,
                keys,
                values,
                selection,
                table,
                requests,
                num_decode_requests=requests_count,
            )

        torch.testing.assert_close(parallel_full(), native_full(), atol=0, rtol=0)
        native_full_us, _, _ = _graph_time(native_full, repeats)
        cp_full_us, _, _ = _graph_time(parallel_full, repeats)
        result.update(
            full_qsa_native_us=round(native_full_us, 3),
            full_qsa_index_cp_us=round(cp_full_us, 3),
            full_qsa_speedup=round(native_full_us / cp_full_us, 3),
            exact_sparse_attention=True,
        )
    dist.barrier()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contexts", nargs="+", type=int, default=[131072, 262144])
    parser.add_argument("--rows", nargs="+", type=int, default=[8, 16, 32, 64])
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--shared-request", action="store_true")
    parser.add_argument("--with-attention", action="store_true")
    parser.add_argument("--ragged", action="store_true")
    parser.add_argument("--zero-index", action="store_true")
    args = parser.parse_args()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    init_distributed_environment(local_rank=local_rank)
    world = dist.get_world_size()
    initialize_model_parallel(tensor_model_parallel_size=world)
    try:
        group = get_tp_group()
        if group.world_size not in (2, 4):
            raise ValueError("the QSA CP benchmark requires two or four ranks")
        os.environ["ATOM_QWEN38_QSA_CP"] = "1"
        warmup_qsa_cp_exchange(torch.device("cuda", local_rank))
        cases = [
            run(
                context,
                rows,
                args.repeats,
                rank,
                group,
                args.shared_request,
                args.with_attention,
                args.ragged,
                args.zero_index,
            )
            for context in args.contexts
            for rows in args.rows
        ]
        if rank == 0:
            print(
                json.dumps(
                    {
                        "gpu": torch.cuda.get_device_name(),
                        "world": world,
                        "cases": cases,
                    },
                    indent=2,
                )
            )
    finally:
        destroy_model_parallel()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
