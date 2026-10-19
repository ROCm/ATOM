# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Time the non-GEMM kernels a V4-Pro TP8 + DSpark step issues.

Attention, the fused compressor and the tensor-parallel allreduce. GEMMs stay
in ``measure_prices``. Each launch is replayed from a CUDA graph. Shapes are
the ones ``kernel_shapes`` looks up, so a priced step does not fall back to a
roofline for these ops.

Allreduce needs every rank of the tensor-parallel group. Run it with
``torch.distributed.run --nproc_per_node=8`` and ``--collectives-only``; the
attention and compressor campaign uses one GPU.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from atom.compass.dsv4_tp8_dspark import (
    DSparkShape,
    V4ProShape,
    kernel_shapes,
)
from atom.compass.price_list import Price, PriceList

_CASES = (
    # batch, prefill tokens, context, speculative tokens
    (1, 1024, 1024, 7),
    (8, 1024, 1024, 7),
    (8, 1024, 4096, 7),
    (8, 256, 256, 7),
    (8, 768, 768, 7),
)


def _cases(shape: V4ProShape, draft: DSparkShape, tp: int) -> list[tuple[str, int, int, int]]:
    found: list[tuple[str, int, int, int]] = []
    seen: set[tuple[str, int, int, int]] = set()
    for batch, prefill, context, spec in _CASES:
        for item in kernel_shapes(
            shape,
            draft,
            tp,
            batch=batch,
            prefill_tokens=prefill,
            context=context,
            num_speculative_tokens=spec,
        ):
            if item not in seen:
                seen.add(item)
                found.append(item)
    return found


def _replay_seconds(launch, *, repeats: int, iters: int) -> float:
    import torch

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for _ in range(2):
            launch()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        graph.capture_begin()
        for _ in range(repeats):
            launch()
        graph.capture_end()
    torch.cuda.current_stream().wait_stream(stream)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / (iters * repeats) / 1e3


def _attn_decode(queries: int, keys: int, heads: int):
    import torch
    from aiter import dtypes

    from atom.model_ops.v4_kernels.paged_decode import sparse_attn_v4_paged_decode

    pool = queries * keys
    q_packed = torch.randint(0, 255, (queries, heads, 512), device="cuda", dtype=torch.uint8).view(
        dtypes.fp8
    )
    q_rope = torch.randn(queries, heads, 64, device="cuda", dtype=torch.bfloat16)
    kv = torch.randint(0, 255, (pool, 512), device="cuda", dtype=torch.uint8).view(dtypes.fp8)
    kv_rope = torch.randn(pool, 64, device="cuda", dtype=torch.bfloat16)
    indptr = torch.arange(queries + 1, device="cuda", dtype=torch.int32) * keys
    indices = torch.arange(pool, device="cuda", dtype=torch.int32)
    qo = torch.arange(queries + 1, device="cuda", dtype=torch.int32)
    sink = torch.zeros(heads, device="cuda", dtype=torch.float32)

    def launch():
        sparse_attn_v4_paged_decode(
            None,
            kv,
            indices,
            indptr,
            sink,
            512**-0.5,
            unified_kv_rope=kv_rope,
            q_packed_in=q_packed,
            q_rope_in=q_rope,
            qo_indptr=qo,
        )

    return launch, "aiter.mla.mla_decode_fwd_v4_nm"


def _attn_prefill(queries: int, keys: int, heads: int):
    import torch
    from aiter import dtypes

    from atom.model_ops.v4_kernels.paged_prefill import sparse_attn_v4_paged_prefill

    pool = queries * keys
    q_packed = torch.randint(0, 255, (queries, heads, 512), device="cuda", dtype=torch.uint8).view(
        dtypes.fp8
    )
    q_rope = torch.randn(queries, heads, 64, device="cuda", dtype=torch.bfloat16)
    kv = torch.randint(0, 255, (pool, 512), device="cuda", dtype=torch.uint8).view(dtypes.fp8)
    kv_rope = torch.randn(pool, 64, device="cuda", dtype=torch.bfloat16)
    prefix_indptr = torch.arange(queries + 1, device="cuda", dtype=torch.int32) * keys
    prefix_indices = torch.arange(pool, device="cuda", dtype=torch.int32)
    # Causal extend over this chunk: token i attends rows 0..i.
    extend_indptr = torch.arange(queries + 1, device="cuda", dtype=torch.int32)
    extend_indptr = extend_indptr * (extend_indptr + 1) // 2
    extend_indices = torch.empty(int(extend_indptr[-1].item()), device="cuda", dtype=torch.int32)
    cursor = 0
    for token in range(queries):
        nxt = cursor + token + 1
        extend_indices[cursor:nxt] = torch.arange(token + 1, device="cuda")
        cursor = nxt
    k_packed = torch.randint(0, 255, (queries, 512), device="cuda", dtype=torch.uint8).view(
        dtypes.fp8
    )
    k_rope = torch.randn(queries, 64, device="cuda", dtype=torch.bfloat16)
    sink = torch.zeros(heads, device="cuda", dtype=torch.float32)

    def launch():
        sparse_attn_v4_paged_prefill(
            None,
            kv,
            prefix_indices,
            prefix_indptr,
            None,
            extend_indices,
            extend_indptr,
            sink,
            512**-0.5,
            unified_kv_rope=kv_rope,
            q_packed=q_packed,
            q_rope=q_rope,
            k_packed=k_packed,
            k_rope=k_rope,
        )

    return launch, "aiter.pa_sparse_prefill_fp8_opus"


class _PlanBuf:
    def __init__(self, rows: int):
        import numpy as np
        import torch

        self.np = np.zeros((rows, 4), dtype=np.int32)
        self._t = torch.from_numpy(self.np).cuda()

    def copy_to_gpu(self, n=None):
        import torch

        self._t.copy_(torch.from_numpy(self.np).cuda())
        return self._t if n is None else self._t[:n]


def _compress(tokens: int, ratio: int, queries_per_seq: int):
    """One production compressor launch.

    Decode (``queries_per_seq < tokens`` or a short query) uses the CUDA-graph
    grid ``batch * ceil(q / ratio)``. Prefill uses the tight eager plan.
    """
    import numpy as np
    import torch
    from aiter import dtypes

    from atom.model_ops.v4_kernels.compress_plan import make_compress_plans
    from atom.model_ops.v4_kernels.fused_compress import fused_compress_attn

    if tokens % queries_per_seq:
        raise ValueError(f"{tokens} tokens is not a multiple of q={queries_per_seq}")
    batch = tokens // queries_per_seq
    overlap = ratio == 4
    head_dim = 512
    rope = 64
    dim_full = head_dim * (2 if overlap else 1)
    k_per_block = 256 // ratio
    # Existing context sits before the new tokens, so a decode step at
    # context 4096 still hits the compressor's state window.
    prefix = 4096 if queries_per_seq < tokens or queries_per_seq <= 16 else 0
    extend = np.full(batch, queries_per_seq, dtype=np.int32)
    context = extend + prefix
    rows = max(batch * queries_per_seq + 8, 64)
    plans = make_compress_plans(
        extend,
        context,
        [(ratio, overlap)],
        plan_buffers={ratio: {"compress": _PlanBuf(rows), "write": _PlanBuf(rows)}},
        running_bs=batch if prefix else None,
        max_q_len=queries_per_seq if prefix else None,
    )
    plan = plans[ratio]
    slots = batch + 2
    k_pool = (2 if overlap else 1) * ratio
    state = torch.randn(slots, k_pool, dim_full, device="cuda")
    score_state = torch.randn(slots, k_pool, dim_full, device="cuda")
    nb = 128
    kv_cache = torch.randint(
        0, 255, (nb, k_per_block, head_dim), device="cuda", dtype=torch.uint8
    ).view(dtypes.fp8)
    kv_rope = torch.zeros(nb, k_per_block, rope, device="cuda", dtype=torch.bfloat16)
    block_tables = torch.arange(batch * 32, device="cuda", dtype=torch.int32).view(batch, 32) % nb
    slots_t = torch.arange(batch, device="cuda", dtype=torch.int32)
    ape = torch.randn(ratio, dim_full, device="cuda")
    rms = torch.rand(head_dim, device="cuda") + 0.5
    cos = torch.randn(8192, rope // 2, device="cuda", dtype=torch.bfloat16)
    sin = torch.randn(8192, rope // 2, device="cuda", dtype=torch.bfloat16)
    # The flydsl compressor reads bf16 projections. The fp32 annotation on the
    # Python wrapper is the Triton reference path.
    kv_in = torch.randn(tokens, dim_full, device="cuda", dtype=torch.bfloat16)
    score_in = torch.randn(tokens, dim_full, device="cuda", dtype=torch.bfloat16)

    def launch():
        fused_compress_attn(
            kv_in=kv_in,
            score_in=score_in,
            kv_state=state,
            score_state=score_state,
            plan=plan,
            state_slot_mapping=slots_t,
            ape=ape,
            rms_weight=rms,
            rms_eps=1e-6,
            cos_cache=cos,
            sin_cache=sin,
            kv_cache=kv_cache,
            block_tables=block_tables,
            k_per_block=k_per_block,
            overlap=overlap,
            ratio=ratio,
            head_dim=head_dim,
            rope_head_dim=rope,
            quant_mode="group_fp8",
            main_2buff_fp8=True,
            kv_cache_rope=kv_rope,
        )

    return launch, "fused_compress_attn"


def _allreduce_seconds(tokens: int, dim: int, *, iters: int) -> float:
    """One allreduce, timed with every rank in the same window.

    A CUDA graph per rank drifts: the replay then waits on whichever rank is
    behind, and that wait is not the collective. The window below starts only
    after a barrier, so the elapsed time is the collective itself.
    """
    import torch
    import torch.distributed as dist

    tensor = torch.randn(tokens, dim, device="cuda", dtype=torch.bfloat16)
    for _ in range(5):
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    dist.barrier()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    end.record()
    torch.cuda.synchronize()
    dist.barrier()
    return start.elapsed_time(end) / iters / 1e3


def _launch(op: str, m: int, k: int, n: int):
    if op == "attn_decode":
        return _attn_decode(m, k, n)
    if op == "attn_prefill":
        return _attn_prefill(m, k, n)
    if op == "compress":
        return _compress(m, k, n)
    raise ValueError(op)


def measure(device_index: int, *, collectives_only: bool, repeats: int = 8, iters: int = 3) -> PriceList:
    import torch

    torch.cuda.set_device(device_index)
    props = torch.cuda.get_device_properties(device_index)
    shape = V4ProShape()
    draft = DSparkShape()
    wanted = [
        item
        for item in _cases(shape, draft, tp=8)
        if (item[0] == "allreduce") == collectives_only
    ]
    entries: dict[tuple[str, int, int, int], Price] = {}
    for op, m, k, n in wanted:
        if op == "allreduce":
            seconds = _allreduce_seconds(m, k, iters=50)
            kernel = "rccl_allreduce"
        else:
            launch, kernel = _launch(op, m, k, n)
            seconds = _replay_seconds(launch, repeats=repeats, iters=iters)
        entries[(op, m, k, n)] = Price(op, m, k, n, seconds, kernel)
        if not collectives_only or torch.distributed.get_rank() == 0:
            print(
                f"{op:14} M={m:5} K={k:6} N={n:6}  {seconds * 1e6:10.2f} us  {kernel}",
                flush=True,
            )
    name = props.name.replace(" ", "_")
    return PriceList(
        entries=entries,
        device=f"{name}:{getattr(props, 'gcnArchName', 'gfx')}",
        method="cuda_graph_replay",
    )


def _merge(path: Path, fresh: PriceList) -> PriceList:
    if not path.exists():
        return fresh
    previous = PriceList.load(path)
    entries = dict(previous.entries)
    entries.update(fresh.entries)
    return PriceList(entries, device=fresh.device, method=fresh.method)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("atom/compass/prices/gfx950_dsv4_tp8_dspark.json"),
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--collectives-only",
        action="store_true",
        help="time allreduce on this rank and merge into --out",
    )
    args = parser.parse_args()
    if args.collectives_only:
        import torch.distributed as dist

        dist.init_process_group("nccl")
        rank = dist.get_rank()
        catalog = measure(rank, collectives_only=True)
        if rank == 0:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            merged = _merge(args.out, catalog)
            merged.save(args.out)
            print(f"wrote {len(merged.entries)} prices to {args.out}")
        dist.barrier()
        dist.destroy_process_group()
        return
    catalog = measure(args.device, collectives_only=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    merged = _merge(args.out, catalog)
    merged.save(args.out)
    print(f"wrote {len(merged.entries)} prices to {args.out}")


if __name__ == "__main__":
    main()
