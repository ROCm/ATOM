# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""gfx1250 K2 bring-up boundaries sharing the final device compute bodies.

The output projection boundary consumes native BF16 attention and performs
inverse RoPE/quant -> grouped wo_a/quant -> wo_b in one cooperative launch.
Its result is a rank partial: TP reduction remains outside this boundary.
The same device schedules are used by the opt-in merged K2 runner. Runtime
GPU validation remains a prerequisite for performance/serving acceptance.
"""

from functools import lru_cache
import os
from atom.models.deepseek_v41.mono.kernels.gfx1250_timeline import mark as timeline_mark

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from aiter.ops.flydsl.kernels.tensor_shim import check_e8m0, ptr_arg
from flydsl.expr import const_expr, range_constexpr

from atom.models.deepseek_v41.mono.kernels.gfx1250_attention import (
    ATTENTION_LDS_BYTES,
    HEAD_TILE,
    KEYS_PER_SPLIT,
    PV_PER_TILE,
    SCORE_PER_TILE,
    SPLITS,
    PRUNE_EMPTY,
    attention_pv_quant_device,
    attention_score_device,
    attention_tasks,
    validate_attention_spans,
)
from atom.models.deepseek_v41.mono.kernels.gfx1250_attn_post import (
    GROUP_K,
    HEAD_DIM,
    HIDDEN,
    O_RANK,
    inverse_rope_quant_device,
    woa_quant_device,
    woa_weight_load,
    woa_tasks,
    wob_device,
)
from atom.models.deepseek_v41.mono.kernels.gfx1250_gemv import gemv_lds_bytes, projection_load
from atom.models.deepseek_v41.mono.kernels.gfx1250_execution_plan import attention_execution_plan
from atom.models.deepseek_v41.mono.kernels.gfx1250_k1 import (
    _await,
    _CooperativeKernel,
    _i32_pointer,
    _publish,
)
from atom.models.deepseek_v41.mono.kernels.gfx1250_mhc import (
    _typed,
    gate_device,
    slice_device,
)
from atom.models.deepseek_v41.mono.kernels.gfx1250_producers import hidden_norm_device
from atom.models.deepseek_v41.mono.sources import SOURCES

WO_PREFETCH = int(os.environ.get("ATOM_DSV41_GFX1250_WO_PREFETCH", "0"))
if WO_PREFETCH not in (0, 1, 2):
    raise ValueError("gfx1250 WO prefetch must be 0, 1 or 2")

SYNC_DEBUG = os.environ.get("ATOM_DSV41_GFX1250_SYNC_DEBUG", "0") == "1"
from atom.mono.device.ops import traced

WORKERS = HIDDEN // 16
OUT_STATE_SIZE = 3 * WORKERS + 1
ATTENTION_STATE_SIZE = 4 * WORKERS + 1
SEAM_WORKERS = 192
SEAM_STATE_SIZE = 4 * SEAM_WORKERS + 1


def _out_name(rows, heads, table_rows):
    return f"atom_gfx1250_k2_out_r{rows}_h{heads}_p{table_rows}_src{SOURCES}_ap{int(PRUNE_EMPTY)}_pf{WO_PREFETCH}"


def out_scratch_specs(rows, heads):
    """Host-only scratch description; no allocation or device discovery."""
    if rows not in (1, 2) or heads not in (16, 32):
        raise ValueError("[gfx1250 K2] output projection supports rows1/2 and TP2/TP4")
    groups = heads // 8
    return {
        "attention_q": ((rows, heads * HEAD_DIM), torch.float8_e4m3fn),
        "attention_s": ((rows, heads * HEAD_DIM // 32), torch.uint8),
        "mid": ((rows, groups * O_RANK), torch.bfloat16),
        "mid_q": ((rows, groups * O_RANK), torch.float8_e4m3fn),
        "mid_s": ((rows, groups * O_RANK // 32), torch.uint8),
        "state": ((OUT_STATE_SIZE,), torch.int32),
    }


@lru_cache(maxsize=16)
def _out_launcher(rows, heads, table_rows):
    workspace = gemv_lds_bytes(rows, GROUP_K)
    name = _out_name(rows, heads, table_rows)

    @flyc.jit
    def launch(
        attention: fx.Pointer,
        positions: fx.Pointer,
        cos: fx.Pointer,
        sin: fx.Pointer,
        woa: fx.Pointer,
        woa_s: fx.Pointer,
        wob: fx.Pointer,
        wob_s: fx.Pointer,
        out: fx.Pointer,
        attention_q: fx.Pointer,
        attention_s: fx.Pointer,
        mid: fx.Pointer,
        mid_q: fx.Pointer,
        mid_s: fx.Pointer,
        state: fx.Pointer,
        stream: fx.Stream,
    ):
        @flyc.kernel(name=name, known_block_size=[256, 1, 1])
        def kernel(
            attention: fx.Pointer,
            positions: fx.Pointer,
            cos: fx.Pointer,
            sin: fx.Pointer,
            woa: fx.Pointer,
            woa_s: fx.Pointer,
            wob: fx.Pointer,
            wob_s: fx.Pointer,
            out: fx.Pointer,
            attention_q: fx.Pointer,
            attention_s: fx.Pointer,
            mid: fx.Pointer,
            mid_q: fx.Pointer,
            mid_s: fx.Pointer,
            state: fx.Pointer,
        ):
            tid, bid = fx.Int32(fx.thread_idx.x), fx.Int32(fx.block_idx.x)
            arena = fx.SharedAllocator(static=False)
            arena.allocate(workspace + 32)
            local = _i32_pointer(arena.base_ptr)
            lds = fx.Int64(fx.ptrtoint(arena.base_ptr))
            status = local + workspace // 4
            flags, diag = _i32_pointer(state), _i32_pointer(state) + OUT_STATE_SIZE - 1
            if tid == 0:
                previous = fx.generic_load(
                    flags + 2 * WORKERS + bid,
                    memory_order=fx.AtomicOrdering.Acquire,
                    syncscope="agent",
                )
                fx.ptr_store(previous + 1, local)
            fx.gpu.barrier()
            epoch = fx.ptr_load(local)
            fx.gpu.barrier()
            if bid < rows * heads:
                position = fx.ptr_load(_typed(positions, fx.Int64) + bid // heads)
                if ((position < 0) | (position >= table_rows)) & (tid == 0):
                    fx.atomic_cas(
                        diag,
                        fx.Int32(0),
                        fx.Int32(9_000_000) + bid,
                        success_ordering=fx.AtomicOrdering.Release,
                        failure_ordering=fx.AtomicOrdering.Monotonic,
                        syncscope="agent",
                    )
                ready = _await(
                    flags,
                    0,
                    0,
                    epoch,
                    diag,
                    local,
                    tid,
                    WORKERS,
                    9,
                    status=status,
                )
                if ready:
                    inverse_rope_quant_device(
                        attention,
                        positions,
                        cos,
                        sin,
                        attention_q,
                        attention_s,
                        tid,
                        bid,
                        rows,
                        heads,
                    )
                    _publish(flags, bid, epoch, synchronized=True)
            if bid < woa_tasks(heads):
                group = bid // (O_RANK // 32)
                ready = fx.Boolean(True)
                for row in range_constexpr(rows):
                    acquired = _await(
                        flags,
                        row * heads + group * 8,
                        8,
                        epoch,
                        diag,
                        local,
                        tid,
                        WORKERS,
                        10,
                        status=status,
                    )
                    ready = ready & acquired
                if ready:
                    woa_quant_device(
                        attention_q,
                        attention_s,
                        woa,
                        woa_s,
                        mid,
                        mid_q,
                        mid_s,
                        lds,
                        tid,
                        bid,
                        rows,
                        heads,
                    )
                    _publish(flags, WORKERS + bid, epoch, synchronized=True)
            ready = _await(
                flags,
                WORKERS,
                woa_tasks(heads),
                epoch,
                diag,
                local,
                tid,
                WORKERS,
                11,
                status=status,
            )
            if ready:
                wob_device(mid_q, mid_s, wob, wob_s, out, lds, tid, bid, rows, heads)
            # Completion is separate from payload flags and sticky diagnostic.
            # A failed launch retires its epoch but never publishes missing data.
            _publish(flags, 2 * WORKERS + bid, epoch)

        kernel(
            attention,
            positions,
            cos,
            sin,
            woa,
            woa_s,
            wob,
            wob_s,
            out,
            attention_q,
            attention_s,
            mid,
            mid_q,
            mid_s,
            state,
        ).launch(grid=(WORKERS, 1, 1), block=(256, 1, 1), stream=stream)

    return launch


def _tensor_spec(tensor, shape, dtype):
    if (
        tuple(tensor.shape) != shape
        or tensor.dtype != dtype
        or not tensor.is_contiguous()
    ):
        raise ValueError(
            f"[gfx1250 K2] requires contiguous {dtype} {shape}; "
            f"got {tensor.dtype} {tuple(tensor.shape)} stride={tensor.stride()}"
        )


def _dispatch(launcher, name, tensors, written_from, rows, workers, smem, cache, key):
    device = tensors[0].device
    if not tensors[0].is_cuda or any(t.device != device for t in tensors):
        raise ValueError("[gfx1250 K2] buffers must share one GPU")
    for index in range(written_from, len(tensors)):
        a = tensors[index]
        if a.numel() == 0:
            continue
        lo, hi = a.data_ptr(), a.data_ptr() + a.numel() * a.element_size()
        if any(
            lo < b.data_ptr() + b.numel() * b.element_size() and b.data_ptr() < hi
            for b in tensors[:index]
            if b.numel() > 0
        ):
            raise ValueError("[gfx1250 K2] written buffers must not overlap")
    if torch.cuda.current_device() != device.index:
        raise RuntimeError("[gfx1250 K2] select the buffer device before dispatch")
    if torch.cuda.get_device_properties(device).gcnArchName.split(":")[0] != "gfx1250":
        raise RuntimeError("[gfx1250 K2] requires gfx1250")
    key = (device.index, *key)
    stream = torch.cuda.current_stream(device).cuda_stream
    if key not in cache:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "[gfx1250 K2] compile and verify residency before capture"
            )
        cache[key] = _CooperativeKernel(
            launcher,
            tuple(ptr_arg(t) for t in tensors) + (stream,),
            name,
            rows,
            workers,
            device.index,
            smem=smem,
            label="gfx1250 K2",
        )
    cache[key](tensors, stream)


_OUT_KERNELS = {}


def out_projection_gfx1250(
    attention, positions, cos, sin, woa, woa_s, wob, wob_s, out, scratch
):
    """One local cooperative launch; caller reduces the returned TP partial.

    Scratch is exclusive to one stream and its int32 state is initially zero.
    Any nonzero state[-1] invalidates outputs and forbids reuse. Inputs/tables
    must be finite; positions are checked on device before table addressing.
    This boundary is deliberately independent of FULL/REUSE index selection.
    """
    if attention.ndim != 3:
        raise ValueError("[gfx1250 K2] expected attention [rows, heads, 512]")
    rows, heads, _ = attention.shape
    specs = out_scratch_specs(rows, heads)
    groups, table_rows = heads // 8, cos.shape[0] if cos.ndim == 2 else 0
    if table_rows <= 0:
        raise ValueError("[gfx1250 K2] empty or invalid RoPE table")
    for tensor, shape, dtype in [
        (attention, (rows, heads, HEAD_DIM), torch.bfloat16),
        (positions, (rows,), torch.int64),
        (cos, (table_rows, 32), torch.float32),
        (sin, (table_rows, 32), torch.float32),
        (woa, (groups * O_RANK, GROUP_K), torch.float8_e4m3fn),
        (wob, (HIDDEN, groups * O_RANK), torch.float8_e4m3fn),
        (out, (rows, HIDDEN), torch.bfloat16),
        *[
            (scratch[name], *spec)
            for name, spec in specs.items()
            if not name.endswith("_s")
        ],
    ]:
        _tensor_spec(tensor, shape, dtype)
    for tensor, shape, name in [
        (woa_s, (groups * O_RANK // 32, GROUP_K // 32), "woa_s"),
        (wob_s, (HIDDEN // 32, groups * O_RANK // 32), "wob_s"),
        (scratch["attention_s"], specs["attention_s"][0], "attention_s"),
        (scratch["mid_s"], specs["mid_s"][0], "mid_s"),
    ]:
        check_e8m0(tensor, name, "gfx1250 K2", shape)
    tensors = [attention, positions, cos, sin, woa, woa_s, wob, wob_s, out] + [
        scratch[name] for name in specs
    ]
    _dispatch(
        _out_launcher(rows, heads, table_rows),
        _out_name(rows, heads, table_rows),
        tensors,
        8,
        rows,
        WORKERS,
        gemv_lds_bytes(rows, GROUP_K) + 32,
        _OUT_KERNELS,
        (rows, heads, table_rows),
    )
    return out


def attention_scratch_specs(rows, heads, debug=False):
    """Score/PV/WO scratch; production has no BF16 attention materialization."""
    specs = out_scratch_specs(rows, heads)
    specs["state"] = ((ATTENTION_STATE_SIZE,), torch.int32)
    return {
        "maxima": ((rows, SPLITS, heads), torch.float32),
        "sums": ((rows, SPLITS, heads), torch.float32),
        "probabilities": (
            (rows, heads // HEAD_TILE, SPLITS, HEAD_TILE, KEYS_PER_SPLIT),
            torch.bfloat16,
        ),
        **specs,
        "debug_out": ((rows, heads, HEAD_DIM) if debug else (0,), torch.bfloat16),
    }


def _attention_name(
    rows,
    heads,
    table_rows,
    pool_rows,
    ring_rows,
    prefix_capacity,
    extend_capacity,
    softmax_scale,
    debug,
):
    scale_bits = (
        float(softmax_scale).hex().replace(".", "_").replace("+", "p").replace("-", "m")
    )
    return (
        f"atom_gfx1250_k2_attention_r{rows}_h{heads}_p{table_rows}"
        f"_kv{pool_rows}_{ring_rows}_i{prefix_capacity}_{extend_capacity}"
        f"_s{scale_bits}_d{int(debug)}_src{SOURCES}_ap{int(PRUNE_EMPTY)}_pf{WO_PREFETCH}"
    )


@lru_cache(maxsize=64)
def _attention_launcher(
    rows,
    heads,
    table_rows,
    pool_rows,
    ring_rows,
    prefix_capacity,
    extend_capacity,
    softmax_scale,
    debug=False,
):
    """One cooperative score -> PV/IRQ -> wo_a/quant -> wo_b launch.

    Every score task publishes all eight splits, including empty ones.
    Each grouped wo_a waits every output-column producer for its head tile.
    Completion epochs have their own bank and never imply payload validity.
    """
    workspace = max(ATTENTION_LDS_BYTES, gemv_lds_bytes(rows, GROUP_K))
    score_tasks, pv_tasks = attention_tasks(rows, heads)
    name = _attention_name(
        rows,
        heads,
        table_rows,
        pool_rows,
        ring_rows,
        prefix_capacity,
        extend_capacity,
        softmax_scale,
        debug,
    )

    @flyc.jit
    def launch(
        query: fx.Pointer,
        pool: fx.Pointer,
        ring: fx.Pointer,
        prefix: fx.Pointer,
        prefix_indptr: fx.Pointer,
        extend: fx.Pointer,
        extend_indptr: fx.Pointer,
        sink: fx.Pointer,
        positions: fx.Pointer,
        cos: fx.Pointer,
        sin: fx.Pointer,
        woa: fx.Pointer,
        woa_s: fx.Pointer,
        wob: fx.Pointer,
        wob_s: fx.Pointer,
        out: fx.Pointer,
        maxima: fx.Pointer,
        sums: fx.Pointer,
        probabilities: fx.Pointer,
        attention_q: fx.Pointer,
        attention_s: fx.Pointer,
        mid: fx.Pointer,
        mid_q: fx.Pointer,
        mid_s: fx.Pointer,
        state: fx.Pointer,
        debug_out: fx.Pointer,
        stream: fx.Stream,
    ):
        @flyc.kernel(name=name, known_block_size=[256, 1, 1])
        def kernel(
            query: fx.Pointer,
            pool: fx.Pointer,
            ring: fx.Pointer,
            prefix: fx.Pointer,
            prefix_indptr: fx.Pointer,
            extend: fx.Pointer,
            extend_indptr: fx.Pointer,
            sink: fx.Pointer,
            positions: fx.Pointer,
            cos: fx.Pointer,
            sin: fx.Pointer,
            woa: fx.Pointer,
            woa_s: fx.Pointer,
            wob: fx.Pointer,
            wob_s: fx.Pointer,
            out: fx.Pointer,
            maxima: fx.Pointer,
            sums: fx.Pointer,
            probabilities: fx.Pointer,
            attention_q: fx.Pointer,
            attention_s: fx.Pointer,
            mid: fx.Pointer,
            mid_q: fx.Pointer,
            mid_s: fx.Pointer,
            state: fx.Pointer,
            debug_out: fx.Pointer,
        ):
            tid, bid = fx.Int32(fx.thread_idx.x), fx.Int32(fx.block_idx.x)
            arena = fx.SharedAllocator(static=False)
            arena.allocate(workspace + 32)
            local = _i32_pointer(arena.base_ptr)
            lds = fx.Int64(fx.ptrtoint(arena.base_ptr))
            status = local + workspace // 4
            flags = _i32_pointer(state)
            diag = flags + ATTENTION_STATE_SIZE - 1
            if tid == 0:
                previous = fx.generic_load(
                    flags + 3 * WORKERS + bid,
                    memory_order=fx.AtomicOrdering.Acquire,
                    syncscope="agent",
                )
                fx.ptr_store(previous + 1, local)
            fx.gpu.barrier()
            epoch = fx.ptr_load(local)
            fx.gpu.barrier()
            run_attention_projection(
                {
                    "query": query,
                    "pool": pool,
                    "ring": ring,
                    "prefix": prefix,
                    "prefix_indptr": prefix_indptr,
                    "extend": extend,
                    "extend_indptr": extend_indptr,
                    "sink": sink,
                    "positions": positions,
                    "cos": cos,
                    "sin": sin,
                    "woa": woa,
                    "woa_s": woa_s,
                    "wob": wob,
                    "wob_s": wob_s,
                    "out": out,
                    "maxima": maxima,
                    "sums": sums,
                    "probabilities": probabilities,
                    "attention_q": attention_q,
                    "attention_s": attention_s,
                    "mid": mid,
                    "mid_q": mid_q,
                    "mid_s": mid_s,
                    "debug_out": debug_out,
                }.__getitem__,
                {
                    "rows": rows,
                    "heads": heads,
                    "table_rows": table_rows,
                    "pool_rows": pool_rows,
                    "ring_rows": ring_rows,
                    "prefix_capacity": prefix_capacity,
                    "extend_capacity": extend_capacity,
                    "softmax_scale": softmax_scale,
                    "debug": debug,
                    "score_tasks": score_tasks,
                    "pv_tasks": pv_tasks,
                    "arena": arena,
                    "local": local,
                    "lds": lds,
                    "status": status,
                    "flags": flags,
                    "diag": diag,
                    "epoch": epoch,
                    "tid": tid,
                    "bid": bid,
                },
            )

        kernel(
            query,
            pool,
            ring,
            prefix,
            prefix_indptr,
            extend,
            extend_indptr,
            sink,
            positions,
            cos,
            sin,
            woa,
            woa_s,
            wob,
            wob_s,
            out,
            maxima,
            sums,
            probabilities,
            attention_q,
            attention_s,
            mid,
            mid_q,
            mid_s,
            state,
            debug_out,
        ).launch(grid=(WORKERS, 1, 1), block=(256, 1, 1), stream=stream)

    return launch


_ATTENTION_KERNELS = {}


def attention_projection_gfx1250(
    query,
    pool,
    ring,
    prefix,
    prefix_indptr,
    extend,
    extend_indptr,
    sink,
    positions,
    cos,
    sin,
    woa,
    woa_s,
    wob,
    wob_s,
    out,
    scratch,
    *,
    softmax_scale,
    debug=False,
    _validate_only=False,
):
    """Consume native sparse lists and emit the attention TP rank partial.

    K1/cache updates and index selection precede this launch on the same
    stream. Pool and ring may alias as readonly native views. The caller owns
    each list's per-layer slot semantics; source row limits alone do not prove
    ring ownership. Every writable buffer must be separate. Initial state is
    zero; a sticky diagnostic forbids consuming or reusing this scratch.
    """
    import math

    if query.ndim != 3:
        raise ValueError("[gfx1250 K2] expected Q [rows, heads, 512]")
    rows, heads, _ = query.shape
    specs = attention_scratch_specs(rows, heads, debug)
    if pool.ndim != 2 or ring.ndim != 2 or prefix.ndim != 1 or extend.ndim != 1:
        raise ValueError("[gfx1250 K2] expected 2D KV sources and flat index lists")
    if max(pool.shape[0], ring.shape[0], prefix.numel(), extend.numel()) >= 2**31:
        raise ValueError("[gfx1250 K2] row IDs and indptr capacities must fit int32")
    table_rows = cos.shape[0] if cos.ndim == 2 else 0
    if table_rows <= 0 or not math.isfinite(softmax_scale) or softmax_scale <= 0:
        raise ValueError("[gfx1250 K2] invalid RoPE table or attention scale")
    groups = heads // 8
    for tensor, shape, dtype in [
        (query, (rows, heads, HEAD_DIM), torch.bfloat16),
        (pool, (pool.shape[0], HEAD_DIM), torch.bfloat16),
        (ring, (ring.shape[0], HEAD_DIM), torch.bfloat16),
        (prefix, (prefix.numel(),), torch.int32),
        (extend, (extend.numel(),), torch.int32),
        (prefix_indptr, (rows + 1,), torch.int32),
        (extend_indptr, (rows + 1,), torch.int32),
        (sink, (heads,), torch.float32),
        (positions, (rows,), torch.int64),
        (cos, (table_rows, 32), torch.float32),
        (sin, (table_rows, 32), torch.float32),
        (woa, (groups * O_RANK, GROUP_K), torch.float8_e4m3fn),
        (wob, (HIDDEN, groups * O_RANK), torch.float8_e4m3fn),
        (out, (rows, HIDDEN), torch.bfloat16),
        *[
            (scratch[name], *spec)
            for name, spec in specs.items()
            if not name.endswith("_s")
        ],
    ]:
        _tensor_spec(tensor, shape, dtype)
    for tensor, shape, name in [
        (woa_s, (groups * O_RANK // 32, GROUP_K // 32), "woa_s"),
        (wob_s, (HIDDEN // 32, groups * O_RANK // 32), "wob_s"),
        (scratch["attention_s"], specs["attention_s"][0], "attention_s"),
        (scratch["mid_s"], specs["mid_s"][0], "mid_s"),
    ]:
        check_e8m0(tensor, name, "gfx1250 K2", shape)
    tensors = [
        query,
        pool,
        ring,
        prefix,
        prefix_indptr,
        extend,
        extend_indptr,
        sink,
        positions,
        cos,
        sin,
        woa,
        woa_s,
        wob,
        wob_s,
        out,
    ] + [scratch[name] for name in specs]
    key = (
        rows,
        heads,
        table_rows,
        pool.shape[0],
        ring.shape[0],
        prefix.numel(),
        extend.numel(),
        float(softmax_scale),
        bool(debug),
    )
    if _validate_only:
        return key, tensors
    _dispatch(
        _attention_launcher(*key),
        _attention_name(*key),
        tensors,
        15,
        rows,
        WORKERS,
        max(ATTENTION_LDS_BYTES, gemv_lds_bytes(rows, GROUP_K)) + 32,
        _ATTENTION_KERNELS,
        key,
    )
    return out


def seam_scratch_specs(rows):
    if rows not in (1, 2):
        raise ValueError("[gfx1250 K2] FFN seam supports rows1/2")
    return {
        "collapse": ((rows, HIDDEN), torch.bfloat16),
        "partial": ((rows, 160, 26), torch.float32),
        "shared_q": ((rows, HIDDEN), torch.float8_e4m3fn),
        "shared_s": ((rows, HIDDEN // 32), torch.uint8),
        "state": ((SEAM_STATE_SIZE,), torch.int32),
    }


def _seam_name(rows):
    return f"atom_gfx1250_k2_ffn_seam_r{rows}_src{SOURCES}_ap{int(PRUNE_EMPTY)}_pf{WO_PREFETCH}"


@lru_cache(maxsize=2)
def _seam_launcher(rows):
    name = _seam_name(rows)

    @flyc.jit
    def launch(
        residual: fx.Pointer,
        output: fx.Pointer,
        pre: fx.Pointer,
        post: fx.Pointer,
        comb: fx.Pointer,
        fn: fx.Pointer,
        scale: fx.Pointer,
        bias: fx.Pointer,
        nw: fx.Pointer,
        residual_out: fx.Pointer,
        pre_out: fx.Pointer,
        post_out: fx.Pointer,
        comb_out: fx.Pointer,
        normed: fx.Pointer,
        shared_q: fx.Pointer,
        shared_s: fx.Pointer,
        collapse: fx.Pointer,
        partial: fx.Pointer,
        state: fx.Pointer,
        stream: fx.Stream,
    ):
        @flyc.kernel(name=name, known_block_size=[256, 1, 1])
        def kernel(
            residual: fx.Pointer,
            output: fx.Pointer,
            pre: fx.Pointer,
            post: fx.Pointer,
            comb: fx.Pointer,
            fn: fx.Pointer,
            scale: fx.Pointer,
            bias: fx.Pointer,
            nw: fx.Pointer,
            residual_out: fx.Pointer,
            pre_out: fx.Pointer,
            post_out: fx.Pointer,
            comb_out: fx.Pointer,
            normed: fx.Pointer,
            shared_q: fx.Pointer,
            shared_s: fx.Pointer,
            collapse: fx.Pointer,
            partial: fx.Pointer,
            state: fx.Pointer,
        ):
            tid, bid = fx.Int32(fx.thread_idx.x), fx.Int32(fx.block_idx.x)
            arena = fx.SharedAllocator(static=False)
            arena.allocate(4096 + 32)
            local = _i32_pointer(arena.base_ptr)
            lds = fx.Int64(fx.ptrtoint(arena.base_ptr))
            status = local + 4096 // 4
            flags = _i32_pointer(state)
            diag = flags + SEAM_STATE_SIZE - 1
            if tid == 0:
                previous = fx.generic_load(
                    flags + 3 * SEAM_WORKERS + bid,
                    memory_order=fx.AtomicOrdering.Acquire,
                    syncscope="agent",
                )
                fx.ptr_store(previous + 1, local)
            fx.gpu.barrier()
            epoch = fx.ptr_load(local)
            fx.gpu.barrier()
            ready = _await(
                flags,
                0,
                0,
                epoch,
                diag,
                local,
                tid,
                SEAM_WORKERS,
                12,
                status=status,
            )
            if (bid < 160) & ready:
                slice_device(
                    residual,
                    output,
                    pre,
                    post,
                    comb,
                    fn,
                    residual_out,
                    collapse,
                    partial,
                    arena.base_ptr,
                    tid,
                    bid,
                    rows,
                    True,
                )
                _publish(flags, bid, epoch, synchronized=True)
            if (bid >= 160) & (bid < 160 + rows):
                ready = _await(
                    flags,
                    0,
                    160,
                    epoch,
                    diag,
                    local,
                    tid,
                    SEAM_WORKERS,
                    12,
                    status=status,
                )
                if ready:
                    hidden_norm_device(
                        collapse,
                        nw,
                        normed,
                        shared_q,
                        shared_s,
                        lds,
                        tid,
                        bid - 160,
                        rows,
                        True,
                        partial,
                    )
                    _publish(flags, SEAM_WORKERS + bid - 160, epoch, synchronized=True)
            if (bid >= 162) & (bid < 162 + rows):
                ready = _await(
                    flags,
                    0,
                    160,
                    epoch,
                    diag,
                    local,
                    tid,
                    SEAM_WORKERS,
                    13,
                    status=status,
                )
                if ready:
                    gate_device(
                        partial,
                        scale,
                        bias,
                        pre_out,
                        post_out,
                        comb_out,
                        arena.base_ptr,
                        tid,
                        bid - 162,
                        rows,
                    )
                    _publish(
                        flags, 2 * SEAM_WORKERS + bid - 162, epoch, synchronized=True
                    )
            _publish(flags, 3 * SEAM_WORKERS + bid, epoch)

        kernel(
            residual,
            output,
            pre,
            post,
            comb,
            fn,
            scale,
            bias,
            nw,
            residual_out,
            pre_out,
            post_out,
            comb_out,
            normed,
            shared_q,
            shared_s,
            collapse,
            partial,
            state,
        ).launch(grid=(SEAM_WORKERS, 1, 1), block=(256, 1, 1), stream=stream)

    return launch


_SEAM_KERNELS = {}


def ffn_seam_gfx1250(
    residual,
    reduced_attention,
    pre,
    post,
    comb,
    fn,
    scale,
    bias,
    nw,
    residual_out,
    pre_out,
    post_out,
    comb_out,
    normed,
    scratch,
):
    """Fold TP-reduced attention into delayed mHC, publish the FFN norm once.

    Returns the normalized MoE input and next delayed gates; the FFN output
    remains pending until the next K1. shared_q/s use ceil group32 scales;
    routed experts must quantize from normed using their own activation ABI.
    The standalone boundary computes gates here; the full K2 will schedule
    that same gate body after shared work, as in gfx950.
    """
    rows = residual.shape[0] if residual.ndim == 3 else 0
    specs = seam_scratch_specs(rows)
    for tensor, shape, dtype in [
        (residual, (rows, 4, HIDDEN), torch.bfloat16),
        (reduced_attention, (rows, HIDDEN), torch.bfloat16),
        (pre, (rows, 4), torch.float32),
        (post, (rows, 4), torch.float32),
        (comb, (rows, 4, 4), torch.float32),
        (fn, (24, 4 * HIDDEN), torch.float32),
        (scale, (3,), torch.float32),
        (bias, (24,), torch.float32),
        (nw, (HIDDEN,), torch.bfloat16),
        (residual_out, (rows, 4, HIDDEN), torch.bfloat16),
        (pre_out, (rows, 4), torch.float32),
        (post_out, (rows, 4), torch.float32),
        (comb_out, (rows, 4, 4), torch.float32),
        (normed, (rows, HIDDEN), torch.bfloat16),
        *[(scratch[name], *spec) for name, spec in specs.items() if name != "shared_s"],
    ]:
        _tensor_spec(tensor, shape, dtype)
    check_e8m0(scratch["shared_s"], "shared_s", "gfx1250 K2", specs["shared_s"][0])
    tensors = [
        residual,
        reduced_attention,
        pre,
        post,
        comb,
        fn,
        scale,
        bias,
        nw,
        residual_out,
        pre_out,
        post_out,
        comb_out,
        normed,
        scratch["shared_q"],
        scratch["shared_s"],
        scratch["collapse"],
        scratch["partial"],
        scratch["state"],
    ]
    _dispatch(
        _seam_launcher(rows),
        _seam_name(rows),
        tensors,
        9,
        rows,
        SEAM_WORKERS,
        4096 + 32,
        _SEAM_KERNELS,
        (rows,),
    )
    return normed, residual_out, pre_out, post_out, comb_out


@traced
def run_attention_projection(a, c, after_projection=None, before_score=None):
    rows = c["rows"]
    heads = c["heads"]
    table_rows = c["table_rows"]
    pool_rows = c["pool_rows"]
    ring_rows = c["ring_rows"]
    prefix_capacity = c["prefix_capacity"]
    extend_capacity = c["extend_capacity"]
    softmax_scale = c["softmax_scale"]
    debug = c["debug"]
    score_tasks = c["score_tasks"]
    pv_tasks = c["pv_tasks"]
    arena = c["arena"]
    local = c["local"]
    lds = c["lds"]
    status = c["status"]
    flags = c["flags"]
    diag = c["diag"]
    epoch = c["epoch"]
    tid = c["tid"]
    bid = c["bid"]
    plan = attention_execution_plan(rows, heads, c.get("placed_attention", False))
    score_task = plan.task(bid, plan.score_base)
    pv_task = plan.task(bid, plan.pv_base)
    woa_task = plan.task(bid, plan.woa_base)
    wob_task = plan.task(bid, plan.wob_base)
    progress = c.get("progress")
    failure_detail = c.get("failure_detail")
    tls = c.get("timeline")
    joined_publication = tls is not None or (c.get("placed_attention", False) and rows == 1)

    def note(stage, published=False):
        if const_expr(progress is not None):
            if tid == 0:
                fx.generic_store(progress + bid * 3, epoch,
                                 memory_order=fx.AtomicOrdering.Release, syncscope="agent")
                fx.generic_store(progress + bid * 3 + (2 if published else 1), fx.Int32(stage),
                                 memory_order=fx.AtomicOrdering.Release, syncscope="agent")

    if score_task < score_tasks:
        timeline_mark(tls, tid, "score_ready")
        note(14)
        row = score_task // (SCORE_PER_TILE * (heads // HEAD_TILE))
        if const_expr(before_score is not None):
            before_score(row)
        validate_attention_spans(
            a("prefix_indptr"),
            a("extend_indptr"),
            row,
            prefix_capacity,
            extend_capacity,
            diag,
            tid,
        )
        position = fx.ptr_load(_typed(a("positions"), fx.Int64) + row)
        if ((position < 0) | (position >= table_rows)) & (tid == 0):
            fx.atomic_cas(
                diag,
                fx.Int32(0),
                fx.Int32(9_000_000) + bid,
                success_ordering=fx.AtomicOrdering.Release,
                failure_ordering=fx.AtomicOrdering.Monotonic,
                syncscope="agent",
            )
        ready = _await(flags, 0, 0, epoch, diag, local, tid, WORKERS, 14, status=status,
                       failure_detail=failure_detail)
        timeline_mark(tls, tid, "score_ready", True)
        if ready:
            timeline_mark(tls, tid, "score")
            attention_score_device(
                a("query"),
                a("pool"),
                a("ring"),
                a("prefix"),
                a("prefix_indptr"),
                a("extend"),
                a("extend_indptr"),
                a("maxima"),
                a("sums"),
                a("probabilities"),
                diag,
                tid,
                score_task,
                rows,
                heads,
                pool_rows,
                ring_rows,
                softmax_scale,
            )
            _publish(flags, score_task, epoch, synchronized=True, joined=joined_publication)
            note(14, True)
            timeline_mark(tls, tid, "score", True)
    if pv_task < pv_tasks:
        timeline_mark(tls, tid, "pv_wait")
        note(16)
        owner = pv_task // PV_PER_TILE
        ready = _await(
            flags,
            owner * SCORE_PER_TILE,
            SCORE_PER_TILE,
            epoch,
            diag,
            local,
            tid,
            WORKERS,
            16,
            status=status,
            failure_detail=failure_detail,
            progress=progress,
        )
        timeline_mark(tls, tid, "pv_wait", True)
        if ready:
            timeline_mark(tls, tid, "pv")
            attention_pv_quant_device(
                a("pool"),
                a("ring"),
                a("prefix"),
                a("prefix_indptr"),
                a("extend"),
                a("extend_indptr"),
                a("sink"),
                a("positions"),
                a("cos"),
                a("sin"),
                a("maxima"),
                a("sums"),
                a("probabilities"),
                a("attention_q"),
                a("attention_s"),
                a("debug_out"),
                arena.base_ptr,
                diag,
                tid,
                pv_task,
                rows,
                heads,
                pool_rows,
                ring_rows,
                debug,
            )
            _publish(flags, WORKERS + pv_task, epoch, synchronized=True, joined=joined_publication)
            note(16, True)
            timeline_mark(tls, tid, "pv", True)
    if woa_task < plan.woa_tasks:
        woa_ops = (woa_weight_load(a("woa"), a("woa_s"), tid, woa_task)
                   if WO_PREFETCH == 2 else None)
        timeline_mark(tls, tid, "woa_wait")
        note(17)
        group = woa_task // (O_RANK // 32)
        ready = fx.Boolean(True)
        for row in range_constexpr(rows):
            owner = row * (heads // HEAD_TILE) + group // 2
            acquired = _await(
                flags,
                WORKERS + owner * PV_PER_TILE,
                PV_PER_TILE,
                epoch,
                diag,
                local,
                tid,
                WORKERS,
                17,
                status=status,
                failure_detail=failure_detail,
                progress=progress,
                progress_bias=plan.pv_base,
            )
            ready = ready & acquired
        timeline_mark(tls, tid, "woa_wait", True)
        if ready:
            timeline_mark(tls, tid, "woa")
            woa_quant_device(
                a("attention_q"),
                a("attention_s"),
                a("woa"),
                a("woa_s"),
                a("mid"),
                a("mid_q"),
                a("mid_s"),
                lds,
                tid,
                woa_task,
                rows,
                heads,
                prefetched=woa_ops,
            )
            _publish(flags, 2 * WORKERS + woa_task, epoch, synchronized=True, joined=joined_publication)
            note(17, True)
            timeline_mark(tls, tid, "woa", True)
    note(18)
    wob_ops = (projection_load(a("wob"), a("wob_s"), tid, wob_task,
                               heads // 8 * O_RANK, HIDDEN)
               if WO_PREFETCH else None)
    timeline_mark(tls, tid, "wob_wait")
    ready = _await(
        flags,
        2 * WORKERS,
        woa_tasks(heads),
        epoch,
        diag,
        local,
        tid,
        WORKERS,
        18,
        status=status,
        failure_detail=failure_detail,
        progress=progress,
        progress_bias=plan.woa_base,
    )
    timeline_mark(tls, tid, "wob_wait", True)
    if ready:
        timeline_mark(tls, tid, "wob")
        wob_device(
            a("mid_q"),
            a("mid_s"),
            a("wob"),
            a("wob_s"),
            a("out"),
            lds,
            tid,
            wob_task,
            rows,
            heads,
            prefetched=wob_ops,
        )
        timeline_mark(tls, tid, "wob", True)
    timeline_mark(tls, tid, "wob_publish")
    _publish(flags, 3 * WORKERS + wob_task, epoch, joined=joined_publication)
    note(18, True)
    timeline_mark(tls, tid, "wob_publish", True)
    if const_expr(after_projection is not None):
        after_projection(ready, wob_task)
