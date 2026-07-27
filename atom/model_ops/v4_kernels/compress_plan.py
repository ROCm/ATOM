# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""SGLang-style packed compression plan for V4 batched compressor kernels.

Each plan slot is a 16-byte struct of 4 int32 fields:
    [ragged_id, batch_id, position, window_len]

  - ragged_id:  token's row index in the ragged input stream (kv_in / score_in)
  - batch_id:   sequence index (→ state_slot_mapping[batch_id], block_table[batch_id])
  - position:   absolute token position (→ RoPE)
  - window_len: number of leading K-loop iterations that read from state cache
                instead of the ragged input. K = K_pool = (1+overlap)*ratio
                (pool window size — the algorithm constant; distinct from
                STATE_SIZE = K_pool + max_spec_steps which is just the ring
                modulus widened by spec slack).

Two plan tensors are produced per `compress_ratio`:
  - compress_plan: rows for tokens whose `(position+1) % ratio == 0`
                   (= compression boundaries). One row per fused-compress kernel
                   program.
  - write_plan:    rows for tokens whose `position` falls in the per-seq
                   "last K_pool positions" window (the only entries the
                   downstream compressor forward will actually read). One row
                   per `update_compressor_states` kernel program.

Each plan is sliced to a kernel-grid length that depends on the mode: tight
`num_compress` / `num_write` for eager, or a fixed `graph_bs * per_seq_bound`
for the decode CUDAGraph path (padding rows sentinel-filled). See
`make_compress_plans` for the exact per-mode capacities.

Caller (per-seq loop) gets `cu_compress_cpu` for slicing the kernel's flat
output `[num_compress, head_dim]` back to per-seq chunks.
"""

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np
import torch


@dataclass
class CompressPlan:
    """Packed metadata for one (compress_ratio, overlap) variant of one fwd.

    `compress_plan_gpu` and `write_plan_gpu` may be either tightly-sized
    `[num_compress, 4]` / `[num_write, 4]` tensors (eager path, fresh
    `from_numpy`) or fixed-capacity buffers from a CpuGpuBuffer pool with
    sentinel-filled tail rows (CUDAGraph path). The kernels skip rows whose
    `position` field (col 2) is negative, so both layouts are correct.
    """

    compress_plan_gpu: torch.Tensor  # [≥num_compress, 4] int32
    write_plan_gpu: torch.Tensor  # [≥num_write, 4]    int32
    num_compress: int  # CPU scalar — actual count for downstream slicing
    num_write: int  # CPU scalar — actual count for downstream slicing
    cu_compress_cpu: (
        np.ndarray
    )  # [bs+1] int32 — per-seq slice into out[num_compress, D]
    # Host copy of the compress rows (only the active head [:num_compress, 4]).
    # Consumed by the indexer-FP8 path to derive a flat slot_mapping for
    # `indexer_k_quant_and_cache`. None for empty fwds.
    compress_plan_cpu: np.ndarray | None = None  # [num_compress, 4] int32 or None


def make_compress_plans(
    extend_lens_cpu: np.ndarray,
    context_lens_cpu: np.ndarray,
    unique_ratios_overlap: Iterable[Tuple[int, bool]],
    *,
    plan_buffers: dict,
    graph_bs: int | None = None,
    max_q_len: int | None = None,
    decode_capacity_per_ratio: dict[int, int] | None = None,
) -> dict[int, CompressPlan]:
    """Build a CompressPlan per (ratio, overlap) variant.

    Args:
      extend_lens_cpu:    np[bs] int — number of tokens this fwd processes per seq.
      context_lens_cpu:   np[bs] int — absolute seq_len per seq AFTER the new
                          extend tokens (= prefix + extend). Internally
                          reconstructs prefix via `context_lens - extend_lens`.
      unique_ratios_overlap: iterable of (ratio, is_overlap) pairs. Typically
                             {(4, True), (128, False)} for V4-Pro; a subset for
                             models with only CSA or only HCA layers.
      plan_buffers: required dict[ratio] -> {"compress": CpuGpuBuffer,
                    "write": CpuGpuBuffer} of pre-allocated fixed-capacity
                    plan buffers. Function writes into the existing buffers
                    and sentinel-fills (-1) trailing rows beyond the actual
                    count, so the returned `compress_plan_gpu` /
                    `write_plan_gpu` views have stable data pointers across
                    calls (CUDAGraph requirement). Fresh per-call alloc is
                    not supported — that pattern caused allocator-churn
                    races (see `write_v4_paged_decode_indices` docstring).
      graph_bs: optional int — the CUDAGraph-padded batch size (>= bs). When
                    PROVIDED this selects the DECODE CUDAGraph path: both
                    `compress_plan_gpu` and `write_plan_gpu` are sliced to a
                    FIXED, content-independent capacity `graph_bs *
                    per_seq_bound` (compress bound = `ceil(max_q_len / ratio)`;
                    write bound = `min(max_q_len, K_pool)`), and rows
                    `[n_actual, cap)` are sentinel-filled. The cap depends only
                    on `graph_bs` and `max_q_len` (both fixed at capture), so
                    capture and replay dispatch identically-shaped kernels; the
                    `[bs, graph_bs)` padding seqs land in the sentinel region.
                    `max_q_len` is required when `graph_bs` is set. Because the
                    decode write count is EXACTLY `bs * min(qlen, K_pool)`
                    (content-independent), no separate write-capacity dict is
                    needed — the write cap is derived from `graph_bs`/`max_q_len`
                    identically to compress.
      max_q_len: optional int — uniform per-seq query length of the padded
                    decode batch (`1 + max_spec_steps`). Required iff `graph_bs`
                    is set; used to compute the per-seq compress/write bounds.
      decode_capacity_per_ratio: optional dict[ratio] -> int — explicit FIXED
                    COMPRESS slice length, for CUDAGraph paths whose per-fwd
                    token count is not the uniform `graph_bs * max_q_len` shape
                    (the extend-shaped target-verify graph, whose buffers are
                    sized to a dynamic token count). Mutually exclusive with
                    `graph_bs`. The write plan then keeps the full-buffer legacy
                    slice (fixed = buffer capacity, sentinel-filled) since that
                    path's write grid is bounded by its buffer sizing.
                    When BOTH this and `graph_bs` are None (eager prefill /
                    eager plugin bridges): compress slice = `n_compress` (tight)
                    and write slice = full buffer (legacy). Slices are
                    contiguous-from-base so data pointers stay stable; only
                    `shape[0]` shrinks. Buffers are always sized to the prefill
                    worst case.

    Returns:
      dict[ratio] -> CompressPlan. On empty fwd (`extend_lens_cpu.sum() == 0`)
      still returns CompressPlans pointing at the pre-allocated buffers
      (fully sentinel-filled), so capture-time addresses match replay-time
      addresses even on a zero-token fwd.
    """
    bs = len(extend_lens_cpu)
    extend_lens_cpu = np.ascontiguousarray(extend_lens_cpu, dtype=np.int32)
    context_lens_cpu = np.ascontiguousarray(context_lens_cpu, dtype=np.int32)
    total = int(extend_lens_cpu.sum())
    out: dict[int, CompressPlan] = {}
    if graph_bs is not None:
        assert max_q_len is not None, "max_q_len is required when graph_bs is set"
        assert (
            decode_capacity_per_ratio is None
        ), "graph_bs and decode_capacity_per_ratio are mutually exclusive"

    def _slices(
        ratio: int,
        is_overlap: bool,
        n_compress: int,
        n_write: int,
        full_wcap: int,
    ) -> tuple[int, int]:
        """(compress_slice, write_slice) — the fixed-or-tight kernel-grid lengths
        for this ratio. Three modes:
          * graph_bs set (uniform decode CG): both = `graph_bs * per_seq_bound`
            (compress ceil(qlen/ratio); write min(qlen,K_pool)) — content-
            independent, so capture/replay dispatch identical shapes and the
            `[n_actual, cap)` region is exactly the `[bs, graph_bs)` padding.
          * decode_capacity_per_ratio set (extend-shaped verify CG): compress =
            explicit cap; write = full buffer (fixed by buffer sizing).
          * neither (eager): compress = n_compress (tight); write = full buffer.
        """
        if graph_bs is not None:
            k_pool = (2 if is_overlap else 1) * ratio
            return (
                graph_bs * ((max_q_len + ratio - 1) // ratio),  # ceil(qlen/ratio)
                graph_bs * min(max_q_len, k_pool),
            )
        if decode_capacity_per_ratio is not None:
            return decode_capacity_per_ratio[ratio], full_wcap
        return n_compress, full_wcap

    if total == 0 or bs == 0:
        # Empty fwd: produce CompressPlans pointing at the pre-allocated
        # buffers so capture-time addresses match replay-time addresses
        # even on a zero-token fwd. Skipped via num_*=0.
        for ratio, is_overlap in unique_ratios_overlap:
            cbuf = plan_buffers[ratio]["compress"]
            wbuf = plan_buffers[ratio]["write"]
            ccap, wcap = _slices(ratio, is_overlap, 0, 0, wbuf.np.shape[0])
            assert ccap <= cbuf.np.shape[0] and wcap <= wbuf.np.shape[0], (
                f"ratio={ratio} empty-fwd caps (compress={ccap}, write={wcap}) "
                f"exceed buffers ({cbuf.np.shape[0]}, {wbuf.np.shape[0]}); "
                f"bump plan-buffer sizing in the builder __init__."
            )
            if ccap > 0:
                cbuf.np[:ccap].fill(-1)
            if wcap > 0:
                wbuf.np[:wcap].fill(-1)
            out[ratio] = CompressPlan(
                compress_plan_gpu=cbuf.copy_to_gpu(ccap),
                write_plan_gpu=wbuf.copy_to_gpu(wcap),
                num_compress=0,
                num_write=0,
                cu_compress_cpu=np.zeros(max(bs, 1) + 1, dtype=np.int32),
                compress_plan_cpu=None,
            )
        return out

    # Per-token columns shared across ratios.
    batch_ids = np.repeat(np.arange(bs, dtype=np.int32), extend_lens_cpu)
    ragged_ids = np.arange(total, dtype=np.int32)
    cu_extend = np.empty(bs + 1, dtype=np.int32)
    cu_extend[0] = 0
    np.cumsum(extend_lens_cpu, out=cu_extend[1:])
    j_in_seq = ragged_ids - cu_extend[batch_ids]
    prefix_lens = context_lens_cpu - extend_lens_cpu
    positions = prefix_lens[batch_ids] + j_in_seq

    for ratio, is_overlap in unique_ratios_overlap:
        K = ratio * (2 if is_overlap else 1)
        # window_len = K - min(j_in_seq + 1, K)
        # Number of leading K-loop iterations that go to state cache.
        window_lens = np.maximum(0, K - np.minimum(j_in_seq + 1, K)).astype(np.int32)
        plan_rows = np.stack(
            [ragged_ids, batch_ids, positions, window_lens], axis=1
        ).astype(np.int32)

        # compress: token at a compression boundary
        compress_mask = (positions + 1) % ratio == 0
        compress_plan = plan_rows[compress_mask]
        # cu_compress: per-seq prefix-sum of boundary counts (for caller slicing).
        # bincount preserves seq order because compress_plan rows are already
        # sorted by ragged_id (and ragged_id increases monotonically with batch_id).
        compress_counts = np.bincount(compress_plan[:, 1], minlength=bs).astype(
            np.int32
        )
        cu_compress = np.empty(bs + 1, dtype=np.int32)
        cu_compress[0] = 0
        np.cumsum(compress_counts, out=cu_compress[1:])

        # write: tokens whose absolute position falls in the per-seq
        # "last STATE_SIZE positions" window. STATE_SIZE = K.
        # write_start[i] = max(0, context_lens[i] - K) — uniform across overlap/non-overlap;
        # the SGLang formula `(seq_len // ratio) * ratio - (ratio if overlap else 0)`
        # is a stricter bound that includes only ratio-aligned writes; the looser
        # `context_len - K` is what ATOM's update_compressor_states already uses
        # (state_writes.py:152-154 docstring) and what the fused kernel's
        # state-cache reader expects.
        write_starts = np.maximum(0, context_lens_cpu - K).astype(np.int32)
        write_mask = positions >= write_starts[batch_ids]
        write_plan = plan_rows[write_mask]

        n_compress = int(compress_plan.shape[0])
        n_write = int(write_plan.shape[0])

        cbuf = plan_buffers[ratio]["compress"]
        wbuf = plan_buffers[ratio]["write"]
        full_ccap = cbuf.np.shape[0]
        full_wcap = wbuf.np.shape[0]
        compress_slice, write_slice = _slices(
            ratio, is_overlap, n_compress, n_write, full_wcap
        )
        assert n_compress <= compress_slice <= full_ccap, (
            f"ratio={ratio} num_compress={n_compress}, slice={compress_slice}, "
            f"buffer={full_ccap}: invariant violated. CG path requires "
            f"n_compress ≤ graph_bs·ceil(qlen/ratio) / decode_cap; eager uses "
            f"n_compress."
        )
        assert n_write <= write_slice <= full_wcap, (
            f"ratio={ratio} num_write={n_write}, slice={write_slice}, "
            f"buffer={full_wcap}: invariant violated. CG path requires "
            f"n_write ≤ graph_bs·min(qlen,K_pool); else uses full buffer."
        )
        if n_compress > 0:
            cbuf.np[:n_compress] = compress_plan
        # Sentinel only within the slice we hand to the kernel; rows beyond
        # the slice are unreachable from this launch. For the CG path the
        # `[n_*, cap)` region is exactly the `[bs, graph_bs)` padding seqs.
        if compress_slice > n_compress:
            cbuf.np[n_compress:compress_slice].fill(-1)
        if n_write > 0:
            wbuf.np[:n_write] = write_plan
        if write_slice > n_write:
            wbuf.np[n_write:write_slice].fill(-1)  # sentinel
        compress_plan_gpu = cbuf.copy_to_gpu(compress_slice)
        write_plan_gpu = wbuf.copy_to_gpu(write_slice)

        out[ratio] = CompressPlan(
            compress_plan_gpu=compress_plan_gpu,
            write_plan_gpu=write_plan_gpu,
            num_compress=n_compress,
            num_write=n_write,
            cu_compress_cpu=cu_compress,
            compress_plan_cpu=compress_plan if n_compress > 0 else None,
        )
    return out
