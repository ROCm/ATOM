# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""DCP (Decode Context Parallel) communication ops for ATOM.

Two backends combine the per-rank partial attention outputs into the global
softmax, selected by ``DCPConfig.comm_backend``:

  * ``ag_rs`` -- AllGather LSE -> correct local output -> ReduceScatter (2 calls)
  * ``a2a``   -- one all-to-all carrying output+LSE, combine locally (1 call)

They are mathematically equivalent but not bitwise identical; see the A2A
section below for why one collective can replace two.
"""

import numpy as np
import torch
import triton
import triton.language as tl

from atom.distributed.dcp_utils import get_dcp_group, get_dcp_world_size

_AG_CUSTOM_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


def _ag_custom_view_dtype(x: torch.Tensor) -> torch.dtype | None:
    """Float dtype to reinterpret `x` as for aiter's custom all-gather.

    The custom kernel dispatches on a float enum (fp32/fp16/bf16) and raises on
    anything else, even though an all-gather is a pure copy that only cares
    about element width. The fp8 query therefore has to be viewed first; 8-bit
    payloads have no 8-bit entry in the enum, so they pair up into fp16, which
    needs an even trailing dim. Returns None when no safe view exists.
    """
    if x.dtype in _AG_CUSTOM_DTYPES:
        return x.dtype
    itemsize = x.element_size()
    if itemsize == 4:
        return torch.float32
    if itemsize == 2:
        return torch.float16
    if itemsize == 1 and x.shape[-1] % 2 == 0:
        return torch.float16
    return None


def dcp_all_gather(cp_group, x: torch.Tensor, dim: int) -> torch.Tensor:
    """AllGather that prefers aiter's custom collective over pynccl.

    ``GroupCoordinator.all_gather`` takes ``use_custom=False`` by default, so
    every DCP gather lands on pynccl. At decode payloads that is the wrong
    trade: the nccl kernel carries a fixed ~5us launch bubble on each side, so
    gathering a few KB of LSE costs more than the custom kernel spends on a
    1.5MB reduce-scatter. The device communicator picks the custom kernel when
    the shape qualifies (dim 0 or last dim, 16B-aligned, within the registered
    pool) and falls back to pynccl otherwise, with identical concat-along-`dim`
    semantics either way.
    """
    device_comm = getattr(cp_group, "device_communicator", None)
    view_dtype = _ag_custom_view_dtype(x) if device_comm is not None else None
    if view_dtype is None:
        return cp_group.all_gather(x, dim=dim)
    if view_dtype is x.dtype:
        return device_comm.all_gather(x, dim)
    # Viewing narrows the trailing dim, so only a last-dim gather sees a
    # different split; both land on the same byte layout after the view back.
    return device_comm.all_gather(x.view(view_dtype), dim).view(x.dtype)


def dcp_all_gather_query_heads(cp_group, q: torch.Tensor) -> torch.Tensor:
    """AllGather decode Q ``[tokens, heads, head_dim]`` over the DCP group's heads.

    Flattened to 2-D so the gather lands on the last dim. aiter's custom
    collective only serves dim 0 and the last dim, and its last-dim variant
    concatenates rank-major -- which for a ``[tokens, heads*head_dim]`` view
    is exactly head-dim concat. Gathering the 3-D tensor on dim=1 instead
    both misses the custom kernel and makes the pynccl path materialise an
    extra reshape copy of the gathered result.
    """
    tokens, _, head_dim = q.shape
    if not q.is_contiguous():
        return cp_group.all_gather(q, dim=1)
    gathered = dcp_all_gather(cp_group, q.reshape(tokens, -1), -1)
    return gathered.view(tokens, -1, head_dim)


class CPTritonContext:
    """Cache compiled Triton kernel to avoid recompilation on every call."""

    def __init__(self):
        self.inner_kernel = None

    def call_kernel(self, kernel, grid, *regular_args, **const_args):
        if self.inner_kernel is None:
            self.inner_kernel = kernel[grid](*regular_args, **const_args)
        else:
            self.inner_kernel[grid](*regular_args)


@triton.jit
def _correct_attn_cp_out_kernel(
    outputs_ptr,
    new_output_ptr,
    lses_ptr,
    vlse_ptr,
    outputs_stride_B,
    outputs_stride_H,
    outputs_stride_D,
    lses_stride_N,
    lses_stride_B,
    lses_stride_H,
    lse_idx,
    HEAD_DIM: tl.constexpr,
    N_ROUNDED: tl.constexpr,
):
    """Correct local rank's attention output using all-gathered LSEs.

    For each (batch, head):
      1. global_lse = logsumexp(lse_0, ..., lse_{N-1})
      2. factor = exp(local_lse - global_lse)
      3. output *= factor

    After ReduceScatter(sum), the corrected outputs from all ranks
    combine into the final attention output.
    """
    batch_idx = tl.program_id(axis=0).to(tl.int64)
    head_idx = tl.program_id(axis=1).to(tl.int64)
    d_offsets = tl.arange(0, HEAD_DIM)
    num_n_offsets = tl.arange(0, N_ROUNDED)

    lse_offsets = (
        num_n_offsets * lses_stride_N
        + batch_idx * lses_stride_B
        + head_idx * lses_stride_H
    )

    lse = tl.load(lses_ptr + lse_offsets)
    lse = tl.where(
        (lse != lse) | (lse == float("inf")),  # noqa: PLR0124 - Triton NaN check
        -float("inf"),
        lse,
    )

    lse_max = tl.max(lse, axis=0)
    lse_max = tl.where(lse_max == -float("inf"), 0, lse_max)
    lse -= lse_max
    lse_exp = tl.exp(lse)
    lse_acc = tl.sum(lse_exp, axis=0)
    global_lse = tl.log(lse_acc) + lse_max

    lse_out_offset = batch_idx * lses_stride_B + head_idx * lses_stride_H
    tl.store(vlse_ptr + lse_out_offset, global_lse)

    local_lse_offset = (
        lse_idx * lses_stride_N + batch_idx * lses_stride_B + head_idx * lses_stride_H
    )
    local_lse = tl.load(lses_ptr + local_lse_offset)
    lse_diff = local_lse - global_lse
    lse_diff = tl.where(
        (lse_diff != lse_diff)  # noqa: PLR0124 - Triton NaN check
        | (lse_diff == float("inf")),
        -float("inf"),
        lse_diff,
    )
    factor = tl.exp(lse_diff)

    output_offsets = (
        batch_idx * outputs_stride_B
        + head_idx * outputs_stride_H
        + d_offsets * outputs_stride_D
    )
    output = tl.load(outputs_ptr + output_offsets)
    output = output * factor
    # A rank that owns no candidate for this row has local_lse=-inf -> factor=0,
    # and its attention output is NaN (0/0 in the kernel's acc/e_sum). NaN*0=NaN
    # would poison the ReduceScatter sum, so force the empty-rank contribution to
    # 0. Matches vLLM's cp merge kernel; covers both decode and prefill so callers
    # need no separate NaN scrub.
    output = tl.where(factor == 0.0, 0.0, output)
    tl.store(new_output_ptr + output_offsets, output)


def correct_attn_out(out, lses, cp_rank, ctx=None):
    """Correct local rank's attention output using all-gathered LSEs.

    Args:
        out: [B, H, D] local attention output
        lses: [N, B, H] all-gathered LSE values
        cp_rank: this rank's index in the CP group
        ctx: optional CPTritonContext to cache compiled kernel

    Returns:
        (out, lse): corrected output [B, H, D] and global LSE [B, H]
    """
    B, H, D = out.shape
    N = lses.shape[0]
    assert (N & (N - 1)) == 0, f"cp world size must be a power of two, got {N}"

    lse = torch.empty_strided(
        (B, H), (lses.stride(1), lses.stride(2)), device=lses.device, dtype=lses.dtype
    )

    grid = (B, H, 1)
    regular_args = (
        out,
        out,
        lses,
        lse,
        out.stride(0),
        out.stride(1),
        out.stride(2),
        lses.stride(0),
        lses.stride(1),
        lses.stride(2),
        cp_rank,
    )
    const_args = {"HEAD_DIM": D, "N_ROUNDED": N}

    if ctx is not None:
        ctx.call_kernel(_correct_attn_cp_out_kernel, grid, *regular_args, **const_args)
    else:
        _correct_attn_cp_out_kernel[grid](*regular_args, **const_args)

    return out, lse


def cp_lse_ag_out_rs(cp_attn_out, cp_attn_lse, cp_group, ctx=None):
    """AG+RS backend: AllGather LSE -> Triton correct -> ReduceScatter output.

    Args:
        cp_attn_out: [B, H_full, D] local attention output (full heads after AG Q)
        cp_attn_lse: [B, H_full] local LSE values
        cp_group: DCP communication group (GroupCoordinator)
        ctx: optional CPTritonContext to cache compiled kernel

    Returns:
        output: [B, H_local, D] corrected output with local heads only
    """
    if cp_group.world_size == 1:
        return cp_attn_out

    cp_attn_lse = cp_attn_lse.contiguous()
    lses = dcp_all_gather(cp_group, cp_attn_lse, 0)
    lses = lses.reshape((cp_group.world_size,) + cp_attn_lse.shape)

    out, _ = correct_attn_out(cp_attn_out, lses, cp_group.rank_in_group, ctx=ctx)

    out = out.movedim(1, 0).contiguous()  # [B, H_full, D] -> [H_full, B, D]
    out = cp_group.reduce_scatter(out, dim=0)
    out = out.movedim(0, 1).contiguous()  # [H_local, B, D] -> [B, H_local, D]
    return out


# ─────────────────────────────────────────────── A2A merge backend ──
#
# All-to-All backend for the DCP output merge.
#
# The AG+RS backend (``cp_lse_ag_out_rs``) needs two collectives, and the reason is
# ``ReduceScatter``'s reduce op: it can only be a predefined ``sum``/``max``, never
# an LSE-weighted combine. To use ``sum`` the weights must already be applied, and
# computing them needs ``global_lse``, which needs every rank's LSE -- hence the
# AllGather LSE that comes first.
#
# A2A sidesteps that by not asking the network to reduce at all. The all-to-all is
# a pure permutation: it relocates data so that every partial for a given head
# lands on the one rank that owns that head. The weighting and the sum then happen
# in a local kernel, where arbitrary math is allowed. One collective instead of two.
#
# Both backends compute the same thing::
#
#     global_lse[b,h] = log sum_r exp(lse_r[b,h])
#     out[b,h]        = sum_r exp(lse_r[b,h] - global_lse[b,h]) * o_r[b,h]
#
# They differ only in where the multiply and the sum happen, so results agree to
# floating-point noise but are NOT bitwise identical (different summation order).
#
# Bytes are roughly the same as AG+RS: ReduceScatter and All-to-All both carry
# scatter semantics, so each moves ~(N-1)/N of the full tensor. What A2A saves is
# one collective's launch and synchronization -- which is worth measuring rather
# than assuming, because the profiling in the DCP notes shows a large part of the
# current ReduceScatter cost scales with row count rather than with bytes.


def _lse_pack_slots(dtype: torch.dtype) -> int:
    """How many buffer slots one fp32 LSE needs at this element width."""
    if dtype == torch.float32:
        return 1
    if dtype in (torch.bfloat16, torch.float16):
        return 2
    raise NotImplementedError(f"a2a merge buffer dtype {dtype} not supported")


@triton.jit
def _dcp_a2a_pack_kernel(
    out_ptr,  # [B, H, D]      this rank's partial attention output
    lse_ptr,  # [B, H]         fp32
    owned_counts_ptr,  # [B]   true rank-local sparse counts, or unused
    send_ptr,  # [N, B, H_LOCAL, D + LSE_PACK]
    out_stride_b,
    out_stride_h,
    lse_stride_b,
    lse_stride_h,
    send_stride_n,
    send_stride_b,
    send_stride_h,
    H_LOCAL: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    LSE_PACK: tl.constexpr,
    HAS_OWNED_COUNTS: tl.constexpr,
):
    """Scatter (output, lse) into the per-destination send buffer.

    One program per (token, GLOBAL head). Head ``h`` belongs to destination rank
    ``h // H_LOCAL`` -- the same contiguous split the ReduceScatter path uses, so
    a rank ends up owning exactly the heads it owned before.
    """
    b = tl.program_id(axis=0).to(tl.int64)
    h = tl.program_id(axis=1).to(tl.int64)

    dst = h // H_LOCAL
    h_local = h % H_LOCAL

    d = tl.arange(0, HEAD_DIM)
    src = tl.load(out_ptr + b * out_stride_b + h * out_stride_h + d)

    dst_base = (
        send_ptr + dst * send_stride_n + b * send_stride_b + h_local * send_stride_h
    )
    tl.store(dst_base + d, src)

    lse = tl.load(lse_ptr + b * lse_stride_b + h * lse_stride_h)
    if HAS_OWNED_COUNTS:
        # Empty sparse-DCP rows carry one dummy KV slot so persistent metadata
        # remains valid. Neutralize that slot while packing the existing A2A
        # payload; this adds no kernel launch to the decode path.
        lse = tl.where(tl.load(owned_counts_ptr + b) == 0, float("-inf"), lse)
    if LSE_PACK == 1:
        tl.store(dst_base + HEAD_DIM, lse)
    else:
        # Split the fp32 into two raw 16-bit halves. Nothing does arithmetic on
        # these slots -- the collective copies bits and the combine kernel puts
        # them back together -- so a bf16 slot is just 16 bits of storage here.
        bits = lse.to(tl.uint32, bitcast=True)
        hi = (bits >> 16).to(tl.uint16).to(dst_base.dtype.element_ty, bitcast=True)
        lo = (bits & 0xFFFF).to(tl.uint16).to(dst_base.dtype.element_ty, bitcast=True)
        tl.store(dst_base + HEAD_DIM, hi)
        tl.store(dst_base + HEAD_DIM + 1, lo)


@triton.jit
def _dcp_a2a_unpack_combine_kernel(
    recv_ptr,  # [N, B, H_LOCAL, D + LSE_PACK]  N = source rank (KV shard)
    out_ptr,  # [B, H_LOCAL, D]
    out_lse_ptr,  # [B, H_LOCAL] fp32, or unused
    recv_stride_n,
    recv_stride_b,
    recv_stride_h,
    out_stride_b,
    out_stride_h,
    lse_stride_b,
    lse_stride_h,
    N_RANKS,
    HEAD_DIM: tl.constexpr,
    LSE_PACK: tl.constexpr,
    N_ROUNDED: tl.constexpr,
    WRITE_LSE: tl.constexpr,
):
    """LSE-weighted combine along the KV-shard axis. One program per (token, head).

    The reduce axis is N (the KV shards), never the head axis -- heads are only
    ever relocated, never summed across. That is what makes the head split above
    free to be any partition.
    """
    b = tl.program_id(axis=0).to(tl.int64)
    h = tl.program_id(axis=1).to(tl.int64)

    n = tl.arange(0, N_ROUNDED)
    valid = n < N_RANKS
    base = (
        recv_ptr
        + n.to(tl.int64) * recv_stride_n
        + b * recv_stride_b
        + h * recv_stride_h
    )

    if LSE_PACK == 1:
        lse = tl.load(base + HEAD_DIM, mask=valid, other=float("-inf"))
        lse = lse.to(tl.float32)
    else:
        hi = tl.load(base + HEAD_DIM, mask=valid, other=0).to(tl.uint16, bitcast=True)
        lo = tl.load(base + HEAD_DIM + 1, mask=valid, other=0).to(
            tl.uint16, bitcast=True
        )
        bits = (hi.to(tl.uint32) << 16) | lo.to(tl.uint32)
        lse = bits.to(tl.float32, bitcast=True)
        lse = tl.where(valid, lse, float("-inf"))

    # A rank that owns no KV for this row reports lse=-inf (and o=NaN). Treat any
    # non-finite lse as "contributed nothing" so it cannot reach the accumulator.
    # `x != x` is the NaN test: this Triton has no tl.math.isnan, and it is the
    # same idiom dcp_ops.py already uses.
    lse_is_nan = lse != lse  # noqa: PLR0124
    lse = tl.where(lse_is_nan | (lse == float("inf")), float("-inf"), lse)

    lse_max = tl.max(lse, axis=0)
    # Every rank empty -> max is -inf; subtracting it would make 0/0 = NaN.
    lse_max = tl.where(lse_max == float("-inf"), 0.0, lse_max)
    global_lse = tl.log(tl.sum(tl.exp(lse - lse_max), axis=0)) + lse_max

    if WRITE_LSE:
        tl.store(out_lse_ptr + b * lse_stride_b + h * lse_stride_h, global_lse)

    factor = tl.exp(lse - global_lse)
    # An empty rank already lands on factor == 0 by itself: its lse is -inf, so
    # exp(-inf - finite) is exactly 0. This line is belt-and-braces for the case
    # where global_lse is itself -inf (every rank empty) and the subtraction
    # yields NaN; the load-bearing NaN guard is the vals mask below.
    factor = tl.where((factor != factor) | (~valid), 0.0, factor)  # noqa: PLR0124

    d = tl.arange(0, HEAD_DIM)
    vals = tl.load(base[:, None] + d[None, :]).to(tl.float32)
    # THIS is what stops an empty rank from poisoning the row. aiter returns
    # o=NaN alongside lse=-inf, and NaN * 0 = NaN, so the NaN has to be replaced
    # BEFORE the multiply -- zeroing the weight is not enough.
    vals = tl.where(factor[:, None] == 0.0, 0.0, vals)
    acc = tl.sum(vals * factor[:, None], axis=0)

    tl.store(
        out_ptr + b * out_stride_b + h * out_stride_h + d,
        acc.to(out_ptr.dtype.element_ty),
    )


def cp_lse_a2a(
    cp_attn_out,
    cp_attn_lse,
    cp_group,
    return_lse: bool = False,
    owned_counts=None,
):
    """A2A backend: pack -> one all-to-all -> local LSE combine.

    Drop-in for ``cp_lse_ag_out_rs``: same inputs, same ``[B, H_local, D]``
    output, same head ownership.

    Args:
        cp_attn_out: ``[B, H, D]`` this rank's partial output over its KV shard.
        cp_attn_lse: ``[B, H]`` matching log-sum-exp, fp32.
        cp_group: DCP GroupCoordinator.
        return_lse: also return the merged ``[B, H_local]`` global LSE.
        owned_counts: optional true rank-local sparse-KV count per row. Empty
            rows are masked inside the existing A2A pack kernel.
    """
    n_ranks = cp_group.world_size
    if n_ranks == 1:
        return (cp_attn_out, cp_attn_lse) if return_lse else cp_attn_out

    b, h_total, head_dim = cp_attn_out.shape
    assert h_total % n_ranks == 0, (
        f"a2a merge needs the head count divisible by the DCP size; "
        f"got H={h_total}, N={n_ranks}"
    )
    h_local = h_total // n_ranks

    dtype = cp_attn_out.dtype
    pack = _lse_pack_slots(dtype)
    dev = cp_attn_out.device

    cp_attn_out = cp_attn_out.contiguous()
    cp_attn_lse = cp_attn_lse.contiguous().to(torch.float32)
    has_owned_counts = owned_counts is not None
    if has_owned_counts:
        owned_counts = owned_counts.contiguous()
        assert owned_counts.numel() >= b
    else:
        # HAS_OWNED_COUNTS removes the load at compile time; use an existing
        # device pointer so no placeholder tensor or allocation is introduced.
        owned_counts = cp_attn_lse

    send = torch.empty((n_ranks, b, h_local, head_dim + pack), dtype=dtype, device=dev)
    _dcp_a2a_pack_kernel[(b, h_total)](
        cp_attn_out,
        cp_attn_lse,
        owned_counts,
        send,
        cp_attn_out.stride(0),
        cp_attn_out.stride(1),
        cp_attn_lse.stride(0),
        cp_attn_lse.stride(1),
        send.stride(0),
        send.stride(1),
        send.stride(2),
        H_LOCAL=h_local,
        HEAD_DIM=head_dim,
        LSE_PACK=pack,
        HAS_OWNED_COUNTS=has_owned_counts,
    )

    # The N axis flips meaning here: it is "destination rank" on the way in and
    # "source rank / KV shard" on the way out.
    recv = torch.empty_like(send)
    torch.distributed.all_to_all_single(recv, send, group=cp_group.device_group)

    out = torch.empty((b, h_local, head_dim), dtype=dtype, device=dev)
    out_lse = (
        torch.empty((b, h_local), dtype=torch.float32, device=dev)
        if return_lse
        else send
    )
    _dcp_a2a_unpack_combine_kernel[(b, h_local)](
        recv,
        out,
        out_lse,
        recv.stride(0),
        recv.stride(1),
        recv.stride(2),
        out.stride(0),
        out.stride(1),
        out_lse.stride(0) if return_lse else 0,
        out_lse.stride(1) if return_lse else 0,
        n_ranks,
        HEAD_DIM=head_dim,
        LSE_PACK=pack,
        N_ROUNDED=triton.next_power_of_2(n_ranks),
        WRITE_LSE=return_lse,
    )
    return (out, out_lse) if return_lse else out


def mask_empty_dcp_lse(
    cp_attn_lse: torch.Tensor, owned_counts: torch.Tensor
) -> torch.Tensor:
    """Mark dummy-backed local rows as absent before the DCP softmax merge."""
    return cp_attn_lse.masked_fill(owned_counts[:, None] == 0, float("-inf"))


def dcp_lse_merge(
    cp_attn_out,
    cp_attn_lse,
    cp_group,
    backend="a2a",
    ctx=None,
    owned_counts=None,
):
    """Reconstruct the global softmax from the per-rank partials.

    Dispatches to one of the two backends above. Both compute the same weighted
    sum over KV shards and leave this rank owning the same head slice; they
    differ only in how many collectives it takes (see the A2A section above for
    why one all-to-all can replace AllGather-LSE + ReduceScatter). Equivalent
    math, not bitwise identical.

    ``owned_counts`` is the optional rank-local sparse-KV count per row. Empty
    prefill rows contain a dummy cache slot to keep persistent metadata valid;
    masking only their small LSE tensor here lets the existing merge kernels
    suppress the corresponding output without another full-output pass.

    ``ctx`` is the Triton context the AG+RS backend caches its launches in; the
    a2a backend does not use one.
    """
    if backend == "a2a":
        return cp_lse_a2a(
            cp_attn_out,
            cp_attn_lse,
            cp_group,
            owned_counts=owned_counts,
        )
    if owned_counts is not None:
        # AG+RS has no existing kernel before its LSE AllGather. The default
        # A2A path above fuses this mask and remains launch-free.
        cp_attn_lse = mask_empty_dcp_lse(cp_attn_lse, owned_counts)
    return cp_lse_ag_out_rs(cp_attn_out, cp_attn_lse, cp_group, ctx=ctx)


def dcp_gather_compressed_kv(
    kv_cache: torch.Tensor, slot_ids: torch.Tensor
) -> torch.Tensor:
    """Gather this rank's compressed KV entries from the paged cache.

    Local-gather step: the MLA cache stores compressed latent KV as
    ``[num_slots, 1, kv_lora_rank + qk_rope_head_dim]`` (or ``[num_slots, d]``),
    so gathering the local rank's interleaved tokens for a chunk is a plain
    index_select over the token-slot axis. This replaces vLLM's
    ``cp_gather_cache`` custom op (unavailable in the aiter server path).

    Args:
        kv_cache: paged compressed KV cache, token-slot major on dim 0.
        slot_ids: int tensor of absolute slot ids to gather (this rank's local
            tokens for the chunk, in per-seq order).

    Returns:
        [len(slot_ids), kv_lora_rank + qk_rope_head_dim] compressed KV.
    """
    gathered = kv_cache.index_select(0, slot_ids)
    # Collapse any singleton head dim -> [toks, kv_lora_rank + qk_rope_head_dim].
    return gathered.reshape(slot_ids.shape[0], -1)


def reorg_kvcache(
    allgatered_kv_c_normed: torch.Tensor,
    allgatered_k_pe: torch.Tensor,
    padded_local_chunk_seq_lens_lst: list,
    local_context_lens_allranks: list,
    sum_seq_len: int,
    max_seq_len: int,
    chunk_size: int,
    chunk_idx: int,
    toks: int,
):
    """Reorg + unpad AllGathered compressed KV into per-sequence contiguous
    layout for the attention kernel.

    The AllGather concatenates every rank's local (padded) chunk gather along
    dim 0, so tokens for one sequence are interleaved across the per-rank
    blocks. This walks each seq's per-rank contribution and concatenates them
    back into the original token order, dropping padding.

    e.g.
    allgatered = [T0_0, T0_1, T0_2, T0_3, T1_0, T1_1, ...,      # rank 0 block
                  T0_4, T0_5, pad, pad, T1_2, pad, ...]         # rank 1 block
    -> reorganized = [T0_0, T0_1, T0_2, T0_3, T0_4, T0_5,
                      T1_0, T1_1, T1_2, ...]

    Args:
        padded_local_chunk_seq_lens_lst: per-seq local chunk lengths (padded)
            under the current CP rank.
        local_context_lens_allranks: per-seq local context lengths on each rank.
        sum_seq_len: sum of the per-seq (global) chunk lengths.
        max_seq_len: max per-seq (global) chunk length.
        chunk_size: local padded max context chunk from metadata building.
        chunk_idx: chunk index of the chunked prefill.
        toks: number of tokens per rank's local gather (one AllGather block).
    """
    kv_c_segments = []
    k_pe_segments = []
    src_token_idx = 0
    max_seq_len_check = 0
    for padded_local_chunk_seq_len, local_context_lens in zip(
        padded_local_chunk_seq_lens_lst, local_context_lens_allranks
    ):
        cur_seq_len = 0
        for rank, local_context_len in enumerate(local_context_lens):
            # We split the context into multiple chunks depending on the
            # workspace size, so the last chunk on a shorter rank may be
            # partial: clamp to what actually remains on that rank.
            local_chunk_len = min(
                max(0, local_context_len - chunk_idx * chunk_size),
                padded_local_chunk_seq_len,
            )
            if local_chunk_len != 0:
                kv_c_segment = allgatered_kv_c_normed[
                    rank * toks
                    + src_token_idx : rank * toks
                    + src_token_idx
                    + local_chunk_len
                ]
                k_pe_segment = allgatered_k_pe[
                    rank * toks
                    + src_token_idx : rank * toks
                    + src_token_idx
                    + local_chunk_len
                ]
                kv_c_segments.append(kv_c_segment)
                k_pe_segments.append(k_pe_segment)
                cur_seq_len += local_chunk_len
        max_seq_len_check = max(max_seq_len_check, cur_seq_len)
        src_token_idx += padded_local_chunk_seq_len
    reorganized_kv_c_normed = torch.cat(kv_c_segments, dim=0)
    reorganized_k_pe = torch.cat(k_pe_segments, dim=0)
    assert reorganized_kv_c_normed.shape[0] == sum_seq_len
    assert reorganized_k_pe.shape[0] == sum_seq_len
    assert max_seq_len_check == max_seq_len
    return reorganized_kv_c_normed, reorganized_k_pe


def get_dcp_local_seq_lens(seq_lens, dcp_size, dcp_rank, cp_kv_cache_interleave_size=1):
    """Compute per-DCP-rank local sequence lengths.

    With interleaved storage, token i is stored on rank
    (i // cp_kv_cache_interleave_size) % dcp_size.

    Args:
        seq_lens: numpy array of sequence lengths
        dcp_size: DCP world size
        dcp_rank: this rank's DCP rank
        cp_kv_cache_interleave_size: interleaving granularity (default 1 = token-level)

    Returns:
        local_seq_lens: numpy array of local sequence lengths
    """
    full_chunks = seq_lens // (cp_kv_cache_interleave_size * dcp_size)
    base = full_chunks * cp_kv_cache_interleave_size

    remainder_total = seq_lens - base * dcp_size
    remainder = np.clip(
        remainder_total - dcp_rank * cp_kv_cache_interleave_size,
        0,
        cp_kv_cache_interleave_size,
    )
    return base + remainder


def dcp_owner_rank(pos, dcp_size, cp_kv_cache_interleave_size=1):
    """Which DCP rank owns global token ``pos`` under interleaved KV storage.

    Interleaving groups tokens into chunks of ``cp_kv_cache_interleave_size`` (= S); chunk
    ``c = pos // S`` is stored on rank ``c % dcp_size``. For S == 1 this reduces
    to the round-robin ``pos % dcp_size``.

    Works elementwise on Python ints, numpy arrays and torch tensors (only ``//``
    and ``%`` are used). Consistent with vLLM's slot kernel
    (``block_table.py`` ``is_local``) because ``block_size * W`` is a multiple of
    ``S * W`` when ``block_size % S == 0``, so computing on the global position
    equals computing on the virtual-block offset.
    """
    return (pos // cp_kv_cache_interleave_size) % dcp_size


def dcp_local_index(pos, dcp_size, cp_kv_cache_interleave_size=1):
    """Local KV-sequence index of global token ``pos`` on its owning rank.

    Each ``S * W`` super-block contributes ``S`` tokens to a rank, so the local
    index is ``(pos // (S*W)) * S + (pos % S)``. For S == 1 this reduces to the
    round-robin ``pos // dcp_size``.

    To map to a physical slot (given ``block_size % S == 0``):
        block_table_index = pos // (block_size * dcp_size)   # == local_index // block_size
        slot_offset       = local_index % block_size
        slot              = block_table[block_table_index] * block_size + slot_offset

    Elementwise over Python ints / numpy / torch.
    """
    sw = cp_kv_cache_interleave_size * dcp_size
    return (pos // sw) * cp_kv_cache_interleave_size + (
        pos % cp_kv_cache_interleave_size
    )


def dcp_global_pos(local_index, dcp_rank, dcp_size, cp_kv_cache_interleave_size=1):
    """Inverse of ``dcp_local_index``: global token position of local KV index
    ``local_index`` held on ``dcp_rank``.

    Local index j on rank r sits in local S-group ``j // S`` at offset ``j % S``;
    that group is global chunk ``(j//S)*W + r``, so the global position is
    ``((j//S)*W + r) * S + (j % S)``. For S == 1 this reduces to the round-robin
    ``j*W + r``. Used to reconstruct globally-unique ids for exchanged sparse
    top-k candidates (the id must be a total order over global positions).

    Elementwise over Python ints / numpy / torch.
    """
    return (
        (local_index // cp_kv_cache_interleave_size) * dcp_size + dcp_rank
    ) * cp_kv_cache_interleave_size + (local_index % cp_kv_cache_interleave_size)


@triton.jit
def _dcp_pack_topk_kernel(
    logits_ptr,  # [rows, l_max] fp32 -- this rank's local score plane
    idx_ptr,  # [rows, k]     int32 -- local top-k column ids
    lens_ptr,  # [rows]        int32 -- this rank's local KV length
    out_ptr,  # [2, rows, k]  fp32  -- plane 0 score, plane 1 int32 gid bits
    logits_stride_r,
    logits_stride_c,
    idx_stride_r,
    idx_stride_c,
    lens_stride,
    out_stride_p,
    out_stride_r,
    out_stride_c,
    K,
    DCP_RANK: tl.constexpr,
    DCP_WORLD: tl.constexpr,
    INTERLEAVE: tl.constexpr,  # cp_kv_cache_interleave_size S (1 = round-robin)
    BLOCK_K: tl.constexpr,
):
    """Fused (bound-check, score gather, global-id) pack. One program per
    (row, BLOCK_K columns).

    Replaces 19 eager kernels; the intermediates they materialised (`valid`,
    `safe`, `sc`, the six `dcp_global_pos` steps) never leave registers, which
    is where nearly all of this segment's HBM traffic came from.
    """
    r = tl.program_id(0).to(tl.int64)
    col = tl.program_id(1) * BLOCK_K + tl.arange(0, BLOCK_K)
    mask = col < K

    idx = tl.load(idx_ptr + r * idx_stride_r + col * idx_stride_c, mask=mask, other=-1)
    ln = tl.load(lens_ptr + r * lens_stride)
    # Bound-check rather than assume a padding convention from the aiter kernel.
    valid = mask & (idx >= 0) & (idx < ln)

    safe = tl.where(valid, idx, 0).to(tl.int64)
    sc = tl.load(
        logits_ptr + r * logits_stride_r + safe * logits_stride_c,
        mask=valid,
        other=0.0,
    )
    sc = tl.where(valid, sc, float("-inf"))

    # Under interleave-S sharding a local index j on rank r is global position
    # ((j//S)*W + r)*S + j%S; S=1 collapses to the round-robin j*W + r.
    if INTERLEAVE == 1:
        gid = idx * DCP_WORLD + DCP_RANK
    else:
        gid = ((idx // INTERLEAVE) * DCP_WORLD + DCP_RANK) * INTERLEAVE + (
            idx % INTERLEAVE
        )
    gid = tl.where(valid, gid, -1)

    base = out_ptr + r * out_stride_r + col * out_stride_c
    tl.store(base, sc, mask=mask)
    # Plane 1 carries the int32 bit pattern in an fp32 slot: the collective only
    # copies bits and the consumer reads it back through an int32 view.
    tl.store(base + out_stride_p, gid.to(tl.float32, bitcast=True), mask=mask)


def dcp_pack_topk_candidates(
    local_logits,
    local_idx,
    local_lens,
    dcp_rank,
    dcp_world_size,
    out_pair,
    cp_kv_cache_interleave_size=1,
):
    """Turn a rank-local top-k into exchangeable (score, global_id) pairs.

    out_pair: fp32 [2, rows, k] -- plane 0 holds scores, plane 1 holds int32
    global ids reinterpreted as fp32 so both travel in one collective. Slots the
    local top-k did not fill get (-inf, -1); the merge sinks them via -inf and
    never selects the -1 gids.

    Under interleave-S sharding a local index j on rank r is global position
    ``((j//S)*W + r)*S + j%S`` (S=1 -> the round-robin j*W + r), so the id is
    globally unique -- which is what makes the tie-break a total order.
    """
    rows, k = local_idx.shape
    BLOCK_K = 256
    _dcp_pack_topk_kernel[(rows, triton.cdiv(k, BLOCK_K))](
        local_logits,
        local_idx,
        local_lens,
        out_pair,
        local_logits.stride(0),
        local_logits.stride(1),
        local_idx.stride(0),
        local_idx.stride(1),
        local_lens.stride(0),
        out_pair.stride(0),
        out_pair.stride(1),
        out_pair.stride(2),
        k,
        DCP_RANK=dcp_rank,
        DCP_WORLD=dcp_world_size,
        INTERLEAVE=cp_kv_cache_interleave_size,
        BLOCK_K=BLOCK_K,
        num_warps=4,
    )


@triton.jit
def _dcp_gather_candidate_gids_kernel(
    recv_ptr,  # [W, 2, rows, k] int32 view of the all-gathered pairs
    cand_ptr,  # [rows, topk]    int32 -- merge output, index into [0, W*k)
    out_ptr,  # [rows, topk]    int32 -- global ids
    recv_stride_w,
    recv_stride_p,
    recv_stride_r,
    recv_stride_c,
    cand_stride_r,
    cand_stride_c,
    out_stride_r,
    out_stride_c,
    TOPK,
    K_LOC: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Map merged candidate indices back to global ids, straight off the
    all-gather buffer.

    The eager version had to `reshape` the permuted ``[rows, W, k]`` gid plane
    into ``[rows, W*k]`` first, which materialises a full rows x W x k copy just
    so `torch.gather` sees a 2-D tensor. Decomposing the candidate index as
    ``(w, j) = divmod(c, K_LOC)`` here reads the same elements in place, so the
    copy, the int64 index cast and the final `copy_` all disappear.
    """
    r = tl.program_id(0).to(tl.int64)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = col < TOPK

    c = tl.load(
        cand_ptr + r * cand_stride_r + col * cand_stride_c, mask=mask, other=0
    ).to(tl.int64)
    w = c // K_LOC
    j = c % K_LOC
    gid = tl.load(
        recv_ptr
        + w * recv_stride_w
        + recv_stride_p  # plane 1 = gid
        + r * recv_stride_r
        + j * recv_stride_c,
        mask=mask,
        other=-1,
    )
    tl.store(out_ptr + r * out_stride_r + col * out_stride_c, gid, mask=mask)


def dcp_gather_candidate_gids(recv_i32, cand_idx, out):
    """``out[r, i] = recv_i32[c // k, 1, r, c % k]`` with ``c = cand_idx[r, i]``.

    ``recv_i32`` is the ``[W, 2, rows, k]`` int32 view of the all-gathered
    (score, gid) pairs; ``cand_idx`` is the merge's row-local candidate index.
    """
    # The eager version was a `copy_` of a gather, so the shapes had to match
    # exactly; keep that contract explicit now that the kernel writes `out`
    # directly and a mismatch would run off the end of it instead of raising.
    assert out.shape == cand_idx.shape, (out.shape, cand_idx.shape)
    rows, topk = cand_idx.shape
    k_loc = recv_i32.shape[3]
    BLOCK = 256
    _dcp_gather_candidate_gids_kernel[(rows, triton.cdiv(topk, BLOCK))](
        recv_i32,
        cand_idx,
        out,
        recv_i32.stride(0),
        recv_i32.stride(1),
        recv_i32.stride(2),
        recv_i32.stride(3),
        cand_idx.stride(0),
        cand_idx.stride(1),
        out.stride(0),
        out.stride(1),
        topk,
        K_LOC=k_loc,
        BLOCK=BLOCK,
        num_warps=4,
    )


def dcp_local_context_lens(
    attn_metadata,
    dcp_rank: int,
    dcp_world_size: int,
    cp_kv_cache_interleave_size: int,
    num_rows: int,
) -> torch.Tensor:
    """This rank's per-request LOCAL KV length under interleave-S sharding.

    Matches get_dcp_local_seq_lens / prepare_decode's slot split: each full S*W
    super-block gives every rank S tokens, and the tail remainder is handed out
    S at a time by rank. S=1 -> the round-robin base + (does this rank own the
    +1 tail?) split.

    Depends only on context_lens / S / W / dcp_rank, so it is the same for every
    layer and prepare_decode already computes it on the host. Prefer that
    published buffer -- deriving it here costs 8 elementwise kernels on every
    full-index layer (21 of them on GLM-5.2). The fallback keeps metadata
    builders that do not publish it working.
    """
    local_ctx = getattr(attn_metadata, "dcp_local_context_lens", None)
    if local_ctx is not None and local_ctx.shape[0] == num_rows:
        return local_ctx
    g_ctx = attn_metadata.context_lens
    S = cp_kv_cache_interleave_size
    W = dcp_world_size
    full_chunks = g_ctx // (S * W)
    base = full_chunks * S
    remainder = (g_ctx - base * W - dcp_rank * S).clamp(0, S)
    return (base + remainder).to(torch.int32)


def dcp_merge_candidates(recv: torch.Tensor, out: torch.Tensor) -> None:
    """Merge the all-gathered candidates into the global top-k, in place.

    ``recv`` is the ``[W, 2, rows, k_loc]`` int32 view of the exchanged
    (score, gid) pairs; ``out`` is the ``[rows, topk]`` int32 destination.

    Everything after the collective lives here rather than inline in
    ``dcp_decode_candidate_exchange`` so that the unit tests drive the same code
    the model runs. They used to keep a hand-written mirror of this block, and
    the two silently diverged when the exchange was moved between modules.
    """
    from aiter import top_k_per_row_decode

    world, _, num_rows, k_loc = recv.shape
    n_cand = world * k_loc
    topk = out.shape[1]
    # Rank is the outer dim after all-gather but the merge wants it on the
    # candidate dim. Selection is provably order-independent, so any consistent
    # permutation works -- but scores and gids must use the SAME one. The gid
    # plane is read in place by dcp_gather_candidate_gids below, so only the
    # score plane is materialised.
    gathered_sc = recv[:, 0].permute(1, 0, 2).reshape(num_rows, n_cand).contiguous()
    cand_idx = torch.empty(num_rows, topk, dtype=torch.int32, device=recv.device)
    cand_lens = torch.full((num_rows,), n_cand, dtype=torch.int32, device=recv.device)
    top_k_per_row_decode(
        gathered_sc.view(torch.float32),
        1,
        cand_lens,
        cand_idx,
        num_rows,
        gathered_sc.stride(0),
        gathered_sc.stride(1),
        topk,
        stable=True,
    )
    # Map row-local candidate index -> global id, reading recv's gid plane in
    # place and writing straight into `out`.
    dcp_gather_candidate_gids(recv, cand_idx, out)


def dcp_decode_candidate_exchange(
    attn_metadata,
    padded_q_fp8_decode_tokens: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    topk_indices: torch.Tensor,
    dcp_rank: int,
    num_decode_tokens: int,
    topk_tokens: int,
    max_model_len: int,
    runner_block_size: int,
    stable_topk: bool,
    cp_kv_cache_interleave_size: int = 1,
) -> None:
    """DCP decode candidate exchange -> deterministic merge into global top-k.

    Each rank holds only 1/W of the KV (index_cache is sharded, same slot_mapping
    as the main KV). Score the LOCAL shard — block_tables already point at it, so
    only context_lens must be localized — take a LOCAL top-k, and all-gather just
    the W*topk (score, global_id) candidates. Because fewer tokens outrank a given
    token locally than globally, a token in the global top-K is in its own rank's
    local top-K, so the merge reconstructs the global top-K. Caveat: the local
    top-k is a score-only radix-select whose tie handling differs from the global
    gid-ordered tie-break, so when the local boundary (2048th) sits on a score tie
    a tied token may be dropped before the exchange -- the merged set can then
    differ from dcp=1 at that boundary. Exact fp32 score ties are rare, so this is
    a negligible boundary effect, not a systematic loss.

    Writes the merged global top-k in place into
    topk_indices[:num_decode_tokens, :topk_tokens].
    """
    # Imported here, not at module scope: this module must stay importable
    # without aiter so the CPU-only tests can exercise the merge kernels.
    from aiter import top_k_per_row_decode
    from aiter.ops.triton.pa_mqa_logits import deepgemm_fp8_paged_mqa_logits

    dcp_world_size = get_dcp_world_size()
    batch_size, next_n = padded_q_fp8_decode_tokens.shape[:2]
    num_rows = batch_size * next_n
    num_padded_tokens = num_rows
    assert attn_metadata.max_seqlen_q == 1, (
        "DCP + DeepSeek-V3.2 sparse indexer (DSA) currently supports "
        "qlen=1 decode only (MTP verify not yet supported)."
    )
    local_ctx = dcp_local_context_lens(
        attn_metadata, dcp_rank, dcp_world_size, cp_kv_cache_interleave_size, num_rows
    )
    l_max = (max_model_len + dcp_world_size - 1) // dcp_world_size
    local_logits = torch.empty([num_rows, l_max], dtype=torch.float32, device="cuda")
    deepgemm_fp8_paged_mqa_logits(
        padded_q_fp8_decode_tokens,
        kv_cache,
        weights[:num_padded_tokens],
        local_logits,
        local_ctx,
        attn_metadata.block_tables,
        l_max,
        KVBlockSize=runner_block_size,
        Preshuffle=True,
    )
    # ---- local top-k -> exchange candidates -> deterministic merge ----
    # k_loc is the constant `topk_tokens`, never the live local length: the
    # exchanged size must be static for CUDAGraph. Short contexts therefore ship
    # (-inf, -1) padding, which the merge drops.
    k_loc = topk_tokens
    local_idx = torch.empty(
        num_rows, k_loc, dtype=torch.int32, device=local_logits.device
    )
    top_k_per_row_decode(
        local_logits,
        next_n,
        local_ctx,
        local_idx,
        num_rows,
        local_logits.stride(0),
        local_logits.stride(1),
        k_loc,
        stable=stable_topk,
    )
    # [2, rows, k_loc]: plane 0 = score, plane 1 = int32 gid bits.
    send = torch.empty(
        2, num_rows, k_loc, dtype=torch.float32, device=local_logits.device
    )
    dcp_pack_topk_candidates(
        local_logits,
        local_idx,
        local_ctx,
        dcp_rank,
        dcp_world_size,
        send,
        cp_kv_cache_interleave_size,
    )
    # Exchange as int32 so the gid bit patterns cannot be touched by any float
    # canonicalization along the way; the score plane is bitcast back at the end
    # (free, no copy).
    recv = (
        get_dcp_group()
        .all_gather(send.view(torch.int32), dim=0)
        .view(dcp_world_size, 2, num_rows, k_loc)
    )
    dcp_merge_candidates(recv, topk_indices[:num_decode_tokens, :topk_tokens])


# ---------------------------------------------------------------------------
# DCP sparse index filter + round-robin localize (decode & prefill).
# Two-pass compacting filter: count this rank's owned top-k, then pack the
# owned slots to the front of each region (no -1 holes -- see cp_lse_ag_out_rs).
# ---------------------------------------------------------------------------


@triton.jit
def _count_owned_dcp_kernel(
    qo_indptr,  # int32 [num_requests + 1]
    global_kv_indptr,  # int32 [num_requests + 1] -- GLOBAL context (column range)
    token_indices_ptr,  # int32 [num_tokens, NUM_TOPK_TOKENS] -- GLOBAL top-k positions
    out_counts,  # int32 [num_requests] -- owned top-k count per request
    out_metadata_counts,  # int32 [num_requests] -- max(out_counts, 1)
    DCP_RANK: tl.constexpr,
    DCP_WORLD: tl.constexpr,
    INTERLEAVE: tl.constexpr,  # cp_kv_cache_interleave_size S (1 = round-robin)
    NUM_TOPK_TOKENS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ti_stride0,
    ti_stride1,
):
    """Pass 1 of the compacting DCP filter: how many of the global top-k
    positions does this rank own, per request? Its exclusive cumsum gives the
    compacted output offsets used by ``_compact_filter_dcp_kernel``.

    qlen==1 only (DCP + sparse + MTP is rejected upstream), so each request has
    exactly one query token. Owner of global position g is rank (g//S)%W
    (S=INTERLEAVE; S=1 -> g%W).
    """
    batch_id = tl.program_id(0)
    token_id = tl.load(qo_indptr + batch_id)

    count = 0
    for tile_start in range(0, NUM_TOPK_TOKENS, BLOCK_N):
        indice_id = tile_start + tl.arange(0, BLOCK_N)
        col_valid = indice_id < NUM_TOPK_TOKENS
        ti_ptr = token_indices_ptr + token_id * ti_stride0 + indice_id * ti_stride1
        tok = tl.load(ti_ptr, mask=col_valid, other=-1)
        owned = col_valid & (tok >= 0) & (((tok // INTERLEAVE) % DCP_WORLD) == DCP_RANK)
        count += tl.sum(owned.to(tl.int32))

    tl.store(out_counts + batch_id, count)
    tl.store(out_metadata_counts + batch_id, tl.maximum(count, 1))


@triton.jit
def _compact_filter_dcp_kernel(
    qo_indptr,  # int32 [num_requests + 1]
    global_kv_indptr,  # int32 [num_requests + 1] -- GLOBAL context (column range)
    out_kv_indptr,  # int32 [num_requests + 1] -- COMPACTED offsets (cumsum of pass 1)
    block_table,  # int32 [num_req, max_num_blocks_per_req] -- logical(global) blocks
    token_indices_ptr,  # int32 [num_tokens, NUM_TOPK_TOKENS] -- GLOBAL top-k positions
    out_kv_indices,  # int32 [>= out_kv_indptr[-1]]
    DCP_RANK: tl.constexpr,
    DCP_WORLD: tl.constexpr,
    INTERLEAVE: tl.constexpr,  # cp_kv_cache_interleave_size S (1 = round-robin)
    PAGE_SIZE: tl.constexpr,  # runner (physical) block size
    NUM_TOPK_TOKENS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ti_stride0,
    ti_stride1,
    bt_stride0: tl.int64,
    bt_stride1: tl.constexpr,
):
    # DCP interleave-S: a GLOBAL position g is owned by rank ``(g // S) % W``; on
    # the owner rank its physical slot follows the virtual-block layout used by
    # _dcp_round_robin_slot / ATOM PR #847 (S=1 -> the original round-robin):
    #     vbs  = PAGE_SIZE * W
    #     vb   = g % vbs
    #     slot = block_table[req, g // vbs] * PAGE_SIZE
    #            + (vb // (W*S)) * S + (vb % S)
    # token_indices holds GLOBAL positions (the indexer scored the full sequence
    # via all-gathered logits). This rank keeps ONLY the positions it owns and
    # writes them COMPACTED to the front of its region -- no -1 holes. Holes are
    # exactly what breaks aiter's lse path (immediate fault on the persistent
    # kernel, silently unwritten lse on the split-KV one).
    #
    # Compaction is order-preserving (tl.cumsum within a tile plus a running
    # offset across tiles) rather than atomic-allocated like vLLM, so the KV
    # order -- and hence the floating-point accumulation order -- is
    # deterministic run to run, which the dcp=1 vs dcp=N comparison relies on.
    #
    # NOTE: the slot is computed from block_table directly (like vLLM) rather
    # than gathered from a precomputed kv_indices -- the DCP round-robin
    # per-token slot array does not exist on the sparse path (dense reads go
    # through block_tables in-kernel).
    batch_id = tl.program_id(0)

    out_kv_start = tl.load(out_kv_indptr + batch_id)
    token_id = tl.load(qo_indptr + batch_id)

    vbs = PAGE_SIZE * DCP_WORLD
    written = 0
    for tile_start in range(0, NUM_TOPK_TOKENS, BLOCK_N):
        indice_id = tile_start + tl.arange(0, BLOCK_N)
        # Full top-k width; `tok >= 0` is the only valid-id guard. Must stay in
        # lock-step with `_count_owned_dcp_kernel` (see the note there on why the
        # old `indice_id < g_kv_len` mask is wrong) -- if the two disagree, the
        # counted offsets and the written entries diverge.
        col_valid = indice_id < NUM_TOPK_TOKENS

        ti_ptr = token_indices_ptr + token_id * ti_stride0 + indice_id * ti_stride1
        tok = tl.load(ti_ptr, mask=col_valid, other=-1)  # GLOBAL position

        idx_valid = (
            col_valid & (tok >= 0) & (((tok // INTERLEAVE) % DCP_WORLD) == DCP_RANK)
        )

        block_id = tok // vbs
        vb = tok % vbs
        inblock_offset = (vb // (DCP_WORLD * INTERLEAVE)) * INTERLEAVE + (
            vb % INTERLEAVE
        )
        physical_block = tl.load(
            block_table + batch_id * bt_stride0 + block_id * bt_stride1,
            mask=idx_valid,
            other=0,
        )
        slot = physical_block * PAGE_SIZE + inblock_offset

        # Exclusive prefix sum of the owned mask -> destination inside this tile.
        owned_i32 = idx_valid.to(tl.int32)
        dst = written + tl.cumsum(owned_i32, axis=0) - owned_i32
        tl.store(out_kv_indices + out_kv_start + dst, slot, mask=idx_valid)
        written += tl.sum(owned_i32)

    # Keep persistent fast-mode metadata valid for a rank that owns no selected
    # KV. The A2A pack kernel neutralizes this dummy row's LSE without adding a
    # separate pointwise launch.
    tl.store(out_kv_indices + out_kv_start, 0, mask=written == 0)


def triton_filter_and_convert_dcp_index(
    qo_indptr: torch.Tensor,  # int32 [num_requests + 1]
    global_kv_indptr: torch.Tensor,  # int32 [num_requests + 1]
    block_table: torch.Tensor,  # int32 [num_req, max_num_blocks_per_req] logical
    token_indices: torch.Tensor,  # int32 [num_tokens, NUM_TOPK_TOKENS] GLOBAL pos
    dcp_rank: int,
    dcp_world_size: int,
    block_size: int,  # runner (physical) block size == PAGE_SIZE
    out_kv_indptr: torch.Tensor,  # int32 [num_requests + 1] COMPACTED, written here
    owned_counts: torch.Tensor,  # int32 [>= num_requests] scratch for pass 1
    NUM_TOPK_TOKENS: int = 2048,
    BLOCK_N: int = 128,
    out: torch.Tensor | None = None,
    cp_kv_cache_interleave_size: int = 1,
):
    """DCP (interleave-S) filter + localize of global top-k positions,
    **compacting** each rank's owned slots to the front of its region.

    ``token_indices[token_id, indice_id]`` is a GLOBAL token position selected by
    the indexer (scored over the full sequence via all-gathered logits). This
    rank keeps a position ``g`` only if ``g % W == dcp_rank`` and maps it to its
    physical slot via the round-robin (virtual-block) layout, computed directly
    from ``block_table`` (like vLLM):
        vbs  = block_size * W
        slot = block_table[req, g // vbs] * block_size + (g % vbs) // W

    Non-owned positions are **dropped**, not marked: the kept slots are packed
    contiguously (original top-k order preserved) and ``out_kv_indptr`` is
    rewritten to the resulting per-request lengths. This replaces the earlier
    "fixed length + -1 sentinel" layout, whose holes broke aiter's lse output.
    Because the kept count depends on the per-layer top-k selection,
    ``out_kv_indptr`` is layer-dependent. Sparse+DCP persistent mode therefore
    rebuilds its work metadata after each full IndexShare layer and reuses that
    plan in the following shared layers.

    The 8 ranks' kept sets are disjoint and their union is exactly the global
    top-k, which is what makes the downstream ``cp_lse_ag_out_rs`` merge valid.
    """
    assert token_indices.dtype == torch.int32
    assert token_indices.shape[1] == NUM_TOPK_TOKENS
    assert NUM_TOPK_TOKENS % BLOCK_N == 0, (
        f"NUM_TOPK_TOKENS ({NUM_TOPK_TOKENS}) must be divisible by"
        f"BLOCK_N ({BLOCK_N})"
    )
    assert 0 <= dcp_rank < dcp_world_size
    assert out is not None, "sparse_kv_indices_buffer (out) is required"

    num_batch = global_kv_indptr.shape[0] - 1

    qo_indptr_c = qo_indptr.contiguous()
    global_kv_indptr_c = global_kv_indptr.contiguous()
    block_table_c = block_table.contiguous()
    token_indices_c = token_indices.contiguous()

    ti_stride0, ti_stride1 = token_indices_c.stride()
    bt_stride0, bt_stride1 = block_table_c.stride()
    grid = (num_batch,)

    # Pass 1: per-request count of owned top-k positions.
    counts = owned_counts[:num_batch]
    metadata_counts = out_kv_indptr[1 : num_batch + 1]
    _count_owned_dcp_kernel[grid](
        qo_indptr_c,
        global_kv_indptr_c,
        token_indices_c,
        counts,
        metadata_counts,
        dcp_rank,
        dcp_world_size,
        cp_kv_cache_interleave_size,
        NUM_TOPK_TOKENS,
        BLOCK_N,
        ti_stride0,
        ti_stride1,
    )

    # Exclusive cumsum -> compacted offsets. Written in place so the caller's
    # tensor (and anything already holding a view of it) sees the update.
    # dtype=int32 keeps the accumulation in int32 (torch would promote integral
    # cumsum to int64 by default, which the kernels' int32 pointers reject).
    # zero_() rather than `out_kv_indptr[0] = 0`: assigning a Python scalar goes
    # through a host->device copy, which HIP rejects while a graph is capturing
    # (hipErrorStreamCaptureUnsupported). Everything here must stay device-side.
    out_kv_indptr[:1].zero_()
    torch.cumsum(
        metadata_counts,
        dim=0,
        dtype=torch.int32,
        out=metadata_counts,
    )

    # Pass 2: write the owned slots packed to the front of each region.
    _compact_filter_dcp_kernel[grid](
        qo_indptr_c,
        global_kv_indptr_c,
        out_kv_indptr,
        block_table_c,
        token_indices_c,
        out,
        dcp_rank,
        dcp_world_size,
        cp_kv_cache_interleave_size,
        block_size,
        NUM_TOPK_TOKENS,
        BLOCK_N,
        ti_stride0,
        ti_stride1,
        bt_stride0,
        bt_stride1,
    )
    return out


@triton.jit
def _count_owned_dcp_prefill_kernel(
    dsa_kv_indptr,  # int32 [num_tokens + 1] -- GLOBAL per-token candidate counts
    token_to_seq_idxs,  # int32 [num_tokens]
    topk_indices,  # int32 [num_tokens, NUM_TOPK_TOKENS] -- FLAT KV indices
    cu_seqlens_k,  # int32 [num_req + 1] -- per-seq base of the flat KV axis
    out_counts,  # int32 [num_tokens]
    out_metadata_counts,  # int32 [num_tokens], max(out_counts, 1)
    DCP_RANK: tl.constexpr,
    DCP_WORLD: tl.constexpr,
    INTERLEAVE: tl.constexpr,  # cp_kv_cache_interleave_size S (1 = round-robin)
    NUM_TOPK_TOKENS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ti_stride0: tl.int64,
    ti_stride1: tl.constexpr,
):
    """Pass 1 of the compacting DCP filter for sparse PREFILL: how many of the
    global top-k candidates does this rank own, per QUERY TOKEN?

    Differs from the decode twin (`_count_owned_dcp_kernel`) in two ways:
    the row unit is a query token rather than a request (prefill has many query
    tokens per request), and `topk_indices` holds a FLAT index into the
    concatenated KV of all sequences, so the within-sequence position -- which
    is what the round-robin owner is derived from -- is `indice - cu_seqlens_k[req]`.
    """
    token_id = tl.program_id(0)
    req_id = tl.load(token_to_seq_idxs + token_id)
    base = tl.load(cu_seqlens_k + req_id)

    count = 0
    for tile_start in range(0, NUM_TOPK_TOKENS, BLOCK_N):
        col_id = tile_start + tl.arange(0, BLOCK_N)
        col_valid = col_id < NUM_TOPK_TOKENS
        indice = tl.load(
            topk_indices + token_id * ti_stride0 + col_id * ti_stride1,
            mask=col_valid,
            other=-1,
        )
        pos = indice - base  # position within the sequence
        owned = (
            col_valid & (indice >= 0) & (((pos // INTERLEAVE) % DCP_WORLD) == DCP_RANK)
        )
        count += tl.sum(owned.to(tl.int32))

    tl.store(out_counts + token_id, count)
    tl.store(out_metadata_counts + token_id, tl.maximum(count, 1))


@triton.jit
def _compact_filter_dcp_prefill_kernel(
    dsa_kv_indptr,  # int32 [num_tokens + 1] -- GLOBAL per-token candidate counts
    out_kv_indptr,  # int32 [num_tokens + 1] -- COMPACTED offsets (cumsum of pass 1)
    token_to_seq_idxs,  # int32 [num_tokens]
    topk_indices,  # int32 [num_tokens, NUM_TOPK_TOKENS] -- FLAT KV indices
    cu_seqlens_k,  # int32 [num_req + 1]
    block_table,  # int32 [num_req, max_num_blocks_per_req] -- logical(global) blocks
    out_kv_indices,  # int32 [>= out_kv_indptr[-1]]
    DCP_RANK: tl.constexpr,
    DCP_WORLD: tl.constexpr,
    INTERLEAVE: tl.constexpr,  # cp_kv_cache_interleave_size S (1 = round-robin)
    PAGE_SIZE: tl.constexpr,  # runner (physical) block size
    NUM_TOPK_TOKENS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ti_stride0: tl.int64,
    ti_stride1: tl.constexpr,
    bt_stride0: tl.int64,
    bt_stride1: tl.constexpr,
):
    """Pass 2 for sparse PREFILL: keep only this rank's owned candidates, map
    them through the interleave-S (virtual-block) layout, and pack them to the
    front of the token's region (S=1 -> the original round-robin):

        vbs  = PAGE_SIZE * W
        vb   = pos % vbs
        slot = block_table[req, pos // vbs] * PAGE_SIZE
               + (vb // (W*S)) * S + (vb % S)

    Same rationale as the decode twin: no `-1` holes (they break aiter's lse
    path) and compaction is order-preserving so the fp accumulation order is
    deterministic.
    """
    token_id = tl.program_id(0)
    req_id = tl.load(token_to_seq_idxs + token_id)
    base = tl.load(cu_seqlens_k + req_id)
    out_kv_start = tl.load(out_kv_indptr + token_id)

    vbs = PAGE_SIZE * DCP_WORLD
    written = 0
    for tile_start in range(0, NUM_TOPK_TOKENS, BLOCK_N):
        col_id = tile_start + tl.arange(0, BLOCK_N)
        # Full-width scan; `indice >= 0` is the only validity guard. Must stay in
        # lockstep with pass 1 -- if the two disagree on which candidates are
        # valid, `out_kv_indptr` (built from pass 1's counts) and the writes here
        # go out of sync and rows overwrite each other.
        col_valid = col_id < NUM_TOPK_TOKENS
        indice = tl.load(
            topk_indices + token_id * ti_stride0 + col_id * ti_stride1,
            mask=col_valid,
            other=-1,
        )
        pos = indice - base
        idx_valid = (
            col_valid & (indice >= 0) & (((pos // INTERLEAVE) % DCP_WORLD) == DCP_RANK)
        )

        block_id = pos // vbs
        vb = pos % vbs
        inblock_offset = (vb // (DCP_WORLD * INTERLEAVE)) * INTERLEAVE + (
            vb % INTERLEAVE
        )
        physical_block = tl.load(
            block_table + req_id * bt_stride0 + block_id * bt_stride1,
            mask=idx_valid,
            other=0,
        )
        slot = physical_block * PAGE_SIZE + inblock_offset

        owned_i32 = idx_valid.to(tl.int32)
        dst = written + tl.cumsum(owned_i32, axis=0) - owned_i32
        tl.store(out_kv_indices + out_kv_start + dst, slot, mask=idx_valid)
        written += tl.sum(owned_i32)

    # Persistent MLA's fast metadata builder does not handle zero-length rows:
    # one empty row can leave that row and several following short rows without
    # valid work descriptors. Reserve one slot for an empty row and point it at
    # a valid dummy cache entry. The attention caller uses the original
    # `owned_counts == 0` to replace this row with the exact softmax identity
    # (O=0, LSE=-inf), so the dummy never contributes to the DCP merge.
    tl.store(out_kv_indices + out_kv_start, 0, mask=written == 0)


def triton_filter_and_convert_dcp_index_prefill(
    dsa_kv_indptr: torch.Tensor,  # int32 [num_tokens + 1] GLOBAL counts
    token_to_seq_idxs: torch.Tensor,  # int32 [num_tokens]
    topk_indices: torch.Tensor,  # int32 [num_tokens, NUM_TOPK_TOKENS] FLAT indices
    cu_seqlens_k: torch.Tensor,  # int32 [num_req + 1]
    block_table: torch.Tensor,  # int32 [num_req, max_num_blocks_per_req] logical
    dcp_rank: int,
    dcp_world_size: int,
    block_size: int,  # runner (physical) block size == PAGE_SIZE
    out_kv_indptr: torch.Tensor,  # int32 [num_tokens + 1] COMPACTED, written here
    owned_counts: torch.Tensor,  # int32 [>= num_tokens] scratch for pass 1
    NUM_TOPK_TOKENS: int = 2048,
    BLOCK_N: int = 128,
    out: torch.Tensor | None = None,
    cp_kv_cache_interleave_size: int = 1,
):
    """Sparse-PREFILL twin of ``triton_filter_and_convert_dcp_index``.

    The decode version keys on requests (qlen==1); prefill has one row per query
    token and its ``topk_indices`` are flat KV indices rather than
    within-sequence positions. Everything else -- two passes, in-place int32
    cumsum, order-preserving compaction -- is identical, and the same
    layer-scoped buffers are reused (they are sized ``max_num_batched_tokens``,
    which bounds the prefill token count too).
    """
    assert topk_indices.dtype == torch.int32
    assert topk_indices.shape[1] == NUM_TOPK_TOKENS
    assert NUM_TOPK_TOKENS % BLOCK_N == 0, (
        f"NUM_TOPK_TOKENS ({NUM_TOPK_TOKENS}) must be divisible by"
        f"BLOCK_N ({BLOCK_N})"
    )
    assert 0 <= dcp_rank < dcp_world_size
    assert out is not None, "sparse_kv_indices_buffer (out) is required"

    num_tokens = dsa_kv_indptr.shape[0] - 1

    dsa_kv_indptr_c = dsa_kv_indptr.contiguous()
    token_to_seq_idxs_c = token_to_seq_idxs.contiguous()
    topk_indices_c = topk_indices.contiguous()
    cu_seqlens_k_c = cu_seqlens_k.contiguous()
    block_table_c = block_table.contiguous()

    ti_stride0, ti_stride1 = topk_indices_c.stride()
    bt_stride0, bt_stride1 = block_table_c.stride()
    grid = (num_tokens,)

    counts = owned_counts[:num_tokens]
    metadata_counts = out_kv_indptr[1 : num_tokens + 1]
    _count_owned_dcp_prefill_kernel[grid](
        dsa_kv_indptr_c,
        token_to_seq_idxs_c,
        topk_indices_c,
        cu_seqlens_k_c,
        counts,
        metadata_counts,
        dcp_rank,
        dcp_world_size,
        cp_kv_cache_interleave_size,
        NUM_TOPK_TOKENS,
        BLOCK_N,
        ti_stride0,
        ti_stride1,
    )

    # The count kernel already wrote max(count, 1) into the destination tail,
    # so this per-full-layer path needs neither an allocation nor another
    # pointwise operator. Exact input/output overlap is supported by cumsum.
    out_kv_indptr[:1].zero_()
    torch.cumsum(
        metadata_counts,
        dim=0,
        dtype=torch.int32,
        out=metadata_counts,
    )

    _compact_filter_dcp_prefill_kernel[grid](
        dsa_kv_indptr_c,
        out_kv_indptr,
        token_to_seq_idxs_c,
        topk_indices_c,
        cu_seqlens_k_c,
        block_table_c,
        out,
        dcp_rank,
        dcp_world_size,
        cp_kv_cache_interleave_size,
        block_size,
        NUM_TOPK_TOKENS,
        BLOCK_N,
        ti_stride0,
        ti_stride1,
        bt_stride0,
        bt_stride1,
    )
    return out
