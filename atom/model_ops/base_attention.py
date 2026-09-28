# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

# from flash_attn import flash_attn_with_kvcache
import functools
import logging
from abc import ABC, abstractmethod

import torch
import triton
import triton.language as tl
from torch import nn

from atom.config import get_current_atom_config
from atom.utils import envs, mark_spliting_op
from atom.utils.selector import Family, get_attn_backend

from .attention_mla import MLAModules, _mla_output_width

logger = logging.getLogger("atom")


# frontend interface class for constructing attention
# op in model file
class Attention:
    def __new__(cls, *args, **kwargs):
        from atom.plugin.prepare import is_rtpllm, is_sglang, is_vllm

        if is_vllm():
            from atom.plugin.vllm.attention.layer import AttentionForVllm

            return AttentionForVllm(*args, **kwargs)
        if is_sglang():
            from atom.plugin.sglang.attention import AttentionForSGLang

            return AttentionForSGLang(*args, **kwargs)
        if is_rtpllm():
            from atom.plugin.rtpllm.attention_backend import AttentionForRTPLLM

            return AttentionForRTPLLM(*args, **kwargs)

        from atom.model_ops.paged_attention import Attention as AttentionForAtom

        return AttentionForAtom(*args, **kwargs)


# Envelopes of the two paged decode kernels wrapped below. Both are multiplied
# by drafting: MiniMax-M3 sits exactly on the gluon group one today (16 x 4 draft
# positions), a gqa=8 model reaches the gluon length one first, and ASM tops out
# lower than either -- past its limit get_heuristic_kernel silently re-runs with
# mtp=1, a kernel built for another query length.
PA_GLUON_MAX_QUERY_LEN = 4
PA_GLUON_MAX_QUERY_GROUP_SIZE = 64
PA_ASM_MAX_QUERY_GROUP_SIZE = 16

# Both are fits on a 256-CU gfx950 and do not scale with the machine, unlike the
# heuristic they bound. TARGET_WG: past it the extra splits only add reduce work.
# MAX is two separate bounds that happen to agree on a number no larger than 32:
# temporary_output is bf16, so each split adds a round trip through the PS
# combine, and the worst shape measured drifts 20pp further from an fp32
# reference at 64 than at 8; and 64 is where the C++ PS reduce stops being built
# at all, with no working fallback under it (see the test that pins this).
PA_DENSE_SPLIT_TARGET_WG = 128
# Not a knob. A planned call is told `plan.max_partitions` instead, so this
# only bounds the gluon fallback; raising it would just enlarge static scratch
# a planned call never reads.
PA_DENSE_SPLIT_MAX = 32


def dense_decode_splits(num_seqs: int, num_kv_heads: int) -> int:
    """KV splits for the dense paged decode.

    aiter's heuristic ends in a flat min(..., 8): at batch 1 it computes 512 and
    returns 8, leaving a call that reads the whole context on 3% of the machine.

    A function of the grid alone, deliberately not of the context length: decode
    runs under a cuda graph, where max_seqlen_k is the model limit rather than the
    real length, so a context term is inert in production and would only over-split
    short requests (+114% measured on a 2K one).

    Only the dense path is routed here. The two MiniMax-M3 sparse call sites are
    excluded on purpose -- their context is a fixed topk window and their num_seqs
    already folds the query tokens in -- as is the vLLM bridge's own copy of this
    dispatch, which is untested against this.
    """
    from aiter.ops.triton.gluon.pa_decode_gluon import get_recommended_splits

    n = max(1, int(num_seqs) * int(num_kv_heads))
    # Power of two, and only the term added here: the PS reduce compiles one
    # variant per distinct count, and a continuous cdiv adds 11 of them that no
    # sweep ever measured. Rounding the RESULT would drop below the heuristic
    # wherever it returns 3, 5, 6 or 7.
    boost = min(PA_DENSE_SPLIT_MAX, triton.cdiv(PA_DENSE_SPLIT_TARGET_WG, n))
    boost = 1 << (boost.bit_length() - 1)
    # The ceiling clamps the result as well. Staying at or above the heuristic is
    # a preference; staying inside what the reduce was built for is not.
    return min(
        PA_DENSE_SPLIT_MAX,
        max(get_recommended_splits(num_seqs, num_kv_heads), boost),
    )


def gluon_decode_over_limit(max_qlen: int, num_heads: int, num_kv_heads: int) -> bool:
    """Whether decode is past what the gluon kernel takes.

    pow2, not the raw product: that is what the kernel indexes its layout table
    with, and it rounds a small group up to fill 16.
    """
    max_qlen = max(1, int(max_qlen))
    qlen_p2 = 1 << (max_qlen - 1).bit_length()
    group_p2 = qlen_p2 * max(
        16 // qlen_p2, 1 << (num_heads // num_kv_heads - 1).bit_length()
    )
    return max_qlen > PA_GLUON_MAX_QUERY_LEN or group_p2 > PA_GLUON_MAX_QUERY_GROUP_SIZE


def run_pa_fwd_asm(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
    qo_indptr: torch.Tensor | None = None,
    max_qlen: int = 1,
    high_precision: int = 0,
    kernel_name: str | None = None,
):
    """Run the AITER paged-attention ASM kernel with explicit metadata.

    ``kernel_name`` bypasses aiter's kernel heuristic; a name absent from
    pa_asm.csv aborts the process rather than raising.
    """

    import aiter

    return aiter.pa_fwd_asm(
        Q=q,
        K=k_cache,
        V=v_cache,
        block_tables=block_tables,
        context_lens=context_lens,
        block_tables_stride0=block_tables.stride(0),
        max_qlen=max_qlen,
        K_QScale=k_scale,
        V_QScale=v_scale,
        out_=out,
        qo_indptr=qo_indptr,
        high_precision=high_precision,
        kernelName=kernel_name,
    )


# Copies of aiter's own limits, pinned against its source by a test. Mirroring
# them is what lets an unsupported call fall back instead of raising inside
# aiter; a copy that drifts either over-rejects or stops protecting.
_FLYDSL_PA_MAX_PARTITIONS = 256
_FLYDSL_PA_TILE = 256
_FLYDSL_PA_BLOCK_SIZES = (16, 64, 128)
_FLYDSL_PA_ARCHS = ("gfx942", "gfx950")
_flydsl_pa_routed: set[tuple] = set()
_flydsl_plan_refused: set[tuple] = set()


@functools.lru_cache(maxsize=1)
def _flydsl_arch_supported() -> bool:
    """Whether FlyDSL builds kernels for the running GPU.

    aiter raises NotImplementedError on anything else, from inside the call.
    """
    from aiter.jit.utils.chip_info import get_gfx_runtime

    return get_gfx_runtime() in _FLYDSL_PA_ARCHS


def _flydsl_pa_decode_num_seqs(
    *,
    output: torch.Tensor,
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    max_seqlen_q: int,
    max_context_partition_num: int,
    context_partition_size: int,
    compute_type: torch.dtype,
    q_scale: torch.Tensor | None,
    alibi_slopes: torch.Tensor | None,
    sinks: torch.Tensor | None,
    sliding_window: int,
    ps: bool,
) -> int | None:
    """Sequence count to run aiter's FlyDSL paged decode with, or None for gluon.

    Mirrors the kernel's own validation, so an unsupported call falls back to
    gluon instead of raising from inside aiter. Every clause below is a hard
    reject there, not a preference.

    Why a count rather than a bool: FlyDSL demands
    ``q.shape[0] == context_lens.shape[0] * query_length`` exactly, but ATOM
    pads its sequence axis (to running_bs, tail zeroed) and its row axis (to
    running_tokens) independently, so the two disagree by a padding slot on an
    ordinary step. gluon absorbs that; FlyDSL raises. Recovering the count the
    way gluon does and slicing the per-sequence arguments to it hands FlyDSL
    the same rectangle.

    Not restricted to ``max_seqlen_q == 1`` on purpose: dense is where FlyDSL's
    headroom over gluon lives, it runs ``num_spec + 1``, and FlyDSL tunes that
    shape (it has a query_length==4 MTP4 grid split).
    """
    import aiter

    if alibi_slopes is not None or sinks is not None or q_scale is not None:
        return None
    if sliding_window > 0 or not ps:
        return None
    if compute_type is not aiter.dtypes.fp8 or k_cache.dtype is not aiter.dtypes.fp8:
        return None
    if k_cache.dim() != 5 or k_cache.shape[-1] != 16:
        return None
    # block_size is dim -2 of the page-16 cache. --block-size 256/1024 reaches
    # this site whenever use_triton_attn is set, which the published recipe does.
    if k_cache.shape[-2] not in _FLYDSL_PA_BLOCK_SIZES:
        return None
    if not _flydsl_arch_supported():
        return None
    if context_partition_size != _FLYDSL_PA_TILE:
        return None
    if not 1 <= max_context_partition_num <= _FLYDSL_PA_MAX_PARTITIONS:
        return None
    head_dim = q.shape[-1]
    if not (head_dim == 64 or (head_dim % 128 == 0 and head_dim <= 1024)):
        return None
    # The cache encodes head_dim as num_hgroups * 16; aiter rejects a q whose
    # own head_dim disagrees, and nothing upstream forces the two to match.
    if k_cache.shape[2] * 16 != head_dim:
        return None
    if q.dtype not in (torch.bfloat16, torch.float16):
        return None
    if output.dtype is not q.dtype or output.shape != q.shape:
        return None
    # aiter checks both head_dim axes (pa_decode.py:443,448); output is
    # whatever the caller handed down, not necessarily contiguous.
    if q.stride(2) != 1 or output.stride(2) != 1:
        return None
    if q.shape[-2] % k_cache.shape[1] != 0:
        return None
    if v_cache.dtype is not k_cache.dtype:
        return None
    if block_tables.dtype is not torch.int32 or context_lens.dtype is not torch.int32:
        return None
    # aiter requires all four contiguous (pa_decode.py:464-470). No in-tree
    # path produces a non-contiguous one -- the cache views come from
    # `.view()`, which would raise first -- but the SGLang bridge's pool is
    # not this tree's to promise, and ATOM_PA_FLYDSL routes it too.
    if not (
        k_cache.is_contiguous()
        and v_cache.is_contiguous()
        and block_tables.is_contiguous()
        and context_lens.is_contiguous()
    ):
        return None
    # The rectangle, recovered the way gluon recovers it. See the docstring.
    if max_seqlen_q < 1:
        return None
    num_seqs, remainder = divmod(q.shape[0], max_seqlen_q)
    if remainder or not 1 <= num_seqs <= context_lens.shape[0]:
        return None
    if num_seqs > block_tables.shape[0]:
        return None
    return num_seqs


_FLYDSL_PLAN_MAX_BATCH = 4096
_FLYDSL_PLAN_SCRATCH: dict[tuple, tuple] = {}


def flydsl_plan_matches(plan, num_seqs: int, num_kv_heads: int) -> bool:
    """Whether a plan built elsewhere fits the call about to be made.

    The plan is built by the metadata builder for the batch it saw; a mismatch
    is aiter's `validate` raising, i.e. a dead worker. Checked here so the call
    can fall back to the static path instead.
    """
    return int(plan.reduce_info.shape[0]) == int(num_seqs) and int(
        plan.num_kv_heads
    ) == int(num_kv_heads)


def _flydsl_plan_scratch(
    plan, query_length, query_group_size, head_dim, out_dtype, device
):
    """Partial-output buffers for a planned call, allocated once per shape.

    Returned in the order the call site passes them: exp_sums, max_logits,
    temporary_output. The first two are interchangeable buffers (same shape,
    same dtype), which is exactly why a swapped unpacking would go unnoticed.

    Planned output is packed [kv_heads, capacity, rows(, D)] where the static
    API wants [num_seqs, kv_heads, partitions, rows(, D)]; passing the static
    ones raises a shape error. Keyed by capacity so a refresh that resized the
    plan gets its own buffers. Allocation stays here because the shapes come
    from the query tensor; only the per-step planner kernel moved out.
    """
    rows = query_length * query_group_size
    want = (int(plan.num_kv_heads), int(plan.capacity), rows)
    # Keyed by the plan itself, not only by shape: capacity is a constant under
    # the workgroup budget, so every capture rung -- and every concurrent
    # ubatch -- would otherwise share one triple of buffers and write it at the
    # same time. The plan is stored alongside so its id cannot be recycled.
    key = (id(plan), *want, head_dim, out_dtype, device.index)
    hit = _FLYDSL_PLAN_SCRATCH.get(key)
    hit = hit[1] if hit is not None else None
    if hit is None:
        hit = (
            torch.empty(want, dtype=torch.float32, device=device),
            torch.empty(want, dtype=torch.float32, device=device),
            torch.empty(*want, head_dim, dtype=out_dtype, device=device),
        )
        _FLYDSL_PLAN_SCRATCH[key] = (plan, hit)
    return hit


def run_pa_decode(
    output: torch.Tensor,
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    softmax_scale: float,
    max_seqlen_q: int,
    max_context_partition_num: int,
    context_partition_size: int,
    compute_type: torch.dtype,
    q_scale: torch.Tensor | None,
    k_scale: torch.Tensor | None,
    v_scale: torch.Tensor | None,
    *,
    exp_sums: torch.Tensor,
    max_logits: torch.Tensor,
    temporary_output: torch.Tensor,
    alibi_slopes: torch.Tensor | None = None,
    sinks: torch.Tensor | None = None,
    sliding_window: int = -1,
    ps: bool = True,
    work_plan=None,
):
    """Run the AITER paged-attention decode kernel.

    Named for what it does rather than for one of the two kernels it can pick:
    it dispatched only gluon until ``ATOM_PA_FLYDSL`` arrived, and four call
    sites -- including the vLLM and SGLang bridges -- import it.

    gluon unless ``ATOM_PA_FLYDSL=1``, and then only where FlyDSL's domain
    covers the call: ``_flydsl_pa_decode_num_seqs`` mirrors the kernel's own
    validation so an unsupported shape falls back here instead of raising
    inside aiter.
    """
    flydsl_seqs = envs.ATOM_PA_FLYDSL and _flydsl_pa_decode_num_seqs(
        output=output,
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        block_tables=block_tables,
        context_lens=context_lens,
        max_seqlen_q=max_seqlen_q,
        max_context_partition_num=max_context_partition_num,
        context_partition_size=context_partition_size,
        compute_type=compute_type,
        q_scale=q_scale,
        alibi_slopes=alibi_slopes,
        sinks=sinks,
        sliding_window=sliding_window,
        ps=ps,
    )
    # Inside the guard, not before it: this runs 63 times per decode step and
    # attention is a piecewise split op, so graph replay does not elide it. A
    # deployment that never enables FlyDSL should pay nothing here.
    #
    # Bounded on purpose: the row counts vary per step -- the sparse sites see a
    # new one on every prefill tail -- so keying on them leaks one entry and one
    # log line per distinct length for the life of the process.
    # `is not False` distinguishes the two falsy cases the `and` above produces:
    # False means the env is off (log nothing -- a deployment that never enables
    # this must not pay for it, and this runs 63x per step on a piecewise split
    # op that graph replay does not elide), None means the env is on and the
    # capability check rejected, which is exactly what the log exists to show.
    # The residual cost when off is one envs read; hoisting that to a module
    # constant would make the env unpatchable, which the tests rely on.
    if (
        flydsl_seqs is not False
        and (route_sig := (bool(flydsl_seqs), max_seqlen_q, q.shape[-1], compute_type))
        not in _flydsl_pa_routed
    ):
        _flydsl_pa_routed.add(route_sig)
        logger.info(
            "pa_decode -> %s (rows=%d max_seqlen_q=%d padded_seqs=%d "
            "head_dim=%d %s)",
            f"flydsl[{flydsl_seqs} seqs]" if flydsl_seqs else "gluon",
            q.shape[0],
            max_seqlen_q,
            context_lens.shape[0],
            q.shape[-1],
            compute_type,
        )

    if flydsl_seqs:
        from aiter.ops.flydsl.pa_decode import pa_decode as _flydsl_pa_decode

        n = flydsl_seqs
        # Handed in by the caller off the ForwardContext it already holds, not
        # re-read from the thread-local one: under TBO a worker thread that
        # never installed its own context would read whatever the other ubatch
        # wrote last, and the shape guard below compares only batch and kv-head
        # count -- two equal-sized ubatches pass. Which call sites pass a plan
        # is the boundary: the sparse sites and the vLLM/SGLang bridges do not.
        if work_plan is not None and not flydsl_plan_matches(
            work_plan, flydsl_seqs, k_cache.shape[1]
        ):
            # Static path: slower, not fatal. Logged because the planner
            # keeps refreshing a plan nothing reads, which looks exactly
            # like "the planner does not help" in an A/B.
            # Keyed on the plan's batch alone, which is a capture-ladder
            # rung and therefore bounded. The op's own count is not: the
            # case this warning exists for is a non-unified DP step, where
            # it tracks the real batch and would mint a new key every step.
            plan_n = int(work_plan.reduce_info.shape[0])
            if plan_n not in _flydsl_plan_refused:
                _flydsl_plan_refused.add(plan_n)
                logger.warning(
                    "flydsl work plan refused, falling back to the static "
                    "path: op wants %d seqs, plan was built for %d",
                    flydsl_seqs,
                    plan_n,
                )
            work_plan = None
        if work_plan is None:
            es, ml, tmp = exp_sums[:n], max_logits[:n], temporary_output[:n]
        else:
            nkv = k_cache.shape[1]
            es, ml, tmp = _flydsl_plan_scratch(
                work_plan,
                max_seqlen_q,
                q.shape[-2] // nkv,
                q.shape[-1],
                output.dtype,
                context_lens.device,
            )

        # Slice off ATOM's sequence-axis padding so the rectangle FlyDSL
        # requires holds. Views, no copy: dim 0 is the outermost axis of each.
        return _flydsl_pa_decode(
            output,
            q,
            k_cache,
            v_cache,
            context_lens[:n],
            block_tables[:n],
            softmax_scale,
            max_seqlen_q,
            # A planned call is told the plan's own ceiling -- the only value
            # `pa_decode` accepts, since it asserts the two are equal. The
            # static one gets the count `get_recommended_splits` sized the
            # scratch for.
            (
                max_context_partition_num
                if work_plan is None
                else int(work_plan.max_partitions)
            ),
            context_partition_size,
            compute_type,
            q_scale,
            k_scale,
            v_scale,
            exp_sums=es,
            max_logits=ml,
            temporary_output=tmp,
            alibi_slopes=alibi_slopes,
            sinks=sinks,
            sliding_window=0,
            work_plan=work_plan,
        )

    return torch.ops.aiter.pa_decode_gluon(
        output,
        q,
        k_cache,
        v_cache,
        context_lens,
        block_tables,
        softmax_scale,
        max_seqlen_q,
        max_context_partition_num,
        context_partition_size,
        compute_type,
        q_scale,
        k_scale,
        v_scale,
        exp_sums=exp_sums,
        max_logits=max_logits,
        temporary_output=temporary_output,
        alibi_slopes=alibi_slopes,
        sinks=sinks,
        sliding_window=sliding_window,
        ps=ps,
    )


# this triton kernel is used to fetch the stored kv in
# kv cache for computing the extend path(chunked prefill)
# and it can be used for both server mode and plugin mode
@triton.jit
def cp_mha_gather_cache_kernel(
    key_cache_ptr,  # [num_blocks, page_size, num_head, head_size]
    value_cache_ptr,  # [num_blocks, page_size, num_head, head_size]
    key_ptr,  # [num_tokens, num_heads, head_size]
    value_ptr,  # [num_tokens, num_heads, head_size]
    block_table_ptr,  # [num_batches, max_block_num]
    cu_seqlens_kv_ptr,  # [num_batches + 1]
    batch_id_per_k_token_ptr,  # [max_cum_tokens]
    seq_start_ptr,  # [num_batches]
    k_scale_ptr,  # [1] / [num_blocks, num_kv_heads, page_size]
    v_scale_ptr,
    k_cache_stride0,
    v_cache_stride0,
    num_heads,
    head_size,
    x,
    max_block_num,
    DEQUANT: tl.constexpr,
    PER_TOKEN_QUANT: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    CACHE_FORMAT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    token_id = tl.program_id(0)
    head_id = tl.program_id(1)
    # BLOCK_SIZE is rounded up to next pow2 at the call site (tl.arange requires
    # pow2); col_mask guards stores/loads when head_size is non-pow2 (e.g. MiMo SWA=192).
    col_offsets = tl.arange(0, BLOCK_SIZE)
    col_mask = col_offsets < head_size

    key_ptr_offset = key_ptr + token_id * head_size * num_heads + head_id * head_size
    value_ptr_offset = (
        value_ptr + token_id * head_size * num_heads + head_id * head_size
    )
    batch_idx = tl.load(batch_id_per_k_token_ptr + token_id)
    batch_start = tl.load(seq_start_ptr + batch_idx)
    token_start = tl.load(cu_seqlens_kv_ptr + batch_idx)
    batch_offset = token_id - token_start + batch_start
    block_offset = batch_offset // PAGE_SIZE
    block_id = tl.load(block_table_ptr + max_block_num * batch_idx + block_offset).to(
        tl.int64
    )
    slot_id = batch_offset % PAGE_SIZE

    if CACHE_FORMAT == "NHD":
        # for kv cache layout as
        # K: [num_blocks, page_size, num_head, head_dim]
        # V: [num_blocks, page_size, num_head, head_dim]
        key_cache_ptr_offset = (
            key_cache_ptr
            + block_id * k_cache_stride0
            + slot_id * num_heads * head_size
            + head_id * head_size
        )
        value_cache_ptr_offset = (
            value_cache_ptr
            + block_id * v_cache_stride0
            + slot_id * num_heads * head_size
            + head_id * head_size
        )
        k_reg = tl.load(key_cache_ptr_offset + col_offsets, mask=col_mask)
        v_reg = tl.load(value_cache_ptr_offset + col_offsets, mask=col_mask)
        if DEQUANT:
            if PER_TOKEN_QUANT:
                scale_offset = (
                    block_id * num_heads * PAGE_SIZE + head_id * PAGE_SIZE + slot_id
                )
                k_scale = tl.load(k_scale_ptr + scale_offset)
                v_scale = tl.load(v_scale_ptr + scale_offset)
            else:
                # per-tensor: one scale per ptr, no offset
                k_scale = tl.load(k_scale_ptr)
                v_scale = tl.load(v_scale_ptr)
            k_reg = k_reg.to(tl.float32) * k_scale
            v_reg = v_reg.to(tl.float32) * v_scale
        tl.store(key_ptr_offset + col_offsets, k_reg, mask=col_mask)
        tl.store(value_ptr_offset + col_offsets, v_reg, mask=col_mask)

    elif CACHE_FORMAT == "SHUFFLE":
        # for kv cache layout as
        # K: [num_blocks, num_head, head_dim // x, page_size, x]
        # V: [num_blocks, num_head, page_size // x, head_dim, x]
        key_cache_ptr_offset = (
            key_cache_ptr
            + block_id * k_cache_stride0
            + head_id * head_size * PAGE_SIZE
            + slot_id * x
        )
        value_cache_ptr_offset = (
            value_cache_ptr
            + block_id * v_cache_stride0
            + head_id * head_size * PAGE_SIZE
            + (slot_id // x) * head_size * x
            + slot_id % x
        )
        k_reg_offset = col_offsets // x * PAGE_SIZE * x + col_offsets % x
        v_reg_offset = col_offsets * x
        k_reg = tl.load(key_cache_ptr_offset + k_reg_offset, mask=col_mask)
        v_reg = tl.load(value_cache_ptr_offset + v_reg_offset, mask=col_mask)
        if DEQUANT:
            if PER_TOKEN_QUANT:
                scale_offset = (
                    block_id * num_heads * PAGE_SIZE + head_id * PAGE_SIZE + slot_id
                )
                k_scale = tl.load(k_scale_ptr + scale_offset)
                v_scale = tl.load(v_scale_ptr + scale_offset)
            else:
                # per-tensor: one scale per ptr, no offset
                k_scale = tl.load(k_scale_ptr)
                v_scale = tl.load(v_scale_ptr)
            k_reg = k_reg.to(tl.float32) * k_scale
            v_reg = v_reg.to(tl.float32) * v_scale
        tl.store(key_ptr_offset + col_offsets, k_reg, mask=col_mask)
        tl.store(value_ptr_offset + col_offsets, v_reg, mask=col_mask)


def cp_mha_gather_cache(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    block_tables: torch.Tensor,
    k_scales: torch.Tensor | None,
    v_scales: torch.Tensor | None,
    cu_seqlens_kv: torch.Tensor,
    batch_id_per_k_token: torch.Tensor,
    seq_starts: torch.Tensor,
    dequant: bool,
    kv_cache_layout: str,
    total_tokens: int,
    per_token_quant: bool = True,
):
    assert kv_cache_layout in [
        "NHD",
        "SHUFFLE",
    ], "kv_cache_layout only support NHD, SHUFFLE"
    if dequant:
        assert k_scales is not None and v_scales is not None
        if k_scales.numel() == 1 and v_scales.numel() == 1:
            per_token_quant = False
        else:
            assert (
                k_scales.numel() > 1 and v_scales.numel() > 1
            ), "k_scales and v_scales must both be scalar or per-token"

    head_dim = key.shape[2]
    x = 16 // key_cache.element_size()
    if kv_cache_layout == "NHD":
        # K: [num_blocks, page_size, num_heads, head_dim]
        assert head_dim == key_cache.shape[3]
        page_size = key_cache.shape[1]
        num_heads = key_cache.shape[2]
    else:
        # SHUFFLE: K [num_blocks, num_heads, head_dim//x, page_size, x]
        assert (
            key_cache.dim() == 5 and head_dim == key_cache.shape[2] * key_cache.shape[4]
        )
        page_size = key_cache.shape[3]
        num_heads = key_cache.shape[1]

    k_cache_stride0 = key_cache.stride(0)
    v_cache_stride0 = value_cache.stride(0)
    grid = lambda meta: (total_tokens, num_heads)
    cp_mha_gather_cache_kernel[grid](
        key_cache,
        value_cache,
        key,
        value,
        block_tables,
        cu_seqlens_kv,
        batch_id_per_k_token,
        seq_starts,
        k_scales,
        v_scales,
        k_cache_stride0,
        v_cache_stride0,
        num_heads,
        head_dim,
        x,
        block_tables.size(1),
        DEQUANT=dequant,
        PER_TOKEN_QUANT=per_token_quant,
        PAGE_SIZE=page_size,
        CACHE_FORMAT=kv_cache_layout,
        BLOCK_SIZE=triton.next_power_of_2(head_dim),
    )


def fake_(
    q: torch.Tensor,
    q_scale: torch.Tensor | None,
    k: torch.Tensor,
    v: torch.Tensor,
    positions: torch.Tensor,
    layer_name: str,
    use_mla: bool,
    qkv: torch.Tensor,
) -> torch.Tensor:
    output_shape = list(q.shape)
    # If we fusion rmsnorm and quant, the input dtype is fp8, but actually we use bf16 for output.
    atom_config = get_current_atom_config()
    if use_mla:
        bound = atom_config.compilation_config.static_forward_context[layer_name]
        impl = getattr(bound, "impl", bound)
        output_shape[-1] = _mla_output_width(impl, atom_config.hf_config.hidden_size)
    output_dtype = atom_config.torch_dtype
    output = torch.zeros(output_shape, dtype=output_dtype, device=q.device)

    return output


# Dynamo will not try to inspect any of the internal operations for prefill or decode
# This way, although attention operation is complicated,
# we can still capture the model's computation graph as a full-graph
@mark_spliting_op(is_custom=True, gen_fake=fake_, mutates_args=[])
def unified_attention_with_output_base(
    q: torch.Tensor,
    q_scale: torch.Tensor | None,
    k: torch.Tensor,
    v: torch.Tensor,
    positions: torch.Tensor,
    layer_name: str,
    use_mla: bool,
    qkv: torch.Tensor,
) -> torch.Tensor:
    atom_config = get_current_atom_config()
    self = atom_config.compilation_config.static_forward_context[layer_name]
    if use_mla:
        return self.impl.forward(
            query=q,
            k_nope=k,
            k_rope=v,
            positions=positions,
            q_scale=q_scale,
        )
    else:
        return self.impl.forward(
            query=q,
            key=k,
            value=v,
            position=positions,
            q_scale=q_scale,
            qkv=qkv,
        )


def linear_attention_with_output_base_fake(
    mixed_qkv: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    core_attn_out: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    return torch.empty_like(core_attn_out)


@mark_spliting_op(
    is_custom=True,
    gen_fake=linear_attention_with_output_base_fake,
    mutates_args=[],
)
def linear_attention_with_output_base(
    mixed_qkv: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    core_attn_out: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    atom_config = get_current_atom_config()
    self = atom_config.compilation_config.static_forward_context[layer_name]
    ret = torch.empty_like(core_attn_out)
    ret = self.impl.forward(mixed_qkv, b, a, ret, layer_name)
    return ret


class BaseAttention(nn.Module, ABC):
    """
    Abstract base class for attention

    This class defines the interface that all attention implementations must follow
    """

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
        kv_cache_dtype="bf16",
        layer_num=0,
        use_mla: bool = False,
        mla_modules: MLAModules | None = None,
        sinks: nn.Parameter | None = None,
        per_layer_sliding_window: int | None = None,
        rotary_emb: torch.nn.Module | None = None,
        prefix: str | None = None,
        **kwargs,
    ):
        super().__init__()

    @abstractmethod
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        positions: torch.Tensor | None = None,
        q_scale: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement the forward() method"
        )


class LinearAttention(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_v_heads,
        num_k_heads,
        head_k_dim,
        head_v_dim,
        key_dim,
        value_dim,
        dt_bias=None,
        A_log=None,
        conv1d=None,
        activation=None,
        layer_num=0,
        prefix: str | None = None,
        **kwargs,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_v_heads = num_v_heads
        self.num_k_heads = num_k_heads
        self.head_k_dim = head_k_dim
        self.head_v_dim = head_v_dim
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.dt_bias = dt_bias
        self.A_log = A_log
        self.conv1d = conv1d
        self.activation = activation
        self.layer_num = layer_num
        self.base_linear_attention = None
        self.prefix = prefix

        atom_config = get_current_atom_config()
        self.attn_backend = get_attn_backend(Family.GDN)
        impl_cls = self.attn_backend.get_impl_cls()
        self.impl = impl_cls(
            self.hidden_size,
            self.num_k_heads,
            self.num_v_heads,
            self.head_k_dim,
            self.head_v_dim,
            self.key_dim,
            self.value_dim,
            dt_bias,
            A_log,
            conv1d,
            activation,
            layer_num,
            **kwargs,
        )

        compilation_config = atom_config.compilation_config
        default_name = f"Linear_{layer_num}"
        self.layer_name = prefix if prefix is not None else default_name
        if self.layer_name in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer: {self.layer_name}")
        compilation_config.static_forward_context[self.layer_name] = self

    def forward(
        self,
        mixed_qkv: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        core_attn_out: torch.Tensor,
    ):
        output = torch.ops.aiter.linear_attention_with_output_base(
            mixed_qkv, b, a, core_attn_out, self.layer_name
        )
        return output
