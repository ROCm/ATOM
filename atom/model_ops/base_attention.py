# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

# from flash_attn import flash_attn_with_kvcache
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
# Overridable only to A/B the cap against aiter's FlyDSL decode (PR #4332),
# which has both the fixed PS reduce and a kernel that keeps improving past 32
# where gluon flattens. 32 remains the default and the shipping value: on
# production aiter neither half of the bound above has moved.
PA_DENSE_SPLIT_MAX = envs.ATOM_PA_DENSE_SPLIT_MAX


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
):
    """Run the AITER paged-attention ASM kernel with explicit metadata."""

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
    )


_FLYDSL_PA_MAX_PARTITIONS = 256
_FLYDSL_PA_TILE = 256
_flydsl_pa_routed: set[tuple] = set()


def _flydsl_pa_decode_num_seqs(
    *,
    q: torch.Tensor,
    k_cache: torch.Tensor,
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

    Mirrors the kernel's own validation so an unsupported call falls to gluon
    instead of raising from inside aiter. Every clause here is a hard reject
    there, not a preference.

    The returned count is the point of the function. FlyDSL indexes ``query``
    as ``[num_seqs, query_length, ...]`` and demands ``q.shape[0] ==
    context_lens.shape[0] * query_length`` exactly, while ATOM pads its two
    axes independently: ``context_lens`` is built to ``running_bs`` (the
    sequence axis, padded for graph identity) with ``[scheduled_bs:running_bs]``
    zeroed, and ``q`` to ``running_tokens`` (the row axis, padded for MoE).
    ``forward_context.Context`` says so outright -- "the ratio is not always
    max_seqlen_q". So the two disagree by a padding slot on a perfectly
    ordinary step, which is what raised

        ValueError: query.shape[0] (12) must equal
                    context_lengths.shape[0] * query_length (4 * 4)

    gluon absorbs this: it derives its own batch as ``q.shape[0] //
    query_length`` and the surplus rows, holding ``context_lens == 0``, do no
    work. Recovering that count here and slicing the per-sequence arguments to
    it hands FlyDSL the same rectangle, dropping exactly the zeroed tail.

    Deliberately NOT restricted to ``max_seqlen_q == 1``. The dense path is
    where FlyDSL's headroom over gluon lives (it splits past the 32 gluon's PS
    reduce caps at), it runs ``max_seqlen_q == num_spec + 1``, and FlyDSL tunes
    that shape specifically -- it has a query_length==4 MTP4 grid split. Only
    the sparse path is naturally ``max_seqlen_q == 1``, having already given
    every query token its own row, table and causal length.

    The cache-layout clause is structural too: the page-16 SHUFFLE cache is
    ``[nb, Hkv, head_dim // x, 16, x]`` with ``x = 16 // element_size``, which
    equals FlyDSL's required ``[nb, Hkv, head_dim // 16, block_size, 16]``
    exactly when the cache is 1 byte per element. A bf16 cache gives x == 8 and
    is rejected -- as is its bf16 ``compute_type``.
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
    if context_partition_size != _FLYDSL_PA_TILE:
        return None
    if not 1 <= max_context_partition_num <= _FLYDSL_PA_MAX_PARTITIONS:
        return None
    head_dim = q.shape[-1]
    if not (head_dim == 64 or (head_dim % 128 == 0 and head_dim <= 1024)):
        return None
    # The rectangle, recovered the way gluon recovers it. See the docstring.
    if max_seqlen_q < 1:
        return None
    num_seqs, remainder = divmod(q.shape[0], max_seqlen_q)
    if remainder or not 1 <= num_seqs <= context_lens.shape[0]:
        return None
    return num_seqs



_FLYDSL_PLAN_MAX_BATCH = 4096
_FLYDSL_PLANS: dict[tuple, object] = {}
_FLYDSL_PLAN_SCRATCH: dict[tuple, tuple] = {}


def _flydsl_work_plan(context_lens, num_kv_heads, max_partitions, query_length,
                      query_group_size, head_dim, out_dtype):
    """aiter #5546's GPU work plan plus its scratch, cached per shape.

    Two things must be allocated once and then only refreshed. The plan, because
    allocation is illegal inside graph capture and a captured graph bakes in the
    pointers -- rebuilding it per step would leave replay reading freed memory
    (the hazard #2227 hit with `n_valid_column_per_row`). And the scratch,
    because a planned call does NOT use the caller's static buffers: planned
    output is packed as [kv_heads, plan.capacity, query_rows(, D)] whereas the
    static API wants [num_seqs, kv_heads, partitions, query_rows(, D)]. Handing
    over the static ones raises

        ValueError: max_logits shape (2, 1, 64, 64) != (1, 128, 64)

    which is what this function existed in a broken form long enough to cause.

    Refreshing the plan is a GPU kernel with no device-to-host readback, so it
    is safe to capture -- that property is what makes the planner usable here.
    """
    from aiter.ops.flydsl.pa_decode import plan_pa_decode

    key = (
        int(context_lens.shape[0]),
        int(num_kv_heads),
        int(max_partitions),
        int(query_length),
        int(query_group_size),
        context_lens.device.index,
    )
    plan = plan_pa_decode(
        context_lens,
        num_kv_heads,
        max_partitions=max_partitions,
        query_length=query_length,
        plan=_FLYDSL_PLANS.get(key),
    )
    _FLYDSL_PLANS[key] = plan

    scratch = _FLYDSL_PLAN_SCRATCH.get(key)
    rows = query_length * query_group_size
    want = (num_kv_heads, int(plan.capacity), rows)
    if scratch is None or tuple(scratch[0].shape) != want:
        dev = context_lens.device
        scratch = (
            torch.empty(want, dtype=torch.float32, device=dev),
            torch.empty(want, dtype=torch.float32, device=dev),
            torch.empty(*want, head_dim, dtype=out_dtype, device=dev),
        )
        _FLYDSL_PLAN_SCRATCH[key] = scratch
    return plan, scratch


def run_pa_decode_gluon(
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
):
    """Run the AITER paged-attention decode kernel.

    ATOM_PA_FLYDSL=1 routes to aiter's FlyDSL implementation (aiter PR #4332)
    instead of the gluon one where FlyDSL's domain covers the call. MEASUREMENT
    SWITCH, not a shipping default: the two take the same arguments and compute
    the same thing, and the env exists to A/B them without two ATOM trees.

    The two are not interchangeable everywhere, and the split is structural
    rather than incidental -- see ``_flydsl_pa_decode_num_seqs``. Callers get
    gluon for anything outside FlyDSL's domain, so enabling the env never turns
    a working configuration into an exception.
    """
    flydsl_seqs = envs.ATOM_PA_FLYDSL and _flydsl_pa_decode_num_seqs(
        q=q,
        k_cache=k_cache,
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
    if envs.ATOM_PA_FLYDSL:
        # Report both routes, once per shape signature. A run where the env is
        # set but every call still lands on gluon is otherwise indistinguishable
        # from one where FlyDSL simply did not help.
        sig = (bool(flydsl_seqs), max_seqlen_q, q.shape[0], context_lens.shape[0])
        if sig not in _flydsl_pa_routed:
            _flydsl_pa_routed.add(sig)
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

        work_plan = None
        es, ml, tmp = exp_sums, max_logits, temporary_output
        # #5546's planner refuses batches past 4096, and M3's sparse call site
        # folds query tokens into num_seqs (`total_q * Hkv`), which reaches
        # 32768 on a prefill-as-decode step. Falling back to the static path
        # keeps those steps running; letting the planner raise killed a worker
        # ~90 s into the run while the server kept answering /metrics, so the
        # client sat in warmup for 66 minutes waiting on a reply that could
        # never come.
        if envs.ATOM_PA_FLYDSL_PLAN and flydsl_seqs <= _FLYDSL_PLAN_MAX_BATCH:
            nkv = k_cache.shape[1]
            work_plan, (ml, es, tmp) = _flydsl_work_plan(
                context_lens[:flydsl_seqs],
                nkv,
                max_context_partition_num,
                max_seqlen_q,
                q.shape[-2] // nkv,
                q.shape[-1],
                output.dtype,
            )

        # Slice off ATOM's sequence-axis padding so the rectangle FlyDSL
        # requires holds. Views, no copy: dim 0 is the outermost axis of each.
        n = flydsl_seqs
        return _flydsl_pa_decode(
            output,
            q,
            k_cache,
            v_cache,
            context_lens[:n],
            block_tables[:n],
            softmax_scale,
            max_seqlen_q,
            max_context_partition_num,
            context_partition_size,
            compute_type,
            q_scale,
            k_scale,
            v_scale,
            exp_sums=es if work_plan is not None else exp_sums[:n],
            max_logits=ml if work_plan is not None else max_logits[:n],
            temporary_output=tmp if work_plan is not None else temporary_output[:n],
            alibi_slopes=alibi_slopes,
            sinks=sinks,
            # FlyDSL spells "no sliding window" as 0; ATOM's gluon path uses -1.
            sliding_window=0,
            ps=ps,
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
