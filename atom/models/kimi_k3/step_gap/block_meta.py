# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The DSpark block's paged metadata in one launch (``ATOM_KIMI_MONO_STEP_LEAN``).

Before every block pass ``DSparkProposer._build_paged_block_metadata`` derives,
from the requests' anchors and the block tables, the block's positions, its KV
write slots, each request's context length and their running sum: for the MLA
draft sharded over DCP, 24 torch elementwise launches of a few integers each
(about 55 us of kernels and 20 us of launch gaps a step on the TP8 profile).
``build_paged_block_metadata`` is that method for that configuration with those
24 launches as one Triton kernel (integer arithmetic: the same values); what
follows them -- the page-0 borrow for a padded tail, ``kv_indices_generate``,
``get_mla_metadata_v1`` and the fields of ``attn_metadata`` -- is the method's
own, statement for statement.

It is a mirror of shared code, so it is installed only on a drafter whose
method is still the one mirrored: ``install`` compares the source of the three
methods it mirrors against the digests recorded here and keeps the drafter's
own method on any difference (an upstream edit turns this off until the mirror
is brought up to date; ``tools/kimi_k3_step_gap_block_meta_check.py`` compares
the two on the GPU).
"""

from __future__ import annotations

import hashlib
import inspect
import logging
import types

import torch
import triton
import triton.language as tl

logger = logging.getLogger("atom")

# sha256[:16] of the mirrored methods' sources (decorator lines and whitespace
# removed): the tree this was written against, then the 1111 runtime image's
# proposer. The image's differs in three ways, none of them in what the covered
# draft computes: it takes the sharded branch on ``dcp_world_size > 1`` alone
# (no ``_draft_is_dcp_sharded``; the same for a draft without a pool of its
# own), it bounds a padded row by ``runner.block_size`` (what
# ``_draft_entry_tokens`` is for that draft), and it fills the padded tail
# after the clamp rather than before (a padded row's length is at most one
# block, which the clamp never lowers).
MIRRORED = {
    "_build_paged_block_metadata": ("153cff81b1235d0f", "93999b618a642e7f"),
    "_block_page_idx": ("e76fa3f1436f4beb",),
    "_paged_pad_ctx_len": ("768a3c03f5374a59", "5dfd009138c467ca"),
}


def source_digest(function) -> str:
    """``MIRRORED``'s digest of a function's source."""
    lines = inspect.getsource(function).splitlines(True)
    body = "".join(ln for ln in lines if not ln.lstrip().startswith("@"))
    return hashlib.sha256("".join(body.split()).encode()).hexdigest()[:16]


@triton.jit(
    do_not_specialize=[
        "scheduled_bs", "stride_bt", "cols", "T", "block_size", "dcp", "rank",
        "pad_ctx_len", "max_ctx",
    ]
)  # fmt: skip
def _paged_block_kernel(
    anchors_ptr,  # [running_bs] int64: each request's anchor position
    block_tables_ptr,  # [running_bs, cols] int32
    positions_ptr,  # out [running_bs, T] int64
    slots_ptr,  # out [running_bs, T] int64
    ctx_lens_ptr,  # out [running_bs] int32
    kv_indptr_ptr,  # out [running_bs + 1] int32; [0] is left alone
    scheduled_bs,
    stride_bt,
    cols,
    T,
    block_size,
    dcp,
    rank,
    pad_ctx_len,
    max_ctx,
    BLOCK_T: tl.constexpr,
    BLOCK_B: tl.constexpr,
):
    """One request a program: its T block positions and slots, its context
    length, and the running sum of the lengths up to it."""
    b = tl.program_id(0)
    anchor = tl.load(anchors_ptr + b)
    t = tl.arange(0, BLOCK_T)
    live = t < T
    # block_positions = anchor + (1 .. T)
    pos = anchor + 1 + t.to(tl.int64)
    tl.store(positions_ptr + b * T + t, pos, mask=live)
    # DCP shards KV token-wise round-robin: an entry covers block_size * dcp
    # global tokens, rank r holding the positions p with p % dcp == r
    virtual = block_size * dcp
    page = pos // virtual
    in_table = page < cols
    page = tl.minimum(page, cols - 1)
    local_off = (pos % virtual) // dcp
    owned = (pos % dcp) == rank
    entry = tl.load(
        block_tables_ptr + b.to(tl.int64) * stride_bt + page, mask=live, other=0
    )
    # (the page id times the block size in int32, as torch computes it)
    slot = (entry * block_size).to(tl.int64) + local_off
    slot = tl.where(in_table & owned & (b < scheduled_bs), slot, -1)
    tl.store(slots_ptr + b * T + t, slot, mask=live)
    # ctx_lens = ceil((anchor + 1 + T - rank) / dcp), a padded row's the
    # warmup row's, clamped to what the table addresses; kv_indptr their sum
    r = tl.arange(0, BLOCK_B)
    upto = r <= b
    others = tl.load(anchors_ptr + r, mask=upto, other=0)
    lens = ((others + (1 + T) + (dcp - 1 - rank)) // dcp).to(tl.int32)
    lens = tl.where(r >= scheduled_bs, pad_ctx_len, lens)
    lens = tl.minimum(lens, max_ctx)
    tl.store(ctx_lens_ptr + b, tl.sum(tl.where(r == b, lens, 0), axis=0))
    tl.store(kv_indptr_ptr + b + 1, tl.sum(tl.where(upto, lens, 0), axis=0))


def lean_refusal(drafter) -> str | None:
    """Why this drafter keeps its own ``_build_paged_block_metadata``, or
    None: the mirror covers the standalone MLA draft sharded over DCP, and
    only the method it was written against."""
    from atom.spec_decode.dspark_proposer import DSparkProposer

    if not isinstance(drafter, DSparkProposer):
        return f"a {type(drafter).__name__}, not a DSparkProposer"
    cls = type(drafter)
    try:
        methods = (
            ("_build_paged_block_metadata", cls._build_paged_block_metadata),
            ("_block_page_idx", cls._block_page_idx),
            ("_paged_pad_ctx_len", cls._paged_pad_ctx_len.fget),
        )
        changed = [
            name
            for name, function in methods
            if source_digest(function) not in MIRRORED[name]
        ]
    except Exception as error:  # noqa: BLE001 -- e.g. no source to read
        # a method that cannot be compared is not known to be the one mirrored
        return (
            f"{cls.__name__}'s methods cannot be compared with the mirror "
            f"({type(error).__name__}: {error})"
        )
    if changed:
        return f"{cls.__name__}.{changed[0]} is not the method mirrored"
    if not drafter._with_draft:
        return "the draft ships inside the target (no paged block)"
    if not drafter.model.block_attention_is_mla:
        return "a draft whose block does not attend through MLA"
    sharded = getattr(drafter, "_draft_is_dcp_sharded", None)
    if sharded is None:
        # a proposer whose method takes its sharded branch on DCP alone
        sharded = drafter.dcp_world_size > 1
    if not sharded:
        return "a draft that is not sharded over DCP"
    return None


def build_paged_block_metadata(
    self,
    forward_context,
    attn_metadata,
    anchor_positions: torch.Tensor,
    *,
    scheduled_bs: int,
    max_seqlen_k: int,
) -> None:
    """``DSparkProposer._build_paged_block_metadata`` for the MLA draft
    sharded over DCP (``self`` the drafter): the same buffers and the same
    ``attn_metadata``, its elementwise arithmetic in one launch."""
    from atom.model_ops.attentions.aiter_mla import get_mla_metadata_v1
    from atom.spec_decode.dspark_proposer import DSparkProposer
    from atom.utils.block_convert import kv_indices_generate_triton

    if (
        self._bound_draft_k_cache(forward_context) is None
        or anchor_positions.dtype != torch.int64
        or not anchor_positions.is_contiguous()
    ):
        # no pool yet (the warmup before the cache is allocated), or anchors
        # the kernel does not read: the method's own path
        return DSparkProposer._build_paged_block_metadata(
            self,
            forward_context,
            attn_metadata,
            anchor_positions,
            scheduled_bs=scheduled_bs,
            max_seqlen_k=max_seqlen_k,
        )
    T = self.draft_tokens_per_seq
    block_size = self.runner.block_size
    running_bs = anchor_positions.shape[0]
    attn_metadata.causal = False
    block_positions = self._blk_positions[:running_bs]
    dtype_q = self._blk_dtype_q
    if dtype_q is None:
        dtype_q, final = self._resolve_dtype_q(forward_context)
        if final:
            self._blk_dtype_q = dtype_q
    attn_metadata.dtype_q = dtype_q

    block_tables = self.runner.forward_vars["block_tables"].gpu[:running_bs]
    slots = self._blk_slots[: running_bs * T].view(running_bs, T)
    ctx_lens = self._blk_ctx_lens[:running_bs]
    kv_indptr = self._blk_kv_indptr[: running_bs + 1]
    _paged_block_kernel[(running_bs,)](
        anchor_positions,
        block_tables,
        block_positions,
        slots,
        ctx_lens,
        kv_indptr,
        scheduled_bs,
        block_tables.stride(0),
        block_tables.shape[1],
        T,
        block_size,
        self.dcp_world_size,
        self.dcp_rank,
        self._paged_pad_ctx_len,
        block_tables.shape[1] * block_size,
        BLOCK_T=triton.next_power_of_2(T),
        BLOCK_B=triton.next_power_of_2(self._blk_ctx_lens.shape[0]),
        num_warps=1,
    )

    attn_metadata.slot_mapping = slots.view(-1)
    attn_metadata.block_tables = block_tables
    attn_metadata.cu_seqlens_q = self._blk_cu_seqlens_q[: running_bs + 1]
    attn_metadata.context_lens = ctx_lens
    attn_metadata.max_seqlen_q = T
    attn_metadata.max_seqlen_k = max_seqlen_k
    index_max_k = (
        max_seqlen_k // self.dcp_world_size + 1
        if self.dcp_world_size > 1
        else max_seqlen_k
    )
    n_pad = running_bs - scheduled_bs
    if n_pad:
        borrowed = block_tables[scheduled_bs:, 0]
        saved_page0 = self._blk_saved_page0[:n_pad]
        saved_page0.copy_(borrowed)
        borrowed.zero_()
    try:
        kv_indices_generate_triton(
            block_tables,
            self._blk_kv_indices,
            kv_indptr,
            block_size,
            index_max_k,
        )
    finally:
        if n_pad:
            block_tables[scheduled_bs:, 0].copy_(saved_page0)
    attn_metadata.kv_indptr = kv_indptr
    attn_metadata.kv_indices = self._blk_kv_indices
    attn_metadata.kv_last_page_lens = self._blk_last_page_lens[:running_bs]
    dtype_kv = dtype_q
    ps = self._init_block_persistent_buffers(dtype_q, dtype_kv)
    get_mla_metadata_v1(
        attn_metadata.cu_seqlens_q,
        kv_indptr,
        attn_metadata.kv_last_page_lens,
        self._blk_padded_heads,
        1,
        False,
        ps["work_meta_data"],
        ps["work_info_set"],
        ps["work_indptr"],
        ps["reduce_indptr"],
        ps["reduce_final_map"],
        ps["reduce_partial_map"],
        page_size=block_size,
        kv_granularity=max(block_size, 16),
        max_seqlen_qo=T,
        uni_seqlen_qo=T,
        fast_mode=True,
        dtype_q=dtype_q,
        dtype_kv=dtype_kv,
        **self._blk_split_kwargs,
    )
    attn_metadata.work_meta_data = ps["work_meta_data"]
    attn_metadata.work_indptr = ps["work_indptr"]
    attn_metadata.work_info_set = ps["work_info_set"]
    attn_metadata.reduce_indptr = ps["reduce_indptr"]
    attn_metadata.reduce_final_map = ps["reduce_final_map"]
    attn_metadata.reduce_partial_map = ps["reduce_partial_map"]


def install(drafter) -> bool:
    """Bind the one-launch build on ``drafter`` where the mirror covers it;
    True when bound. Both of the drafter's callers reach the method through
    the instance, so the binding is all it takes. Never raises: any refusal,
    a mirrored method that changed or cannot be read included, leaves the
    drafter's own method serving."""
    try:
        why = lean_refusal(drafter)
    except Exception as error:  # noqa: BLE001 -- the installer must not fail
        why = f"{type(error).__name__}: {error}"
    if why is not None:
        logger.info(
            "Kimi-K3 step lean: the block metadata stays the drafter's (%s)", why
        )
        return False
    drafter._build_paged_block_metadata = types.MethodType(
        build_paged_block_metadata, drafter
    )
    logger.info("Kimi-K3 step lean: the block's paged metadata in one launch")
    return True
