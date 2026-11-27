# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""V4.1 draft-block kernels: bidirectional attention, the prologue, the KV tail.

The block's own geometry -- where a draft row sits, what it may attend to, how
a ragged batch reaches the RoPE interface -- is `draft_block`, which is plain
torch and importable without Triton. `fused_draft_kv_tail` is the epilogue of
the *context* KV write: the target rows the drafter absorbs after every target
forward.
"""

from functools import cache

import torch
import triton
import triton.language as tl
from aiter.jit.utils.chip_info import get_gfx_runtime

from atom.model_ops.blockscale_kernels.quantization import FP8_DTYPE, fp8_groups
from atom.model_ops.deepseek_v41.draft_block import DraftSrcIndices
from atom.model_ops.deepseek_v41.mhc import (
    write_peer_stream_state,
    write_stream_state,
)
from atom.model_ops.sparse_attn_v4 import sparse_attn
from atom.model_ops.v4_kernels import sparse_attn_v4_paged_2src


def draft_attention(query, context_kv, draft_kv, sink, step, scale):
    """All draft rows attend to the whole draft block, plus valid target rows.

    With `step.src_indices` (`draft_src_indices`), the window and the block
    are the two sources of `sparse_attn_v4_paged_2src`, the sparse kernel the
    target's BF16 path uses: it reads both where they lie, so there is no
    concatenated copy. Without `step.src_indices`, joins the two by
    concatenating. Neither touches `context_kv`.
    """
    src = step.src_indices
    if src is None:
        keys = torch.cat((context_kv, draft_kv), dim=1)
        return sparse_attn(query, keys, sink, step.indices, scale)
    # `DraftAttention.forward` hands over a GEMM output, the gathered window
    # and the keys it has just written, all dense. Anything else is a caller
    # bug, not a reason to take the slower path without saying so.
    assert (
        query.is_contiguous()
        and context_kv.is_contiguous()
        and draft_kv.is_contiguous()
    ), "two-source draft attention needs a contiguous query, window and block"
    batch, width, heads, dim = query.shape
    output = sparse_attn_v4_paged_2src(
        query.view(batch * width, heads, dim),
        context_kv.view(-1, dim),
        src.context,
        src.context_ptr,
        draft_kv.view(-1, dim),
        src.block,
        src.block_ptr,
        sink,
        scale,
    )
    return output.view(batch, width, heads, dim)


@cache
def two_source_draft() -> bool:
    """Whether the BF16 draft stages read the window and the block as two
    sources.

    Only measured on gfx1250, and not bit-identical to the concatenating path
    (it can move draft acceptance), so every other arch keeps that path. The
    FP8 draft has no concatenating path and does not ask.
    """
    return get_gfx_runtime() == "gfx1250"


@triton.jit
def _history_slots(anchor, slot, ring, window):
    """The absolute position each ring slot holds for `anchor`, and whether a
    draft row may attend to it.

    Normalized because Triton's remainder carries the DIVIDEND's sign where
    torch's follows the divisor, and `anchor - slot` is negative over most of a
    partly-filled ring. Getting this wrong does not fail -- it points the draft
    at the wrong rows.
    """
    residue = (anchor - slot) % ring
    residue = tl.where(residue < 0, residue + ring, residue)
    context = anchor - residue
    valid = (
        (slot < ring)
        & (context >= 0)
        & (context <= anchor)
        & (context > anchor - window)
    )
    return context, valid


@triton.jit(do_not_specialize=["batch_size"])
def _draft_src_kernel(
    positions_ptr,  # [B] anchor position per request
    context_ptr,  # out: window rows, packed
    context_indptr_ptr,  # out: [B * width + 1]
    block_ptr,  # out: [B * width * width] block rows
    block_indptr_ptr,  # out: [B * width + 1]
    batch_size,
    ring,
    window,
    width,
    BLOCK_B: tl.constexpr,  # next_pow2(B)
    RING_BLOCK: tl.constexpr,  # next_pow2(ring)
    WIDTH_BLOCK: tl.constexpr,  # next_pow2(width)
):
    """One program per draft row: its window rows, its block rows, and both
    indptr entries.

    A request's slots hold positions `anchor - ring + 1 .. anchor`, one each,
    so exactly `min(anchor + 1, window, ring)` of them are valid (none for a
    negative anchor). Every row of a request shares that list, which makes the
    row's window offset `width * (rows before b) + t * count[b]`: a closed form
    over the anchors, recomputed per program, so the build is one launch with
    no count pass and no cumsum between kernels -- the same reason
    `DSparkIndexBuffers.build` is one launch.
    """
    row = tl.program_id(0)  # flat (request, draft position)
    request = row // width
    t = row % width

    everyone = tl.arange(0, BLOCK_B)
    live = everyone < batch_size
    anchors = tl.load(positions_ptr + everyone, mask=live, other=0).to(tl.int64)
    counts = tl.minimum(tl.minimum(anchors + 1, window), ring)
    counts = tl.where(live, tl.maximum(counts, 0), 0)
    before = tl.sum(tl.where(everyone < request, counts, 0))
    count = tl.sum(tl.where(everyone == request, counts, 0))
    start = before * width + t * count

    anchor = tl.load(positions_ptr + request).to(tl.int64)
    slot = tl.arange(0, RING_BLOCK)
    _, valid = _history_slots(anchor, slot, ring, window)
    at = start + tl.cumsum(valid.to(tl.int64), 0) - 1
    tl.store(context_ptr + at, (request * ring + slot).to(tl.int32), mask=valid)
    tl.store(context_indptr_ptr + row, start.to(tl.int32))

    # The whole block, for every row: its indptr is a ramp of `width`.
    steps = tl.arange(0, WIDTH_BLOCK)
    tl.store(
        block_ptr + row.to(tl.int64) * width + steps,
        (request * width + steps).to(tl.int32),
        mask=steps < width,
    )
    tl.store(block_indptr_ptr + row, (row * width).to(tl.int32))
    if row == 0:
        rows = batch_size * width
        tl.store(context_indptr_ptr + rows, (tl.sum(counts) * width).to(tl.int32))
        tl.store(block_indptr_ptr + rows, (rows * width).to(tl.int32))


class DraftSrcBuffers:
    """`draft_src_indices`' outputs, allocated once at the largest batch.

    The `DSparkIndexBuffers` idiom: sized at `max_num_seqs` requests of the
    full block width and only ever sliced, every slice prefix-stable in the
    batch, so a captured draft graph replays the build. Allocated on the first
    build, where the ring is known; a BF16 draft that keeps the concatenating
    path never allocates. The indptrs start zeroed: their first entry is
    always 0, so an empty batch's one-entry slice reads 0 with nothing built.
    """

    def __init__(self, max_batch: int, max_width: int):
        self.max_batch, self.max_width = max_batch, max_width
        self.ring = None
        self.context = self.block = self.indptr = None

    def claim(self, batch: int, width: int, ring: int, device) -> DraftSrcIndices:
        """This batch's slices, allocating the bundle on first use."""
        if batch > self.max_batch:
            raise ValueError(
                f"Draft source buffers hold {self.max_batch} requests, not {batch}; "
                "they are sized at max_num_seqs."
            )
        if width > self.max_width:
            raise ValueError(
                f"Draft source buffers hold blocks {self.max_width} wide, not "
                f"{width}; they are sized at the DSpark block width."
            )
        if self.context is None:
            rows = self.max_batch * self.max_width
            i32 = {"dtype": torch.int32, "device": device}
            self.context = torch.empty(rows * ring, **i32)
            self.block = torch.empty(rows * self.max_width, **i32)
            self.indptr = torch.zeros(2, rows + 1, **i32)
            self.ring = ring
        elif ring != self.ring:
            raise ValueError(f"Draft source buffers were cut for ring {self.ring}")
        rows = batch * width
        return DraftSrcIndices(
            self.context[: rows * ring],
            self.indptr[0, : rows + 1],
            self.block[: rows * width],
            self.indptr[1, : rows + 1],
        )


def draft_src_indices(
    positions, ring, window, width, buffers: DraftSrcBuffers, *, fp8: bool = False
):
    """`draft_step_indices`' index list as the two sources both draft
    attentions read, or None where the BF16 stages keep the concatenating path.

    The one builder for both drafts. `draft_attention_fp8` has only the
    two-source kernel, so with `fp8` the lists are always built. For the BF16
    draft this is where the path is chosen, by `two_source_draft()` alone: None
    off it, and nothing is allocated. Otherwise fills `buffers` in one launch
    with no host read; built once per step, every stage reads the same rows.
    With no rows nothing is launched, and the indptrs read 0.
    """
    if not (fp8 or two_source_draft()):
        return None
    batch = positions.shape[0]
    src = buffers.claim(batch, width, ring, positions.device)
    if batch:
        _draft_src_kernel[(batch * width,)](
            positions,
            src.context,
            src.context_ptr,
            src.block,
            src.block_ptr,
            batch,
            ring,
            window,
            width,
            BLOCK_B=triton.next_power_of_2(batch),
            RING_BLOCK=triton.next_power_of_2(ring),
            WIDTH_BLOCK=triton.next_power_of_2(width),
            num_warps=4,
        )
    return src


def draft_attention_fp8(query, context_kv, draft_kv, sink, step, scale):
    """OPUS A8W8 over MXFP8 query, history and the whole current draft block.

    Reads the window and the block through `step.src_indices`, which
    `draft_src_indices(..., fp8=True)` builds once per draft step.
    """
    from atom.model_ops.attentions.deepseek_v41.fp8_attention import (
        sparse_attn_v41_paged_2src,
    )

    src = step.src_indices
    assert src is not None, "FP8 draft attention needs the step's source lists"
    output = torch.empty_like(query[0], dtype=torch.bfloat16)
    sparse_attn_v41_paged_2src(
        tuple(p.flatten(0, 1) for p in query),
        tuple(p.flatten(0, 1) for p in context_kv),
        src.context,
        src.context_ptr,
        tuple(p.flatten(0, 1) for p in draft_kv),
        src.block,
        src.block_ptr,
        sink,
        scale,
        out=output.flatten(0, 1),
    )
    return output


# ---------------------------------------------------------------------------
# The stages' rolling windows, gathered in one coalesced pass.
#
# The stages read the same rows of the same tensor at different layers, so the
# read is `window[layers[:, None], slots[None, :]]` -- one broadcast index. Torch
# serves that through its generic advanced-indexing path, which walks the index
# arithmetic per element as an opaque 2-byte type: measured on MI355X at a decode
# batch it spends 13.2us moving 6.3MB, against the ~1.6us the bytes themselves
# cost at HBM speed.
#
# Nothing about the read needs that generality. A row is `head_dim` wide and
# contiguous at both ends, so one program per output row makes it a single
# coalesced load and store.
# ---------------------------------------------------------------------------


@triton.jit
def _gather_window_kernel(
    window_ptr,  # [L_all, S, R, D]
    layers_ptr,  # [L]
    slots_ptr,  # [B]
    out_ptr,  # [L, B, R, D]
    stride_layer,
    stride_slot,
    batch,
    span,  # ring * D, the contiguous run one (layer, slot) owns
    BLOCK: tl.constexpr,
):
    pair = tl.program_id(0)  # flat (stage, request)
    stage = pair // batch
    request = pair % batch
    layer = tl.load(layers_ptr + stage).to(tl.int64)
    slot = tl.load(slots_ptr + request).to(tl.int64)

    # A request's whole window is one contiguous run in both tensors -- the
    # ring is the last axis but one and `D` the last -- so the copy is a run
    # of `ring * D`, not `ring` separate rows. Moving it a row at a time left
    # each program with 1KB and the gather at a quarter of a plain copy's
    # speed.
    src = window_ptr + layer * stride_layer + slot * stride_slot
    dst = out_ptr + pair.to(tl.int64) * span
    for start in tl.range(0, span, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < span
        tl.store(dst + offs, tl.load(src + offs, mask=mask), mask=mask)


def gather_window_rows(window, layers, slots):
    """`window[layers[:, None], slots[None, :]]`, as a run copy per request.

    Args:
        window: ``[L_all, S, ring, D]`` the pool's window view.
        layers: ``[L]`` int64 layer ids, on device.
        slots: ``[B]`` int64 slot per request.

    Returns:
        ``[L, B, ring, D]``, contiguous.
    """
    ring, dim = window.shape[-2], window.shape[-1]
    stages, batch = layers.numel(), slots.numel()
    out = window.new_empty((stages, batch, ring, dim))
    if not (stages and batch and ring):
        return out
    if window.stride(-1) != 1 or window.stride(-2) != dim:
        raise ValueError("Window gather needs each request's window contiguous")
    _gather_window_kernel[(stages * batch,)](
        window,
        layers,
        slots,
        out,
        window.stride(0),
        window.stride(1),
        batch,
        ring * dim,
        BLOCK=4096,
        num_warps=8,
    )
    return out


@triton.jit
def _draft_step_kernel(
    positions_ptr,  # [B] anchor position per request
    out_positions_ptr,  # [B, width]
    out_indices_ptr,  # [B, width, ring + width] int32
    out_context_ptr,  # [B, ring] slot -> absolute position
    ring,
    window,
    width,
    WIDTH_BLOCK: tl.constexpr,  # next_pow2(width)
    ENTRIES: tl.constexpr,  # next_pow2(ring + width)
):
    batch = tl.program_id(0)
    row = tl.program_id(1)  # the draft position this index row belongs to
    anchor = tl.load(positions_ptr + batch).to(tl.int64)

    if row == 0:
        # Anchors locate the last processed target token, so the block starts
        # at +1. One row of the grid writes these; the rest only do indices.
        steps = tl.arange(0, WIDTH_BLOCK)
        tl.store(
            out_positions_ptr + batch.to(tl.int64) * width + steps,
            anchor + 1 + steps,
            mask=steps < width,
        )

    slot = tl.arange(0, ENTRIES)
    total = ring + width
    is_history = slot < ring

    # Shared with `_draft_src_kernel`, so the two index forms cannot disagree.
    context, valid = _history_slots(anchor, slot, ring, window)
    if row == 0:
        tl.store(
            out_context_ptr + batch.to(tl.int64) * ring + slot,
            context,
            mask=is_history,
        )

    index = tl.where(valid, slot, -1)
    # Past the ring the block's own rows index themselves.
    index = tl.where(is_history, index, slot)
    tl.store(
        out_indices_ptr + (batch.to(tl.int64) * width + row) * total + slot,
        index.to(tl.int32),
        mask=slot < total,
    )


def draft_step_indices(positions: torch.Tensor, ring: int, window: int, width: int):
    """The draft block's positions, attention indices and slot map, in one launch.

    `block_backbone` derived the ring-slot -> position map and `draft_step`
    turned it into the block's bidirectional index list. Between them that was
    ~16 torch launches over `[B, ring]` and `[B, width]` tensors -- at the
    batches a draft runs, every one of them costs the ~4us launch floor rather
    than any arithmetic, and four of them were `arange` rebuilding a constant.
    It is all a closed form over `(positions, ring, window, width)`.

    The precedent is `v4_kernels/dspark_fp8_indices.DSparkIndexBuffers.build`,
    which does this for V4 DSpark's FP8 path: one launch, shapes statically
    known, no `.item()` and no data-dependent allocation, so a captured graph
    replays it.

    The slot map is returned rather than kept internal: it is only an
    intermediate for the indices, but `block_backbone` publishes it to
    `draft_hidden` and a test pins that contract, and writing it is one store
    of a row the kernel already holds in registers.
    """
    batch = positions.shape[0]
    out_positions = torch.empty(
        batch, width, device=positions.device, dtype=positions.dtype
    )
    out_indices = torch.empty(
        batch, width, ring + width, device=positions.device, dtype=torch.int32
    )
    out_context = torch.empty(
        batch, ring, device=positions.device, dtype=positions.dtype
    )
    if batch:
        _draft_step_kernel[(batch, width)](
            positions,
            out_positions,
            out_indices,
            out_context,
            ring,
            window,
            width,
            WIDTH_BLOCK=triton.next_power_of_2(width),
            ENTRIES=triton.next_power_of_2(ring + width),
            num_warps=4,
        )
    return out_positions, out_indices, out_context


def build_block_state(anchor_embed, noise_embed, width, hc_mult):
    """The draft block's `(residual, pre_mix)`, from the anchors alone.

    `draft_hidden` used to build a `[B, width]` id block -- anchor in column
    zero, a noise token everywhere else -- embed all of it, broadcast the
    result over the `hc_mult` streams and zero a pre-mix beside it. Five
    launches, and an embedding that is a TP all-reduce over `B * width` rows of
    which only `B` differ: the noise row is the same every step, so it is
    embedded once and cached, and the collective shrinks by `width`.

    Returns what `SinglePassHCState.from_embeddings` returned, in one launch
    of the kernel that builds it (`write_stream_state`, anchor then noise):
    `residual [B, width, hc_mult, H]` and `pre_mix [B, width, hc_mult]`.
    """
    batch, hidden = anchor_embed.shape
    residual = torch.empty(
        batch,
        width,
        hc_mult,
        hidden,
        device=anchor_embed.device,
        dtype=anchor_embed.dtype,
    )
    pre_mix = torch.empty(
        batch, width, hc_mult, device=anchor_embed.device, dtype=torch.float32
    )
    rows = batch * width
    write_stream_state(
        residual.view(rows, hc_mult, hidden),
        pre_mix.view(rows, hc_mult),
        anchor_embed,
        fill=noise_embed,
        width=width,
        # The tile this launch was measured at for draft batches.
        block_h=1024,
    )
    return residual, pre_mix


def build_peer_block_state(peer, anchor_ids, noise_embed, width, hc_mult):
    """`build_block_state` with the anchors read from ``peer``'s vocab shards
    (`PeerEmbedding`) in the same launch: the anchors' lookup and its TP
    all-reduce are gone."""
    batch, hidden = anchor_ids.numel(), noise_embed.numel()
    residual = torch.empty(
        batch,
        width,
        hc_mult,
        hidden,
        device=noise_embed.device,
        dtype=noise_embed.dtype,
    )
    pre_mix = torch.empty(
        batch, width, hc_mult, device=noise_embed.device, dtype=torch.float32
    )
    rows = batch * width
    write_peer_stream_state(
        residual.view(rows, hc_mult, hidden),
        pre_mix.view(rows, hc_mult),
        peer,
        anchor_ids,
        noise_embed,
        width=width,
        block_h=1024,
    )
    return residual, pre_mix


_GROUP = 32  # quantize_fp8's group, fixed by the V4.1 QAT the cache stores


@triton.jit
def _draft_prologue_kernel(
    positions_ptr,  # [B] anchor position per request
    ids_ptr,  # [B] anchor token id per request
    embed_ptr,  # [V, H] the embedding table
    noise_ptr,  # [H] the noise token's row
    out_positions_ptr,  # [B, width]
    out_indices_ptr,  # [B, width, ring + width] int32
    out_context_ptr,  # [B, ring]
    residual_ptr,  # [B, width, M, H]
    pre_mix_ptr,  # [B, width, M] fp32
    hidden,
    ring,
    window,
    width,
    stride_embed_row,
    M: tl.constexpr,  # hc_mult
    BLOCK_H: tl.constexpr,
    H_BLOCKS: tl.constexpr,  # cdiv(hidden, BLOCK_H)
    WIDTH_BLOCK: tl.constexpr,  # next_pow2(width)
    ENTRIES: tl.constexpr,  # next_pow2(ring + width)
):
    """`_draft_step_kernel`, the anchors' embedding gather and
    `build_block_state` in one launch.

    Grid ``(B * width, H_BLOCKS + 1)``: program ``(row, part)`` with
    ``part < H_BLOCKS`` writes one H tile of one block row's state, as
    `build_block_state` does, reading the anchor's row straight out of the
    table; ``part == H_BLOCKS`` writes that row's index list, as
    `_draft_step_kernel` does. Copies and integer arithmetic only, so the five
    outputs are the three launches' own, bit for bit.
    """
    row = tl.program_id(0)  # flat (batch, draft position)
    part = tl.program_id(1)
    batch = row // width
    pos = row % width

    if part == H_BLOCKS:
        anchor = tl.load(positions_ptr + batch).to(tl.int64)
        if pos == 0:
            steps = tl.arange(0, WIDTH_BLOCK)
            tl.store(
                out_positions_ptr + batch.to(tl.int64) * width + steps,
                anchor + 1 + steps,
                mask=steps < width,
            )
        slot = tl.arange(0, ENTRIES)
        total = ring + width
        is_history = slot < ring
        context, valid = _history_slots(anchor, slot, ring, window)
        if pos == 0:
            tl.store(
                out_context_ptr + batch.to(tl.int64) * ring + slot,
                context,
                mask=is_history,
            )
        index = tl.where(valid, slot, -1)
        index = tl.where(is_history, index, slot)
        tl.store(
            out_indices_ptr + row.to(tl.int64) * total + slot,
            index.to(tl.int32),
            mask=slot < total,
        )
    else:
        offs = part * BLOCK_H + tl.arange(0, BLOCK_H)
        mask = offs < hidden
        token = tl.load(ids_ptr + batch).to(tl.int64)
        src = tl.where(pos == 0, embed_ptr + token * stride_embed_row, noise_ptr)
        value = tl.load(src + offs, mask=mask)
        dst = residual_ptr + row.to(tl.int64) * M * hidden
        for stream in tl.static_range(M):
            tl.store(dst + stream * hidden + offs, value, mask=mask)
        if part == 0:
            streams = tl.arange(0, M)
            tl.store(
                pre_mix_ptr + row.to(tl.int64) * M + streams,
                tl.where(streams == 0, 1.0, 0.0).to(tl.float32),
            )


def fused_draft_prologue(
    positions: torch.Tensor,
    anchor_ids: torch.Tensor,
    embed_weight: torch.Tensor,
    noise_embed: torch.Tensor,
    ring: int,
    window: int,
    width: int,
    hc_mult: int,
):
    """`draft_step_indices` and `build_block_state(embed(anchor_ids), ...)` in
    one launch instead of three.

    For an embedding that is a plain local table (no TP all-reduce): the kernel
    reads each anchor's row where it lies. ``anchor_ids`` must be in range, as
    `F.embedding` needs them (the proposer clamps the staged ids).

    Returns ``(step_positions, indices, context_positions, (residual, pre_mix))``
    -- what the two calls return.
    """
    batch = positions.shape[0]
    hidden = embed_weight.shape[1]
    device = positions.device
    out_positions = torch.empty(batch, width, device=device, dtype=positions.dtype)
    out_indices = torch.empty(
        batch, width, ring + width, device=device, dtype=torch.int32
    )
    out_context = torch.empty(batch, ring, device=device, dtype=positions.dtype)
    residual = torch.empty(
        batch, width, hc_mult, hidden, device=device, dtype=embed_weight.dtype
    )
    pre_mix = torch.empty(batch, width, hc_mult, device=device, dtype=torch.float32)
    if batch:
        block = 1024
        h_blocks = triton.cdiv(hidden, block)
        _draft_prologue_kernel[(batch * width, h_blocks + 1)](
            positions,
            anchor_ids,
            embed_weight,
            noise_embed,
            out_positions,
            out_indices,
            out_context,
            residual,
            pre_mix,
            hidden,
            ring,
            window,
            width,
            embed_weight.stride(0),
            M=hc_mult,
            BLOCK_H=block,
            H_BLOCKS=h_blocks,
            WIDTH_BLOCK=triton.next_power_of_2(width),
            ENTRIES=triton.next_power_of_2(ring + width),
            num_warps=4,
        )
    return out_positions, out_indices, out_context, (residual, pre_mix)


@triton.jit
def _fused_draft_kv_tail_kernel(
    kv_ptr,  # [W, L, D] fused-GEMM output, rows contiguous
    norm_weight_ptr,  # [L, D] the stages' kv_norm weights, stacked
    positions_ptr,  # [W]
    cos_ptr,  # [max_position, PE_DIM // 2]
    sin_ptr,  # [max_position, PE_DIM // 2]
    out_ptr,  # [L, W, D] fp8 codes, or QAT activations when DEQUANT
    scale_ptr,  # [L, W, D // 32] ue8m0 exponents; unused when DEQUANT
    window_ptrs,  # [layers, 2] each layer's ring (data, scale) planes
    window_layers,  # [L] a stage's layer in `window_ptrs`
    step_positions,  # [W] the following four: the step's, as `_write_planes`
    step_batches,
    step_cu,
    step_slots,
    cached_until,  # [requests] when CACHED
    width,  # W, a runtime value: it changes every step
    max_position,
    stride_kv_w,
    stride_kv_l,
    stride_cos_p,
    eps,
    D: tl.constexpr,
    PE_DIM: tl.constexpr,
    GROUPS: tl.constexpr,
    DEQUANT: tl.constexpr,
    WINDOW: tl.constexpr,
    CACHED: tl.constexpr,
    RING: tl.constexpr,
):
    # One program per output row. The grid is flat over (stage, token) in that
    # order, so `row` IS the stage-major output row: the transpose the cat used
    # to do is this indexing, and the store below needs no stride.
    row = tl.program_id(0)
    stage = row // width
    token = row % width
    if WINDOW:
        # `write_window_planes`' rule: a padding row and a row its ring
        # already holds, or will not keep, are written nowhere.
        batch = tl.load(step_batches + token)
        if batch < 0:
            return
        if token < tl.load(step_cu + batch + 1) - RING:
            return
        ring_pos = tl.load(step_positions + token)
        if CACHED:  # noqa: SIM102
            if ring_pos < tl.load(cached_until + batch):
                return
        layer = tl.load(window_layers + stage)
        values = tl.load(window_ptrs + layer * 2).to(tl.pointer_type(tl.uint8))
        scales = tl.load(window_ptrs + layer * 2 + 1).to(tl.pointer_type(tl.uint8))
        out_row = tl.load(step_slots + batch).to(tl.int64) * RING + ring_pos % RING
    else:
        values, scales, out_row = out_ptr, scale_ptr, row.to(tl.int64)

    offs = tl.arange(0, D)
    src = kv_ptr + token.to(tl.int64) * stride_kv_w + stage.to(tl.int64) * stride_kv_l
    stage_weight = norm_weight_ptr + stage.to(tl.int64) * D

    x = tl.load(src + offs).to(tl.float32)
    w = tl.load(stage_weight + offs).to(tl.float32)
    rstd = tl.rsqrt(tl.sum(x * x, axis=0) / D + eps)
    # The original model's rounding points: kv_norm writes bf16, RoPE reads
    # and writes bf16, the QAT quantizes that.
    y = (x * rstd * w).to(tl.bfloat16).to(tl.float32)

    # RoPE over the tail lanes only, GPT-J interleaved: (2i, 2i+1) share
    # frequency i and lane 2i takes its partner negated. `pe_lane` is negative
    # outside the tail and Triton's `%` keeps the sign of the DIVIDEND, so it is
    # `is_pe` -- not the parity -- that decides whether a lane rotates.
    pe_lane = offs - (D - PE_DIM)
    is_pe = pe_lane >= 0
    even = pe_lane % 2 == 0
    pair = tl.where(even, offs + 1, offs - 1)
    # A masked lane still forms its address, so clamp rather than trust the mask.
    pair = tl.where(is_pe, pair, offs)
    freq = tl.where(is_pe, pe_lane // 2, 0)

    # The partner is the *normed* value, so unlike the MLA context kernel this
    # cannot re-read it from memory. Recomputing it from the same x, w and rstd
    # is exact (identical fp32 expression) and keeps the kernel free of
    # cross-lane ops; under the mask it is 64 lanes off a line already resident.
    x_pair = tl.load(src + pair, mask=is_pe, other=0.0).to(tl.float32)
    w_pair = tl.load(stage_weight + pair, mask=is_pe, other=0.0).to(tl.float32)
    y_pair = (x_pair * rstd * w_pair).to(tl.bfloat16).to(tl.float32)

    pos = tl.load(positions_ptr + token).to(tl.int64)
    pos = tl.minimum(tl.maximum(pos, 0), max_position - 1)
    cos = tl.load(cos_ptr + pos * stride_cos_p + freq).to(tl.float32)
    sin = tl.load(sin_ptr + pos * stride_cos_p + freq).to(tl.float32)
    rot = tl.where(even, -y_pair, y_pair)
    y = tl.where(is_pe, (y * cos + rot * sin).to(tl.bfloat16).to(tl.float32), y)

    # quantize_fp8's group-32 ue8m0 ceil scale, through its own groups helper
    # so the two cannot drift. The reshape is free: `offs` already runs
    # group-major.
    q, code, scale = fp8_groups(tl.reshape(y, (GROUPS, 32)))

    group_offs = tl.arange(0, GROUPS)
    dst = values + out_row * D + group_offs[:, None] * 32 + tl.arange(0, 32)
    if DEQUANT:
        tl.store(dst, (q.to(tl.float32) * scale[:, None]).to(values.dtype.element_ty))
    else:
        tl.store(dst, q.to(values.dtype.element_ty, bitcast=True))
        tl.store(scales + out_row * GROUPS + group_offs, code.to(tl.uint8))


def fused_draft_kv_tail(
    kv: torch.Tensor,
    norm_weight: torch.Tensor,
    positions: torch.Tensor,
    cos_cache: torch.Tensor,
    sin_cache: torch.Tensor,
    eps: float,
    *,
    packed: bool,
    window=None,
    step=None,
):
    """Per-stage RMSNorm + RoPE + FP8 quantize, emitted stage-major, in one launch.

    The epilogue of `write_context_kv`. Once the stages' `wkv` weights are one
    GEMM, its tail is six kernels -- three RMSNorms, a cat, a RoPE, a quantize --
    each re-reading the D-wide row from HBM only to hand it to the next. At 4.5us
    of launch floor apiece that is ~39us of a 163us drafting step doing no
    arithmetic worth the traffic. The cat does not survive as a kernel at all:
    one program owns one (stage, token) row and writes it straight to its
    stage-major slot, which is the layout both window writers need since they
    address rows by width.

    It rounds where the original model does: the normed row and the rotated
    tail land in bf16 before the QAT, as the op chain's RMSNorm and RoPE
    outputs do. Only the fp32 reduction tree and `rsqrt` differ from the HIP
    kernel's, so a code moves only where that last-ulp difference crosses a
    bf16 rounding boundary (`tests/model_ops/test_fused_dspark_ctx_kv.py`).

    aiter has nothing to reuse here: its fused RMSNorm+quant kernels emit a
    per-token or block scale rather than the group-32 ue8m0 ceil scale V4.1's
    QAT is defined against, and none of them carry a rotation.

    Args:
        kv: ``[..., W, L, D]`` fused-``wkv`` output; leading dims flatten into
            ``W``. Rows must be contiguous -- a stage's D lanes are what one
            program reads.
        norm_weight: ``[L, D]`` the stages' ``kv_norm.weight``, stacked.
        positions: ``[W]`` absolute positions indexing ``cos_cache``; every
            stage of a token rotates by the same one, which is what
            ``RotaryEmbedding._rotate_cuda``'s ``positions.repeat(batch)``
            spells out for the per-op path.
        cos_cache, sin_cache: ``RotaryEmbedding``'s own buffers, so YaRN scaling
            and cache dtype come along unchanged instead of being recomputed.
        eps: the stages' shared ``kv_norm.eps``.
        packed: the cache layout. ``True`` returns ``(fp8 codes, ue8m0 scales)``
            for ``write_packed_window``; ``False`` returns QAT activations in
            ``kv.dtype``. This is ``quantize_fp8(..., dequantize=not packed)``.
        window: the cache's ``window_target`` for the stages' layers, with
            ``step``: each row goes straight to its layer's fp8 ring by
            ``write_window``'s rule instead of to a returned tensor (packed
            only).

    Returns:
        Stage-major ``[L, W, D]`` (and ``[L, W, D // 32]`` scales when packed),
        so that each ``keys[i : i + 1]`` is contiguous; None with ``window``.
    """
    stages, dim = norm_weight.shape
    kv = kv.reshape(-1, stages, dim)
    tokens = kv.shape[0]
    pe_dim = cos_cache.shape[-1] * 2
    if kv.stride(-1) != 1 or norm_weight.stride(-1) != 1:
        raise ValueError("Fused draft KV needs contiguous rows")
    # One position per token, checked rather than masked: a short `positions`
    # is a caller contract violation, and the kernel indexes it by token with
    # no bound of its own -- so the alternative to this line is a silent
    # out-of-bounds read that faults only once the shapes line up to make it.
    if positions.numel() < tokens:
        raise ValueError(
            f"Fused draft KV needs {tokens} positions, got {positions.numel()}"
        )
    if dim % _GROUP or dim & (dim - 1) or pe_dim > dim or pe_dim % 2:
        raise ValueError(f"Fused draft KV needs a power-of-two {dim} >= {pe_dim}")

    if window is not None and not packed:
        raise ValueError("Fused draft KV writes only fp8 windows")

    values = scales = None
    if window is None:
        values = torch.empty(
            (stages, tokens, dim),
            device=kv.device,
            dtype=FP8_DTYPE if packed else kv.dtype,
        )
        if packed:
            scales = torch.empty(
                (stages, tokens, dim // _GROUP),
                device=kv.device,
                dtype=torch.float8_e8m0fnu,
            )
    ptrs, layers, ring, cached = window or (None, None, 1, None)
    if tokens:
        _fused_draft_kv_tail_kernel[(stages * tokens,)](
            kv,
            norm_weight,
            positions,
            cos_cache,
            sin_cache,
            values,
            None if scales is None else scales.view(torch.uint8),
            ptrs,
            layers,
            None if window is None else step.positions,
            None if window is None else step.batch_ids,
            None if window is None else step.cu_seqlens_q,
            None if window is None else step.slots,
            cached,
            tokens,
            cos_cache.shape[0],
            kv.stride(0),
            kv.stride(1),
            cos_cache.stride(0),
            eps,
            D=dim,
            PE_DIM=pe_dim,
            GROUPS=dim // _GROUP,
            DEQUANT=not packed,
            WINDOW=window is not None,
            CACHED=cached is not None,
            RING=ring,
            # 4 waves over a 512-wide row leave each lane a handful of elements,
            # which is where the fp32 reduction stops dominating.
            num_warps=4,
            num_stages=2,
        )
    if window is not None:
        return None
    return (values, scales) if packed else values
