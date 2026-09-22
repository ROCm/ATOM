# SPDX-License-Identifier: MIT
"""V4.1 draft-block kernels: bidirectional attention, the prologue, the KV tail.

The block's own geometry -- where a draft row sits, what it may attend to, how
a ragged batch reaches the RoPE interface -- is `draft_block`, which is plain
torch and importable without Triton. `fused_draft_kv_tail` is the epilogue of
the *context* KV write: the target rows the drafter absorbs after every target
forward.
"""

import torch
import triton
import triton.language as tl

from atom.model_ops.blockscale_kernels.quantization import _ceil_pow2_code
from atom.model_ops.sparse_attn_v4 import sparse_attn


def draft_attention(query, context_kv, draft_kv, sink, step, scale):
    """All draft rows attend to the whole draft block, plus valid target rows.

    Joins the two by concatenating and does not touch `context_kv`. A caller
    whose window already reserves room for the block writes into that tail and
    calls `sparse_attn` itself -- see `DraftAttention.forward` -- which is what
    keeps this one free of that assumption.
    """
    keys = torch.cat((context_kv, draft_kv), dim=1)
    return sparse_attn(query, keys, sink, step.indices, scale)


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

    # The absolute position a ring slot currently holds. Normalized because
    # Triton's remainder carries the DIVIDEND's sign where torch's follows the
    # divisor, and `anchor - slot` is negative over most of a partly-filled
    # ring. Getting this wrong does not fail -- it points the draft at the
    # wrong rows.
    residue = (anchor - slot) % ring
    residue = tl.where(residue < 0, residue + ring, residue)
    context = anchor - residue
    if row == 0:
        tl.store(
            out_context_ptr + batch.to(tl.int64) * ring + slot,
            context,
            mask=is_history,
        )

    valid = (
        is_history & (context >= 0) & (context <= anchor) & (context > anchor - window)
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


@triton.jit
def _block_state_kernel(
    anchor_ptr,  # [B, H] the anchor token's embedding
    noise_ptr,  # [H] the noise token's, identical for every step
    residual_ptr,  # [B, T, M, H]
    pre_mix_ptr,  # [B, T, M] fp32
    hidden,
    width,
    M: tl.constexpr,  # hc_mult
    BLOCK_H: tl.constexpr,
):
    row = tl.program_id(0)  # flat (batch, draft position)
    batch = row // width
    pos = row % width
    offs = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    mask = offs < hidden

    # Position 0 carries the anchor; the rest carry the same noise row, which
    # is why only the anchors are embedded per step.
    src = tl.where(pos == 0, anchor_ptr + batch.to(tl.int64) * hidden, noise_ptr)
    value = tl.load(src + offs, mask=mask)

    # The mHC state starts as `hc_mult` copies of the embedding. Writing them
    # here is the same bytes the `expand().contiguous()` wrote, minus reading
    # the row back out of HBM to do it.
    dst = residual_ptr + row.to(tl.int64) * M * hidden
    for stream in tl.static_range(M):
        tl.store(dst + stream * hidden + offs, value, mask=mask)

    if tl.program_id(1) == 0:
        streams = tl.arange(0, M)
        tl.store(
            pre_mix_ptr + row.to(tl.int64) * M + streams,
            tl.where(streams == 0, 1.0, 0.0).to(tl.float32),
        )


def build_block_state(anchor_embed, noise_embed, width, hc_mult):
    """The draft block's `(residual, pre_mix)`, from the anchors alone.

    `draft_hidden` used to build a `[B, width]` id block -- anchor in column
    zero, a noise token everywhere else -- embed all of it, broadcast the
    result over the `hc_mult` streams and zero a pre-mix beside it. Five
    launches, and an embedding that is a TP all-reduce over `B * width` rows of
    which only `B` differ: the noise row is the same every step, so it is
    embedded once and cached, and the collective shrinks by `width`.

    Returns what `SinglePassHCState.from_embeddings` returned, in one launch:
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
    if rows:
        block = 1024
        _block_state_kernel[(rows, triton.cdiv(hidden, block))](
            anchor_embed,
            noise_embed,
            residual,
            pre_mix,
            hidden,
            width,
            M=hc_mult,
            BLOCK_H=block,
            num_warps=4,
        )
    return residual, pre_mix


_GROUP = 32  # quantize_fp8's group, fixed by the V4.1 QAT the cache stores


@triton.jit
def _fused_draft_kv_tail_kernel(
    kv_ptr,  # [W, L, D] fused-GEMM output, rows contiguous
    norm_weight_ptr,  # [L, D] the stages' kv_norm weights, stacked
    positions_ptr,  # [W]
    cos_ptr,  # [max_position, PE_DIM // 2]
    sin_ptr,  # [max_position, PE_DIM // 2]
    out_ptr,  # [L, W, D] fp8 codes, or QAT activations when DEQUANT
    scale_ptr,  # [L, W, D // 32] ue8m0 exponents; unused when DEQUANT
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
):
    # One program per output row. The grid is flat over (stage, token) in that
    # order, so `row` IS the stage-major output row: the transpose the cat used
    # to do is this indexing, and the store below needs no stride.
    row = tl.program_id(0)
    stage = row // width
    token = row % width

    offs = tl.arange(0, D)
    src = kv_ptr + token.to(tl.int64) * stride_kv_w + stage.to(tl.int64) * stride_kv_l
    stage_weight = norm_weight_ptr + stage.to(tl.int64) * D

    x = tl.load(src + offs).to(tl.float32)
    w = tl.load(stage_weight + offs).to(tl.float32)
    rstd = tl.rsqrt(tl.sum(x * x, axis=0) / D + eps)
    y = x * rstd * w

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
    y_pair = x_pair * rstd * w_pair

    pos = tl.load(positions_ptr + token).to(tl.int64)
    pos = tl.minimum(tl.maximum(pos, 0), max_position - 1)
    cos = tl.load(cos_ptr + pos * stride_cos_p + freq).to(tl.float32)
    sin = tl.load(sin_ptr + pos * stride_cos_p + freq).to(tl.float32)
    rot = tl.where(even, -y_pair, y_pair)
    y = tl.where(is_pe, y * cos + rot * sin, y)

    # quantize_fp8's group-32 ue8m0 ceil scale, sharing its exponent helper so
    # the two cannot drift. The reshape is free: `offs` already runs group-major.
    groups = tl.reshape(y, (GROUPS, 32))
    amax = tl.maximum(tl.max(tl.abs(groups), 1), 1e-4)
    code = _ceil_pow2_code(amax * (1.0 / 448.0))
    scale = (code << 23).to(tl.float32, bitcast=True)
    q = tl.minimum(tl.maximum(groups / scale[:, None], -448.0), 448.0).to(tl.float8e4nv)

    group_offs = tl.arange(0, GROUPS)
    dst = out_ptr + row.to(tl.int64) * D + group_offs[:, None] * 32 + tl.arange(0, 32)
    if DEQUANT:
        tl.store(dst, (q.to(tl.float32) * scale[:, None]).to(out_ptr.dtype.element_ty))
    else:
        tl.store(dst, q)
        tl.store(scale_ptr + row.to(tl.int64) * GROUPS + group_offs, code.to(tl.uint8))


def fused_draft_kv_tail(
    kv: torch.Tensor,
    norm_weight: torch.Tensor,
    positions: torch.Tensor,
    cos_cache: torch.Tensor,
    sin_cache: torch.Tensor,
    eps: float,
    *,
    packed: bool,
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

    Numerics are not the op chain's and cannot be from Triton -- the fp32
    reduction tree and `rsqrt` differ from the HIP kernel's -- and this path
    also rounds less, carrying fp32 where the chain lands in bf16 three times.
    So ~3.4% of E4M3 codes move, essentially all of them toward an fp64
    reference: over 7.3M elements, 7 come out nearer under the chain and all 7
    straddle a code boundary. Per quantization group the fused value is never
    the further one, which is what `tests/model_ops/test_fused_dspark_ctx_kv.py`
    asserts.

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

    Returns:
        Stage-major ``[L, W, D]`` (and ``[L, W, D // 32]`` scales when packed),
        so that each ``keys[i : i + 1]`` is contiguous.
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

    values = torch.empty(
        (stages, tokens, dim),
        device=kv.device,
        dtype=torch.float8_e4m3fn if packed else kv.dtype,
    )
    scales = (
        torch.empty(
            (stages, tokens, dim // _GROUP),
            device=kv.device,
            dtype=torch.float8_e8m0fnu,
        )
        if packed
        else None
    )
    if tokens:
        _fused_draft_kv_tail_kernel[(stages * tokens,)](
            kv,
            norm_weight,
            positions,
            cos_cache,
            sin_cache,
            values,
            None if scales is None else scales.view(torch.uint8),
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
            # 4 waves over a 512-wide row leave each lane a handful of elements,
            # which is where the fp32 reduction stops dominating.
            num_warps=4,
            num_stages=2,
        )
    return (values, scales) if packed else values
