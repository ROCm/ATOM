# SPDX-License-Identifier: MIT
"""The KV seam between the projections and attention, in one launch.

RoPE, FP8 quantization and the window row used to be three kernels whose
per-launch cost on MI355X was 5.5-6.8us each whatever shape they ran --
dispatch, not work. Q's rotation rides along because it is the one piece of
the seam that cannot move: it waits on `wq_b`, so it has to run here anyway,
and a program this far under the launch floor does not notice the extra rows.

The RMSNorm ahead of this stays where it is. It shares a launch with Q's, so
pulling its half in saves nothing and would leave Q's on a different kernel
than the one whose bytes the checkpoint was validated against.

The row this writes is `packed_rows`': `head_dim` FP8 values then
`head_dim // 32` E8M0 scales. Where it lands is the cache's, not this file's.
"""

import torch
import triton
import triton.language as tl

from atom.model_ops.blockscale_kernels.quantization import (
    AMAX_FLOOR,
    FP8_MAX,
    FP8_TL_DTYPE,
    ceil_pow2_code,
)
from atom.model_ops.v4_kernels.pool_index import window_row


@triton.jit
def _rotate_head(query, token, slot, position, cos, sin, HEADS, D, ROPE):
    """One query head's RoPE tail, in place. GPT-J pairs, as V4 rotates."""
    half: tl.constexpr = ROPE // 2
    pair = tl.arange(0, half)
    cosine = tl.load(cos + position * half + pair).to(tl.float32)
    sine = tl.load(sin + position * half + pair).to(tl.float32)
    head = query + (token * HEADS + slot) * D + (D - ROPE)
    tail = tl.arange(0, ROPE)
    even, odd = tl.split(tl.reshape(tl.load(head + tail).to(tl.float32), (half, 2)))
    tl.store(
        head + tail,
        tl.reshape(
            tl.join(even * cosine - odd * sine, even * sine + odd * cosine), (ROPE,)
        ).to(query.dtype.element_ty),
    )


@triton.jit
def _quantized_row(row, position, cos, sin, D, ROPE):
    """Rotate and quantize one row (a normed KV row or a query head); returns
    its bytes, E8M0 codes and power-of-two steps."""
    half: tl.constexpr = ROPE // 2
    dim = tl.arange(0, D)
    activation = row.dtype.element_ty
    normed = tl.load(row + dim).to(tl.float32)

    even, odd = tl.split(tl.reshape(normed, (D // 2, 2)))
    phase = tl.arange(0, D // 2) - (D - ROPE) // 2
    # A NoPE pair reads no table entry, and `other=` leaves it alone: cos 1
    # with sin 0 is the identity rotation.
    cosine = tl.load(cos + position * half + phase, phase >= 0, other=1.0).to(
        tl.float32
    )
    sine = tl.load(sin + position * half + phase, phase >= 0, other=0.0).to(tl.float32)
    # Back through the activation dtype first: the kernels this replaces each
    # returned BF16, so the amax has to see the rounded row.
    groups = (
        tl.reshape(
            tl.join(even * cosine - odd * sine, even * sine + odd * cosine),
            (D // 32, 32),
        )
        .to(activation)
        .to(tl.float32)
    )

    amax = tl.maximum(tl.max(tl.abs(groups), 1), AMAX_FLOOR)
    code = ceil_pow2_code(amax * (1.0 / FP8_MAX))
    step = (code << 23).to(tl.float32, bitcast=True)
    quantized = tl.minimum(tl.maximum(groups / step[:, None], -FP8_MAX), FP8_MAX).to(
        FP8_TL_DTYPE
    )
    return quantized, code, step


@triton.jit
def _rope_window_kernel(
    query,
    kv,
    cos,
    sin,
    positions,
    qat,
    values,
    scales,
    values_stride,
    scales_stride,
    query_values,
    query_scales,
    pool,
    batches,
    cu,
    slots,
    ring_start,
    kv_stride,
    HEADS: tl.constexpr,
    D: tl.constexpr,
    ROPE: tl.constexpr,
    RING_SLOTS: tl.constexpr,
    SLOT_ROWS: tl.constexpr,
    RING_STRIDE: tl.constexpr,
    RUN_ROWS: tl.constexpr,
    ROW_UNITS: tl.constexpr,
    EMIT_QAT: tl.constexpr,
    EMIT_PACKED: tl.constexpr,
    EMIT_QUERY_FP8: tl.constexpr,
    WRITE_RING: tl.constexpr,
    PACKED_RING: tl.constexpr,
):
    """One program per (token, slot); the last slot of a token is its KV.

    `HEADS` slots rotate a query head in place, slot `HEADS` runs the whole KV
    chain. Splitting the grid this way rather than one program per token is
    what keeps a decode bucket -- a few dozen rows -- wide enough to fill the
    device; the two branches are otherwise independent.
    """
    token = tl.program_id(0)
    slot = tl.program_id(1)
    position = tl.load(positions + token)
    dim = tl.arange(0, D)
    if slot < HEADS:
        if EMIT_QUERY_FP8:
            # The FP8 pool's attention reads the query as MXFP8, by the KV
            # row's rule; the BF16 head is not written.
            head = token * HEADS + slot
            quantized, code, _ = _quantized_row(
                query + head * D, position, cos, sin, D, ROPE
            )
            tl.store(query_values + head * D + dim, tl.reshape(quantized, (D,)))
            tl.store(query_scales + head * (D // 32) + tl.arange(0, D // 32), code)
        else:
            _rotate_head(query, token, slot, position, cos, sin, HEADS, D, ROPE)
    else:
        quantized, code, step = _quantized_row(
            kv + token * kv_stride, position, cos, sin, D, ROPE
        )

        if EMIT_QAT:
            tl.store(
                qat + token * D + dim,
                tl.reshape(quantized.to(tl.float32) * step[:, None], (D,)),
            )
        if EMIT_PACKED:
            tl.store(values + token * values_stride + dim, tl.reshape(quantized, (D,)))
            tl.store(scales + token * scales_stride + tl.arange(0, D // 32), code)
        if WRITE_RING:
            # Two exits, the ones both window writers already have: a padding
            # row owns no request, and a row the ring has scrolled past has
            # nowhere to land. The second is also what keeps two tokens of one
            # request off the same row -- it admits exactly the last
            # `RING_SLOTS` of them. The grid stays the forward's width either
            # way, because that is the width a captured replay runs.
            batch = tl.load(batches + token)
            if batch >= 0 and token >= tl.load(cu + batch + 1) - RING_SLOTS:
                base = (
                    window_row(
                        tl.load(slots + batch).to(tl.int64),
                        position,
                        ring_start,
                        RING_SLOTS,
                        SLOT_ROWS,
                        RING_STRIDE,
                        RUN_ROWS,
                    )
                    * ROW_UNITS
                )
                if PACKED_RING:
                    tl.store(
                        pool + base + dim,
                        tl.reshape(quantized, (D,)).to(tl.uint8, bitcast=True),
                    )
                    tl.store(pool + base + D + tl.arange(0, D // 32), code)
                else:
                    tl.store(
                        pool + base + dim,
                        tl.reshape(quantized.to(tl.float32) * step[:, None], (D,)),
                    )


def rope_quant_window(
    query,
    kv,
    cos,
    sin,
    positions,
    *,
    rope_dim,
    qat=None,
    values=None,
    scales=None,
    ring=None,
    query_fp8=None,
):
    """Rotate `query` in place and run the KV row to whichever sinks are given.

    `qat` takes the dequantized BF16 row an extend pass attends to, `values`
    and `scales` the FP8 pair (rows of any stride) a later window write or an
    FP8 attention consumes, and `ring` the `(pool, step, window)` a decode
    stores straight into. With none of them there is no KV half, and `kv` may
    be None. `query_fp8`: `(values, scales)` the rotated query is quantized
    into instead of being rotated in place.

    One address serves both ring layouts: `window_row` returns a byte offset
    for a byte pool (FP8 values then codes, `V41PoolGeometry.window_bytes`)
    and a row index for a `[rows, head_dim]` BF16 one.
    """
    tokens, heads, dim = query.shape
    kv_half = ring is not None or qat is not None or values is not None
    pool, step, window = ring or (None, None, None)
    byte_ring = pool is not None and pool.dtype == torch.uint8
    query_values, query_scales = query_fp8 or (None, None)
    _rope_window_kernel[(tokens, heads + kv_half)](
        query,
        kv if kv_half else query,
        cos,
        sin,
        positions,
        qat,
        values,
        None if scales is None else scales.view(torch.uint8),
        0 if values is None else values.stride(-2),
        0 if scales is None else scales.stride(-2),
        query_values,
        None if query_scales is None else query_scales.view(torch.uint8),
        pool,
        None if step is None else step.batch_ids,
        None if step is None else step.cu_seqlens_q,
        None if step is None else step.slots,
        0 if window is None else window.ring_start,
        kv.stride(0) if kv_half else 0,
        HEADS=heads,
        D=dim,
        ROPE=rope_dim,
        RING_SLOTS=1 if window is None else window.ring_slots,
        SLOT_ROWS=0 if window is None else window.slot_rows,
        RING_STRIDE=1 if window is None else window.ring_stride,
        RUN_ROWS=0 if window is None else window.run_rows,
        ROW_UNITS=1 if pool is None or byte_ring else dim,
        PACKED_RING=byte_ring,
        EMIT_QAT=qat is not None,
        EMIT_PACKED=values is not None,
        EMIT_QUERY_FP8=query_fp8 is not None,
        WRITE_RING=ring is not None,
    )
