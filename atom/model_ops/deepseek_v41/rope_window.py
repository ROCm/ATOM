# SPDX-License-Identifier: MIT
"""The KV seam between the projections and attention, in one launch.

RoPE, FP8 quantization and the window row used to be three kernels whose
per-launch cost on MI355X was 5.5-6.8us each whatever shape they ran --
dispatch, not work. Q's rotation rides along because it is the one piece of
the seam that cannot move: it waits on `wq_b`, so it has to run here anyway,
and a program this far under the launch floor does not notice the extra rows.

The RMSNorm ahead of this stays where it is. It shares a launch with Q's, so
pulling its half in saves nothing and would leave Q's on a different kernel
than the one whose bytes the checkpoint was validated against. The exception
is fused attention, which rotates Q itself: this launch is then the KV half
alone, and `norm_rope_quant_window` runs it inside the dual RMSNorm instead,
Q's half kept op for op with AITER's `_fused_dual_rmsnorm_mxfp8_quant_kernel`.

The row this writes is `packed_rows`': `head_dim` FP8 values then
`head_dim // 32` E8M0 scales. Where it lands is the cache's, not this file's.
"""

import functools

import torch
import triton
import triton.language as tl
from aiter.ops.triton._triton_kernels.quant.quant import _mxfp8_quant_op

from atom.model_ops.blockscale_kernels.quantization import fp8_groups
from atom.model_ops.v4_kernels.pool_index import window_row

# A token's query heads rotated by one program rather than one each. The
# per-head programs each move one 64-element tail, so a launch of hundreds of
# tokens is nearly all program scheduling; on MI455X the whole launch took
# 2.00 -> 1.51 us at 32 tokens and 16.2 -> 2.2 us at 768, never slower from 1
# token up, with the same bytes out. False only for the tests' per-head
# reference.
_HEAD_BLOCK = True

# The MXFP8 query's heads quantized by one program rather than one each, with
# the same bytes out. 8 heads on 2 warps, 64 heads at 1344 / 16384 tokens:
# MI455X 31.8 -> 16.2 / 422 -> 205 us (c1024 serving +4.1%), MI355X 22.4 ->
# 21.6 / 322 -> 287 us. 1 only for the tests' per-head reference.
_FP8_HEADS = 8
# A program a head below this many grouped query programs a CU: measured on
# MI455X and MI355X (256 CUs each), not derived. 64/32/16 heads cross over at
# about this on MI455X and between 1x and 1.5x it on MI355X (at 1x up to 3%,
# 0.1 us, slower); 8 heads below it on both. Other CU counts are unmeasured.
_FP8_HEADS_MIN_PROGRAMS_PER_CU = 2


@functools.lru_cache
def _cu_count(device):
    return torch.cuda.get_device_properties(device).multi_processor_count


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
def _rotate_heads(query, token, position, cos, sin, HEADS, D, ROPE):
    """Every query head's RoPE tail of one token, in place: `_rotate_head`
    over a `[HEADS, ROPE]` tile, the cos / sin row loaded once."""
    half: tl.constexpr = ROPE // 2
    pair = tl.arange(0, half)
    cosine = tl.load(cos + position * half + pair).to(tl.float32)
    sine = tl.load(sin + position * half + pair).to(tl.float32)
    heads = tl.arange(0, HEADS)
    tail = tl.arange(0, ROPE)
    ptr = query + (token * HEADS + heads[:, None]) * D + (D - ROPE) + tail[None, :]
    even, odd = tl.split(tl.reshape(tl.load(ptr).to(tl.float32), (HEADS, half, 2)))
    tl.store(
        ptr,
        tl.reshape(
            tl.join(
                even * cosine[None, :] - odd * sine[None, :],
                even * sine[None, :] + odd * cosine[None, :],
            ),
            (HEADS, ROPE),
        ).to(query.dtype.element_ty),
    )


@triton.jit
def _quantized_row(row, position, cos, sin, D, ROPE):
    """Rotate and quantize one row (a normed KV row or a query head); returns
    its bytes, E8M0 codes and power-of-two steps."""
    dim = tl.arange(0, D)
    normed = tl.load(row + dim).to(tl.float32)
    return _quantized_values(normed, row.dtype.element_ty, position, cos, sin, D, ROPE)


@triton.jit
def _quantized_values(normed, activation, position, cos, sin, D, ROPE):
    """`_quantized_row` of a row already in registers, as fp32 that
    `activation` holds exactly."""
    half: tl.constexpr = ROPE // 2
    even, odd = tl.split(tl.reshape(normed, (D // 2, 2)))
    # Every pair reads the table row, unmasked so the reads stay contiguous;
    # a NoPE pair then takes cos 1 with sin 0, the identity rotation.
    # `pair % half` is a RoPE pair's table entry only when the NoPE pairs
    # fill whole table rows.
    tl.static_assert((D - ROPE) % ROPE == 0)
    pair = tl.arange(0, D // 2)
    rope_pair = pair >= (D - ROPE) // 2
    cosine = tl.where(
        rope_pair, tl.load(cos + position * half + pair % half).to(tl.float32), 1.0
    )
    sine = tl.where(
        rope_pair, tl.load(sin + position * half + pair % half).to(tl.float32), 0.0
    )
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
    return fp8_groups(groups)


@triton.jit
def _quantize_heads(
    query, query_values, query_scales, row0, position, cos, sin, HB, D, ROPE
):
    """`_quantized_row` over query heads `row0 .. row0 + HB` as one `[HB, D]`
    tile, stored as MXFP8: the same arithmetic per element, so the same bytes."""
    half: tl.constexpr = ROPE // 2
    rows = (row0 + tl.arange(0, HB))[:, None]
    dim = tl.arange(0, D)[None, :]
    activation = query.dtype.element_ty
    normed = tl.load(query + rows * D + dim).to(tl.float32)
    even, odd = tl.split(tl.reshape(normed, (HB, D // 2, 2)))
    tl.static_assert((D - ROPE) % ROPE == 0)
    pair = tl.arange(0, D // 2)
    rope_pair = pair >= (D - ROPE) // 2
    cosine = tl.where(
        rope_pair, tl.load(cos + position * half + pair % half).to(tl.float32), 1.0
    )[None, :]
    sine = tl.where(
        rope_pair, tl.load(sin + position * half + pair % half).to(tl.float32), 0.0
    )[None, :]
    groups = (
        tl.reshape(
            tl.join(even * cosine - odd * sine, even * sine + odd * cosine),
            (HB * D // 32, 32),
        )
        .to(activation)
        .to(tl.float32)
    )
    quantized, code, _ = fp8_groups(groups)
    tl.store(query_values + rows * D + dim, tl.reshape(quantized, (HB, D)))
    tl.store(
        query_scales + rows * (D // 32) + tl.arange(0, D // 32)[None, :],
        tl.reshape(code, (HB, D // 32)),
    )


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
    tokens,
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
    HEAD_BLOCK: tl.constexpr,
    FP8_HEADS: tl.constexpr,
    QUERY_PROGRAMS: tl.constexpr,  # a token's query slots
):
    """One program per (token, slot); slot `QUERY_PROGRAMS` of a token is its KV.

    With `HEAD_BLOCK` (the in-place BF16 query) slot 0 rotates all of a
    token's heads and slot 1 runs the whole KV chain: one query program per
    token. Otherwise `QUERY_PROGRAMS` slots rotate or quantize the query
    heads and slot `QUERY_PROGRAMS` is the KV. Either way the KV
    chain has its own program rather than sharing the token's, which keeps a
    decode bucket -- a few dozen rows -- wide enough to fill the device; the
    two branches are otherwise independent. `FP8_HEADS` > 1: each MXFP8 query
    slot quantizes that many consecutive heads.

    One grid axis (the token count may pass the second's limit): every
    query program, then every KV program, each run in memory order, so
    neighbouring programs read and write neighbouring rows.
    """
    # `HEAD_BLOCK` is the in-place BF16 rotation only; an MXFP8-emitting query
    # groups its heads with `FP8_HEADS` instead.
    tl.static_assert(not (HEAD_BLOCK and EMIT_QUERY_FP8))
    pid = tl.program_id(0)
    query_rows = tokens * QUERY_PROGRAMS
    is_query = pid < query_rows
    token = tl.where(is_query, pid // max(QUERY_PROGRAMS, 1), pid - query_rows)
    position = tl.load(positions + token)
    dim = tl.arange(0, D)
    if is_query:
        slot = pid % max(QUERY_PROGRAMS, 1)
        if HEAD_BLOCK:
            _rotate_heads(query, token, position, cos, sin, HEADS, D, ROPE)
        elif EMIT_QUERY_FP8 and FP8_HEADS > 1:
            _quantize_heads(
                query,
                query_values,
                query_scales,
                token * HEADS + slot * FP8_HEADS,
                position,
                cos,
                sin,
                FP8_HEADS,
                D,
                ROPE,
            )
        elif EMIT_QUERY_FP8:
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
            _store_ring(
                quantized,
                code,
                step,
                token,
                position,
                pool,
                batches,
                cu,
                slots,
                ring_start,
                D,
                RING_SLOTS,
                SLOT_ROWS,
                RING_STRIDE,
                RUN_ROWS,
                ROW_UNITS,
                PACKED_RING,
            )


@triton.jit
def _store_ring(
    quantized,
    code,
    step,
    token,
    position,
    pool,
    batches,
    cu,
    slots,
    ring_start,
    D,
    RING_SLOTS,
    SLOT_ROWS,
    RING_STRIDE,
    RUN_ROWS,
    ROW_UNITS,
    PACKED_RING,
):
    """A decode's KV row into its window ring.

    Two exits, the ones both window writers already have: a padding row owns
    no request, and a row the ring has scrolled past has nowhere to land. The
    second is also what keeps two tokens of one request off the same row -- it
    admits exactly the last `RING_SLOTS` of them. The grid stays the forward's
    width either way, because that is the width a captured replay runs.
    """
    dim = tl.arange(0, D)
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
    # The tile's shape must be powers of two; anything else, a query-less KV
    # launch (no heads: slot 0 must stay the KV), and the MXFP8 query keep a
    # program a head.
    head_block = (
        _HEAD_BLOCK
        and query_fp8 is None
        and heads > 0
        and (heads & (heads - 1)) == 0
        and rope_dim >= 2
        and (rope_dim & (rope_dim - 1)) == 0
    )
    fp8_heads = _FP8_HEADS
    if not (
        query_fp8 is not None
        and fp8_heads > 1
        and (fp8_heads & (fp8_heads - 1)) == 0
        and heads % fp8_heads == 0
        and tokens * heads // fp8_heads
        >= _FP8_HEADS_MIN_PROGRAMS_PER_CU * _cu_count(query.device)
    ):
        fp8_heads = 1
    query_programs = 1 if head_block else heads // fp8_heads
    _rope_window_kernel[(tokens * (query_programs + kv_half),)](
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
        tokens,
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
        HEAD_BLOCK=head_block,
        FP8_HEADS=fp8_heads,
        QUERY_PROGRAMS=query_programs,
        # The most waves a row splits over with every lane still loading 16 B.
        num_warps=(
            2 if fp8_heads > 1 else max(1, dim * query.element_size() // (64 * 16))
        ),
    )


@triton.jit
def _norm_rope_window_kernel(
    q,
    kv,
    q_weight,
    kv_weight,
    q_values,
    q_scales,
    cos,
    sin,
    positions,
    values,
    scales,
    values_stride,
    scales_stride,
    pool,
    batches,
    cu,
    slots,
    ring_start,
    q_stride,
    kv_stride,
    eps_q,
    eps_kv,
    KQ: tl.constexpr,
    BLOCK_KQ: tl.constexpr,
    SPLIT_Q: tl.constexpr,
    Q_AMAX_FLOOR: tl.constexpr,
    D: tl.constexpr,
    ROPE: tl.constexpr,
    RING_SLOTS: tl.constexpr,
    SLOT_ROWS: tl.constexpr,
    RING_STRIDE: tl.constexpr,
    RUN_ROWS: tl.constexpr,
    ROW_UNITS: tl.constexpr,
    EMIT_PACKED: tl.constexpr,
    WRITE_RING: tl.constexpr,
    PACKED_RING: tl.constexpr,
):
    """One program per token: Q's RMSNorm to MXFP8, and the KV row's RMSNorm
    through `_rope_window_kernel`'s KV chain. `SPLIT_Q` holds Q as a
    `BLOCK_KQ` head and a power-of-two tail rather than one padded tile.

    This extends AITER's dual latent RMSNorm with ATOM's window address and
    byte-layout contract. AITER's QK norm/RoPE/SWA op instead normalizes
    projected per-head Q and emits BF16 rows: it cannot consume Q's latent
    width or emit this MXFP8 ring. Quantization reuses `_mxfp8_quant_op`;
    the KV chain reuses the shared `_quantized_values` and `_store_ring` helpers.
    """
    groups: tl.constexpr = BLOCK_KQ // 32
    cols = tl.arange(0, BLOCK_KQ)
    mask = cols < KQ
    group = tl.arange(0, groups)
    dim = tl.arange(0, D)
    activation = kv.dtype.element_ty
    g = tl.load(q_weight + cols, mask=mask, other=0.0).to(tl.float32)
    gk = tl.load(kv_weight + dim).to(tl.float32)
    if SPLIT_Q:
        TAIL: tl.constexpr = KQ - BLOCK_KQ
        cols_t = BLOCK_KQ + tl.arange(0, TAIL)
        g_t = tl.load(q_weight + cols_t).to(tl.float32)
    token = tl.program_id(0).to(tl.int64)
    x = tl.load(q + token * q_stride + cols, mask=mask, other=0.0).to(tl.float32)
    if SPLIT_Q:
        x_t = tl.load(q + token * q_stride + cols_t).to(tl.float32)
        norm = tl.math.rsqrt(
            ((tl.sum(x * x, axis=-1) + tl.sum(x_t * x_t, axis=-1)) / KQ) + eps_q
        )
        y_t = tl.reshape(x_t * norm * g_t, (TAIL // 32, 32))
        t_code, t_step = _mxfp8_quant_op(y_t, QUANT_AXIS=1, AMAX_FLOOR=Q_AMAX_FLOOR)
        tl.store(
            q_values + token * KQ + cols_t,
            tl.reshape(y_t * t_step, (TAIL,)).to(q_values.dtype.element_ty),
        )
        tl.store(
            q_scales + token * (KQ // 32) + groups + tl.arange(0, TAIL // 32),
            tl.reshape(t_code, (TAIL // 32,)),
        )
    else:
        norm = tl.math.rsqrt((tl.sum(x * x, axis=-1) / KQ) + eps_q)
    y = x * norm * g
    y = tl.reshape(y, (groups, 32))
    q_code, q_step = _mxfp8_quant_op(y, QUANT_AXIS=1, AMAX_FLOOR=Q_AMAX_FLOOR)
    tl.store(
        q_values + token * KQ + cols,
        tl.reshape(y * q_step, (BLOCK_KQ,)).to(q_values.dtype.element_ty),
        mask=mask,
    )
    tl.store(
        q_scales + token * (KQ // 32) + group,
        tl.reshape(q_code, (groups,)),
        mask=group < KQ // 32,
    )

    xk = tl.load(kv + token * kv_stride + dim).to(tl.float32)
    # Rounded where the dual RMSNorm stored its BF16 row, for the same bytes.
    normed = (
        (xk * tl.math.rsqrt((tl.sum(xk * xk, axis=-1) / D) + eps_kv) * gk)
        .to(activation)
        .to(tl.float32)
    )
    position = tl.load(positions + token)
    quantized, code, step = _quantized_values(
        normed, activation, position, cos, sin, D, ROPE
    )
    if EMIT_PACKED:
        tl.store(values + token * values_stride + dim, tl.reshape(quantized, (D,)))
        tl.store(scales + token * scales_stride + tl.arange(0, D // 32), code)
    if WRITE_RING:
        _store_ring(
            quantized,
            code,
            step,
            token,
            position,
            pool,
            batches,
            cu,
            slots,
            ring_start,
            D,
            RING_SLOTS,
            SLOT_ROWS,
            RING_STRIDE,
            RUN_ROWS,
            ROW_UNITS,
            PACKED_RING,
        )


def norm_rope_quant_window(
    q,
    kv,
    q_norm,
    kv_norm,
    cos,
    sin,
    positions,
    *,
    rope_dim,
    q_amax_floor=0.0,
    values=None,
    scales=None,
    ring=None,
):
    """The dual RMSNorm and `rope_quant_window`'s KV half in one launch.

    Returns Q's `(e4m3, e8m0)` pair, as `DualRMSNormMXFP8` does; the KV row
    goes to `values`/`scales` (an FP8 attention's extend rows) or `ring`
    (`(pool, step, window)`, a decode's store), and is not returned.
    """
    tokens, q_dim = q.shape
    dim = kv.shape[-1]
    q_values = torch.empty_like(q, dtype=torch.float8_e4m3fn)
    q_scales = torch.empty(
        (tokens, q_dim // 32), device=q.device, dtype=torch.float8_e8m0fnu
    )
    if not tokens:
        return q_values, q_scales
    pool, step, window = ring or (None, None, None)
    byte_ring = pool is not None and pool.dtype == torch.uint8
    # A long prefill is bound by the padded lanes of a 2048-wide tile for a
    # 1280-wide Q; a decode bucket is too short to notice them.
    block_kq = triton.next_power_of_2(q_dim)
    tail = q_dim - block_kq // 2
    # Both pieces are reshaped into group-32 rows. In particular, a 32-wide
    # Q must stay unsplit: two 16-wide pieces contain no quantization group.
    split_q = (
        tokens >= 768
        and block_kq // 2 % 32 == 0
        and tail > 0
        and tail % 32 == 0
        and tail & (tail - 1) == 0
    )
    if split_q:
        block_kq //= 2
    _norm_rope_window_kernel[(tokens,)](
        q,
        kv,
        q_norm.weight,
        kv_norm.weight,
        q_values,
        q_scales.view(torch.uint8),
        cos,
        sin,
        positions,
        values,
        None if scales is None else scales.view(torch.uint8),
        0 if values is None else values.stride(-2),
        0 if scales is None else scales.stride(-2),
        pool,
        None if step is None else step.batch_ids,
        None if step is None else step.cu_seqlens_q,
        None if step is None else step.slots,
        0 if window is None else window.ring_start,
        q.stride(0),
        kv.stride(0),
        q_norm.eps,
        kv_norm.eps,
        KQ=q_dim,
        BLOCK_KQ=block_kq,
        SPLIT_Q=split_q,
        Q_AMAX_FLOOR=q_amax_floor,
        D=dim,
        ROPE=rope_dim,
        RING_SLOTS=1 if window is None else window.ring_slots,
        SLOT_ROWS=0 if window is None else window.slot_rows,
        RING_STRIDE=1 if window is None else window.ring_stride,
        RUN_ROWS=0 if window is None else window.run_rows,
        ROW_UNITS=1 if pool is None or byte_ring else dim,
        EMIT_PACKED=values is not None,
        WRITE_RING=ring is not None,
        PACKED_RING=byte_ring,
        # Four warps hide a decode bucket's latency; past a few hundred rows
        # the grid fills the device and one-warp programs win (MI355X, 8192
        # rows: 14.6us at four warps, 10.5us at one with `SPLIT_Q`).
        num_warps=4 if tokens < 768 else 1,
    )
    return q_values, q_scales
