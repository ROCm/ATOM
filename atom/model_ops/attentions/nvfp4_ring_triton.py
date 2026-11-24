# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Triton launches for the NVFP4 BF16 tail and FP4 cache writes.

`ring_append_quantize` appends each decode row's new token to its ring row and
does what ``mla_nvfp4_update_cache`` does to its open tile, writing it straight
into the page64 cache (N16 AUX layout), so a layer's whole cache write is one
launch. `prefill_quantize` does the same for a prefill step's tiles, gathering
their rows from the split latent.
"""

import torch
import triton
import triton.language as tl

TILE = 16
PAGE = 64
NOPE = 512
ROPE = 64
NOPE_BYTES = 16384
AUX_BYTES = 8192
KV_GAIN = tl.constexpr(89.6)


def rcp32(gain: float) -> float:
    """RN(1 / fp32(gain)) in FP32, the reciprocal `_fp4` divides by `gain` with."""
    one = torch.ones((), dtype=torch.float32)
    return float(one / torch.tensor(gain, dtype=torch.float32))


KV_RCP = tl.constexpr(rcp32(89.6))


@triton.jit
def _e4m3(v):
    """Round FP32 ``v`` in [2**-9, 448] to E4M3, nearest even. Returns the
    value and its code. Exact: the step is a power of two."""
    bits = v.to(tl.int32, bitcast=True)
    e = tl.maximum(((bits >> 23) & 0xFF) - 127, -6)
    step = ((e - 3 + 127) << 23).to(tl.float32, bitcast=True)
    x = v * ((3 - e + 127) << 23).to(tl.float32, bitcast=True)
    f = tl.floor(x)
    d = x - f
    odd = (f.to(tl.int32) & 1) == 1
    m = f + tl.where((d > 0.5) | ((d == 0.5) & odd), 1.0, 0.0)
    # Subnormals (e == -6, m < 8) land on code m; m == 16 carries into e + 1.
    code = (e + 7) * 8 + (m.to(tl.int32) - 8)
    return m * step, code


@triton.jit
def _e2m1(v):
    """E2M1 of finite ``v >= 0``, nearest even; saturates at 6. Value and code.
    Rounds v onto the grid and reads the code off the bits: below 1 to
    multiples of 0.5 (adding 2**22 leaves an ulp of 0.5), from 1 up to one
    mantissa bit."""
    small = (v + 4194304.0) - 4194304.0
    bits = v.to(tl.int32, bitcast=True)
    big = ((bits + 0x1FFFFF + ((bits >> 22) & 1)) & -4194304).to(
        tl.float32, bitcast=True
    )
    q = tl.minimum(tl.where(v < 1.0, small, big), 6.0)
    qb = q.to(tl.int32, bitcast=True)
    # q >= 1: 2 * exponent + mantissa bit + 2 (1 -> 2, 1.5 -> 3, ..., 6 -> 7).
    code = tl.where(
        q >= 1.0, ((qb >> 23) - 127) * 2 + ((qb >> 22) & 1) + 2, (q != 0).to(tl.int32)
    )
    return q, code


@triton.jit
def _div_rn_by(a, b, recip):
    """``a / b`` correctly rounded, given ``recip = RN(1 / b)``: ``a * recip``
    is within an ulp and one FMA correction rounds it exactly (Markstein; no
    over/underflow here). A division per block instead of per element."""
    q = a * recip
    return tl.fma(tl.fma(-b, q, a), recip, q)


@triton.jit
def _fp4(x, amax, ideal: tl.constexpr, gain: tl.constexpr, rcp_gain: tl.constexpr):
    """Quantize ``x`` against its block ``amax`` (broadcastable) at ``gain``.
    Returns the signed dequantized value, the 4-bit codes and the E4M3 scale
    code. ``ideal`` codes over ``amax / 6``, otherwise over the rounded scale.
    ``rcp_gain = rcp32(gain)``: the divisions by ``gain`` (of E4M3 scales and
    of E2M1 x E4M3 products) are `_div_rn_by`, exact for all those values; it
    returns +0 for -0, so the sign goes on after (-0 for a negative zero code)."""
    div = tl.where(amax == 0, 1.0, tl.math.div_rn(amax, 6.0))
    scale, scale_code = _e4m3(tl.clamp(div * gain, 1.0 / 512, 448.0))
    if not ideal:
        div = _div_rn_by(scale, gain, rcp_gain)
    q, code = _e2m1(_div_rn_by(tl.abs(x), div, tl.math.div_rn(1.0, div)))
    neg = (x.to(tl.int32, bitcast=True) >> 31) & 1
    mag = _div_rn_by(q * scale, gain, rcp_gain)
    return tl.where(neg == 1, -mag, mag), code | (neg << 3), scale_code


@triton.jit
def _rope_gptj(
    x, pos, cos_ptr, sin_ptr, stride_cos_t, ROWS: tl.constexpr, ROPE: tl.constexpr
):
    """Interleaved RoPE of BF16 ``x [ROWS, ROPE]`` at ``pos [ROWS]`` with the
    model's BF16 cos/sin cache (front half, GPT-J pairs), rounded as AITER's
    `_rope_kernel_cached_thd_2c_gqa_onehead_fwd` is on gfx1250 when it lowers
    ``x * cos + rot * sin`` to ``v_pk_mul_bf16`` then ``v_pk_fma_bf16``:
    ``bf16(x * cos + bf16(rot * sin))``, one rounding of the sum. Here the
    BF16 products are exact in FP32, and TwoSum recovers what the FP32 add
    drops, so the final rounding to BF16 is the same. A Triton build that
    lowers AITER's kernel otherwise lands some elements one BF16 ulp apart."""
    d = tl.arange(0, ROPE)
    cs = pos[:, None] * stride_cos_t + (d // 2)[None, :]
    cos = tl.load(cos_ptr + cs).to(tl.float32)
    sin = tl.load(sin_ptr + cs).to(tl.float32)
    xf = x.to(tl.float32)
    rot = tl.where((d % 2 == 0)[None, :], xf, -xf)
    rot = tl.reshape(tl.flip(tl.reshape(rot, (ROWS, ROPE // 2, 2)), 2), (ROWS, ROPE))
    a = xf * cos  # exact: 8-bit x 8-bit significands
    b = (rot * sin).to(tl.bfloat16).to(tl.float32)  # v_pk_mul_bf16
    s = a + b
    bb = s - a
    err = (a - (s - bb)) + (b - bb)  # exact a + b == s + err
    # Round s to BF16, nearest even, but break an exact-midpoint tie with err.
    bits = s.to(tl.int32, bitcast=True)
    low = bits & 0xFFFF
    up = (err > 0) == (s > 0)  # the exact sum lies farther from zero than s
    tie = (low == 0x8000) & (err != 0)
    trunc = bits & -65536
    fixed = tl.where(up, trunc + 65536, trunc).to(tl.float32, bitcast=True)
    return tl.where(tie, fixed, s).to(x.dtype)


@triton.jit
def _pack(codes, ROWS: tl.constexpr, COLS: tl.constexpr):
    """[ROWS, COLS] 4-bit codes to [ROWS, COLS // 2] bytes, low nibble first."""
    lo, hi = tl.split(tl.reshape(codes, (ROWS, COLS // 2, 2)))
    return (lo | (hi << 4)).to(tl.uint8)


@triton.jit
def _k_token_setup(
    n,
    ring_ptr,
    req_ptr,
    pos_ptr,
    slot_ptr,
    nope_ptr,
    aux_ptr,
    stride_req,
    stride_slot,
    RING: tl.constexpr,
    PAGE: tl.constexpr,
    TILE: tl.constexpr,
):
    """Token ``n``'s tail rows, tile and cache addresses."""
    req = tl.load(req_ptr + n).to(tl.int64)
    pos = tl.load(pos_ptr + n).to(tl.int64)
    slot = tl.load(slot_ptr + n).to(tl.int64)
    rows = tl.arange(0, TILE)
    tile_start = (pos // TILE) * TILE
    valid = pos - tile_start + 1
    row_base = ring_ptr + req * stride_req
    ring_rows = row_base + ((tile_start + rows) % RING)[:, None] * stride_slot
    new_row = row_base + (pos % RING) * stride_slot
    page = slot // PAGE
    tile = (pos % PAGE) // TILE
    nope_page = nope_ptr + page * 16384
    aux_page = aux_ptr + page * 8192
    return req, pos, slot, rows, valid, ring_rows, new_row, tile, nope_page, aux_page


@triton.jit
def _k_nope_finish(
    c0,
    new,
    old,
    rows,
    valid,
    tile,
    nope_page,
    aux_page,
    TILE: tl.constexpr,
    CHK: tl.constexpr,
):
    """Channels ``c0 + [0, CHK)`` of the tile: one scale per 16-token x
    16-channel block, codes over amax / 6, from the earlier rows ``old`` (the
    ring) and the new token ``new`` (its source), the rest zero."""
    newest = rows == valid - 1
    blocks = tl.reshape(
        tl.where(newest[:, None], new[None, :], old).to(tl.float32),
        (TILE, CHK // 16, 16),
    )
    amax = tl.max(tl.max(tl.abs(blocks), axis=2), axis=0)
    _, codes, scale_code = _fp4(blocks, amax[None, :, None], True, KV_GAIN, KV_RCP)
    scale_code = tl.reshape(scale_code, (CHK // 16,))
    # 16-token x 32-channel bricks: tile, channel group, token, byte.
    b = c0 // 2 + tl.arange(0, CHK // 2)
    off = (
        tile * 4096 + (b // 16)[None, :] * 256 + rows[:, None] * 16 + (b % 16)[None, :]
    )
    tl.store(nope_page + off, _pack(tl.reshape(codes, (TILE, CHK)), TILE, CHK))
    _k_nope_scales(
        c0 // 16 + tl.arange(0, CHK // 16),
        scale_code,
        rows,
        valid,
        tile,
        aux_page,
        TILE,
        CHK // 16,
    )


@triton.jit
def _k_nope_scales(
    sc,
    scale_code,
    rows,
    valid,
    tile,
    aux_page,
    TILE: tl.constexpr,
    NB: tl.constexpr,
):
    """The tile's scales of blocks ``sc [NB]``: ``scale_code`` for its live
    rows, 0x7F past them. Two 16 x 16-byte bricks: k * 256 + row * 16 + s for
    index k * 16 + s."""
    scales = tl.where((rows < valid)[:, None], scale_code[None, :], 127).to(tl.uint8)
    soff = 1024 + (sc // 16)[None, :] * 256 + rows[:, None] * 16 + (sc % 16)[None, :]
    tl.store(aux_page + tile * 2048 + soff, scales)


@triton.jit
def _k_rope(
    n,
    pos,
    rows,
    valid,
    new_row,
    tile,
    aux_page,
    src_rope_ptr,
    stride_src_rope,
    cos_ptr,
    sin_ptr,
    stride_cos_t,
    TILE: tl.constexpr,
    NOPE: tl.constexpr,
    ROPE: tl.constexpr,
    ROPE_FUSED: tl.constexpr,
):
    """The new token's RoPE part: per token, 16-channel blocks, plus a
    residual quantized alike. Its scales are per token, so only the new row
    changes: the earlier rows' codes and scales stay as written. A tile
    started here (valid == 1) first gets its later rows and padding
    initialized as the producer leaves them (zero codes, scale 0x7F)."""
    rc = tl.arange(0, ROPE)
    new_rope = tl.load(src_rope_ptr + n * stride_src_rope + rc)
    if ROPE_FUSED:
        new_rope = tl.reshape(
            _rope_gptj(
                tl.reshape(new_rope, (1, ROPE)),
                tl.zeros((1,), tl.int64) + pos,
                cos_ptr,
                sin_ptr,
                stride_cos_t,
                1,
                ROPE,
            ),
            (ROPE,),
        )
    tl.store(new_row + NOPE + rc, new_rope)
    rb = tl.reshape(new_rope.to(tl.float32), (1, ROPE // 16, 16))
    base, base_codes, base_scale = _fp4(
        rb, tl.max(tl.abs(rb), axis=2)[:, :, None], False, KV_GAIN, KV_RCP
    )
    res = rb - base
    _, res_codes, res_scale = _fp4(
        res, tl.max(tl.abs(res), axis=2)[:, :, None], False, KV_GAIN, KV_RCP
    )
    r = valid - 1
    rbytes = tl.arange(0, ROPE // 2)
    row_brick = (rbytes // 16) * 256 + r * 16 + (rbytes % 16)
    base_bytes = tl.reshape(
        _pack(tl.reshape(base_codes, (1, ROPE)), 1, ROPE), (ROPE // 2,)
    )
    res_bytes = tl.reshape(
        _pack(tl.reshape(res_codes, (1, ROPE)), 1, ROPE), (ROPE // 2,)
    )
    blk = tl.arange(0, ROPE // 16)
    base_sf = tl.reshape(base_scale, (ROPE // 16,)).to(tl.uint8)
    res_sf = tl.reshape(res_scale, (ROPE // 16,)).to(tl.uint8)
    later = (rows >= 1)[:, None]
    brick = (rbytes // 16)[None, :] * 256 + rows[:, None] * 16 + (rbytes % 16)[None, :]
    zero_codes = tl.zeros((TILE, ROPE // 2), tl.uint8)
    unset_sf = tl.full((TILE, ROPE // 16), 127, tl.uint8)
    pad = tl.arange(0, 512)
    # Codes 0 / 512, then one 16-byte row per token at 1536: four RoPE
    # scales, four residual scales, eight padding bytes; 1792.. padding.
    aux_tile = aux_page + tile * 2048
    tl.store(aux_tile + row_brick, base_bytes)
    tl.store(aux_tile + 512 + row_brick, res_bytes)
    tl.store(aux_tile + 1536 + r * 16 + blk, base_sf)
    tl.store(aux_tile + 1540 + r * 16 + blk, res_sf)
    if valid == 1:
        rs_off = 1536 + rows[:, None] * 16 + blk[None, :]
        tl.store(aux_tile + brick, zero_codes, mask=later)
        tl.store(aux_tile + 512 + brick, zero_codes, mask=later)
        tl.store(aux_tile + rs_off, unset_sf, mask=later)
        tl.store(aux_tile + rs_off + 4, unset_sf, mask=later)
        poff = tl.where(
            pad < 128, 1536 + (pad // 8) * 16 + 8 + pad % 8, 1792 + pad - 128
        )
        tl.store(aux_tile + poff, tl.full((512,), 127, tl.uint8), mask=pad < 384)


@triton.jit
def _k_rope_tile(
    rope,
    rows,
    valid,
    tile,
    aux_page,
    TILE: tl.constexpr,
    ROPE: tl.constexpr,
):
    """The RoPE part of a whole tile, ``rope [TILE, ROPE]``: what `_k_rope`
    writes for one token, for every row below ``valid``; zero codes and scale
    0x7F past it, and the tile's padding bytes as the producer leaves them."""
    rb = tl.reshape(rope.to(tl.float32), (TILE, ROPE // 16, 16))
    base, base_codes, base_scale = _fp4(
        rb, tl.max(tl.abs(rb), axis=2)[:, :, None], False, KV_GAIN, KV_RCP
    )
    res = rb - base
    _, res_codes, res_scale = _fp4(
        res, tl.max(tl.abs(res), axis=2)[:, :, None], False, KV_GAIN, KV_RCP
    )
    live = (rows < valid)[:, None]
    rbytes = tl.arange(0, ROPE // 2)
    brick = (rbytes // 16)[None, :] * 256 + rows[:, None] * 16 + (rbytes % 16)[None, :]
    zero_codes = tl.zeros((TILE, ROPE // 2), tl.uint8)
    base_bytes = _pack(tl.reshape(base_codes, (TILE, ROPE)), TILE, ROPE)
    res_bytes = _pack(tl.reshape(res_codes, (TILE, ROPE)), TILE, ROPE)
    aux_tile = aux_page + tile * 2048
    tl.store(aux_tile + brick, tl.where(live, base_bytes, zero_codes))
    tl.store(aux_tile + 512 + brick, tl.where(live, res_bytes, zero_codes))
    blk = tl.arange(0, ROPE // 16)
    base_sf = tl.reshape(base_scale, (TILE, ROPE // 16)).to(tl.uint8)
    res_sf = tl.reshape(res_scale, (TILE, ROPE // 16)).to(tl.uint8)
    unset_sf = tl.full((TILE, ROPE // 16), 127, tl.uint8)
    rs_off = 1536 + rows[:, None] * 16 + blk[None, :]
    tl.store(aux_tile + rs_off, tl.where(live, base_sf, unset_sf))
    tl.store(aux_tile + rs_off + 4, tl.where(live, res_sf, unset_sf))
    pad = tl.arange(0, 512)
    poff = tl.where(pad < 128, 1536 + (pad // 8) * 16 + 8 + pad % 8, 1792 + pad - 128)
    tl.store(aux_tile + poff, tl.full((512,), 127, tl.uint8), mask=pad < 384)


@triton.jit
def _prefill_quantize_kernel(
    src_nope_ptr,  # [T, 512] latent NoPE
    src_rope_ptr,  # [T, 64] latent RoPE, after RoPE
    gather_ptr,  # [M * TILE] token of every tile row
    page_ptr,
    tile_id_ptr,
    valid_ptr,
    nope_ptr,  # [pages, 16384] of one layer
    aux_ptr,  # [pages, 8192] of one layer
    stride_src_nope,
    stride_src_rope,
    TILE: tl.constexpr,
    NOPE: tl.constexpr,
    ROPE: tl.constexpr,
    CH: tl.constexpr,
):
    """One tile of `prefill_quantize`: its rows gathered from the latent,
    quantized and written into its page, CH NoPE channels at a time."""
    t = tl.program_id(0)
    page = tl.load(page_ptr + t).to(tl.int64)
    if page >= 0:
        tile = tl.load(tile_id_ptr + t).to(tl.int64)
        valid = tl.load(valid_ptr + t).to(tl.int64)
        rows = tl.arange(0, TILE)
        live = (rows < valid)[:, None]
        tokens = tl.load(gather_ptr + t * TILE + rows).to(tl.int64)
        newest = tl.load(gather_ptr + t * TILE + valid - 1).to(tl.int64)
        nope_page = nope_ptr + page * 16384
        aux_page = aux_ptr + page * 8192
        for c0 in tl.static_range(0, NOPE, CH):
            cc = c0 + tl.arange(0, CH)
            old = tl.load(
                src_nope_ptr + tokens[:, None] * stride_src_nope + cc[None, :],
                mask=live,
                other=0.0,
            )
            new = tl.load(src_nope_ptr + newest * stride_src_nope + cc)
            _k_nope_finish(
                c0, new, old, rows, valid, tile, nope_page, aux_page, TILE, CH
            )
        rc = tl.arange(0, ROPE)
        rope = tl.load(
            src_rope_ptr + tokens[:, None] * stride_src_rope + rc[None, :],
            mask=live,
            other=0.0,
        )
        _k_rope_tile(rope, rows, valid, tile, aux_page, TILE, ROPE)


@triton.jit
def _append_quantize_token(
    n,  # the token (batch row) this program writes
    ring_ptr,  # [reqs, ring, D] of one layer
    src_nope_ptr,  # [N, 512] new latent NoPE
    src_rope_ptr,  # [N, 64] new latent RoPE (pre-RoPE under ROPE_FUSED)
    req_ptr,
    pos_ptr,
    slot_ptr,
    nope_ptr,  # [pages, 16384] of one layer
    aux_ptr,  # [pages, 8192] of one layer
    stride_req,
    stride_slot,
    stride_src_nope,
    stride_src_rope,
    cos_ptr,
    sin_ptr,
    stride_cos_t,
    RING: tl.constexpr,
    PAGE: tl.constexpr,
    TILE: tl.constexpr,
    NOPE: tl.constexpr,
    ROPE: tl.constexpr,
    ROPE_FUSED: tl.constexpr,
    CH: tl.constexpr = 256,
):
    """One token of `ring_append_quantize`: append, refresh its tile, write.

    The NoPE blocks are independent along channels, so the 16-token tile is
    handled CH channels at a time to bound the registers live."""
    _, pos, slot, rows, valid, ring_rows, new_row, tile, nope_page, aux_page = (
        _k_token_setup(
            n,
            ring_ptr,
            req_ptr,
            pos_ptr,
            slot_ptr,
            nope_ptr,
            aux_ptr,
            stride_req,
            stride_slot,
            RING,
            PAGE,
            TILE,
        )
    )
    # A padding row (slot -1) has nothing to keep: no tail write, no tile.
    # Every padding row shares the scratch tail row, so writing it would
    # serialize them on the same addresses.
    if slot >= 0:
        older = (rows < valid - 1)[:, None]
        for c0 in tl.static_range(0, NOPE, CH):
            cc = c0 + tl.arange(0, CH)
            new = tl.load(src_nope_ptr + n * stride_src_nope + cc)
            old = tl.load(ring_rows + cc[None, :], mask=older, other=0.0)
            tl.store(new_row + cc, new)
            _k_nope_finish(
                c0, new, old, rows, valid, tile, nope_page, aux_page, TILE, CH
            )
        _k_rope(
            n,
            pos,
            rows,
            valid,
            new_row,
            tile,
            aux_page,
            src_rope_ptr,
            stride_src_rope,
            cos_ptr,
            sin_ptr,
            stride_cos_t,
            TILE,
            NOPE,
            ROPE,
            ROPE_FUSED,
        )


@triton.jit
def _append_quantize_kernel(
    ring_ptr,
    src_nope_ptr,
    src_rope_ptr,
    req_ptr,
    pos_ptr,
    slot_ptr,
    nope_ptr,
    aux_ptr,
    stride_req,
    stride_slot,
    stride_src_nope,
    stride_src_rope,
    cos_ptr,
    sin_ptr,
    stride_cos_t,
    RING: tl.constexpr,
    PAGE: tl.constexpr,
    TILE: tl.constexpr,
    NOPE: tl.constexpr,
    ROPE: tl.constexpr,
    ROPE_FUSED: tl.constexpr,
):
    _append_quantize_token(
        tl.program_id(0),
        ring_ptr,
        src_nope_ptr,
        src_rope_ptr,
        req_ptr,
        pos_ptr,
        slot_ptr,
        nope_ptr,
        aux_ptr,
        stride_req,
        stride_slot,
        stride_src_nope,
        stride_src_rope,
        cos_ptr,
        sin_ptr,
        stride_cos_t,
        RING,
        PAGE,
        TILE,
        NOPE,
        ROPE,
        ROPE_FUSED,
    )


def ring_append_quantize(
    ring: torch.Tensor,
    req_ids: torch.Tensor,
    positions: torch.Tensor,
    values: torch.Tensor,
    slots: torch.Tensor,
    kv_nope: torch.Tensor,
    kv_aux: torch.Tensor,
    rope: torch.Tensor | None = None,
    cos: torch.Tensor | None = None,
    sin: torch.Tensor | None = None,
) -> None:
    """The tail append and ``mla_nvfp4_update_cache`` in one launch.

    ``ring`` is one layer of the tail ``[reqs, ring_size, 576]``; ``kv_nope``
    ``[P, 16384]`` and ``kv_aux`` ``[P, 8192]`` are one layer of the cache.
    The new tokens are ``values [N, 576]``, or ``values [N, 512]`` NoPE with
    pre-RoPE ``rope [N, 64]`` and the BF16 cos/sin caches, in which case the
    kernel applies interleaved RoPE (no concat, no separate RoPE launch).
    Rows with slot -1 update nothing. Graph-safe.
    """
    fused = rope is not None
    nope_src = values if fused else values[:, :NOPE]
    rope_src = rope if fused else values[:, NOPE:]
    n = nope_src.shape[0]
    if nope_src.shape[1] != NOPE or rope_src.shape != (n, ROPE):
        raise ValueError(
            f"NVFP4 tail rows are {NOPE} + {ROPE} wide, got "
            f"{tuple(nope_src.shape)} + {tuple(rope_src.shape)}."
        )
    if ring.stride(-1) != 1 or nope_src.stride(-1) != 1 or rope_src.stride(-1) != 1:
        raise ValueError("NVFP4 ring and values need a unit inner stride.")
    if not (kv_nope.is_contiguous() and kv_aux.is_contiguous()):
        raise ValueError("NVFP4 cache planes must be contiguous.")
    if fused and (
        cos is None
        or cos.dtype != torch.bfloat16
        or cos.shape[-1] != ROPE // 2
        or cos.stride(-1) != 1
        or sin.stride() != cos.stride()
    ):
        raise ValueError(
            "Fused NVFP4 RoPE needs the BF16 [max_pos, ..., 32] cos/sin caches."
        )
    if n:
        _append_quantize_kernel[(n,)](
            ring,
            nope_src,
            rope_src,
            req_ids,
            positions,
            slots,
            kv_nope,
            kv_aux,
            ring.stride(0),
            ring.stride(1),
            nope_src.stride(0),
            rope_src.stride(0),
            cos if fused else ring,
            sin if fused else ring,
            cos.stride(0) if fused else 0,
            RING=ring.shape[1],
            PAGE=PAGE,
            TILE=TILE,
            NOPE=NOPE,
            ROPE=ROPE,
            ROPE_FUSED=fused,
            num_warps=8,
        )


def prefill_quantize(
    k_nope: torch.Tensor,
    k_rope: torch.Tensor,
    gather: torch.Tensor,
    pages: torch.Tensor,
    tile_ids: torch.Tensor,
    valid: torch.Tensor,
    kv_nope: torch.Tensor,
    kv_aux: torch.Tensor,
) -> None:
    """``mla_nvfp4_update_cache`` of a prefill step's tiles in one launch,
    read straight from the split latent.

    ``k_nope [T, 512]`` and ``k_rope [T, 64]`` are the step's BF16 latent after
    RoPE. ``gather [M * 16]``, ``pages``, ``tile_ids`` and ``valid [M]`` are
    the step's plan (`Nvfp4MlaCache.plan_prefill`). ``kv_nope [P, 16384]`` and
    ``kv_aux [P, 8192]`` are one layer of the cache. The bytes written are the
    producer's for the gathered ``[M, 16, 576]`` tiles, which are never built.
    Tiles with page -1 update nothing.
    """
    tiles = pages.numel()
    if k_nope.dtype != torch.bfloat16 or k_rope.dtype != torch.bfloat16:
        raise ValueError("NVFP4 prefill latent must be BF16.")
    if (
        k_nope.dim() != 2
        or k_nope.shape[1] != NOPE
        or k_rope.shape != (k_nope.shape[0], ROPE)
    ):
        raise ValueError(
            f"NVFP4 prefill latent is {NOPE} + {ROPE} wide, got "
            f"{tuple(k_nope.shape)} + {tuple(k_rope.shape)}."
        )
    if k_nope.stride(-1) != 1 or k_rope.stride(-1) != 1:
        raise ValueError("NVFP4 prefill latent needs a unit inner stride.")
    if gather.numel() != tiles * TILE:
        raise ValueError(
            f"NVFP4 prefill plan has {tiles} tiles but gathers "
            f"{gather.numel()} rows."
        )
    plan_and_cache = (gather, pages, tile_ids, valid, kv_nope, kv_aux)
    if not all(t.is_contiguous() for t in plan_and_cache):
        raise ValueError("NVFP4 prefill plan and cache planes must be contiguous.")
    if tiles:
        _prefill_quantize_kernel[(tiles,)](
            k_nope,
            k_rope,
            gather,
            pages,
            tile_ids,
            valid,
            kv_nope,
            kv_aux,
            k_nope.stride(0),
            k_rope.stride(0),
            TILE=TILE,
            NOPE=NOPE,
            ROPE=ROPE,
            CH=128,
            num_warps=4,
        )
