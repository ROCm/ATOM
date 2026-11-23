# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Triton launches for a decode step's BF16 tail.

`ring_append_gather` appends each row's new token to its ring row and returns
its open tile, ready for ``mla_nvfp4_update_cache``; it replaces the
index_put / gather / masked_fill and index arithmetic of
`Nvfp4HpRing.append_tokens`. `ring_append_quantize` also does what
``mla_nvfp4_update_cache`` does, writing the tile straight into the page64
cache, so a layer's whole cache write is one launch.
"""

import torch
import triton
import triton.language as tl

TILE = 16
BLOCK_D = 64
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
def _append_gather_kernel(
    ring_ptr,  # [reqs, ring, D] of one layer
    values_ptr,  # [N, D]
    req_ptr,  # [N]
    pos_ptr,  # [N]
    slot_ptr,  # [N] cache slot, -1 for padding
    tiles_ptr,  # [N, TILE, D] out
    valid_ptr,  # [N] out
    tile_ids_ptr,  # [N] out
    pages_ptr,  # [N] out
    stride_req,
    stride_slot,
    stride_val,
    D,
    RING: tl.constexpr,
    PAGE: tl.constexpr,
    TILE: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    n = tl.program_id(0)
    block = tl.program_id(1)
    req = tl.load(req_ptr + n).to(tl.int64)
    pos = tl.load(pos_ptr + n).to(tl.int64)
    cols = block * BLOCK_D + tl.arange(0, BLOCK_D)
    in_d = cols < D
    rows = tl.arange(0, TILE)
    tile_start = (pos // TILE) * TILE
    valid = pos - tile_start + 1
    row_base = ring_ptr + req * stride_req
    # Earlier rows come from the ring. The new token comes from `values`, not
    # from the ring row this program also stores: no read-after-write.
    old = tl.load(
        row_base + ((tile_start + rows) % RING)[:, None] * stride_slot + cols[None, :],
        mask=(rows[:, None] < valid - 1) & in_d[None, :],
        other=0.0,
    )
    new = tl.load(values_ptr + n * stride_val + cols, mask=in_d, other=0.0)
    tile = tl.where((rows == valid - 1)[:, None], new[None, :], old)
    tl.store(
        tiles_ptr + n * (TILE * D) + rows[:, None] * D + cols[None, :],
        tile,
        mask=in_d[None, :],
    )
    tl.store(row_base + (pos % RING) * stride_slot + cols, new, mask=in_d)
    if block == 0:
        slot = tl.load(slot_ptr + n).to(tl.int64)
        # Triton integer division truncates; a padding slot must map to -1.
        page = tl.where(slot < 0, -1, slot // PAGE)
        tl.store(valid_ptr + n, valid.to(tl.int32))
        tl.store(tile_ids_ptr + n, ((pos % PAGE) // TILE).to(tl.int32))
        tl.store(pages_ptr + n, page.to(tl.int32))


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
    big = ((bits + 0x1FFFFF + ((bits >> 22) & 1)) & -4194304).to(tl.float32, bitcast=True)
    q = tl.minimum(tl.where(v < 1.0, small, big), 6.0)
    qb = q.to(tl.int32, bitcast=True)
    # q >= 1: 2 * exponent + mantissa bit + 2 (1 -> 2, 1.5 -> 3, ..., 6 -> 7).
    code = tl.where(q >= 1.0, ((qb >> 23) - 127) * 2 + ((qb >> 22) & 1) + 2, (q != 0).to(tl.int32))
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
def _rope_gptj(x, pos, cos_ptr, sin_ptr, stride_cos_t, ROWS: tl.constexpr, ROPE: tl.constexpr):
    """Interleaved RoPE of BF16 ``x [ROWS, ROPE]`` at ``pos [ROWS]`` with the
    model's BF16 cos/sin cache (front half, GPT-J pairs), bit-identical to
    AITER's `_rope_kernel_cached_thd_2c_gqa_onehead_fwd` on gfx1250. That
    kernel lowers ``x * cos + rot * sin`` to ``v_pk_mul_bf16`` then
    ``v_pk_fma_bf16``: ``bf16(x * cos + bf16(rot * sin))``, one rounding of
    the sum. Here the BF16 products are exact in FP32, and TwoSum recovers
    what the FP32 add drops, so the final rounding to BF16 is the same."""
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
    n, ring_ptr, req_ptr, pos_ptr, slot_ptr, nope_ptr, aux_ptr, stride_req, stride_slot,
    RING: tl.constexpr, PAGE: tl.constexpr, TILE: tl.constexpr,
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
    c0, new, old, rows, valid, tile, nope_page, aux_page,
    TILE: tl.constexpr, CHK: tl.constexpr, N16: tl.constexpr,
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
    off = tile * 4096 + (b // 16)[None, :] * 256 + rows[:, None] * 16 + (b % 16)[None, :]
    tl.store(nope_page + off, _pack(tl.reshape(codes, (TILE, CHK)), TILE, CHK))
    _k_nope_scales(c0 // 16 + tl.arange(0, CHK // 16), scale_code, rows, valid, tile, aux_page, TILE, CHK // 16, N16)


@triton.jit
def _k_nope_scales(
    sc, scale_code, rows, valid, tile, aux_page,
    TILE: tl.constexpr, NB: tl.constexpr, N16: tl.constexpr,
):
    """The tile's scales of blocks ``sc [NB]``: ``scale_code`` for its live
    rows, 0x7F past them."""
    scales = tl.where((rows < valid)[:, None], scale_code[None, :], 127).to(tl.uint8)
    if N16:
        # Two 16 x 16-byte bricks: k * 256 + row * 16 + s for index k * 16 + s.
        soff = 1024 + (sc // 16)[None, :] * 256 + rows[:, None] * 16 + (sc % 16)[None, :]
        tl.store(aux_page + tile * 2048 + soff, scales)
    else:
        # Per token: k * 256 + token * 8 + s for scale index k * 8 + s.
        token = (tile % 2) * TILE + rows
        soff = 2048 + (sc // 8)[None, :] * 256 + token[:, None] * 8 + (sc % 8)[None, :]
        tl.store(aux_page + (tile // 2) * 4096 + soff, scales)


@triton.jit
def _k_nope_incr(
    c0, new, ring_rows, cc, rows, valid, tile, nope_page, aux_page, amax_row, tag_ok,
    TILE: tl.constexpr, CHK: tl.constexpr, N16: tl.constexpr,
):
    """`_k_nope_finish` writing only what the new token changes. ``amax_row``
    holds each 16 x 16 block's amax over the tile's earlier rows (valid when
    ``tag_ok``). A block whose amax the new token does not raise keeps its
    scale, so its earlier rows' codes stand: they are neither read nor
    written. A raised block, a tile started here, or a stale
    state reads the earlier rows and rewrites the block as `_k_nope_finish`
    does. Same bytes as `_k_nope_finish` either way."""
    NB: tl.constexpr = CHK // 16
    r = valid - 1
    blk = c0 // 16 + tl.arange(0, NB)
    j = tl.arange(0, 16)
    new3 = tl.reshape(new, (1, NB, 16))
    newmax = tl.max(tl.max(tl.abs(new3.to(tl.float32)), axis=2), axis=0)
    prev = tl.load(amax_row + blk)
    fresh = (r == 0) | (tag_ok == 0)
    changed = fresh | (newmax > prev)
    # Masks stay [TILE, NB, 1] broadcast over a block's channels so the
    # accesses keep their vector width.
    earlier = rows[:, None, None] < r
    old = tl.load(
        ring_rows[:, :, None] + blk[None, :, None] * 16 + j[None, None, :],
        mask=earlier & changed[None, :, None],
        other=0.0,
    )
    blocks = tl.where(rows[:, None, None] == r, new3, old).to(tl.float32)
    amax = tl.where(changed, tl.max(tl.max(tl.abs(blocks), axis=2), axis=0), prev)
    _, codes, scale_code = _fp4(blocks, amax[None, :, None], True, KV_GAIN, KV_RCP)
    scale_code = tl.reshape(scale_code, (NB,))
    # A kept block's earlier rows keep their codes; every other row is
    # rewritten: the new one, a rebuilt block's, and the rows past the new one
    # (zero codes, as each step leaves them).
    lo, hi = tl.split(tl.reshape(codes, (TILE, NB, 8, 2)))
    k8 = tl.arange(0, 8)
    off = tile * 4096 + (blk // 2)[None, :, None] * 256 + rows[:, None, None] * 16 + (blk % 2)[None, :, None] * 8 + k8[None, None, :]
    tl.store(nope_page + off, (lo | (hi << 4)).to(tl.uint8), mask=~(earlier & ~changed[None, :, None]))
    # A kept block's scale is its stored one, so all scales are written.
    _k_nope_scales(blk, scale_code, rows, valid, tile, aux_page, TILE, NB, N16)
    tl.store(amax_row + blk, amax)


@triton.jit
def _k_rope(
    n, pos, rows, valid, new_row, tile, aux_page, src_rope_ptr, stride_src_rope,
    cos_ptr, sin_ptr, stride_cos_t,
    TILE: tl.constexpr, NOPE: tl.constexpr, ROPE: tl.constexpr,
    ROPE_FUSED: tl.constexpr, N16: tl.constexpr,
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
    base, base_codes, base_scale = _fp4(rb, tl.max(tl.abs(rb), axis=2)[:, :, None], False, KV_GAIN, KV_RCP)
    res = rb - base
    _, res_codes, res_scale = _fp4(res, tl.max(tl.abs(res), axis=2)[:, :, None], False, KV_GAIN, KV_RCP)
    r = valid - 1
    rbytes = tl.arange(0, ROPE // 2)
    row_brick = (rbytes // 16) * 256 + r * 16 + (rbytes % 16)
    base_bytes = tl.reshape(_pack(tl.reshape(base_codes, (1, ROPE)), 1, ROPE), (ROPE // 2,))
    res_bytes = tl.reshape(_pack(tl.reshape(res_codes, (1, ROPE)), 1, ROPE), (ROPE // 2,))
    blk = tl.arange(0, ROPE // 16)
    base_sf = tl.reshape(base_scale, (ROPE // 16,)).to(tl.uint8)
    res_sf = tl.reshape(res_scale, (ROPE // 16,)).to(tl.uint8)
    later = (rows >= 1)[:, None]
    brick = (rbytes // 16)[None, :] * 256 + rows[:, None] * 16 + (rbytes % 16)[None, :]
    zero_codes = tl.zeros((TILE, ROPE // 2), tl.uint8)
    unset_sf = tl.full((TILE, ROPE // 16), 127, tl.uint8)
    pad = tl.arange(0, 512)
    if N16:
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
            poff = tl.where(pad < 128, 1536 + (pad // 8) * 16 + 8 + pad % 8, 1792 + pad - 128)
            tl.store(aux_tile + poff, tl.full((512,), 127, tl.uint8), mask=pad < 384)
    else:
        aux_half = aux_page + (tile // 2) * 4096
        sub = tile % 2
        token = sub * TILE + rows
        tok_r = sub * TILE + r
        tl.store(aux_half + sub * 512 + row_brick, base_bytes)
        tl.store(aux_half + 1024 + sub * 512 + row_brick, res_bytes)
        tl.store(aux_half + 3072 + tok_r * 4 + blk, base_sf)
        tl.store(aux_half + 3200 + tok_r * 4 + blk, res_sf)
        if valid == 1:
            rs_off = token[:, None] * 4 + blk[None, :]
            tl.store(aux_half + sub * 512 + brick, zero_codes, mask=later)
            tl.store(aux_half + 1024 + sub * 512 + brick, zero_codes, mask=later)
            tl.store(aux_half + 3072 + rs_off, unset_sf, mask=later)
            tl.store(aux_half + 3200 + rs_off, unset_sf, mask=later)
            tl.store(
                aux_half + 3328 + sub * 384 + pad,
                tl.full((512,), 127, tl.uint8),
                mask=pad < 384,
            )


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
    N16: tl.constexpr = False,
    KSPLIT: tl.constexpr = 1,
    part=0,
    CH: tl.constexpr = 256,
    amax_ptr=None,
    tag_ptr=None,
    stride_amax=0,
    stride_tag=0,
    INCR: tl.constexpr = False,
):
    """One token of `ring_append_quantize`: append, refresh its tile, write.

    The NoPE blocks are independent along channels, so the 16-token tile is
    handled CH channels at a time to bound the registers live, and a token can
    be split over KSPLIT programs: ``part`` does NoPE channels
    ``[part, part + 1) * NOPE / KSPLIT`` and the last part also the RoPE.
    ``N16`` selects the AUX byte layout: one 2048-byte block per 16-token tile
    (AITER PR 8201), else per-field runs over 32-token half pages (PR 7987).
    ``INCR``: per-tile block amax state (`_k_nope_incr`) at ``amax_ptr`` /
    ``tag_ptr`` (the tile's last position folded in), one row per tail row."""
    req, pos, slot, rows, valid, ring_rows, new_row, tile, nope_page, aux_page = _k_token_setup(
        n, ring_ptr, req_ptr, pos_ptr, slot_ptr, nope_ptr, aux_ptr, stride_req, stride_slot,
        RING, PAGE, TILE,
    )
    # A padding row (slot -1) has nothing to keep: no tail write, no tile.
    # Every padding row shares the scratch tail row, so writing it would
    # serialize them on the same addresses.
    if slot >= 0:
        older = (rows < valid - 1)[:, None]
        if INCR:
            # Another part may already have stamped this step (tag == pos);
            # each part owns its blocks' amax, so its state is still pos - 1's.
            tag = tl.load(tag_ptr + req * stride_tag)
            tag_ok = ((tag == pos - 1) | (tag == pos)).to(tl.int32)
        # CH channels per pass, or the whole part when it is narrower.
        CHK: tl.constexpr = CH if CH <= NOPE // KSPLIT else NOPE // KSPLIT
        for c0 in tl.static_range(0, NOPE // KSPLIT, CHK):
            c0 = c0 + part * (NOPE // KSPLIT)
            cc = c0 + tl.arange(0, CHK)
            new = tl.load(src_nope_ptr + n * stride_src_nope + cc)
            if INCR:
                _k_nope_incr(
                    c0, new, ring_rows, cc, rows, valid, tile, nope_page, aux_page,
                    amax_ptr + req * stride_amax, tag_ok, TILE, CHK, N16,
                )
                tl.store(new_row + cc, new)
            else:
                old = tl.load(ring_rows + cc[None, :], mask=older, other=0.0)
                tl.store(new_row + cc, new)
                _k_nope_finish(c0, new, old, rows, valid, tile, nope_page, aux_page, TILE, CHK, N16)
        if INCR and part == KSPLIT - 1:
            tl.store(tag_ptr + req * stride_tag, pos.to(tl.int32))
        if part == KSPLIT - 1:
            _k_rope(
                n, pos, rows, valid, new_row, tile, aux_page, src_rope_ptr, stride_src_rope,
                cos_ptr, sin_ptr, stride_cos_t, TILE, NOPE, ROPE, ROPE_FUSED, N16,
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
    N16: tl.constexpr,
    amax_ptr=None,
    tag_ptr=None,
    stride_amax=0,
    stride_tag=0,
    INCR: tl.constexpr = False,
):
    _append_quantize_token(
        tl.program_id(0), ring_ptr, src_nope_ptr, src_rope_ptr, req_ptr, pos_ptr,
        slot_ptr, nope_ptr, aux_ptr, stride_req, stride_slot, stride_src_nope,
        stride_src_rope, cos_ptr, sin_ptr, stride_cos_t,
        RING, PAGE, TILE, NOPE, ROPE, ROPE_FUSED, N16,
        amax_ptr=amax_ptr, tag_ptr=tag_ptr, stride_amax=stride_amax,
        stride_tag=stride_tag, INCR=INCR,
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
    n16: bool = False,
    tile_state: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> None:
    """`ring_append_gather` and ``mla_nvfp4_update_cache`` in one launch.

    ``ring`` is one layer of the tail ``[reqs, ring_size, 576]``; ``kv_nope``
    ``[P, 16384]`` and ``kv_aux`` ``[P, 8192]`` are one layer of the cache.
    The new tokens are ``values [N, 576]``, or ``values [N, 512]`` NoPE with
    pre-RoPE ``rope [N, 64]`` and the BF16 cos/sin caches, in which case the
    kernel applies interleaved RoPE (no concat, no separate RoPE launch).
    Rows with slot -1 update only the tail. ``n16`` selects the AUX layout and
    must match ``mla_nvfp4_update_cache``'s (`aiter_aux_n16`). ``tile_state``
    as in `decode_prep`. Graph-safe.
    """
    fused = rope is not None
    incr = tile_state is not None
    t_amax, t_tag = tile_state if incr else (ring, ring)
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
        cos is None or cos.dtype != torch.bfloat16 or cos.shape[-1] != ROPE // 2
        or cos.stride(-1) != 1 or sin.stride() != cos.stride()
    ):
        raise ValueError("Fused NVFP4 RoPE needs the BF16 [max_pos, ..., 32] cos/sin caches.")
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
            N16=n16,
            amax_ptr=t_amax,
            tag_ptr=t_tag,
            stride_amax=t_amax.stride(0) if incr else 0,
            stride_tag=t_tag.stride(0) if incr else 0,
            INCR=incr,
            num_warps=8,
        )


def ring_append_gather(
    ring: torch.Tensor,
    req_ids: torch.Tensor,
    positions: torch.Tensor,
    values: torch.Tensor,
    slots: torch.Tensor,
    page_size: int,
):
    """``ring`` is one layer of the tail, ``[reqs, ring_size, D]`` BF16.

    Writes ``values [N, D]`` at each row's position and returns
    ``(tiles [N, 16, D], valid, tile_ids, pages)``, int32 indices, tile rows
    past ``valid`` zero and ``pages`` -1 where ``slots`` is -1. Graph-safe.
    """
    n, dim = values.shape
    if ring.stride(-1) != 1 or values.stride(-1) != 1:
        raise ValueError("NVFP4 ring and values need a unit inner stride.")
    tiles = torch.empty((n, TILE, dim), dtype=ring.dtype, device=ring.device)
    ints = {"dtype": torch.int32, "device": ring.device}
    valid, tile_ids, pages = (torch.empty(n, **ints) for _ in range(3))
    if n:
        _append_gather_kernel[(n, triton.cdiv(dim, BLOCK_D))](
            ring,
            values,
            req_ids,
            positions,
            slots,
            tiles,
            valid,
            tile_ids,
            pages,
            ring.stride(0),
            ring.stride(1),
            values.stride(0),
            dim,
            RING=ring.shape[1],
            PAGE=page_size,
            TILE=TILE,
            BLOCK_D=BLOCK_D,
        )
    return tiles, valid, tile_ids, pages
