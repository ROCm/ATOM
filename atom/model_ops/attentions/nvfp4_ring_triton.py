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
    """E2M1 of ``v >= 0``, nearest even; saturates at 6. Value and code."""
    code = (
        (v > 0.25).to(tl.int32)
        + (v >= 0.75).to(tl.int32)
        + (v > 1.25).to(tl.int32)
        + (v >= 1.75).to(tl.int32)
        + (v > 2.5).to(tl.int32)
        + (v >= 3.5).to(tl.int32)
        + (v > 5.0).to(tl.int32)
    )
    q = tl.where(code <= 4, code.to(tl.float32) * 0.5, 0.0)
    q = tl.where(code == 5, 3.0, q)
    q = tl.where(code == 6, 4.0, q)
    q = tl.where(code == 7, 6.0, q)
    return q, code


@triton.jit
def _fp4(x, amax, ideal: tl.constexpr):
    """Quantize ``x`` against its block ``amax`` (broadcastable). Returns the
    signed dequantized value, the 4-bit codes and the E4M3 scale code."""
    div = tl.where(amax == 0, 1.0, tl.math.div_rn(amax, 6.0))
    scale, scale_code = _e4m3(tl.clamp(div * KV_GAIN, 1.0 / 512, 448.0))
    if not ideal:
        div = tl.math.div_rn(scale, KV_GAIN)
    q, code = _e2m1(tl.math.div_rn(tl.abs(x), div))
    neg = (x.to(tl.int32, bitcast=True) >> 31) & 1
    signed = tl.where(neg == 1, -q, q)
    return tl.math.div_rn(signed * scale, KV_GAIN), code | (neg << 3), scale_code


@triton.jit
def _pack(codes, ROWS: tl.constexpr, COLS: tl.constexpr):
    """[ROWS, COLS] 4-bit codes to [ROWS, COLS // 2] bytes, low nibble first."""
    lo, hi = tl.split(tl.reshape(codes, (ROWS, COLS // 2, 2)))
    return (lo | (hi << 4)).to(tl.uint8)


@triton.jit
def _append_quantize_kernel(
    ring_ptr,  # [reqs, ring, D] of one layer
    values_ptr,  # [N, D]
    req_ptr,
    pos_ptr,
    slot_ptr,
    nope_ptr,  # [pages, 16384] of one layer
    aux_ptr,  # [pages, 8192] of one layer
    stride_req,
    stride_slot,
    stride_val,
    RING: tl.constexpr,
    PAGE: tl.constexpr,
    TILE: tl.constexpr,
    NOPE: tl.constexpr,
    ROPE: tl.constexpr,
):
    n = tl.program_id(0)
    req = tl.load(req_ptr + n).to(tl.int64)
    pos = tl.load(pos_ptr + n).to(tl.int64)
    slot = tl.load(slot_ptr + n).to(tl.int64)
    rows = tl.arange(0, TILE)
    tile_start = (pos // TILE) * TILE
    valid = pos - tile_start + 1
    live = rows < valid
    newest = rows == valid - 1
    row_base = ring_ptr + req * stride_req
    ring_rows = row_base + ((tile_start + rows) % RING)[:, None] * stride_slot

    # The tile in FP32: earlier rows from the ring, the new token from
    # `values` (not re-read from the ring row stored below), the rest zero.
    nc = tl.arange(0, NOPE)
    rc = tl.arange(0, ROPE)
    new_nope = tl.load(values_ptr + n * stride_val + nc)
    new_rope = tl.load(values_ptr + n * stride_val + NOPE + rc)
    old_nope = tl.load(
        ring_rows + nc[None, :], mask=(rows < valid - 1)[:, None], other=0.0
    )
    old_rope = tl.load(
        ring_rows + NOPE + rc[None, :], mask=(rows < valid - 1)[:, None], other=0.0
    )
    nope = tl.where(newest[:, None], new_nope[None, :], old_nope).to(tl.float32)
    rope = tl.where(newest[:, None], new_rope[None, :], old_rope).to(tl.float32)
    tl.store(row_base + (pos % RING) * stride_slot + nc, new_nope)
    tl.store(row_base + (pos % RING) * stride_slot + NOPE + rc, new_rope)

    if slot >= 0:
        page = slot // PAGE
        tile = (pos % PAGE) // TILE
        half = tile // 2
        sub = tile % 2
        nope_page = nope_ptr + page * 16384
        aux_half = aux_ptr + page * 8192 + half * 4096

        # NoPE: one scale per 16-token x 16-channel block, codes over amax / 6.
        blocks = tl.reshape(nope, (TILE, NOPE // 16, 16))
        amax = tl.max(tl.max(tl.abs(blocks), axis=2), axis=0)
        _, codes, scale_code = _fp4(blocks, amax[None, :, None], True)
        scale_code = tl.reshape(scale_code, (NOPE // 16,))
        packed = _pack(tl.reshape(codes, (TILE, NOPE)), TILE, NOPE)
        b = tl.arange(0, NOPE // 2)
        # 16-token x 32-channel bricks: tile, channel group, token, byte.
        nope_off = (
            tile * 4096
            + (b // 16)[None, :] * 256
            + rows[:, None] * 16
            + (b % 16)[None, :]
        )
        tl.store(nope_page + nope_off, packed)
        # Per token: k * 256 + token * 8 + s for scale index k * 8 + s.
        sc = tl.arange(0, NOPE // 16)
        token = sub * TILE + rows
        scale_off = (
            2048 + (sc // 8)[None, :] * 256 + token[:, None] * 8 + (sc % 8)[None, :]
        )
        scales = tl.where(live[:, None], scale_code[None, :], 127)
        tl.store(aux_half + scale_off, scales.to(tl.uint8))

        # RoPE: per token, 16-channel blocks, plus a residual quantized alike.
        rb = tl.reshape(rope, (TILE, ROPE // 16, 16))
        base, base_codes, base_scale = _fp4(
            rb, tl.max(tl.abs(rb), axis=2)[:, :, None], False
        )
        res = rb - base
        _, res_codes, res_scale = _fp4(
            res, tl.max(tl.abs(res), axis=2)[:, :, None], False
        )
        base_scale = tl.reshape(base_scale, (TILE, ROPE // 16))
        res_scale = tl.reshape(res_scale, (TILE, ROPE // 16))
        rbytes = tl.arange(0, ROPE // 2)
        code_off = (
            sub * 512
            + (rbytes // 16)[None, :] * 256
            + rows[:, None] * 16
            + (rbytes % 16)[None, :]
        )
        tl.store(
            aux_half + code_off, _pack(tl.reshape(base_codes, (TILE, ROPE)), TILE, ROPE)
        )
        tl.store(
            aux_half + 1024 + code_off,
            _pack(tl.reshape(res_codes, (TILE, ROPE)), TILE, ROPE),
        )
        blk = tl.arange(0, ROPE // 16)
        rs_off = token[:, None] * 4 + blk[None, :]
        tl.store(
            aux_half + 3072 + rs_off,
            tl.where(live[:, None], base_scale, 127).to(tl.uint8),
        )
        tl.store(
            aux_half + 3200 + rs_off,
            tl.where(live[:, None], res_scale, 127).to(tl.uint8),
        )
        pad = tl.arange(0, 512)
        tl.store(
            aux_half + 3328 + sub * 384 + pad,
            tl.full((512,), 127, tl.uint8),
            mask=pad < 384,
        )


def ring_append_quantize(
    ring: torch.Tensor,
    req_ids: torch.Tensor,
    positions: torch.Tensor,
    values: torch.Tensor,
    slots: torch.Tensor,
    kv_nope: torch.Tensor,
    kv_aux: torch.Tensor,
) -> None:
    """`ring_append_gather` and ``mla_nvfp4_update_cache`` in one launch.

    ``ring`` is one layer of the tail ``[reqs, ring_size, 576]``; ``kv_nope``
    ``[P, 16384]`` and ``kv_aux`` ``[P, 8192]`` are one layer of the cache.
    Rows with slot -1 update only the tail. Graph-safe.
    """
    n, dim = values.shape
    if dim != NOPE + ROPE:
        raise ValueError(f"NVFP4 tail rows are {NOPE + ROPE} wide, got {dim}.")
    if ring.stride(-1) != 1 or values.stride(-1) != 1:
        raise ValueError("NVFP4 ring and values need a unit inner stride.")
    if not (kv_nope.is_contiguous() and kv_aux.is_contiguous()):
        raise ValueError("NVFP4 cache planes must be contiguous.")
    if n:
        _append_quantize_kernel[(n,)](
            ring,
            values,
            req_ids,
            positions,
            slots,
            kv_nope,
            kv_aux,
            ring.stride(0),
            ring.stride(1),
            values.stride(0),
            RING=ring.shape[1],
            PAGE=PAGE,
            TILE=TILE,
            NOPE=NOPE,
            ROPE=ROPE,
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
