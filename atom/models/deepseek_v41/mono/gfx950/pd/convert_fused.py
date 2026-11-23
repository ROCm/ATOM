# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""TileRT wire -> existing TkRank cache, without temporary decode planes.

The independent arithmetic/layout reference remains ``pd.convert``. Main keeps
the wire FP8 round trip; window also keeps the reference's BF16 rounding before
E4M3 quantization. Each rank reads its mapped receive buffer on its own stream.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from . import convert as C
from . import wire as W


@triton.jit
def _wire_row(Wire, row, dim, valid, CB: tl.constexpr):
    page = row // CB
    within = row % CB
    base = page * (CB * 584) + within * 576
    raw = tl.load(Wire + base + dim, valid & (dim < 448), other=0)
    exponent = tl.load(
        Wire + page * (CB * 584) + CB * 576 + within * 8 + dim // 64,
        valid & (dim < 448),
        other=127,
    ).to(tl.uint32)
    # exp2(integer) exactly, including 2**-127 and overflow at code 255.
    bits = tl.where(exponent == 0, 0x00400000, exponent << 23)
    scale = bits.to(tl.float32, bitcast=True)
    nope = raw.to(tl.float8e4nv, bitcast=True).to(tl.float32) * scale
    rope_ptr = (Wire + base + 448).to(tl.pointer_type(tl.bfloat16))
    rope = tl.load(rope_ptr + dim - 448, valid & (dim >= 448), other=0)
    return tl.where(dim < 448, nope, rope.to(tl.float32))


@triton.jit
def _main(Wire, Values, Scales, N, CB: tl.constexpr):
    groups = tl.program_id(0) * 128 + tl.arange(0, 128)
    row = groups // 32
    dim = (groups % 32)[:, None] * 16 + tl.arange(0, 16)[None, :]
    x = _wire_row(Wire, row[:, None], dim, (row < N)[:, None], CB)
    amax = tl.maximum(tl.max(tl.abs(x), 1), 6.0 * 2.0**-9)
    scale = tl.div_rn(amax, 6.0).to(tl.float8e4nv).to(tl.float32)
    normalized = tl.minimum(tl.maximum(tl.div_rn(x, scale[:, None]), -6.0), 6.0)
    a = tl.abs(normalized)
    code = (a > 0.25).to(tl.uint8)
    code += (a >= 0.75).to(tl.uint8)
    code += (a > 1.25).to(tl.uint8)
    code += (a >= 1.75).to(tl.uint8)
    code += (a > 2.5).to(tl.uint8)
    code += (a >= 3.5).to(tl.uint8)
    code += (a > 5.0).to(tl.uint8)
    sign = (normalized.to(tl.uint32, bitcast=True) >> 31).to(tl.uint8)
    pairs = tl.reshape(code | (sign << 3), (128, 8, 2))
    lo, hi = tl.split(pairs)
    out = groups[:, None] * 8 + tl.arange(0, 8)[None, :]
    tl.store(Values + out, lo | (hi << 4), (row < N)[:, None])
    tl.store(
        Scales + groups, scale.to(tl.float8e4nv).to(tl.uint8, bitcast=True), row < N
    )


@triton.jit
def _index(Wire, Values, Scales, N, CB: tl.constexpr):
    row = tl.program_id(0) * 16 + tl.arange(0, 16)
    b = tl.arange(0, 64)
    within = row % CB
    page = row // CB * (CB * 68)
    voff = (page + within // 32 * 2048)[:, None] + b[None, :] // 16 * 512
    voff += (within % 32)[:, None] * 16 + b[None, :] % 16
    value = tl.load(Wire + voff, (row < N)[:, None], other=0)
    tl.store(Values + row[:, None] * 64 + b[None, :], value, (row < N)[:, None])
    s = tl.arange(0, 4)
    soff = (page + CB * 64 + within // 32 * 128)[:, None] + s[None, :] % 2 * 64
    soff += (within % 32)[:, None] * 2 + s[None, :] // 2
    scale = tl.load(Wire + soff, (row < N)[:, None], other=0)
    tl.store(Scales + row[:, None] * 4 + s[None, :], scale, (row < N)[:, None])


@triton.jit
def _window(Wire, Pointers, N, NWIN, FIRST, LAYER_BYTES, CB: tl.constexpr):
    layer = tl.program_id(1)
    groups = tl.program_id(0) * 64 + tl.arange(0, 64)
    slot = groups // 16
    pos = N - 1 - ((N - 1 - slot) & 255)
    valid = (layer < NWIN) & (pos >= tl.maximum(0, N - 128)) & (pos < N)
    dim = (groups % 16)[:, None] * 32 + tl.arange(0, 32)[None, :]
    x = _wire_row(
        Wire + layer * LAYER_BYTES, (pos - FIRST * CB)[:, None], dim, valid[:, None], CB
    )
    # Reference to_window() casts the dequantized wire to BF16 first.
    x = x.to(tl.bfloat16).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(x), 1), 0.0001)
    unrounded = amax * 0.0022321429569274187  # reference float32(1 / 448)
    bits = unrounded.to(tl.uint32, bitcast=True)
    exponent = (bits >> 23) + ((bits & 0x7FFFFF) != 0).to(tl.uint32)
    scale = (exponent << 23).to(tl.float32, bitcast=True)
    q = tl.minimum(tl.maximum(tl.div_rn(x, scale[:, None]), -448.0), 448.0)
    value = q.to(tl.float8e4nv).to(tl.uint8, bitcast=True)
    dst_v = tl.load(Pointers + layer * 2).to(tl.pointer_type(tl.uint8))
    dst_s = tl.load(Pointers + layer * 2 + 1).to(tl.pointer_type(tl.uint8))
    tl.store(dst_v + slot[:, None] * 512 + dim, tl.where(valid[:, None], value, 0))
    tl.store(dst_s + groups, tl.where(valid, exponent, 0).to(tl.uint8))


@triton.jit
def _state_hist(
    Wire, Pointers, TokenMap, Hist, N, RING_ROWS, NTOKEN, RING_OFFSET, LAYER_BYTES
):
    pid = tl.program_id(0)
    if pid < 48:
        layer = pid // 16
        e = (pid % 16) * 1024 + tl.arange(0, 1024)
        active = ((N & 1) != 0) & (e // 1024 == ((N - 1) & 15))
        source_row = (N - 1) % tl.maximum(RING_ROWS, 1)
        src = (Wire + RING_OFFSET + layer * LAYER_BYTES).to(tl.pointer_type(tl.float32))
        x = tl.load(src + source_row * 1024 + e % 1024, active, other=0)
        empty = tl.where(e % 1024 < 512, 0.0, float("-inf"))
        dst = tl.load(Pointers + layer).to(tl.pointer_type(tl.float32))
        tl.store(dst + e, tl.where(active, x, empty))
    else:
        slot = tl.arange(0, 16)
        k = tl.minimum(N, 16)
        pos = N - 1 - ((N - 1 - slot) & 15)
        valid = pos >= N - k
        tail = (Wire + 32 * 4).to(tl.pointer_type(tl.int32))
        token = tl.load(tail + pos - (N - k), valid, other=0)
        token = tl.minimum(tl.maximum(token, 0), NTOKEN - 1)
        value = tl.load(TokenMap + token, valid, other=0)
        tl.store(Hist + slot, value.to(tl.int32))


class Converter:
    """Own constant device pointer tables; enqueue into caller-owned cache/stream."""

    def __init__(self, tensors: dict, token_map: torch.Tensor, max_seq_len: int):
        self.tensors = tensors
        self.device = tensors["temp:hist"].device
        self.regions = W.regions(max_seq_len)
        self.token_map = token_map.to(device=self.device, dtype=torch.int64)
        self.windows = torch.tensor(
            [
                tensors[f"cache:win.{i}.{name}"].data_ptr()
                for i in range(C.NUM_WIN)
                for name in ("kv", "s")
            ],
            dtype=torch.int64,
            device=self.device,
        )
        self.states = torch.tensor(
            [tensors[f"cache:state.{s}"].data_ptr() for s in W.R2],
            dtype=torch.int64,
            device=self.device,
        )

    @torch.inference_mode()
    def import_wire(self, buffer: torch.Tensor, seq_len: int) -> None:
        """The receive slot must remain claimed until the caller synchronizes."""
        h = W.parse_header(buffer)
        n = h["n"]
        if n != seq_len or n < 0:
            raise ValueError(f"KV header length {n} != admitted length {seq_len}")
        if not 0 <= h["nwin"] <= C.NUM_WIN:
            raise ValueError(f"Invalid KV window layers: {h['nwin']}")
        if len(h["tail"]) != min(n, C.HIST):
            raise ValueError("Incomplete KV token history")
        if not 0 <= h["ring_rows"] <= W.RING_ROWS_MAX or (
            n % 2 and h["ring_rows"] == 0
        ):
            raise ValueError("Invalid KV compressor ring rows")
        t = self.tensors
        # Validate every region before the first direct cache write.
        for pl in W.PAGE_PLANES:
            cb = h["cb"][pl]
            ne = n // W.PLANE_RATIO[pl]
            region = self.regions[pl]
            if (
                cb <= 0
                or h["pages"][pl] * cb < ne
                or h["pages"][pl] * cb * W.PLANE_REC[pl] > region.layer_bytes
            ):
                raise ValueError(f"Invalid KV pages for {pl}")
            index = pl.startswith("index")
            if index and cb % 32:
                raise ValueError("Index page row count must be a multiple of 32")
            name = "index_k" if index else "main_kv"
            for i, source in enumerate(W.PLANE_LAYERS[pl]):
                if ne > t[f"cache:{name}.{source}.0"].shape[0]:
                    raise ValueError("KV cache capacity exceeded")
                if (
                    region.offset
                    + i * region.layer_bytes
                    + h["pages"][pl] * cb * W.PLANE_REC[pl]
                    > buffer.numel()
                ):
                    raise ValueError("Incomplete KV wire buffer")
        cb, first = h["win_cb"], h["win_first"]
        lo = max(0, n - C.WINDOW)
        if (
            cb <= 0
            or first < 0
            or lo < first * cb
            or n - first * cb > h["win_blocks"] * cb
        ):
            raise ValueError("Invalid KV window page range")
        region = self.regions["swa"]
        if h["win_blocks"] * cb * W.V4_REC > region.layer_bytes:
            raise ValueError("KV window region capacity exceeded")
        for i in range(h["nwin"]):
            if (
                region.offset + i * region.layer_bytes + h["win_blocks"] * cb * W.V4_REC
                > buffer.numel()
            ):
                raise ValueError("Incomplete KV window buffer")
        ring = self.regions["ring"]
        if (
            n % 2
            and ring.offset + 2 * ring.layer_bytes + h["ring_rows"] * W.RING_ROW_BYTES
            > buffer.numel()
        ):
            raise ValueError("Incomplete KV compressor ring buffer")
        for pl in W.PAGE_PLANES:
            page_cb, ne = h["cb"][pl], n // W.PLANE_RATIO[pl]
            source_region = self.regions[pl]
            index = pl.startswith("index")
            kernel, name = (_index, "index_k") if index else (_main, "main_kv")
            for i, source in enumerate(W.PLANE_LAYERS[pl]):
                if ne:
                    kernel[(triton.cdiv(ne, 16 if index else 4),)](
                        buffer[source_region.offset + i * source_region.layer_bytes :],
                        t[f"cache:{name}.{source}.0"],
                        t[f"cache:{name}.{source}.1"],
                        ne,
                        page_cb,
                        num_warps=4,
                        enable_fp_fusion=False,
                    )
        _window[(64, C.NUM_WIN)](
            buffer[region.offset :],
            self.windows,
            n,
            h["nwin"],
            first,
            region.layer_bytes,
            cb,
            num_warps=4,
            enable_fp_fusion=False,
        )
        region = self.regions["ring"]
        _state_hist[(49,)](
            buffer,
            self.states,
            self.token_map,
            t["temp:hist"],
            n,
            h["ring_rows"],
            self.token_map.numel(),
            region.offset,
            region.layer_bytes,
            num_warps=4,
            enable_fp_fusion=False,
        )


def warm(device) -> None:
    """Compile and exercise the six variants in small temporary buffers."""
    n = 1024
    buffer = torch.zeros(W.wire_bytes(n), dtype=torch.uint8, device=device)
    header = torch.zeros(W.HDR_WORDS, dtype=torch.int32)
    header[W.H_MAGIC], header[W.H_VER], header[W.H_N] = W.WIRE_MAGIC, W.WIRE_VERSION, n
    header[W.H_RING_ROWS], header[W.H_NWIN] = 16, C.NUM_WIN
    header[W.H_WIN_FIRST], header[W.H_WIN_BLOCKS] = (
        C.window_first_block(n),
        W.WIN_BLOCKS,
    )
    for i, pl in enumerate(W.PAGE_PLANES):
        cb = W.BLOCK // W.PLANE_RATIO[pl]
        header[W.H_CB + i], header[W.H_PAGES + i] = cb, triton.cdiv(
            n // W.PLANE_RATIO[pl], cb
        )
    header[W.H_CB + 4], header[W.H_NTAIL] = W.SWA_CB, C.HIST
    header[W.H_TAIL : W.H_TAIL + C.HIST] = torch.arange(C.HIST, dtype=torch.int32)
    buffer[: W.HDR_BYTES].copy_(header.view(torch.uint8))
    from ..tk import alloc

    tensors = {
        spec.name: torch.empty(
            spec.shape, dtype=getattr(torch, spec.dtype), device=device
        )
        for spec in alloc.cache_specs(n)
    }
    tensors["temp:hist"] = torch.empty(C.HIST, dtype=torch.int32, device=device)
    converter = Converter(tensors, torch.arange(C.HIST, dtype=torch.int64), n)
    converter.import_wire(buffer, n)
    torch.cuda.synchronize(device)
