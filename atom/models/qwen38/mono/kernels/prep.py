# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The small launches around a Mono step's layer programs (Triton, graph-capturable).

``VerifyPrep``: before a target verify step, one launch writes the embedding rows into the first layer's (registered)
input buffer, repeated into every residual stream, and the step's derived GDN metadata as contiguous int32 buffers.
Per request ``b`` (``idx = spec_state_indices [batch, spec]``, ``a = num_accepted_tokens``): ``valid[b] = idx[b, 0] >= 0
and slot_mapping[b * spec] >= 0`` (a capture's rows carry PAD_SLOT_ID: nothing written), ``prev[b] = max(a[b] - 1,
0)``, ``slot[b] = idx[b, 0]``, ``ssm_read[b] = idx_in[b, prev[b]]``, ``rope_off[b] = positions[b * spec] -
(context_lens[b] - spec)`` (ATOM's RoPE position of the request's first row minus its cache row: an image request's
MRoPE offset, 0 for text and for an invalid request). Per row ``r``: ``out[r, s * H + c] = embed[ids[r], c]`` for every
stream ``s`` (the rows of ``embed(ids).repeat(1, streams)``, bit for bit).

``fused_puts``: the target step's metadata args (``step_meta.target_puts``) and every kind's step counter in one launch.
A put is ``dst[i] = (valid[i // group] if masked else True) ? src[i // inner * stride + i % inner] - sub : fill`` for
``i < n``: ``dst`` an int32 buffer, ``src`` int32 or int64 (rounded to int32 as ``copy_`` does), ``valid`` the step's
``[batch]`` bool.

``draft_meta``: the MTP drafter's step metadata and step counter in one launch.
"""

import torch
import triton
import triton.language as tl

from atom.models.qwen38.mono.step_meta import Put


@triton.jit
def _prep_kernel(
    ids,
    emb,
    emb_stride,
    out,
    out_stride,
    acc,
    idx,
    idx_stride,
    idx_in,
    idx_in_stride,
    valid,
    prev,
    slot,
    ssm_read,
    slots,
    pos,
    ctx,
    rope_off,
    spec,
    rows,
    batch,
    H: tl.constexpr,
    STREAMS: tl.constexpr,
    BLOCK: tl.constexpr,
    BB: tl.constexpr,
):
    r = tl.program_id(0)
    if r < rows:
        tok = tl.load(ids + r).to(tl.int64)
        c = tl.arange(0, BLOCK)
        for c0 in range(0, H, BLOCK):
            m = c0 + c < H
            v = tl.load(emb + tok * emb_stride + c0 + c, mask=m)
            for s in tl.static_range(STREAMS):
                tl.store(out + r * out_stride + s * H + c0 + c, v, mask=m)
    else:
        b = tl.arange(0, BB)
        m = b < batch
        a = tl.load(acc + b, mask=m, other=1)
        p = tl.maximum(a - 1, 0)
        s0 = tl.load(idx + b * idx_stride, mask=m, other=-1)
        sr = tl.load(idx_in + b * idx_in_stride + p, mask=m, other=-1)
        sm = tl.load(slots + b * spec, mask=m, other=-1)
        ok = (s0 >= 0) & (sm >= 0)
        # ATOM's position of the request's first row minus its cache row (context - spec)
        p0 = tl.load(pos + b * spec, mask=m, other=0)
        cl = tl.load(ctx + b, mask=m, other=0)
        off = tl.where(ok, p0 - (cl.to(tl.int64) - spec), 0)
        tl.store(rope_off + b, off.to(tl.int32), mask=m)
        tl.store(valid + b, ok, mask=m)
        tl.store(prev + b, p.to(tl.int32), mask=m)
        tl.store(slot + b, s0.to(tl.int32), mask=m)
        tl.store(ssm_read + b, sr.to(tl.int32), mask=m)


class VerifyPrep:
    """The persistent outputs (graph-stable addresses) for up to ``max_batch`` requests."""

    def __init__(self, max_batch: int, device="cuda"):
        i32 = torch.int32
        self.valid = torch.zeros(max_batch, dtype=torch.bool, device=device)
        self.prev = torch.zeros(max_batch, dtype=i32, device=device)
        self.slot = torch.zeros(max_batch, dtype=i32, device=device)
        self.ssm_read = torch.zeros(max_batch, dtype=i32, device=device)
        self.rope_off = torch.zeros(max_batch, dtype=i32, device=device)

    def __call__(
        self,
        ids,
        emb,
        out,
        accepted,
        idx,
        idx_in,
        batch: int,
        streams: int,
        slot_mapping,
        spec: int,
        positions,
        context_lens,
    ):
        rows, h = ids.numel(), emb.shape[1]
        # (MRoPE positions are [3, N]: a text row's three are equal, and the step's rows are text)
        pos = positions[0] if positions.dim() == 2 else positions
        assert pos.stride(-1) == 1
        assert out.stride(1) == 1 and emb.stride(1) == 1 and out.shape[1] == streams * h
        assert (
            idx.stride(1) == 1 and idx_in.stride(1) == 1 and batch <= self.valid.numel()
        )
        _prep_kernel[(rows + 1,)](
            ids, emb, emb.stride(0), out, out.stride(0), accepted, idx, idx.stride(0), idx_in, idx_in.stride(0),
            self.valid, self.prev, self.slot, self.ssm_read, slot_mapping, pos, context_lens, self.rope_off,
            spec, rows, batch,
            H=h, STREAMS=streams, BLOCK=1024, BB=max(2, triton.next_power_of_2(batch)),
        )  # fmt: skip
        return (
            self.valid[:batch],
            self.prev[:batch],
            self.slot[:batch],
            self.ssm_read[:batch],
            self.rope_off[:batch],
        )


MAX_PUTS = 12  # put slots of the kernel (the target step fills 6)
MAX_BUMPS = 4  # step counters bumped by the same launch
BLOCK = 1024


@triton.jit
def _put(
    d, s, n, group, inner, stride, sub, fill, masked, valid, chunk, BLOCK: tl.constexpr
):
    i = chunk * BLOCK + tl.arange(0, BLOCK)
    live = i < n
    src = tl.load(
        s + (i // inner).to(tl.int64) * stride + i % inner, mask=live, other=0
    )
    v = src.to(tl.int64) - sub
    ok = tl.load(valid + i // group, mask=live & (masked != 0), other=1) != 0
    tl.store(d + i, tl.where(ok, v, fill).to(tl.int32), mask=live)


_INTS = [
    f"{k}{j}"
    for j in range(MAX_PUTS)
    for k in ("n", "g", "in", "st", "sub", "fill", "m")
]


# fmt: off
@triton.jit(do_not_specialize=_INTS)
def _fused_puts_kernel(
    valid,
    c0,
    c1,
    c2,
    c3,
    d0, s0, n0, g0, in0, st0, sub0, fill0, m0,
    d1, s1, n1, g1, in1, st1, sub1, fill1, m1,
    d2, s2, n2, g2, in2, st2, sub2, fill2, m2,
    d3, s3, n3, g3, in3, st3, sub3, fill3, m3,
    d4, s4, n4, g4, in4, st4, sub4, fill4, m4,
    d5, s5, n5, g5, in5, st5, sub5, fill5, m5,
    d6, s6, n6, g6, in6, st6, sub6, fill6, m6,
    d7, s7, n7, g7, in7, st7, sub7, fill7, m7,
    d8, s8, n8, g8, in8, st8, sub8, fill8, m8,
    d9, s9, n9, g9, in9, st9, sub9, fill9, m9,
    d10, s10, n10, g10, in10, st10, sub10, fill10, m10,
    d11, s11, n11, g11, in11, st11, sub11, fill11, m11,
    NPUTS: tl.constexpr,
    NBUMPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    p = tl.program_id(0)
    chunk = tl.program_id(1)
    if p == NPUTS:  # the last program bumps the step counters
        if chunk == 0:
            if NBUMPS > 0:
                tl.store(c0, tl.load(c0) + 1)
            if NBUMPS > 1:
                tl.store(c1, tl.load(c1) + 1)
            if NBUMPS > 2:
                tl.store(c2, tl.load(c2) + 1)
            if NBUMPS > 3:
                tl.store(c3, tl.load(c3) + 1)
    elif p == 0:
        _put(d0, s0, n0, g0, in0, st0, sub0, fill0, m0, valid, chunk, BLOCK)
    elif p == 1:
        _put(d1, s1, n1, g1, in1, st1, sub1, fill1, m1, valid, chunk, BLOCK)
    elif p == 2:
        _put(d2, s2, n2, g2, in2, st2, sub2, fill2, m2, valid, chunk, BLOCK)
    elif p == 3:
        _put(d3, s3, n3, g3, in3, st3, sub3, fill3, m3, valid, chunk, BLOCK)
    elif p == 4:
        _put(d4, s4, n4, g4, in4, st4, sub4, fill4, m4, valid, chunk, BLOCK)
    elif p == 5:
        _put(d5, s5, n5, g5, in5, st5, sub5, fill5, m5, valid, chunk, BLOCK)
    elif p == 6:
        _put(d6, s6, n6, g6, in6, st6, sub6, fill6, m6, valid, chunk, BLOCK)
    elif p == 7:
        _put(d7, s7, n7, g7, in7, st7, sub7, fill7, m7, valid, chunk, BLOCK)
    elif p == 8:
        _put(d8, s8, n8, g8, in8, st8, sub8, fill8, m8, valid, chunk, BLOCK)
    elif p == 9:
        _put(d9, s9, n9, g9, in9, st9, sub9, fill9, m9, valid, chunk, BLOCK)
    elif p == 10:
        _put(d10, s10, n10, g10, in10, st10, sub10, fill10, m10, valid, chunk, BLOCK)
    elif p == 11:
        _put(d11, s11, n11, g11, in11, st11, sub11, fill11, m11, valid, chunk, BLOCK)
# fmt: on


def _args(put: Put) -> list:
    dst, src = put.dst, put.src
    if dst.dtype != torch.int32 or not dst.is_contiguous():
        raise ValueError(
            f"a put's destination must be a contiguous int32 buffer, not {dst.dtype} {dst.stride()}"
        )
    if src.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"a put's source must be int32 or int64, not {src.dtype}")
    n = dst.numel()
    inner = put.inner if put.inner is not None else max(n, 1)
    if put.stride is not None:
        stride = put.stride
    else:
        if not src.is_contiguous():
            src = src.contiguous()
        stride = inner
    if put.inner is None and src.numel() < n:
        raise ValueError(f"a put's source has {src.numel()} elements for {n}")
    return [dst, src, n, put.group, inner, stride, put.sub, put.fill, int(put.masked)]


def fused_puts(
    valid: torch.Tensor, puts: list[Put], counters: list[torch.Tensor] = ()
) -> None:
    """Every put of ``puts`` and ``counter += 1`` for each of ``counters`` (int32 ``[1]``), one
    launch on the current stream (graph-capturable)."""
    if len(puts) > MAX_PUTS or len(counters) > MAX_BUMPS:
        raise ValueError(
            f"{len(puts)} puts / {len(counters)} counters: at most {MAX_PUTS} / {MAX_BUMPS}"
        )
    if valid.dtype != torch.bool or not valid.is_contiguous():
        raise ValueError("valid must be a contiguous bool tensor")
    flat = []
    for put in puts:
        flat += _args(put)
    dummy = puts[0].dst if puts else valid.new_zeros(1, dtype=torch.int32)
    # unused slots: n = 0
    flat += [dummy, dummy, 0, 1, 1, 1, 0, 0, 0] * (MAX_PUTS - len(puts))
    cs = list(counters) + [counters[0] if counters else dummy] * (
        MAX_BUMPS - len(counters)
    )
    chunks = max([triton.cdiv(p.dst.numel(), BLOCK) for p in puts] + [1])
    _fused_puts_kernel[(len(puts) + 1, chunks)](
        valid, *cs, *flat, NPUTS=len(puts), NBUMPS=len(counters), BLOCK=BLOCK
    )


@triton.jit
def _draft_meta_kernel(
    pos,
    slots,
    bt,
    bt_stride,
    sel_src,
    seq_lens,
    block_table,
    sel,
    counter,
    width,
    spec,
    HAS_SEL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    b = tl.program_id(0)
    c = tl.program_id(1)
    first = tl.load(pos + b * spec)
    ok = (first >= 0) & (tl.load(slots + b * spec) >= 0)
    j = c * BLOCK + tl.arange(0, BLOCK)
    m = j < width
    v = tl.load(bt + b * bt_stride + j, mask=m, other=-1)
    tl.store(block_table + b * width + j, tl.where(ok, v, -1).to(tl.int32), mask=m)
    if c == 0:
        tl.store(seq_lens + b, tl.where(ok, first, 0).to(tl.int32))
        if HAS_SEL:
            tl.store(sel + b, tl.load(sel_src + b).to(tl.int32))
        if b == 0:
            tl.store(counter, tl.load(counter) + 1)


def draft_meta(
    m: dict, counters: list, pos, slots, bt, select, batch: int, spec: int
) -> None:
    """The MTP drafter's step metadata into ``m`` (``runner.Kernels.begin_step_buffers``) and its step counter + 1, in
    one launch: ``seq_lens[b]`` = request ``b``'s first cache position, its block-table row, ``sel`` (the first step's
    selected rows); a request whose first row is padding (position or slot -1) gets 0 / -1, so the kernels touch no
    pool for it."""
    assert len(counters) == 1, "one program kind (mtp)"
    width = m["block_table"].shape[1]
    assert bt.shape[1] == width and bt.stride(1) == 1
    block = 1024
    _draft_meta_kernel[(batch, triton.cdiv(width, block))](
        pos, slots, bt, bt.stride(0), select if select is not None else pos, m["seq_lens"], m["block_table"], m["sel"],
        counters[0], width, spec, HAS_SEL=select is not None, BLOCK=block,
    )  # fmt: skip
