# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""ATOM_PROLOGUE_PACKED_H2D=1 (default off; needs ATOM_HOSTNEXT_FUSED_PROLOGUE): the decode
metadata build's host->device uploads in ONE copy + ONE scatter launch.

With the fused prologue (ATOM_HOSTNEXT_EARLY_META + ATOM_HOSTNEXT_FUSED_PROLOGUE) no GPU work
between the start of the GDN decode build and `early_prologue` reads the build's uploads: the
MHA builder defers its early fix and kv_indices, the GDN builder its accepted counts, pads and
commit, all into the prologue launch. The build still issues its uploads one by one as it goes
(block tables: a packed H2D + scatter; M-RoPE positions, the GDN accepted counts and the two GDN
state-index rows: one hipMemcpyAsync each -- six ops a step). Here, inside that window, each
upload's source bytes are copied on the host into a pinned arena (so a source may change after
the call, as with the direct copies once enqueued) together with its destination, and `flush`,
right before the prologue launch, uploads the arena once and scatters every member to its
destination in one kernel (`_scatter_arena_kernel`). The destinations receive the bytes the
direct copies would have written; the first reader (the prologue) runs after the scatter.

Members whose destinations overlap are not reordered: such a window falls back to one copy a
member, in order, from the snapshots.
"""

from __future__ import annotations

import os

import numpy as np
import torch
import triton
import triton.language as tl

_ENABLED = os.environ.get("ATOM_PROLOGUE_PACKED_H2D", "0") == "1"
_CHECK = os.environ.get("ATOM_PROLOGUE_PACKED_H2D_CHECK", "0") == "1"
_RING = 8  # pinned arenas in rotation (the host runs at most ~1 step ahead; see spec_rejection)
_ALIGN = 16


def enabled() -> bool:
    return _ENABLED


@triton.jit
def _scatter_arena_kernel(arena, BLOCK: tl.constexpr):
    # member m: header int64 [offset, length, destination] at arena + 24 * m
    m = tl.program_id(0)
    header = arena.to(tl.pointer_type(tl.int64))
    start = tl.load(header + 3 * m)
    length = tl.load(header + 3 * m + 1)
    dst = tl.load(header + 3 * m + 2).to(tl.pointer_type(tl.uint8))
    offset = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(arena + start + offset, offset < length, other=0)
    tl.store(dst + offset, values, offset < length)


class _State:
    def __init__(self):
        self.active = False
        self.device = None
        self.host = []  # pinned uint8 arenas (ring)
        self.next = 0
        self.dev = None  # device uint8 arena
        self.members = []  # (dst tensor, src snapshot offset, nbytes)
        self.buf = None  # this window's pinned arena (numpy view)
        self.buf_t = None
        self.used = 0
        self.kernel = None
        self.windows = 0


_S = _State()


def active() -> bool:
    return _S.active


def _arena(nbytes: int):
    """A pinned ring slot of at least `nbytes` (grown slots replace their predecessor)."""
    if len(_S.host) < _RING:
        t = torch.empty(max(nbytes, 1 << 20), dtype=torch.uint8).pin_memory()
        _S.host.append(t)
        return t
    i = _S.next
    _S.next = (i + 1) % _RING
    t = _S.host[i]
    if t.numel() < nbytes:
        t = torch.empty(max(nbytes, 2 * t.numel()), dtype=torch.uint8).pin_memory()
        _S.host[i] = t
    return t


def begin(device) -> None:
    """Open a window: uploads go to the arena until `flush`."""
    assert not _S.active
    _S.active = True
    _S.device = device
    _S.members = []
    _S.used = 0
    _S.windows += 1


def _reserve(nbytes: int) -> int:
    """Room for `nbytes` in the host scratch (pageable, host-only: the snapshots are copied
    into a pinned ring slot at `flush`)."""
    off = _S.used
    end = off + (nbytes + _ALIGN - 1) // _ALIGN * _ALIGN
    if _S.buf_t is None or end > _S.buf_t.numel():
        old = _S.buf_t
        t = torch.empty(max(4 * end, 1 << 16), dtype=torch.uint8)
        if old is not None and off:
            t[:off].copy_(old[:off])
        _S.buf_t = t
    _S.used = end
    return off


def add(dst: torch.Tensor, src: torch.Tensor) -> None:
    """Upload `src` (host) into `dst` (device, contiguous, same byte size) at `flush`."""
    assert _S.active
    assert dst.is_contiguous() and src.is_contiguous() and src.device.type == "cpu"
    nbytes = src.numel() * src.element_size()
    assert nbytes == dst.numel() * dst.element_size(), (tuple(src.shape), tuple(dst.shape))
    if nbytes == 0:
        return
    off = _reserve(nbytes)
    _S.buf_t[off : off + nbytes].copy_(src.view(-1).view(torch.uint8))
    _S.members.append((dst, off, nbytes))


def flush() -> None:
    """Close the window: one H2D of header + payload, one scatter launch."""
    assert _S.active
    _S.active = False
    members = _S.members
    _S.members = []
    if not members:
        return
    n = len(members)
    hdr = (24 * n + _ALIGN - 1) // _ALIGN * _ALIGN
    payload = _S.used
    total = hdr + payload
    # overlapping destinations: keep the order (one copy a member from the snapshots)
    spans = sorted((d.data_ptr(), d.data_ptr() + nb) for d, _, nb in members)
    overlap = any(b0 < a1 for (_, a1), (b0, _) in zip(spans, spans[1:]))
    t = _arena(total)
    t[hdr:total].copy_(_S.buf_t[:payload])
    if overlap:
        for d, off, nb in members:
            d.view(-1).view(torch.uint8).copy_(t[hdr + off : hdr + off + nb], non_blocking=True)
        return
    h = t[: 24 * n].view(torch.int64).numpy()
    largest = 0
    for i, (d, off, nb) in enumerate(members):
        h[3 * i] = hdr + off
        h[3 * i + 1] = nb
        h[3 * i + 2] = d.data_ptr()
        largest = max(largest, nb)
    if _S.dev is None or _S.dev.numel() < total:
        _S.dev = torch.empty(max(total, 1 << 20), dtype=torch.uint8, device=_S.device)
    _S.dev[:total].copy_(t[:total], non_blocking=True)
    grid = (n, (largest + 1023) // 1024, 1)
    _scatter_arena_kernel[grid](_S.dev, BLOCK=1024)
    if _CHECK:
        torch.cuda.synchronize()
        for d, off, nb in members:
            got = d.view(-1).view(torch.uint8).cpu()
            want = t[hdr + off : hdr + off + nb]
            if not torch.equal(got, want):
                raise RuntimeError(f"ATOM_PROLOGUE_PACKED_H2D check: member of {nb} B differs")
        _S.checked = getattr(_S, "checked", 0) + n
