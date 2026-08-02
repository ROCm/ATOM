# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""One request's attention state as a contiguous byte range.

A stateful attention type keeps several tensors per request — DeepSeek-V4's
compressor keeps a `kv_state`/`score_state` pair for each of its three
compressor flavors, GDN keeps a recurrent k and v. The natural way to write
that down is one tensor per family, layer outermost and the request slot
inside: `[layers, entries, ...]`. Every kernel then binds one layer's slice
and indexes it by slot.

That layout spreads a single request's state across as many disjoint
allocations as there are families, which is fine as long as nothing ever
needs the state *as a whole*. Three things do:

  - saving it as a prefix-cache checkpoint, which wants one `copy_`;
  - relocating it when the pool boundary moves, which needs an entry to be
    the unit of movement;
  - shipping it over RDMA, which wants one registered range per entry.

`StateArena` keeps the same per-layer views the kernels already take, but
backs them with one allocation laid out entry-major: entry `i` owns
`buf[i * entry_bytes : (i + 1) * entry_bytes]`, and inside it each field is
laid out layer-major. So a per-layer view is the same shape as before with a
larger slot stride, and an entry is a contiguous slice.

Backends stay in charge of what the fields are; this module only owns the
arithmetic. The layout is deliberately the one DeepSeek-V4's PD staging path
already builds by hand on every transfer (`_make_gather_slot`) — making it
physical is what lets that gather collapse into a copy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

# Field offsets inside an entry, and `entry_bytes` itself, are rounded up to
# this. 256 B is the torch caching allocator's own granularity and a multiple
# of every element size in play, so a field pointer is always safe for the
# widest vector load a kernel might use. Real state shapes are already much
# coarser than this, so the rounding is normally free.
_ALIGN = 256


def _round_up(n: int, to: int = _ALIGN) -> int:
    return -(-n // to) * to


@dataclass(frozen=True)
class StateField:
    """One tensor family inside an entry.

    `shape` is what ONE (layer, entry) pair holds — the same trailing shape
    the backend passes today, without the leading layer and slot dims.
    """

    name: str
    layers: int
    shape: tuple[int, ...]
    dtype: torch.dtype
    # Value the field is initialized to. Score states start at -inf so an
    # unwritten ring position loses the softmax; kv states start at zero.
    fill: float = 0.0

    @property
    def per_layer_numel(self) -> int:
        return math.prod(self.shape)

    @property
    def bytes_per_entry(self) -> int:
        """Bytes this field occupies in one entry, across all its layers."""
        return self.layers * self.per_layer_numel * self.dtype.itemsize


def entry_bytes_for(fields: list[StateField]) -> int:
    """Bytes one entry costs, including inter-field alignment.

    Sizing calls this before any GPU allocation exists, so it is a free
    function rather than a property of a built arena — the byte budget and
    the allocation must come from the same expression or the two drift.
    """
    total = 0
    for field in fields:
        total = _round_up(total) + field.bytes_per_entry
    return _round_up(total)


class StateArena:
    """`entries` fixed-size state entries in one allocation.

    Exposes the per-layer views kernels expect (`view(name)` →
    `[layers, entries, *shape]`) and the whole-entry byte range that
    checkpointing, relocation and RDMA need (`entry(i)`).
    """

    def __init__(
        self,
        fields: list[StateField],
        entries: int,
        device,
    ):
        if not fields:
            raise ValueError("a state arena needs at least one field")
        names = [f.name for f in fields]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate field names: {names}")

        self.fields = list(fields)
        self.entries = entries
        self.entry_bytes = entry_bytes_for(fields)

        offset = 0
        self._offsets: dict[str, int] = {}
        for field in self.fields:
            offset = _round_up(offset)
            self._offsets[field.name] = offset
            offset += field.bytes_per_entry

        self._by_name = {f.name: f for f in self.fields}
        # Zeroed, not `empty`: alignment padding falls outside every field
        # view, and an entry is copied whole by checkpointing and RDMA, so
        # uninitialized padding would travel. One memset at startup.
        self.buf = torch.zeros(
            entries * self.entry_bytes, dtype=torch.uint8, device=device
        )
        for field in self.fields:
            self.view(field.name).fill_(field.fill)

    @property
    def total_bytes(self) -> int:
        return self.entries * self.entry_bytes

    def view(self, name: str) -> torch.Tensor:
        """`[layers, entries, *shape]` — a drop-in for the standalone tensor.

        Only the slot stride differs from a standalone allocation: it is the
        whole entry rather than this field alone. Kernels that take the slot
        stride as an argument (both V4 compressor kernels do) are unaffected;
        one that assumes contiguity is not, and has to be checked.
        """
        field = self._by_name[name]
        itemsize = field.dtype.itemsize
        # `buf` starts at storage offset 0, so byte offsets convert to element
        # offsets by plain division. `_ALIGN` is a multiple of every itemsize,
        # which is what makes that division exact.
        typed = self.buf.view(field.dtype)
        inner: tuple[int, ...] = ()
        acc = 1
        for dim in reversed(field.shape):
            inner = (acc,) + inner
            acc *= dim
        return typed.as_strided(
            (field.layers, self.entries) + field.shape,
            (field.per_layer_numel, self.entry_bytes // itemsize) + inner,
            self._offsets[field.name] // itemsize,
        )

    def entry(self, index: int) -> torch.Tensor:
        """One entry's whole state as a contiguous 1-D uint8 slice."""
        start = index * self.entry_bytes
        return self.buf[start : start + self.entry_bytes]

    def field_offset(self, name: str) -> int:
        """Byte offset of a field from the start of an entry."""
        return self._offsets[name]
