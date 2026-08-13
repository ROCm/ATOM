# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""One request's attention state as few byte ranges as the layout allows.

A stateful attention type keeps several tensors per request — DeepSeek-V4's
compressor keeps a `kv_state`/`score_state` pair for each of its three
compressor flavors, GDN keeps a recurrent k and v. The natural way to write
that down is one tensor per family, layer outermost and the request slot
inside: `[layers, entries, ...]`. Every kernel then binds one layer's slice
and indexes it by slot.

That layout spreads a single request's state across as many disjoint
allocations as there are families, which is fine as long as nothing ever
needs the state *as a whole*. Three things do:

  - saving it as a prefix-cache checkpoint, which wants one `copy_` per range;
  - relocating it when the pool boundary moves, which needs an entry to be
    the unit of movement;
  - shipping it over RDMA, which wants one registered range per entry.

`StateArena` keeps the same per-layer views the kernels already take, but
backs them with one allocation laid out entry-major: entry `i` starts at
`i * slot_stride`, and inside it each field is laid out layer-major. So a
per-layer view is the same shape as before with a larger slot stride, and an
entry is a contiguous slice.

The stride is the entry's own size when the arena owns its buffer. It is not
when the arena lives at the front of a slot in a shared plane — there the
entries are a slot apart and the space between them belongs to whatever else
the plane holds, so `live_entries` says which part of the index range is
really the caller's and nothing outside it is ever written.

A row space with planes of differing width cannot hold one entry contiguously
at all: a field is one strided tensor, so it lands in one plane or the other.
`plan_field_planes` decides which, `SplitStateArena` hides the split from
consumers asking for a field by name, and what stays contiguous is a *slot* —
which is the range a checkpoint copies and a PD transfer registers anyway.

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


def _align_up(n: int, to: int = _ALIGN) -> int:
    return -(-n // to) * to


def plan_regions(sizes: list[int]) -> tuple[list[int], int]:
    """Byte offsets for regions packed back to back in one allocation.

    Returns `(offsets, total)`. Both the offsets and `total` are `_ALIGN`-
    aligned, so `plan_regions(a) + plan_regions(b)` shifted by `a`'s total
    lays out exactly as `plan_regions(a + b)` would — plan groups separately
    and concatenate rather than slicing one flat result positionally. An
    empty list plans to `([], 0)`, so an absent group needs no special case.

    Lives beside the arena because `_ALIGN` does: whoever carves the arena out
    of a shared allocation has to place every other region on the boundary the
    arena's own fields assume.
    """
    offsets: list[int] = []
    offset = 0
    for nbytes in sizes:
        offset = _align_up(offset)
        offsets.append(offset)
        offset += nbytes
    return offsets, _align_up(offset)


def plan_field_planes(
    fields: list[StateField], plane_row_bytes: list[int]
) -> tuple[list[list[StateField]], int]:
    """Split fields across the planes of one row space, in the fewest rows.

    Every plane materializes the same rows at its own width, so a slot that
    reserves `r` rows offers `r * plane_row_bytes[p]` bytes in plane `p` — and
    the same `r` in all of them, because a row index has to mean one thing
    across planes. A field cannot straddle two, since its view is one strided
    tensor, so the question is which plane each one goes in.

    Returns `(per_plane_fields, rows)`. Enumerating every assignment rather
    than packing greedily: `2^len(fields)` is 64 for DeepSeek-V4 and this runs
    once at startup, so there is no reason to settle for a heuristic answer to
    a question with an exact one. Ties go to the first assignment found, which
    keeps a given field list mapping to the same layout every run.

    Field order inside a plane stays the declared order, which is the order a
    checkpoint and a PD transfer see the bytes in.
    """
    if not plane_row_bytes:
        raise ValueError("a row space needs at least one plane")
    num_planes = len(plane_row_bytes)
    assignments = num_planes ** len(fields)
    if assignments > 1 << 16:
        raise ValueError(
            f"refusing to enumerate {assignments} assignments of "
            f"{len(fields)} fields over {num_planes} planes"
        )

    best: tuple[list[list[StateField]], int] | None = None
    for code in range(assignments):
        groups: list[list[StateField]] = [[] for _ in plane_row_bytes]
        rest = code
        for field in fields:
            groups[rest % num_planes].append(field)
            rest //= num_planes
        rows = max(
            -(-entry_bytes_for(group) // row_bytes)
            for group, row_bytes in zip(groups, plane_row_bytes)
        )
        if best is None or rows < best[1]:
            best = (groups, rows)
    assert best is not None
    return best


class SplitStateArena:
    """One request's state, spread over the planes of a row space.

    A row space materializes the same rows at several widths, and a field is
    one strided tensor so it cannot straddle two of them — see
    `plan_field_planes`. Consumers still want to ask for a field by name
    without knowing which plane it landed in, which is all this is.

    There is deliberately no whole-entry accessor. When the state shares a slot
    with that request's windows, the range worth copying is the slot, and the
    caller who knows the geometry takes it from the plane directly.
    """

    def __init__(self, arenas: list[StateArena]):
        if not arenas:
            raise ValueError("a split arena needs at least one plane")
        self.arenas = list(arenas)
        self._by_field: dict[str, StateArena] = {}
        for arena in self.arenas:
            for field in arena.fields:
                if field.name in self._by_field:
                    raise ValueError(f"field {field.name!r} is in two planes")
                self._by_field[field.name] = arena

    @property
    def entry_bytes(self) -> int:
        """Bytes one request's state takes, summed over the planes."""
        return sum(a.entry_bytes for a in self.arenas)

    def view(self, name: str) -> torch.Tensor:
        return self._by_field[name].view(name)

    def field_offset(self, name: str) -> int:
        """Bytes into the plane's slot where field `name` begins."""
        return self._by_field[name].field_offset(name)


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
    # Byte alignment the field's offset inside an entry must satisfy, over and
    # above `_ALIGN`. A field read as a plain strided tensor needs nothing more
    # than the retype boundary; one also read as rows of a *retyped plane* — a
    # window whose row is several plane rows wide — needs its offset to land on
    # one of those wider rows, or the row index the kernel computes is off by a
    # fraction of a row and nothing about the view says so.
    align: int = 0

    def __post_init__(self):
        if self.align and self.align % _ALIGN:
            raise ValueError(
                f"{self.name}: alignment {self.align} must be a multiple of "
                f"{_ALIGN}, which every field already has"
            )

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
        total = _align_up(total, max(_ALIGN, field.align)) + field.bytes_per_entry
    return _align_up(total)


class StateArena:
    """`entries` fixed-size state entries, one stride apart.

    Exposes the per-layer views kernels expect (`view(name)` →
    `[layers, entries, *shape]`) and the whole-entry byte range that
    checkpointing, relocation and RDMA need (`entry(i)`).

    The stride between entries is `entry_bytes` when the arena owns a buffer
    of its own, and whatever the caller says when it does not — an arena
    living at the front of a slot in a shared plane is strided by the slot,
    not by its own size.
    """

    def __init__(
        self,
        fields: list[StateField],
        entries: int,
        device,
        buf: torch.Tensor | None = None,
        slot_stride: int | None = None,
        live_entries: int | None = None,
    ):
        if not fields:
            raise ValueError("a state arena needs at least one field")
        names = [f.name for f in fields]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate field names: {names}")

        self.fields = list(fields)
        self.entries = entries
        self.entry_bytes = entry_bytes_for(fields)
        self.slot_stride = self.entry_bytes if slot_stride is None else slot_stride
        if self.slot_stride < self.entry_bytes:
            raise ValueError(
                f"slot stride {self.slot_stride} is under the "
                f"{self.entry_bytes} an entry takes"
            )
        # Every entry repeats the first entry's field offsets, so whatever
        # alignment the strictest field asks for has to survive the stride and
        # the buffer's own start as well — otherwise only entry 0 satisfies it.
        self._align = max([_ALIGN] + [f.align for f in self.fields])
        if self.slot_stride % self._align:
            raise ValueError(
                f"slot stride {self.slot_stride} must be a multiple of "
                f"{self._align}, or entries past the first fall off the boundary "
                "their field views retype from"
            )
        # Entries the caller will actually hand out, counted from the END:
        # a pool that grows takes the next index down, so the top of the range
        # is the part that is live. The rest is addressable but belongs to
        # whatever else shares the buffer, and must not be written here.
        self.live_entries = entries if live_entries is None else live_entries
        if not 0 <= self.live_entries <= entries:
            raise ValueError(f"live_entries {self.live_entries} outside 0..{entries}")

        offset = 0
        self._offsets: dict[str, int] = {}
        for field in self.fields:
            offset = _align_up(offset, max(_ALIGN, field.align))
            self._offsets[field.name] = offset
            offset += field.bytes_per_entry

        self._by_name = {f.name: f for f in self.fields}
        # Zeroed, not `empty`: alignment padding falls outside every field
        # view, and an entry is copied whole by checkpointing and RDMA, so
        # uninitialized padding would travel. One memset at startup.
        #
        # `buf` lets a caller carve the arena out of a larger allocation it also
        # carves the paged pools from, so the two are one contiguous region
        # whose internal boundary can move. It must already be zeroed for the
        # same reason. Owning the allocation stays the default — the tests and
        # any single-pool backend construct arenas standalone.
        want = (entries - 1) * self.slot_stride + self.entry_bytes if entries else 0
        if buf is None:
            self.buf = torch.zeros(want, dtype=torch.uint8, device=device)
        else:
            if buf.dtype is not torch.uint8 or buf.numel() < want:
                raise ValueError(
                    f"buf must hold at least {want} uint8 elements, got "
                    f"{buf.numel()} {buf.dtype}"
                )
            if not buf.is_contiguous():
                raise ValueError("buf must be contiguous")
            if buf.storage_offset() % self._align:
                raise ValueError(
                    f"buf must start on a {self._align}B boundary, got storage "
                    f"offset {buf.storage_offset()}: field views retype the "
                    "buffer, which needs the offset to divide every itemsize"
                )
            self.buf = buf
        for field in self.fields:
            self.view(field.name)[:, entries - self.live_entries :].fill_(field.fill)

    @property
    def total_bytes(self) -> int:
        """Bytes the entries span, from the first to the end of the last.

        Equal to `entries * entry_bytes` only when the arena owns its buffer;
        with a wider slot stride the gaps between entries belong to whoever
        else shares it, and this counts them.
        """
        return (self.entries - 1) * self.slot_stride + self.entry_bytes

    def view(self, name: str) -> torch.Tensor:
        """`[layers, entries, *shape]` — a drop-in for the standalone tensor.

        Only the slot stride differs from a standalone allocation: it is the
        whole entry rather than this field alone. Kernels that take the slot
        stride as an argument (both V4 compressor kernels do) are unaffected;
        one that assumes contiguity is not, and has to be checked.
        """
        field = self._by_name[name]
        itemsize = field.dtype.itemsize
        # `as_strided`'s storage_offset is ABSOLUTE, so `typed`'s own offset
        # has to be added: omit it and a carved arena addresses from the front
        # of the host allocation and writes through whatever precedes it. An
        # owned buffer sits at offset 0, which is what hides this.
        #
        # Byte offsets convert to element offsets by plain division: `_ALIGN`
        # is a multiple of every itemsize, which is what makes that exact.
        typed = self.buf.view(field.dtype)
        inner: tuple[int, ...] = ()
        acc = 1
        for dim in reversed(field.shape):
            inner = (acc,) + inner
            acc *= dim
        return typed.as_strided(
            (field.layers, self.entries) + field.shape,
            (field.per_layer_numel, self.slot_stride // itemsize) + inner,
            typed.storage_offset() + self._offsets[field.name] // itemsize,
        )

    def entry(self, index: int) -> torch.Tensor:
        """One entry's whole state as a contiguous 1-D uint8 slice."""
        start = index * self.slot_stride
        return self.buf[start : start + self.entry_bytes]

    def field_offset(self, name: str) -> int:
        """Byte offset of a field from the start of an entry."""
        return self._offsets[name]
