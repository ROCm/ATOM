# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Where every DeepSeek-V4 KV row lives, in rows rather than bytes.

V4 keeps two kinds of KV in the same pool: compressed history addressed by
block, and a per-request sliding window addressed by slot. They used to be two
regions of one per-layer tensor, which made the split between them a startup
constant — shrinking the block count moves the start of all 46 layers, so
neither kind can ever give space back to the other.

This module states the layout as a single **row space** that both kinds share,
and lets each pool materialize it at its own row width. A row is one token's KV
at one layer; what a row *costs* differs per plane (512 B packed fp8 NoPE,
128 B bf16 RoPE) but a row *index* means the same thing in both. That is a hard
requirement, not a convenience: `pa_sparse_prefill_fp8_opus` and
`mla_decode_fwd_v4_nm` each take an NoPE pool, a RoPE pool, and ONE shared
index buffer. It is also why NoPE and RoPE cannot be two sub-blocks of one
envelope — an envelope of `E` bytes puts a block's row at `b*E/512 + r` in one
plane and `b*E/128 + r` in the other, and those agree only for `E == 0`.

The row space::

    row 0                                       plane_rows
    ├─────────────────────────┬  gap  ┬────────────────────┤
    │ num_blocks envelopes    │       │ num_slots slots    │
    │ (compressed history)    │       │ (state + windows)  │
    └─────────────────────────┴───────┴────────────────────┘
                                        slot j occupies
                                        [j*S, (j+1)*S)

Slots take the high end so the two regions grow towards each other and neither
moves what the other has placed: a new slot takes the next position *down*, a
new block the next row up. **Slots are numbered by position, not by age** —
slot `j` is at `j * slot_rows` whatever the split is, and `physical_slot` maps
a pool group to one. Numbering from `plane_rows` downwards instead, which is
how a two-ended layout reads most naturally, would make a slot's address
*decrease* with its index; the compressor state is handed to its kernels as a
tensor strided by slot, and that stride would be negative. A per-layer
view anchors at that layer's own base row and runs to the end of the plane, so
both regions are addressed through one base pointer and moving the boundary is
a host-side counter change — no re-carve, no re-capture.

A slot holds a request's compressor state first and its sliding windows after.
The state is bytes rather than rows and does not divide by compress class, so
it takes whole rows off the front of the slot and each plane materializes its
share at that plane's row width. Putting it first is what frees the window row
count from having to be even: only a slot's own start has to land on an
alignment boundary, and `slot_rows` is rounded up to keep it there.

Inside an envelope the layers are grouped by compress class rather than
interleaved, because within a class every layer contributes the same number of
rows. That regularity makes a class's layer stride a constant, which is what
lets one index formula serve every layer of the class — and the index buffers
the attention path already builds (`prefix_swa_indices` / `prefix_hca_indices`
/ `prefix_csa_indices`) are already split exactly that way.

Dense layers keep no compressed KV at all, so they appear only in the entry.
A layer whose window is wider than a plane's row — a DSpark draft layer, whose
block attention wants unquantized KV where the pool is packed — appears in
neither: it is `ABSENT_RATIO` here and its ring is a state field instead, still
addressed by `WindowParams` through `field_window_params`.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

# Compress ratios, as they appear in the model config's per-layer list.
DENSE_RATIO = 0
CSA_RATIO = 4
HCA_RATIO = 128

# A token sees the compressed groups closed at or before its OWN position. The
# per-sequence `ctx // ratio` is the LAST token's count, and an MTP or DSpark
# sequence hands `1 + k` tokens to one forward, so the earlier ones would read
# groups holding their own future drafts. It is not a second cap either: it
# cannot bind while `pos < ctx`, which `require_step_within_full_q` keeps true
# at ATOM's producers of per-seq step lengths. The plugin bridges inherit the
# same premise from the host engine's `positions` and do not check it.
#
# Every host-side count comes from these two. The Triton kernels take the ratio
# as a `constexpr` and divide inline; keep the spellings identical.


def visible_csa(pos):
    """CSA groups visible to the token at `pos` — int, numpy array or tensor."""
    return (pos + 1) // CSA_RATIO


def visible_hca(pos):
    """HCA groups visible to the token at `pos` — int, numpy array or tensor."""
    return (pos + 1) // HCA_RATIO


def require_step_within_full_q(longest: int, full_q: int, source: str) -> None:
    """A sequence may not forward more tokens in one step than `full_q`.

    This is what `pos < ctx` reduces to, so ATOM's three producers of per-seq
    forward lengths check it rather than assume it. Two things break together
    when it fails. `positions` are anchored at `ctx - full_q`, so a longer step
    puts its last token at or past `ctx` -- the one case where the per-sequence
    bound the rule above drops would have bound something. And the DSpark
    rectangle's `lead = full_q - len` goes negative, so
    `_v4_decode_indptr_kernel` maps a whole band and reads tokens past the
    sequence's span, stamping one sequence's visibility onto another's rows.

    `ValueError`, not `assert`: the second failure is a device-side
    out-of-bounds read, and `python -O` strips asserts.
    """
    if longest > full_q:
        raise ValueError(
            f"{source} forwards {longest} tokens for some sequence but full_q "
            f"is {full_q}, which would place a token past its own context"
        )


# Not a compress ratio: a layer whose KV this row space does not hold at all.
# A DSpark draft layer wants its window at a width the planes do not offer —
# unquantized where the pool is packed — so the window becomes a state field
# instead, and the layer reserves neither an envelope nor an entry here. It is
# spelled as a ratio because that is the per-layer vocabulary the caller
# already has; it never reaches `classes`, so asking one for its layout raises.
ABSENT_RATIO = -1

# Envelope order puts the narrow class first so a future one can be appended
# without moving it. The entry leads with dense, the one class that has no
# envelope part at all, which is what frees the two orders from having to agree.
_ENVELOPE_ORDER = (HCA_RATIO, CSA_RATIO)
_ENTRY_ORDER = (DENSE_RATIO, HCA_RATIO, CSA_RATIO)
_KNOWN_RATIOS = frozenset(_ENTRY_ORDER) | {ABSENT_RATIO}


def merge_abutting(runs: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    """`(start, count)` pairs in order, with the ones that touch joined.

    Rows of a row space and bytes of a copy are both described this way, and
    both want as few of them as possible: every range a checkpoint copy is cut
    into costs a span, and a span costs grid.

    Ascending and disjoint is required, not assumed. Each run is compared only
    against the one before it, so an out-of-order input merges nothing and
    reads as legal: `[(0, 400), (256, 64), (320, 64)]` used to return
    `[(0, 400), (256, 128)]`, which double-counts 128 bytes and overlaps the
    first range. Nothing downstream can see that -- `checkpoint_image_bytes`
    would over-count, and both the op validator and the sizing cross-check
    compare against that same wrong number -- so the ordering is checked here
    rather than left as a property of whoever happens to build the list.
    """
    merged: list[tuple[int, int]] = []
    for start, count in runs:
        if count < 0 or start < 0:
            raise ValueError(f"a run must be non-negative, got ({start}, {count})")
        if merged:
            end = merged[-1][0] + merged[-1][1]
            if start < end:
                raise ValueError(
                    f"runs must ascend and not overlap: ({start}, {count}) "
                    f"starts inside the run ending at {end}"
                )
            if start == end:
                merged[-1] = (merged[-1][0], merged[-1][1] + count)
                continue
        merged.append((start, count))
    return merged


def ring_offset_for(num_layers: int, ring_stride: int, ring_pos: int) -> int:
    """The layer-independent part of a window row, `f(q)`.

    Every layer's window row has to be `i * ring_stride + f(q)` for one shared
    `f`: the layer term is forced by the compress side, which cannot pad, and a
    shared `f` is what keeps the index layer-independent. Whenever the window
    is longer than `ring_stride` the layers overlap if laid out contiguously,
    so the window is cut into `ring_stride`-sized runs and the runs interleaved
    across the class.

    Give each run the class's full height. Two positions then collide only if
    they share a run *and* a position inside it: positions in one run differ by
    less than `ring_stride` and so cannot be a whole layer apart, and positions
    in different runs differ by at least the class height and so cannot be
    either.
    """
    chunk, within = divmod(ring_pos, ring_stride)
    return chunk * (num_layers * ring_stride) + within


def entry_rows_for(num_layers: int, ring_stride: int, ring_slots: int) -> int:
    """Height the interleaved construction reaches — and the least possible.

    `ring_offset_for` peaks at the last window position (a later run always
    outweighs a longer tail inside an earlier one), so the tallest row belongs
    to the last layer at that position.

    No construction does better. Count rows by residue mod `ring_stride`: a
    height of `L` offers each residue at most `ceil(L / ring_stride)` rows, and
    every one of the `num_layers` layers needs a row of its own for each window
    position falling in that residue. Working that inequality back over all
    `ring_stride` residues yields exactly this height.
    """
    return (
        (num_layers - 1) * ring_stride
        + ring_offset_for(num_layers, ring_stride, ring_slots - 1)
        + 1
    )


@dataclass(frozen=True)
class WindowParams:
    """A class's window-row formula as five scalars a kernel can carry.

    `index(slot, pos)` is the whole of it::

        q = pos % ring_slots
        slot * slot_rows + ring_start + (q // ring_stride) * run_rows
                                      + (q %  ring_stride)

    All five are constant for the life of an allocation, so a kernel may carry
    them by value even inside a captured graph. `ring_start` is an ordinary
    argument rather than a `constexpr` only to keep a reallocation from forcing
    a Triton recompile; moving the compress/window boundary does not change it.
    See `UnifiedPoolGeometry.window_params` for why.

    `slot` is a *physical* slot — a position in the plane, not a pool group.
    `UnifiedPoolGeometry.physical_slot` is the one place the two meet, and
    everything downstream of it speaks positions: the windows, the compressor
    state's strided view, and the DSpark draft's separate plane.

    `pos` is an absolute token position and the wrap happens here, matching
    `pool_index.window_row` term for term — the two are transcriptions of one
    formula and `tests/test_pool_index.py` is where they meet.
    """

    ring_start: int
    slot_rows: int
    ring_slots: int
    ring_stride: int
    run_rows: int

    def index(self, slot: int, pos: int) -> int:
        chunk, within = divmod(pos % self.ring_slots, self.ring_stride)
        return slot * self.slot_rows + self.ring_start + chunk * self.run_rows + within


@dataclass(frozen=True)
class ClassLayout:
    """One compress class's rows, inside an envelope and inside an entry.

    `layers` are global layer ids in class order; a layer's position in that
    list is the only per-layer input any address formula takes.

    `block_rows` is how many rows one block compresses to in one layer of this
    class (`block_size // ratio`), and doubles as the class's **layer stride**:
    layer `i` starts `i * block_rows` rows into the class's part of an
    envelope. A dense class compresses to nothing, so it is free to pack its
    window rows layer after layer and takes `ring_slots` as its stride instead.
    """

    ratio: int
    layers: tuple[int, ...]
    block_rows: int
    ring_stride: int
    ring_slots: int
    envelope_offset: int
    entry_offset: int

    @property
    def num_layers(self) -> int:
        return len(self.layers)

    @property
    def envelope_rows(self) -> int:
        """Rows this class takes in one envelope, across all its layers."""
        return self.num_layers * self.block_rows

    @property
    def entry_rows(self) -> int:
        """Rows this class's sliding windows take in one entry."""
        return entry_rows_for(self.num_layers, self.ring_stride, self.ring_slots)

    def layer_index(self, layer_id: int) -> int:
        return self.layers.index(layer_id)

    def ring_offset(self, ring_pos: int) -> int:
        return ring_offset_for(self.num_layers, self.ring_stride, ring_pos)

    def ring_row(self, layer_index: int, ring_pos: int) -> int:
        """Row of a window position, relative to the class's part of an entry."""
        return layer_index * self.ring_stride + self.ring_offset(ring_pos)

    def entry_row_runs(self) -> list[tuple[int, int]]:
        """`(start, count)` runs of entry rows some `ring_row` reaches.

        The interleave is by ring position, not by layer: run `c` of the ring
        owns the rows `[c*run_rows, (c+1)*run_rows)` and every layer of the
        class has its positions for that run inside them. So all the whole
        runs together are one contiguous range, and only the ring's last,
        partial run is scattered — there each layer reaches
        `ring_slots % ring_stride` rows of its own `ring_stride` slice and
        leaves the rest.

        Those leftovers are `entry_rows` less `num_layers * ring_slots`: what
        a layer-independent index formula costs (`entry_rows_for`). No
        `(layer, position)` pair maps to one, so nothing writes or reads them,
        which is what lets a copy that only has to preserve windows — a
        checkpoint image is gathered back into a slot, never read by an
        attention kernel — leave them out.
        """
        run_rows = self.num_layers * self.ring_stride
        whole, partial = divmod(self.ring_slots, self.ring_stride)
        runs = [(0, whole * run_rows)] if whole else []
        if partial:
            runs += [
                (whole * run_rows + i * self.ring_stride, partial)
                for i in range(self.num_layers)
            ]
        return runs


class UnifiedPoolGeometry:
    """The row space, and every address formula that reads from it.

    Sole owner of the arithmetic: allocation, view binding, the index kernels
    and their references all take their offsets from here rather than each
    re-deriving them. A carve is only ever as correct as the agreement between
    the party that splits the buffer and the parties that compute offsets into
    it, and that agreement is not something a test of the carve alone can see.
    """

    def __init__(
        self,
        ratios: list[int],
        num_blocks: int,
        num_slots: int,
        ring_slots: int,
        block_size: int,
        plane_rows: int | None = None,
        arena_rows: int = 0,
        slot_positions: int | None = None,
        slot_align_rows: int = 2,
    ):
        """`ring_slots` is `win_with_spec` — the sliding window plus the draft
        lookahead,
        i.e. how many of a request's most recent writes stay addressable.

        `arena_rows` is how many rows off the front of a slot the compressor
        state takes; 0 leaves a slot holding nothing but windows, which is what
        the window-only tests want.

        `plane_rows` sizes a plane above what `num_blocks` and `num_slots`
        currently need, leaving the surplus as the gap the two ends grow into.
        It defaults to exactly what they need, which pins the boundary.

        `slot_align_rows` is the row multiple a slot's size is rounded to. Two
        is enough for the compressor state alone (see `slot_rows`); a window
        materialized at a width wider than a plane's row needs the slot to be a
        multiple of that many rows as well, or the retyped view it is addressed
        through does not divide evenly. The caller owns that arithmetic because
        it owns the plane widths.
        """
        if not ratios:
            raise ValueError("a V4 pool needs at least one layer")
        unknown = set(ratios) - _KNOWN_RATIOS
        if unknown:
            raise ValueError(f"unknown V4 compress ratios {sorted(unknown)}")
        if slot_align_rows < 1:
            raise ValueError(f"slot_align_rows must be positive, got {slot_align_rows}")
        if ring_slots < 1:
            raise ValueError(f"ring_slots must be positive, got {ring_slots}")
        if arena_rows < 0:
            raise ValueError(f"arena_rows must not be negative, got {arena_rows}")

        self.arena_rows = arena_rows
        self.ratios = list(ratios)
        self.num_layers = len(ratios)
        self.num_blocks = num_blocks
        self.num_slots = num_slots
        self.ring_slots = ring_slots
        self.block_size = block_size

        def layers_of(ratio: int) -> tuple[int, ...]:
            return tuple(i for i, r in enumerate(ratios) if r == ratio)

        def stride_of(ratio: int) -> int:
            return ring_slots if ratio == DENSE_RATIO else block_size // ratio

        # Both orders are walked before any layout is built, so each class
        # learns its offset in the envelope and in the entry at once.
        envelope_offsets: dict[int, int] = {}
        cursor = 0
        for ratio in _ENVELOPE_ORDER:
            if layers_of(ratio):
                envelope_offsets[ratio] = cursor
                cursor += len(layers_of(ratio)) * (block_size // ratio)
        self.envelope_rows = cursor

        entry_offsets: dict[int, int] = {}
        cursor = 0
        for ratio in _ENTRY_ORDER:
            layers = layers_of(ratio)
            if layers:
                entry_offsets[ratio] = cursor
                cursor += entry_rows_for(len(layers), stride_of(ratio), ring_slots)
        self.entry_rows = cursor

        self.classes: dict[int, ClassLayout] = {
            ratio: ClassLayout(
                ratio=ratio,
                layers=layers_of(ratio),
                block_rows=0 if ratio == DENSE_RATIO else block_size // ratio,
                ring_stride=stride_of(ratio),
                ring_slots=ring_slots,
                envelope_offset=envelope_offsets.get(ratio, 0),
                entry_offset=entry_offsets[ratio],
            )
            for ratio in _ENTRY_ORDER
            if layers_of(ratio)
        }

        # A slot's start is the one address the compressor state's alignment
        # rests on, so it is rounded up: the narrow plane is 128 B wide and
        # `StateArena` retypes from a 256 B boundary, which alone wants an even
        # row. Rounding the slot rather than the window is what keeps
        # `entry_rows` free of the constraint — the state sits at the front,
        # offset zero.
        self.slot_align_rows = slot_align_rows
        rows = arena_rows + self.entry_rows
        self.slot_rows = -(-rows // slot_align_rows) * slot_align_rows

        block_rows = num_blocks * self.envelope_rows
        if plane_rows is None:
            whole = num_slots + -(-block_rows // self.slot_rows)
            self.plane_rows = whole * self.slot_rows
        else:
            self.plane_rows = plane_rows
        # Whole slots the plane holds. Slot `j` sits at `j * slot_rows`, so the
        # pool hands out the top `num_slots` of these positions and the
        # remainder under one slot at the very top is the only waste the layout
        # carries — it does not scale with anything.
        held = self.plane_rows // self.slot_rows
        if num_slots > held or (held - num_slots) * self.slot_rows < block_rows:
            raise ValueError(
                f"plane of {self.plane_rows} rows cannot hold {num_blocks} "
                f"blocks ({self.envelope_rows} rows each) plus {num_slots} "
                f"slots ({self.slot_rows} rows each)"
            )
        self.slot_positions = held if slot_positions is None else slot_positions
        # Positions this plane does not reach, because it took its numbering
        # from one with more room. Subtracted back out of every row it computes.
        self.slot_origin = self.slot_positions - held
        if self.slot_origin < 0:
            raise ValueError(
                f"plane of {self.plane_rows} rows holds {held} slots, more than "
                f"the {slot_positions} positions it was told to number by"
            )

    def with_capacity(
        self, num_blocks: int, num_slots: int, plane_rows: int | None = None
    ) -> UnifiedPoolGeometry:
        """The same layout at a different split.

        Sizing has to state a block's and a slot's cost before either count is
        known, and the counts it produces then have to describe the same
        layout. Building one shape at capacity zero and re-sizing it here keeps
        that a single object rather than two constructions that must agree.
        """
        return UnifiedPoolGeometry(
            self.ratios,
            num_blocks,
            num_slots,
            self.ring_slots,
            self.block_size,
            plane_rows,
            self.arena_rows,
            slot_align_rows=self.slot_align_rows,
        )

    # ---- byte helpers ---------------------------------------------------

    def plane_bytes(self, row_bytes: int) -> int:
        return self.plane_rows * row_bytes

    def block_bytes(self, row_bytes: int) -> int:
        """Bytes one block costs in a plane of this row width."""
        return self.envelope_rows * row_bytes

    def slot_bytes(self, row_bytes: int) -> int:
        """Bytes one request costs in a plane of this row width.

        The whole slot: compressor state and windows both, since the two are
        allocated and relinquished together and no caller can have one without
        the other.
        """
        return self.slot_rows * row_bytes

    # ---- addressing -----------------------------------------------------

    def layer_class(self, layer_id: int) -> ClassLayout:
        ratio = self.ratios[layer_id]
        if ratio == ABSENT_RATIO:
            raise ValueError(
                f"layer {layer_id} keeps no KV in this row space; its window is "
                "a state field, so it has no class, no base row and no view"
            )
        return self.classes[ratio]

    def layer_base_row(self, layer_id: int) -> int:
        """Row a layer's view anchors at; it runs from there to the plane end.

        Compressed classes anchor inside envelope 0, so a compress index needs
        no layer term. Dense layers have no envelope part, so they anchor at
        the offset their windows carry inside an entry instead — the same
        cancellation, arranged from the other side.
        """
        cls = self.layer_class(layer_id)
        index = cls.layer_index(layer_id)
        if cls.ratio == DENSE_RATIO:
            return index * cls.ring_stride
        return cls.envelope_offset + index * cls.block_rows

    def layer_view_rows(self, layer_id: int) -> int:
        """Rows a layer's view spans — its base to the end of the plane."""
        return self.plane_rows - self.layer_base_row(layer_id)

    def compress_index(self, layer_id: int, block: int, row: int) -> int:
        """Index of a compressed row, relative to the layer's own base.

        Layer-independent by construction: the layer's contribution already
        sits in the base, leaving the envelope stride times the block.
        """
        cls = self.layer_class(layer_id)
        if cls.ratio == DENSE_RATIO:
            raise ValueError(f"layer {layer_id} is dense and keeps no compressed KV")
        if not 0 <= row < cls.block_rows:
            raise ValueError(
                f"row {row} outside 0..{cls.block_rows} for ratio {cls.ratio}"
            )
        return block * self.envelope_rows + row

    def window_index(self, layer_id: int, slot: int, ring_pos: int) -> int:
        """Index of a sliding-window row, relative to the layer's own base."""
        if not 0 <= ring_pos < self.ring_slots:
            raise ValueError(f"ring position {ring_pos} outside 0..{self.ring_slots}")
        return self.window_params(self.ratios[layer_id]).index(slot, ring_pos)

    def window_params(self, ratio: int) -> WindowParams:
        """The same formula, flattened into what a Triton kernel can take.

        Kernels get this rather than the class layout so there is one
        expression for a window row, not one here and a transcription of it in
        each kernel.

        **Moving the boundary changes nothing here.** `ring_start` is an offset
        inside a slot and a compress row counts up from row 0, so neither
        address knows where the other region stops; `num_blocks` and `num_slots`
        only gate which block ids and slots the allocator may hand out. That is
        what makes the layout safe under CUDA-graph capture, where a by-value
        scalar is frozen exactly as hard as a `constexpr` — the argument-vs-
        constexpr choice buys a recompile, not a re-capture.

        The one thing that must not move is `plane_rows`, because
        `physical_slot` counts back from the topmost position and that is how
        many the plane holds. It defaults to a tight fit around the current
        split, which is right while the boundary is pinned; an elastic pool has
        to pass its own fixed capacity instead, or every group lands somewhere
        else the moment the split it was supposed to be independent of changes.
        """
        cls = self.classes[ratio]
        anchor = 0 if ratio == DENSE_RATIO else cls.envelope_offset
        return WindowParams(
            ring_start=self.arena_rows
            + cls.entry_offset
            - anchor
            - self.slot_origin * self.slot_rows,
            slot_rows=self.slot_rows,
            ring_slots=self.ring_slots,
            ring_stride=cls.ring_stride,
            run_rows=cls.num_layers * cls.ring_stride,
        )

    def field_window_params(
        self, field_offset_rows: int, rows_per_window_row: int
    ) -> WindowParams:
        """A window that lives in a state field rather than in the entry.

        A layer whose KV is wider than a plane's row needs
        `rows_per_window_row` of them per ring position, so it takes bytes off
        the front of the slot like the compressor state and is read through a
        view of the plane retyped to its own width, where those rows are one.
        The same five scalars still describe it, counted in that view's rows:
        the slot is that many times shorter and the ring starts at the field's
        own offset.

        A ratio of one — a layer whose KV happens to match the plane — is not a
        special case: the expression below collapses to the field's offset,
        which is what `window_params` would give a class placed there. Nothing
        here branches on a dtype.
        """
        if rows_per_window_row < 1:
            raise ValueError(
                f"rows per window row must be positive, got {rows_per_window_row}"
            )
        for name, rows in (
            ("slot", self.slot_rows),
            ("field offset", field_offset_rows),
        ):
            if rows % rows_per_window_row:
                raise ValueError(
                    f"{name} of {rows} rows does not divide by the "
                    f"{rows_per_window_row} plane rows one window row takes"
                )
        slot_rows = self.slot_rows // rows_per_window_row
        return WindowParams(
            ring_start=field_offset_rows // rows_per_window_row
            - self.slot_origin * slot_rows,
            slot_rows=slot_rows,
            ring_slots=self.ring_slots,
            # One run: the ring is contiguous here, since a field is not
            # interleaved with anything the way a class's layers are.
            ring_stride=self.ring_slots,
            run_rows=self.ring_slots,
        )

    def entry_row_runs(self) -> list[tuple[int, int]]:
        """`(start, count)` runs of entry rows any window reaches, in order.

        Every class's runs, offset into the entry and merged where they abut.
        The complement is interleave padding — see `ClassLayout.entry_row_runs`
        for why nothing reaches it. `classes` is built in entry order, so
        walking it is walking the entry.
        """
        return merge_abutting(
            (cls.entry_offset + start, count)
            for cls in self.classes.values()
            for start, count in cls.entry_row_runs()
        )

    def physical_slot(self, group: int) -> int:
        """Where pool group `group` sits in the plane.

        Groups run 0..num_slots-1 with 0 the one furthest from the compressed
        blocks, so growing the pool takes the next position *down* and shrinking
        gives back the one that abuts the gap. Group order and position order
        are therefore opposites, and this is the only place they meet.
        """
        if not 0 <= group < self.num_slots:
            raise ValueError(f"group {group} outside 0..{self.num_slots}")
        return self.slot_positions - 1 - group

    def slot_span(self, slot: int) -> tuple[int, int]:
        """`[start, stop)` plane rows a request occupies, state and windows.

        One range rather than one per class or per layer: a slot is the unit a
        checkpoint copies and a PD transfer registers.
        """
        start = (slot - self.slot_origin) * self.slot_rows
        return start, start + self.slot_rows

    def arena_span(self, slot: int) -> tuple[int, int]:
        """`[start, stop)` plane rows holding a request's compressor state."""
        start = (slot - self.slot_origin) * self.slot_rows
        return start, start + self.arena_rows

    def absolute_row(self, layer_id: int, index: int) -> int:
        """Plane row an index reaches — what the two address spaces share."""
        return self.layer_base_row(layer_id) + index
