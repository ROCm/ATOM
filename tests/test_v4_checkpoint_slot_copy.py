# SPDX-License-Identifier: MIT

"""Which bytes of a DeepSeek-V4 Active Slot a checkpoint carries, and that a
store/restore round trip moves exactly those and nothing else.

`checkpoint_ranges_for` (tested in `test_state_arena.py`) says which bytes of the
compressor *arena* are live. This file covers the step after it: composing
those with the sliding-window rows that share the slot, turning the result into
byte segments at a slot's real address, and round-tripping them through the
copy planner.

The builder is exercised through unbound methods on a stub rather than a real
`DeepseekV4AttentionMetadataBuilder`, which would want a ModelRunner, a model
and a GPU. What the stub supplies is exactly what these methods read, so the
arithmetic under test is the shipped arithmetic.
"""

from __future__ import annotations

import ctypes
from itertools import pairwise

import pytest
import torch

from atom.model_ops.attentions.deepseek_v4_attn import (
    DeepseekV4AttentionMetadataBuilder as Builder,
)
from atom.model_ops.attentions.paged_state_copy import (
    ByteSegment,
    plan_segmented_copy,
)
from atom.model_ops.attentions.state_arena import (
    StateField,
    checkpoint_bytes_for,
    checkpoint_ranges_for,
    entry_bytes_for,
    field_extents,
)

NEG_INF = float("-inf")
ROW_BYTES = 64
SLOTS = 3

# DeepSeek-V4's field list in miniature: two CSA families, one HCA family that
# a checkpoint owes nothing, and a draft window that is a sliding window and so
# stays whole. Same order as `_state_fields`, which is the order the bytes are
# seen in.
FIELDS = [
    StateField("csa_main_kv", 2, (4, 8), torch.float32),
    StateField("csa_main_score", 2, (4, 8), torch.float32, NEG_INF),
    StateField("hca_main_kv", 2, (16, 8), torch.float32, in_checkpoint=False),
    StateField(
        "hca_main_score", 2, (16, 8), torch.float32, NEG_INF, in_checkpoint=False
    ),
    StateField("state_window", 1, (6, 8), torch.float32),
]
ARENA_BYTES = entry_bytes_for(FIELDS)
ARENA_ROWS = -(-ARENA_BYTES // ROW_BYTES)
ENTRY_ROWS = 5  # the windows that share the slot, in row-space rows
SLOT_ROWS = ARENA_ROWS + ENTRY_ROWS + 1  # +1 so there is tail padding to drop
SLOT_BYTES = SLOT_ROWS * ROW_BYTES


class _Geo:
    """Only the three numbers `_checkpoint_slot_ranges` and `_slot_views` read."""

    arena_rows = ARENA_ROWS
    entry_rows = ENTRY_ROWS
    slot_rows = SLOT_ROWS

    def slot_span(self, physical: int) -> tuple[int, int]:
        return physical * SLOT_ROWS, (physical + 1) * SLOT_ROWS

    def physical_slot(self, group: int) -> int:
        return group


class _StubBuilder:
    """Stands in for the parts of the builder these two methods touch."""

    # The real methods, not copies of them: between them they read nothing the
    # stub does not supply, and a reimplementation here would stop tracking the
    # ones it stands in for.
    _assert_ratios_divide_block = Builder._assert_ratios_divide_block
    _checkpoint_slot_ranges = Builder._checkpoint_slot_ranges
    _checkpoint_slot_segments = Builder._checkpoint_slot_segments
    checkpoint_image_bytes = Builder.checkpoint_image_bytes

    def __init__(self, plane: torch.Tensor):
        self.pool_geometry = _Geo()
        self._arena_planes = [FIELDS]
        self._checkpoint_range_cache = None
        self.block_size = 256
        self._plane = plane

    def _plane_row_widths(self):
        return [ROW_BYTES]

    def _slot_views(self):
        geo = self.pool_geometry
        return [
            [self._plane[slice(*geo.slot_span(geo.physical_slot(g)))]]
            for g in range(SLOTS)
        ]


@pytest.fixture
def plane():
    # uint8 so a row is ROW_BYTES elements and byte offsets are element offsets.
    return torch.zeros(SLOTS * SLOT_ROWS, ROW_BYTES, dtype=torch.uint8)


def field_spans() -> dict[str, tuple[int, int]]:
    """Each field's own byte range inside an entry, padding excluded.

    Spelled out here rather than read back from the arena so the round trip is
    checked against the layout it is supposed to have, not against the same
    arithmetic it is exercising.
    """
    spans = {}
    offset = 0
    for field in FIELDS:
        align = max(256, field.align)
        offset = -(-offset // align) * align
        spans[field.name] = (offset, offset + field.bytes_per_entry)
        offset += field.bytes_per_entry
    return spans


def dead_arena_span() -> tuple[int, int]:
    """Byte range spanned by the two HCA fields, which are dead together."""
    spans = field_spans()
    return spans["hca_main_kv"][0], spans["hca_main_score"][1]


def execute(spans):
    """Run the planner's spans as plain host copies."""
    for span in spans:
        ctypes.memmove(span.dst_ptr, span.src_ptr, span.num_bytes)


class TestCheckpointSlotRanges:
    def test_the_ranges_skip_the_dead_fields_and_the_slot_tail(self, plane):
        (ranges,) = Builder._checkpoint_slot_ranges(_StubBuilder(plane))
        dead_start, dead_end = dead_arena_span()

        covered = set()
        for start, nbytes in ranges:
            covered |= set(range(start, start + nbytes))

        assert not covered & set(range(dead_start, dead_end)), "HCA is carried"
        # The windows share the slot and are a sliding window: all of them.
        window = range(ARENA_ROWS * ROW_BYTES, (ARENA_ROWS + ENTRY_ROWS) * ROW_BYTES)
        assert set(window) <= covered
        # Neither the padding the arena is rounded up by nor the slot's own
        # tail alignment belongs to anyone.
        assert not covered & set(
            range((ARENA_ROWS + ENTRY_ROWS) * ROW_BYTES, SLOT_BYTES)
        )

    def test_the_ranges_are_ordered_disjoint_and_inside_the_slot(self, plane):
        (ranges,) = Builder._checkpoint_slot_ranges(_StubBuilder(plane))

        assert ranges == sorted(ranges)
        for (a_start, a_bytes), (b_start, _) in pairwise(ranges):
            assert a_start + a_bytes <= b_start
        for start, nbytes in ranges:
            assert 0 <= start and start + nbytes <= SLOT_BYTES

    def test_the_image_size_is_the_arena_live_bytes_plus_the_windows(self, plane):
        stub = _StubBuilder(plane)

        assert Builder.checkpoint_image_bytes(stub) == (
            checkpoint_bytes_for(FIELDS) + ENTRY_ROWS * ROW_BYTES
        )
        assert Builder.checkpoint_image_bytes(stub) < SLOT_BYTES

    def test_a_block_size_a_ratio_does_not_divide_is_refused(self, plane):
        """HCA owes nothing only while a checkpoint lands on a pool boundary."""
        stub = _StubBuilder(plane)
        stub.block_size = 64  # 64 % 128 != 0

        with pytest.raises(ValueError, match="not a multiple of compress"):
            Builder.checkpoint_image_bytes(stub)


class TestRoundTrip:
    """Store slot 0 into PAGE units, gather it back into slot 1."""

    def test_the_live_bytes_survive_and_the_dead_bytes_are_not_touched(self, plane):
        stub = _StubBuilder(plane)
        live_bytes = Builder.checkpoint_image_bytes(stub)

        # Two distinguishable fills, so a byte that failed to move and a byte
        # that moved when it should not have both show up.
        plane[0 * SLOT_ROWS : 1 * SLOT_ROWS] = 0xA5
        plane[1 * SLOT_ROWS : 2 * SLOT_ROWS] = 0x3C
        before = plane[1 * SLOT_ROWS : 2 * SLOT_ROWS].clone()

        # A checkpoint image: arbitrary units, deliberately not contiguous and
        # not in address order, which is the whole point of PAGE backing.
        units = torch.full((4, live_bytes // 2), 0x11, dtype=torch.uint8)
        image = [ByteSegment(units[i].data_ptr(), units[i].numel()) for i in (2, 0, 3)]

        src = Builder._checkpoint_slot_segments(stub, 0)
        execute(plan_segmented_copy(src, image, live_bytes))
        dst = Builder._checkpoint_slot_segments(stub, 1)
        execute(plan_segmented_copy(image, dst, live_bytes))

        after = plane[1 * SLOT_ROWS : 2 * SLOT_ROWS].reshape(-1)
        source = plane[0 * SLOT_ROWS : 1 * SLOT_ROWS].reshape(-1)
        untouched = before.reshape(-1)

        # Derive what must have moved from the layout, NOT from
        # `_checkpoint_slot_ranges`. Checking the ranges against themselves passes
        # for any self-consistent mistake — including a range shifted the same
        # way on both sides, which is exactly the bug worth catching here.
        dead_start, dead_end = dead_arena_span()
        spans = field_spans()
        for start, end in (
            spans["csa_main_kv"],
            spans["csa_main_score"],
            spans["state_window"],  # a draft window, behind the dead fields
            (ARENA_ROWS * ROW_BYTES, (ARENA_ROWS + ENTRY_ROWS) * ROW_BYTES),
        ):
            assert torch.equal(
                after[start:end], source[start:end]
            ), f"bytes [{start}, {end}) did not survive the round trip"

        # The dead fields must still hold slot 1's own fill: carrying them
        # would be waste, writing them from anywhere else would be a bug.
        assert torch.equal(
            after[dead_start:dead_end], untouched[dead_start:dead_end]
        ), "the dead HCA fields were overwritten"
        tail = (ARENA_ROWS + ENTRY_ROWS) * ROW_BYTES
        assert torch.equal(
            after[tail:], untouched[tail:]
        ), "the slot's tail padding was overwritten"

    def test_a_restore_does_not_read_past_the_image(self, plane):
        """`total_bytes` is the image, so the tail of the last unit is spare."""
        stub = _StubBuilder(plane)
        live_bytes = Builder.checkpoint_image_bytes(stub)
        units = torch.full((live_bytes + 4096,), 0x77, dtype=torch.uint8)
        image = [ByteSegment(units.data_ptr(), units.numel())]

        spans = plan_segmented_copy(
            image, Builder._checkpoint_slot_segments(stub, 2), live_bytes
        )

        assert sum(s.num_bytes for s in spans) == live_bytes
        end = max(s.src_ptr + s.num_bytes for s in spans)
        assert end - units.data_ptr() == live_bytes


class TestTheBuilderDeclaresWhatItDrops:
    """`_state_fields` is where the rule actually lives.

    Everything above uses its own field list, so none of it says anything
    about what DeepSeek-V4 itself declares — flipping `in_checkpoint` back on
    in the builder left every one of those tests green. This is the one that
    notices.
    """

    @staticmethod
    def build_fields() -> list[StateField]:
        class _Stub:
            _state_dtype = torch.float32
            csa_layers = (2, 4, 6)
            hca_layers = (3, 5)
            csa_main_state_shape = (13, 1024)
            csa_idx_state_shape = (13, 256)
            hca_main_state_shape = (133, 512)
            head_dim = 512
            win_with_spec = 133
            _field_window_dtype = torch.bfloat16
            _field_window_layers = (43,)
            _window_field_row_bytes = Builder._window_field_row_bytes

        return Builder._state_fields(_Stub())

    def test_hca_is_the_only_thing_dropped(self):
        dropped = {f.name for f in self.build_fields() if not f.in_checkpoint}

        assert dropped == {"hca_main_kv", "hca_main_score"}, (
            "HCA pools [P, P+128) with no overlap, so a resumer writes every "
            "row it reads; nothing else here is known to be dead at a boundary"
        )

    def test_the_dropped_fields_are_outside_every_range(self):
        fields = self.build_fields()
        carried = set()
        for start, nbytes in checkpoint_ranges_for(fields):
            carried |= set(range(start, start + nbytes))

        for field, start, end in field_extents(fields):
            overlap = carried & set(range(start, end))
            if field.in_checkpoint:
                assert overlap, f"{field.name} is carried but has no range"
            else:
                assert not overlap, f"{field.name} is dropped but has one"

    def test_what_is_dropped_is_versioned_into_the_layout_id(self):
        """Two workers disagreeing on the rule read one image two ways."""
        fields = self.build_fields()
        nocopy = ",".join(f.name for f in fields if not f.in_checkpoint)

        # `state_transfer` builds the id from exactly this expression, so a
        # change to the rule cannot leave the id saying what it used to.
        assert nocopy == "hca_main_kv,hca_main_score"
