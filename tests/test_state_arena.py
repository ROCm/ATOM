# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Layout arithmetic for `StateArena`.

The property under test throughout is that the two ways of looking at the
same memory agree: the per-layer views the kernels bind, and the contiguous
per-entry byte range that checkpointing and RDMA use. Everything runs on CPU
— the arena is pure indexing, no kernels.
"""

import math
from dataclasses import replace
from itertools import pairwise

import pytest
import torch

from atom.model_ops.attentions.pool_layout.state_arena import (
    StateArena,
    StateField,
    checkpoint_ranges_for,
    entry_bytes_for,
    field_extents,
    plan_field_planes,
    plan_regions,
)

# Shaped after DeepSeek-V4's compressor state, scaled down: three families,
# each a (kv, score) pair, two of them on the CSA layer count and one on HCA.
NEG_INF = float("-inf")
V4_LIKE = [
    StateField("csa_main_kv", 3, (8, 32), torch.float32),
    StateField("csa_main_score", 3, (8, 32), torch.float32, fill=NEG_INF),
    StateField("csa_idx_kv", 3, (8, 16), torch.float32),
    StateField("csa_idx_score", 3, (8, 16), torch.float32, fill=NEG_INF),
    StateField("hca_main_kv", 2, (128, 32), torch.float32),
    StateField("hca_main_score", 2, (128, 32), torch.float32, fill=NEG_INF),
]


def build(fields=V4_LIKE, entries=5) -> StateArena:
    return StateArena(fields, entries, device="cpu")


def carried_bytes(fields) -> int:
    """Bytes of an entry a checkpoint image holds, the long way round.

    Spelled out here rather than imported: the module used to export this and
    nothing but these tests called it, and a helper kept alive by its own
    tests is not an interface.
    """
    return sum(nbytes for _, nbytes in checkpoint_ranges_for(fields))


class TestEntryBytes:

    def test_sum_of_fields_when_naturally_aligned(self):
        """Real state shapes are coarse multiples of the alignment, so the
        budget is a plain sum — sizing must not pay for padding it will not
        get."""
        expected = sum(f.bytes_per_entry for f in V4_LIKE)
        assert expected % 256 == 0
        assert entry_bytes_for(V4_LIKE) == expected

    def test_pads_between_misaligned_fields(self):
        odd = [
            StateField("a", 1, (3,), torch.float32),  # 12 B
            StateField("b", 1, (3,), torch.float32),  # 12 B
        ]
        # Each field starts on its own 256 B boundary, and the entry as a
        # whole is rounded so entry i+1 starts aligned too.
        assert entry_bytes_for(odd) == 512

    def test_sizing_and_allocation_use_the_same_expression(self):
        """`entry_bytes_for` is what the byte budget is computed from before
        any GPU exists; the built arena must not disagree with it."""
        arena = build()
        assert arena.entry_bytes == entry_bytes_for(V4_LIKE)
        assert arena.total_bytes == arena.entries * arena.entry_bytes
        assert arena.buf.numel() == arena.total_bytes


class TestViewsAreDropInShapes:

    def test_shape_matches_the_standalone_tensor(self):
        arena = build()
        for field in V4_LIKE:
            view = arena.view(field.name)
            assert view.shape == (field.layers, arena.entries) + field.shape
            assert view.dtype == field.dtype

    def test_slot_stride_is_the_whole_entry(self):
        """The only difference from a standalone allocation. Kernels that
        take the slot stride as an argument are unaffected by this; one that
        assumes contiguity is not."""
        arena = build()
        view = arena.view("csa_main_kv")
        itemsize = torch.float32.itemsize
        assert view.stride(1) == arena.entry_bytes // itemsize
        assert view.stride(0) == math.prod((8, 32))
        assert view.stride(-1) == 1

    def test_trailing_dims_stay_contiguous(self):
        """Kernels index the innermost dim with a bare `+ d`."""
        arena = build()
        for field in V4_LIKE:
            per_layer_slot = arena.view(field.name)[0, 0]
            assert per_layer_slot.is_contiguous()


class TestViewsAndEntriesAgree:

    def test_write_through_view_lands_in_that_entry(self):
        arena = build()
        arena.view("hca_main_kv").zero_()
        arena.view("hca_main_kv")[1, 3].fill_(7.0)

        touched = arena.entry(3).view(torch.float32)
        assert (touched == 7.0).sum() == 128 * 32
        for other in (0, 1, 2, 4):
            assert (arena.entry(other).view(torch.float32) == 7.0).sum() == 0

    def test_entries_are_contiguous_and_disjoint(self):
        arena = build()
        for i in range(arena.entries):
            assert arena.entry(i).is_contiguous()
            assert arena.entry(i).numel() == arena.entry_bytes
        base = arena.buf.data_ptr()
        for i in range(arena.entries):
            assert arena.entry(i).data_ptr() == base + i * arena.entry_bytes

    def test_entry_bytes_equal_the_hand_rolled_gather(self):
        """The layout DeepSeek-V4's PD path builds per transfer today:
        each field's `[:, slot]` flattened, concatenated in field order.
        Making that physical is the whole point of the arena, so the two
        must be byte-identical."""
        arena = build()
        torch.manual_seed(0)
        for field in V4_LIKE:
            arena.view(field.name).copy_(
                torch.randn((field.layers, arena.entries) + field.shape)
            )

        slot = 2
        gathered = torch.cat([arena.view(f.name)[:, slot].reshape(-1) for f in V4_LIKE])
        assert torch.equal(arena.entry(slot).view(torch.float32), gathered)

    def test_field_offsets_are_ascending_and_inside_the_entry(self):
        arena = build()
        offsets = [arena.field_offset(f.name) for f in V4_LIKE]
        assert offsets == sorted(offsets)
        last = V4_LIKE[-1]
        assert arena.field_offset(last.name) + last.bytes_per_entry <= arena.entry_bytes


class TestInitialFill:

    def test_kv_zero_score_neg_inf(self):
        arena = build()
        for field in V4_LIKE:
            view = arena.view(field.name)
            if field.fill == 0.0:
                assert torch.equal(view, torch.zeros_like(view))
            else:
                assert torch.isneginf(view).all()

    def test_alignment_padding_is_initialized_too(self):
        """Padding falls outside every field view, but an entry is copied
        whole by checkpointing and RDMA — so it must not be whatever the
        allocator last left there."""
        arena = build([StateField("a", 1, (3,), torch.float32)], entries=2)
        assert arena.entry_bytes == 256  # 12 B of field, 244 B of padding
        arena.view("a").fill_(1.0)
        for i in range(arena.entries):
            assert (arena.entry(i)[12:] == 0).all()


class TestMixedDtypes:

    def test_fields_may_differ_in_dtype(self):
        """GDN keeps its recurrent k and v in different dtypes."""
        fields = [
            StateField("k", 2, (4, 8), torch.bfloat16),
            StateField("v", 2, (4, 8), torch.float32),
        ]
        arena = StateArena(fields, 3, device="cpu")
        assert arena.view("k").dtype == torch.bfloat16
        assert arena.view("v").dtype == torch.float32
        arena.view("k")[1, 2].fill_(1.5)
        arena.view("v")[1, 2].fill_(2.5)
        assert arena.view("k")[1, 2].eq(1.5).all()
        assert arena.view("v")[1, 2].eq(2.5).all()
        assert arena.view("k")[1, 1].eq(0).all()


class TestPlanRegions:
    """Packing for the one allocation every per-request pool is carved from.

    Kept here rather than beside the V4 backend so it runs without importing
    AITER. That matters most for the fp8 shape: it carves a RoPE pool per
    layer on top of the unified pools, and cannot be exercised end to end
    while the fused fp8 SWA write is paged-only.
    """

    def test_regions_are_aligned_disjoint_and_in_order(self):
        sizes = [1, 255, 256, 257, 4096, 3]
        offsets, total = plan_regions(sizes)
        assert len(offsets) == len(sizes)
        for off in offsets:
            assert off % 256 == 0
        for (a, n), b in zip(zip(offsets, sizes), offsets[1:]):
            assert a + n <= b, "region overruns the next one"
        assert offsets[-1] + sizes[-1] <= total
        assert total % 256 == 0

    def test_total_is_alignable_so_plans_concatenate(self):
        a, total_a = plan_regions([100, 200])
        b, total_b = plan_regions([300])
        joint, total_joint = plan_regions([100, 200, 300])
        assert joint[:2] == a
        assert joint[2] == total_a + b[0]
        assert total_joint == total_a + total_b

    def test_empty_plan(self):
        assert plan_regions([]) == ([], 0)

    def test_zero_sized_region_still_gets_an_offset(self):
        offsets, _ = plan_regions([256, 0, 256])
        assert len(offsets) == 3
        assert offsets[1] == offsets[2] == 256

    @pytest.mark.parametrize("with_rope", [False, True])
    def test_v4_shaped_layout(self, with_rope):
        """bf16 carves one region per layer plus the arena; fp8 carves two."""
        layers, head_dim, rope_dim = 4, 512, 64
        pages = [1000, 1000, 5000, 3000]
        sizes = [p * head_dim * 2 for p in pages]
        if with_rope:
            sizes += [p * rope_dim * 2 for p in pages]
        arena_bytes = entry_bytes_for(V4_LIKE) * 7
        sizes.append(arena_bytes)

        offsets, total = plan_regions(sizes)
        kv = offsets[:layers]
        rope = offsets[layers : 2 * layers] if with_rope else []
        arena = offsets[-1]

        assert len(rope) == (layers if with_rope else 0)
        # The arena must clear every pool, which is the invariant that broke
        # when `StateArena.view()` addressed from the host allocation's base.
        assert arena >= max(o + s for o, s in zip(offsets[:-1], sizes[:-1]))
        assert arena + arena_bytes <= total
        assert all(a < b for a, b in pairwise(kv))


class TestCarvedBuf:
    """An arena carved out of a larger buffer must stay inside its slice.

    `view()` reaches the storage through `as_strided`, whose storage_offset is
    absolute; an owned buffer sits at offset 0, so forgetting to add the
    slice's own offset is invisible until someone passes `buf`. What it costs
    when it is not caught: every field view starts at the front of the host
    allocation and the arena writes through whatever was carved before it.
    """

    @staticmethod
    def _carve(head_bytes: int, entries: int = 5):
        want = entry_bytes_for(V4_LIKE) * entries
        host = torch.zeros(head_bytes + want, dtype=torch.uint8)
        arena = StateArena(V4_LIKE, entries, device="cpu", buf=host[head_bytes:])
        return host, arena

    def test_views_start_inside_the_slice_not_at_the_host_base(self):
        host, arena = self._carve(4096)
        for field in V4_LIKE:
            offset = arena.view(field.name).data_ptr() - host.data_ptr()
            assert offset >= 4096, (
                f"{field.name} view starts {4096 - offset} bytes before the "
                "arena — it is addressing from the host allocation's base"
            )

    def test_head_of_the_host_allocation_is_untouched(self):
        head_bytes = 4096
        host, arena = self._carve(head_bytes)
        host[:head_bytes] = 0xAB
        for field in V4_LIKE:
            arena.view(field.name).fill_(1.0)
        assert bool((host[:head_bytes] == 0xAB).all()), (
            "writing through the field views modified memory carved before " "the arena"
        )

    def test_rejects_a_misaligned_slice(self):
        want = entry_bytes_for(V4_LIKE) * 2
        host = torch.zeros(8 + want, dtype=torch.uint8)
        with pytest.raises(ValueError, match="boundary"):
            StateArena(V4_LIKE, 2, device="cpu", buf=host[8:])

    def test_carved_and_owned_agree_field_for_field(self):
        _, carved = self._carve(4096)
        owned = build()
        for field in V4_LIKE:
            c, o = carved.view(field.name), owned.view(field.name)
            assert c.shape == o.shape
            assert c.stride() == o.stride()
            assert c.data_ptr() - carved.buf.data_ptr() == (
                o.data_ptr() - owned.buf.data_ptr()
            )


class TestRejectsBadFieldLists:

    def test_empty(self):
        with pytest.raises(ValueError, match="at least one field"):
            StateArena([], 4, device="cpu")

    def test_duplicate_names(self):
        dup = [
            StateField("a", 1, (4,), torch.float32),
            StateField("a", 1, (4,), torch.float32),
        ]
        with pytest.raises(ValueError, match="duplicate field names"):
            StateArena(dup, 4, device="cpu")


# ── An arena strided by something bigger than itself ───────────────────────
#
# When the compressor state moves into the front of a slot in a shared plane,
# consecutive entries are a slot apart rather than an entry apart, and only
# the top of the index range belongs to the pool — the rest of the plane is
# compressed blocks.


class TestSlotStride:
    def build(self, entries=5, live=None, stride=None):
        stride = stride or (entry_bytes_for(V4_LIKE) + 4 * 256)
        return StateArena(
            V4_LIKE, entries, device="cpu", slot_stride=stride, live_entries=live
        )

    def test_entries_land_one_stride_apart(self):
        arena = self.build()
        starts = [
            arena.view("csa_main_kv")[0, i].storage_offset() * 4
            for i in range(arena.entries)
        ]
        assert [b - a for a, b in pairwise(starts)] == [arena.slot_stride] * 4

    def test_an_entry_is_still_its_own_contiguous_range(self):
        arena = self.build()
        arena.view("hca_main_kv")[:, 2] = 7.0
        assert arena.entry(2).view(torch.float32).eq(7.0).any()
        assert not arena.entry(3).view(torch.float32).eq(7.0).any()

    def test_the_gap_between_entries_is_never_written(self):
        """It belongs to whatever else shares the plane — blocks, in V4."""
        arena = self.build()
        stride, size = arena.slot_stride, arena.entry_bytes
        for i in range(arena.entries - 1):
            gap = arena.buf[i * stride + size : (i + 1) * stride]
            assert gap.numel() > 0 and not gap.any()

    def test_only_the_live_tail_is_initialized(self):
        """A pool that grows takes the next index DOWN, so the live entries are
        the top of the range. Filling the rest would write over the blocks the
        boundary has not given up yet."""
        arena = self.build(entries=5, live=2)
        score = arena.view("csa_main_score")
        assert torch.isinf(score[:, 3:]).all()
        assert (score[:, :3] == 0).all()

    def test_a_stride_under_one_entry_is_rejected(self):
        with pytest.raises(ValueError, match="under the"):
            StateArena(V4_LIKE, 2, device="cpu", slot_stride=256)

    def test_a_misaligned_stride_is_rejected(self):
        with pytest.raises(ValueError, match="multiple of"):
            StateArena(
                V4_LIKE, 2, device="cpu", slot_stride=entry_bytes_for(V4_LIKE) + 8
            )


class TestPlanFieldPlanes:
    """Splitting the fields across planes of differing row width.

    A slot reserves the same row count in every plane, so what it costs is set
    by whichever plane its share overflows first.
    """

    def rows_used(self, groups, widths):
        return [-(-entry_bytes_for(g) // w) if g else 0 for g, w in zip(groups, widths)]

    def test_one_plane_takes_everything(self):
        groups, rows = plan_field_planes(V4_LIKE, [512])
        assert groups == [V4_LIKE]
        assert rows == -(-entry_bytes_for(V4_LIKE) // 512)

    def test_two_planes_beat_one_of_the_same_total_width(self):
        """640 B of row split 512/128 holds a slot in fewer rows than 512 alone
        — which is the whole reason the state goes in the row space at all."""
        _, wide_only = plan_field_planes(V4_LIKE, [512])
        _, split = plan_field_planes(V4_LIKE, [512, 128])
        assert split < wide_only

    def test_every_field_lands_in_exactly_one_plane(self):
        groups, _ = plan_field_planes(V4_LIKE, [512, 128])
        placed = [f.name for g in groups for f in g]
        assert sorted(placed) == sorted(f.name for f in V4_LIKE)

    def test_the_answer_is_the_best_of_every_assignment(self):
        widths = [512, 128]
        groups, rows = plan_field_planes(V4_LIKE, widths)
        assert max(self.rows_used(groups, widths)) == rows
        for code in range(2 ** len(V4_LIKE)):
            trial = [[], []]
            for bit, field in enumerate(V4_LIKE):
                trial[(code >> bit) & 1].append(field)
            assert max(self.rows_used(trial, widths)) >= rows

    def test_field_order_inside_a_plane_is_the_declared_one(self):
        groups, _ = plan_field_planes(V4_LIKE, [512, 128])
        for group in groups:
            names = [f.name for f in group]
            assert names == [f.name for f in V4_LIKE if f.name in names]

    def test_a_row_space_with_no_planes_is_rejected(self):
        with pytest.raises(ValueError, match="at least one plane"):
            plan_field_planes(V4_LIKE, [])


class TestCheckpointRanges:
    """Which bytes of an entry a checkpoint image holds.

    A field declaring `in_checkpoint=False` is dead at a checkpoint boundary —
    the resumer writes every row of it before reading any — so the image must
    not carry it, and must not carry its neighbours' padding by accident
    either. The flag is a bool because "some of its rows" would depend on the
    position the checkpoint was taken at, which this module is never given.
    """

    @staticmethod
    def without_hca():
        return [
            replace(f, in_checkpoint=False) if f.name.startswith("hca_") else f
            for f in V4_LIKE
        ]

    def test_an_all_carried_entry_is_one_range(self):
        assert checkpoint_ranges_for(V4_LIKE) == [(0, entry_bytes_for(V4_LIKE))]
        assert carried_bytes(V4_LIKE) == entry_bytes_for(V4_LIKE)

    def test_a_dropped_field_is_not_in_the_image(self):
        fields = self.without_hca()
        arena = StateArena(fields, 5, device="cpu")

        (start, nbytes), *rest = checkpoint_ranges_for(fields)
        assert not rest, "the four CSA fields are adjacent, so they merge"
        assert start == 0
        # Stops at the first dropped field rather than running to entry_bytes.
        assert nbytes <= arena.field_offset("hca_main_kv")
        assert carried_bytes(fields) < entry_bytes_for(fields)

    def test_a_dropped_field_breaks_the_run_it_sits_in(self):
        """Merging across it would put it back in the image."""
        fields = [
            StateField("a", 1, (4, 8), torch.float32),
            StateField("dead", 1, (64, 8), torch.float32, in_checkpoint=False),
            StateField("b", 1, (4, 8), torch.float32),
        ]
        arena = StateArena(fields, 2, device="cpu")
        dead_start = arena.field_offset("dead")
        dead_end = dead_start + fields[1].bytes_per_entry

        ranges = checkpoint_ranges_for(fields)

        assert [start for start, _ in ranges] == [
            arena.field_offset("a"),
            arena.field_offset("b"),
        ]
        for start, nbytes in ranges:
            assert start >= dead_end or start + nbytes <= dead_start

    def test_the_image_spans_the_carried_run_padding_included(self):
        """From the first carried field's start to the last one's end.

        Derived from `field_extents` rather than from the function under
        test, and not from `sum(bytes_per_entry)` either: the alignment
        between two carried fields rides along, because splitting a range to
        shave it costs more descriptor than it saves.
        """
        fields = self.without_hca()
        carried = [(s, e) for f, s, e in field_extents(fields) if f.in_checkpoint]

        assert carried_bytes(fields) == carried[-1][1] - carried[0][0]
        assert carried_bytes(fields) >= sum(
            f.bytes_per_entry for f in fields if f.in_checkpoint
        )

    def test_the_ranges_land_where_the_arena_put_the_fields(self):
        """Sizing and the copy have to read one layout, not two."""
        fields = self.without_hca()
        arena = StateArena(fields, 5, device="cpu")

        carried = [f for f in fields if f.in_checkpoint]
        (start, nbytes), *rest = checkpoint_ranges_for(fields)
        assert not rest
        assert start == arena.field_offset(carried[0].name)
        last = carried[-1]
        assert start + nbytes == arena.field_offset(last.name) + last.bytes_per_entry

    def test_a_wholly_dropped_entry_has_no_ranges(self):
        fields = [StateField("dead", 1, (8, 8), torch.float32, in_checkpoint=False)]

        assert checkpoint_ranges_for(fields) == []
        assert carried_bytes(fields) == 0

    def test_a_zero_byte_carried_run_is_not_a_range(self):
        """A range of no bytes is refused downstream, and only on first use.

        `plan_segmented_copy` rejects empty segments, and it is reached lazily
        on the first checkpoint copy -- so a field list that produced one would
        size, cross-check and start cleanly, then abort mid-serving on the
        first request to cross a rung.
        """
        empty = StateField("no_layers", 0, (4, 4), torch.float32)
        dead = StateField("dead", 1, (8, 8), torch.float32, in_checkpoint=False)

        assert checkpoint_ranges_for([empty]) == []
        # Between two dropped fields it is a run of its own, so nothing merges
        # it away either.
        assert checkpoint_ranges_for([dead, empty, dead]) == []
        assert all(n > 0 for _, n in checkpoint_ranges_for([empty, *V4_LIKE]))
