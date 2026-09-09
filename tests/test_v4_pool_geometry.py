# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Address arithmetic for the unified V4 pool.

The property that matters is not that any one formula is pretty but that the
formulas agree with each other and with the carve: every compressed row and
every sliding-window row has to land on its own plane row, inside the layer
view that addresses it. So the tests enumerate addresses and check them
pairwise rather than comparing against a restatement of the same expression.
"""

from itertools import pairwise

import pytest

from atom.model_ops.attentions.pool_layout.v4_pool_geometry import (
    ABSENT_RATIO,
    CSA_RATIO,
    DENSE_RATIO,
    HCA_RATIO,
    ClassLayout,
    UnifiedPoolGeometry,
    entry_rows_for,
    merge_abutting,
    ring_offset_for,
)

# V4-Flash-DSpark: 5 dense, 21 CSA, 20 HCA, 256-token blocks, MTP-3 lookahead.
FLASH_RATIOS = [0, 0] + [4, 128] * 20 + [4] + [0, 0, 0]
FLASH_BLOCK_SIZE = 256
FLASH_WINDOW = 131  # 128 window + 3 draft steps


def flash_geometry(num_blocks=3, num_slots=2, **kwargs):
    return UnifiedPoolGeometry(
        FLASH_RATIOS,
        num_blocks=num_blocks,
        num_slots=num_slots,
        ring_slots=FLASH_WINDOW,
        block_size=FLASH_BLOCK_SIZE,
        **kwargs,
    )


def all_addresses(geo):
    """Every row the pool can address, as `(what, absolute plane row)`.

    `what` identifies the logical slot so a collision report names both sides.
    """
    out = []
    for layer_id in range(geo.num_layers):
        cls = geo.layer_class(layer_id)
        if cls.ratio != DENSE_RATIO:
            for block in range(geo.num_blocks):
                for row in range(cls.block_rows):
                    index = geo.compress_index(layer_id, block, row)
                    out.append(
                        (
                            ("compress", layer_id, block, row),
                            index,
                            geo.absolute_row(layer_id, index),
                        )
                    )
        for group in range(geo.num_slots):
            slot = geo.physical_slot(group)
            for pos in range(geo.ring_slots):
                index = geo.window_index(layer_id, slot, pos)
                out.append(
                    (
                        ("ring_slots", layer_id, slot, pos),
                        index,
                        geo.absolute_row(layer_id, index),
                    )
                )
    return out


def minimum_entry_rows(num_layers, ring_stride, ring_slots):
    """Least height any interleaving could reach, by counting residues.

    A window row's residue mod `ring_stride` is the ring position's own
    residue, because the layer term is a whole multiple of the stride and so is
    each run's height. So all `num_layers` layers compete for rows of one
    residue, and a height of `L` only supplies `len(range(r, L, stride))` of
    them. This is an independent derivation of the number
    `entry_rows_for` returns in closed form — the two agreeing is the point.
    """
    height = 0
    for residue in range(ring_stride):
        want = num_layers * len(range(residue, ring_slots, ring_stride))
        need = residue + 1
        while len(range(residue, need, ring_stride)) < want:
            need += 1
        height = max(height, need)
    return height


class TestRingConstruction:
    def test_matches_the_residue_counting_lower_bound(self):
        for num_layers in (1, 2, 3, 5, 20, 21):
            for stride in (1, 2, 3, 7, 64):
                for ring_slots in (1, 2, 5, 64, 128, 131):
                    if ring_slots < stride:
                        continue
                    assert entry_rows_for(num_layers, stride, ring_slots) == (
                        minimum_entry_rows(num_layers, stride, ring_slots)
                    ), (num_layers, stride, ring_slots)

    def test_a_window_that_divides_the_stride_wastes_nothing(self):
        # 128 is what `win_with_spec` would be without speculative decoding,
        # and it is a multiple of both class strides — the whole overhead this
        # layout carries comes from the draft lookahead pushing past that.
        assert entry_rows_for(21, 64, 128) == 21 * 128
        assert entry_rows_for(20, 2, 128) == 20 * 128

    def test_offsets_are_injective_across_layers_and_positions(self):
        for num_layers, stride, ring_slots in (
            (21, 64, 131),
            (20, 2, 131),
            (5, 131, 131),
        ):
            seen = {}
            for layer in range(num_layers):
                for pos in range(ring_slots):
                    row = layer * stride + ring_offset_for(num_layers, stride, pos)
                    assert row not in seen, (row, seen.get(row), (layer, pos))
                    seen[row] = (layer, pos)
                    assert row < entry_rows_for(num_layers, stride, ring_slots)


class TestTheRowsAWindowActuallyReaches:
    """`entry_row_runs` against the addresses, not against its own algebra.

    A checkpoint image leaves out whatever these runs do not cover, so the
    claim under test is exactly "no `(layer, position)` maps there". Getting
    it wrong loses window rows silently — a resumer reads stale KV at the far
    end of its window, which costs a fraction of a point and crashes nothing.
    """

    @staticmethod
    def reached(num_layers, stride, ring_slots):
        return {
            layer * stride + ring_offset_for(num_layers, stride, pos)
            for layer in range(num_layers)
            for pos in range(ring_slots)
        }

    @staticmethod
    def covered(runs):
        return {row for start, count in runs for row in range(start, start + count)}

    def test_the_runs_are_exactly_the_reachable_rows(self):
        # `ring_slots < stride` is included on purpose: it is the `whole == 0`
        # branch, where the construction is nothing but per-layer partials,
        # and a window shorter than `block_size // CSA_RATIO` takes it.
        for num_layers in (1, 2, 3, 5, 20, 21):
            for stride in (1, 2, 3, 7, 64):
                for ring_slots in (1, 2, 5, 64, 128, 131, 133):
                    cls = ClassLayout(
                        ratio=4,
                        layers=tuple(range(num_layers)),
                        block_rows=0,
                        ring_stride=stride,
                        ring_slots=ring_slots,
                        envelope_offset=0,
                        entry_offset=0,
                    )
                    what = (num_layers, stride, ring_slots)
                    assert self.covered(cls.entry_row_runs()) == self.reached(
                        num_layers, stride, ring_slots
                    ), what

    def test_the_runs_are_ordered_disjoint_and_inside_the_entry(self):
        for cls in flash_geometry().classes.values():
            runs = cls.entry_row_runs()
            assert runs == sorted(runs)
            for (a_start, a_count), (b_start, _) in pairwise(runs):
                assert a_start + a_count <= b_start
            for start, count in runs:
                assert 0 <= start and start + count <= cls.entry_rows

    def test_what_they_leave_out_is_what_the_interleave_costs(self):
        for cls in flash_geometry().classes.values():
            packed = sum(count for _, count in cls.entry_row_runs())
            assert packed == cls.num_layers * cls.ring_slots
            assert cls.entry_rows - packed >= 0

    def test_a_window_that_divides_the_stride_leaves_nothing_out(self):
        """No partial ring run, so the whole class is one contiguous range."""
        cls = ClassLayout(
            ratio=4,
            layers=tuple(range(21)),
            block_rows=0,
            ring_stride=64,
            ring_slots=128,
            envelope_offset=0,
            entry_offset=0,
        )

        assert cls.entry_row_runs() == [(0, 21 * 128)]

    def test_the_pool_offsets_each_class_and_merges_where_they_meet(self):
        geo = flash_geometry()

        runs = geo.entry_row_runs()
        covered = self.covered(runs)

        assert sum(count for _, count in runs) == sum(
            c.num_layers * c.ring_slots for c in geo.classes.values()
        )
        assert covered == {
            cls.entry_offset + row
            for cls in geo.classes.values()
            for row in self.reached(cls.num_layers, cls.ring_stride, cls.ring_slots)
        }
        assert max(row for row in covered) < geo.entry_rows

    def test_every_window_index_lands_in_a_run(self):
        """The formula the kernels use, checked against the runs directly."""
        geo = flash_geometry()
        covered = self.covered(geo.entry_row_runs())

        for layer_id, ratio in enumerate(FLASH_RATIOS):
            cls = geo.layer_class(layer_id)
            for pos in range(geo.ring_slots):
                row = cls.entry_offset + cls.ring_row(cls.layer_index(layer_id), pos)
                assert row in covered, (layer_id, ratio, pos, row)


class TestFlashLayout:
    def test_class_shapes(self):
        geo = flash_geometry()
        assert [len(c.layers) for c in geo.classes.values()] == [5, 20, 21]
        assert geo.classes[CSA_RATIO].block_rows == 64
        assert geo.classes[HCA_RATIO].block_rows == 2
        assert geo.classes[DENSE_RATIO].block_rows == 0

    def test_envelope_and_entry_heights(self):
        geo = flash_geometry()
        assert geo.envelope_rows == 20 * 2 + 21 * 64
        assert geo.classes[DENSE_RATIO].entry_rows == 655
        assert geo.classes[HCA_RATIO].entry_rows == 2639
        assert geo.classes[CSA_RATIO].entry_rows == 3971
        assert geo.entry_rows == 655 + 2639 + 3971

    def test_byte_accounting(self):
        # 512 B packed fp8 NoPE + 128 B bf16 RoPE per row. The block figure has
        # to stay what the paged pool costs today minus the indexer, which is
        # its own region and not part of the row space.
        geo = flash_geometry()
        assert geo.block_bytes(512) + geo.block_bytes(128) == 885_760
        # A slot with no compressor state is its windows and, when the row
        # count comes out odd, the one row that rounds it to an even start.
        assert geo.slot_bytes(512) + geo.slot_bytes(128) == geo.slot_rows * 640

    def test_dense_layers_have_no_compressed_rows(self):
        geo = flash_geometry()
        dense = geo.classes[DENSE_RATIO].layers[0]
        with pytest.raises(ValueError, match="dense"):
            geo.compress_index(dense, 0, 0)


class TestNoTwoRowsShareAnAddress:
    @pytest.mark.parametrize(
        "num_blocks,num_slots,plane_rows",
        [(3, 2, None), (1, 1, None), (5, 3, None), (3, 2, 60_000)],
    )
    def test_every_address_is_distinct(self, num_blocks, num_slots, plane_rows):
        geo = flash_geometry(
            num_blocks=num_blocks, num_slots=num_slots, plane_rows=plane_rows
        )
        seen = {}
        for what, _index, row in all_addresses(geo):
            assert row not in seen, f"{what} collides with {seen[row]} at row {row}"
            seen[row] = what

    def test_every_address_lands_inside_its_own_layer_view(self):
        geo = flash_geometry()
        for what, index, row in all_addresses(geo):
            layer_id = what[1]
            assert 0 <= index < geo.layer_view_rows(layer_id), what
            assert 0 <= row < geo.plane_rows, what

    def test_compressed_rows_stay_in_their_own_envelope(self):
        geo = flash_geometry()
        for what, _index, row in all_addresses(geo):
            if what[0] != "compress":
                continue
            _, layer_id, block, _row = what
            cls = geo.layer_class(layer_id)
            start = block * geo.envelope_rows + cls.envelope_offset
            assert start <= row < start + cls.envelope_rows, what

    def test_window_rows_stay_in_their_own_entry(self):
        geo = flash_geometry()
        for what, _index, row in all_addresses(geo):
            if what[0] != "ring_slots":
                continue
            _, layer_id, slot, _pos = what
            cls = geo.layer_class(layer_id)
            entry_start = geo.slot_span(slot)[0] + geo.arena_rows
            start = entry_start + cls.entry_offset
            assert start <= row < start + cls.entry_rows, what


class TestIndicesAreLayerIndependent:
    """One index buffer serves every layer of a class — the whole reason the
    envelope groups layers by class. If either formula grew a layer term the
    three per-class index buffers would have to become per-layer."""

    def test_compress_index_is_shared_within_a_class(self):
        geo = flash_geometry()
        for ratio in (CSA_RATIO, HCA_RATIO):
            layers = geo.classes[ratio].layers
            for block in (0, 1, 2):
                for row in (0, geo.classes[ratio].block_rows - 1):
                    values = {geo.compress_index(lid, block, row) for lid in layers}
                    assert len(values) == 1, (ratio, block, row, values)

    def test_window_index_is_shared_within_a_class(self):
        geo = flash_geometry()
        for ratio, layers in ((r, c.layers) for r, c in geo.classes.items()):
            for slot in range(geo.num_slots):
                for pos in (0, 1, 63, 64, 130):
                    values = {geo.window_index(lid, slot, pos) for lid in layers}
                    assert len(values) == 1, (ratio, slot, pos, values)


class TestTheArenaTakesTheFrontOfASlot:
    """The compressor state shares a slot with that request's windows.

    It is bytes rather than rows and does not divide by compress class, so it
    takes whole rows off the front and each plane materializes its share at its
    own width. Being at the front is what keeps the window row count free of
    the alignment constraint — see `slot_rows`.
    """

    ARENA = 37

    def geo(self, **kwargs):
        return flash_geometry(arena_rows=self.ARENA, **kwargs)

    def test_a_slot_is_its_state_plus_its_windows_rounded_to_an_even_start(self):
        geo = self.geo()
        raw = geo.arena_rows + geo.entry_rows
        assert geo.slot_rows == raw + raw % 2
        assert geo.slot_rows % 2 == 0

    def test_windows_sit_after_the_state_in_every_slot(self):
        geo = self.geo(num_blocks=3, num_slots=4)
        for group in range(geo.num_slots):
            slot = geo.physical_slot(group)
            start, stop = geo.slot_span(slot)
            assert geo.arena_span(slot) == (start, start + geo.arena_rows)
            for layer_id in range(geo.num_layers):
                for pos in range(geo.ring_slots):
                    row = geo.absolute_row(
                        layer_id, geo.window_index(layer_id, slot, pos)
                    )
                    assert start + geo.arena_rows <= row < stop

    def test_no_state_row_is_a_compressed_row_or_another_slot(self):
        geo = self.geo(num_blocks=3, num_slots=4)
        seen = {}
        for what, _index, row in all_addresses(geo):
            seen[row] = what
        for group in range(geo.num_slots):
            slot = geo.physical_slot(group)
            start, stop = geo.arena_span(slot)
            for row in range(start, stop):
                assert row not in seen, (slot, row, seen[row])
                seen[row] = ("arena", slot)

    def test_groups_fill_the_plane_from_the_top_down(self):
        geo = self.geo(num_blocks=3, num_slots=4)
        positions = [geo.physical_slot(g) for g in range(geo.num_slots)]
        assert positions == sorted(positions, reverse=True)
        assert positions[0] == geo.slot_positions - 1
        # The last group abuts the gap, so shrinking hands back exactly the
        # rows the blocks would grow into.
        assert geo.slot_span(positions[-1])[0] >= geo.num_blocks * geo.envelope_rows

    def test_a_group_outside_the_pool_is_rejected(self):
        geo = self.geo(num_slots=2)
        with pytest.raises(ValueError, match="group 2 outside"):
            geo.physical_slot(2)

    def test_the_state_widens_a_slot_in_both_planes(self):
        without = flash_geometry()
        with_state = self.geo()
        grew = with_state.slot_rows - without.slot_rows
        for row_bytes in (512, 128):
            assert (
                with_state.slot_bytes(row_bytes) - without.slot_bytes(row_bytes)
                == grew * row_bytes
            )


class TestAWindowCarriedAsAField:
    """A window whose row is wider than a plane's lives in the state region.

    A DSpark draft layer wants unquantized KV where the pool is packed, so its
    ring cannot be rows of a plane; it takes bytes off the front of the slot
    like the compressor state and is read through a view of that plane retyped
    to its own width. The property to hold is that the rows it names, converted
    back to plane rows, land inside that slot's state region and nowhere else —
    the retype is the one step where an off-by-a-fraction-of-a-row would look
    like a valid address.
    """

    def geometry(self, rows_per_window_row, layers=1):
        """A pool whose state region holds `layers` rings and nothing else."""
        return flash_geometry(
            num_blocks=9,
            num_slots=3,
            arena_rows=FLASH_WINDOW * rows_per_window_row * layers,
            slot_align_rows=rows_per_window_row,
        )

    def plane_rows_named(self, geo, params, slot, pos, rows_per_window_row):
        """Plane rows the window row for `(slot, pos)` covers."""
        first = params.index(slot, pos) * rows_per_window_row
        return range(first, first + rows_per_window_row)

    @pytest.mark.parametrize("rows_per_window_row", [1, 2, 8])
    def test_every_row_lands_in_its_own_slot_state_region(self, rows_per_window_row):
        geo = self.geometry(rows_per_window_row)
        params = geo.field_window_params(0, rows_per_window_row)
        for group in range(geo.num_slots):
            slot = geo.physical_slot(group)
            start, stop = geo.arena_span(slot)
            for pos in range(geo.ring_slots):
                for row in self.plane_rows_named(
                    geo, params, slot, pos, rows_per_window_row
                ):
                    assert start <= row < stop, (group, pos, row, (start, stop))

    def test_two_layers_at_different_offsets_never_collide(self):
        rows = 2
        geo = self.geometry(rows, layers=2)
        # Two layers of one field: the second starts a whole ring further in.
        first = geo.field_window_params(0, rows)
        second = geo.field_window_params(geo.ring_slots * rows, rows)
        seen = set()
        for params in (first, second):
            for group in range(geo.num_slots):
                slot = geo.physical_slot(group)
                for pos in range(geo.ring_slots):
                    row = params.index(slot, pos)
                    assert row not in seen, (group, pos, row)
                    seen.add(row)

    def test_a_slot_that_does_not_divide_is_refused(self):
        # Whatever the slot came out to, the first width it does not divide by
        # has to be refused rather than silently truncated — that width is what
        # `slot_align_rows` exists to rule out at construction.
        geo = self.geometry(2)
        rows = 2
        while geo.slot_rows % rows == 0:
            rows *= 2
        with pytest.raises(ValueError, match="does not divide"):
            geo.field_window_params(0, rows)

    def test_a_field_offset_that_does_not_divide_is_refused(self):
        geo = self.geometry(2)
        with pytest.raises(ValueError, match="does not divide"):
            geo.field_window_params(1, 2)


class TestBoundary:
    def test_a_gap_moves_no_compressed_row(self):
        """Blocks grow from row 0, so widening the plane must not disturb them
        — that is what lets the boundary move without a re-carve."""
        tight = flash_geometry()
        loose = flash_geometry(plane_rows=60_000)
        for layer_id in range(tight.num_layers):
            if tight.layer_class(layer_id).ratio == DENSE_RATIO:
                continue
            assert tight.compress_index(layer_id, 2, 1) == (
                loose.compress_index(layer_id, 2, 1)
            )
            assert tight.layer_base_row(layer_id) == loose.layer_base_row(layer_id)

    def test_a_gap_moves_no_window_row(self):
        """A position is at `slot * slot_rows` whatever the plane holds, so
        widening it moves nothing — the same property blocks get from counting
        up from row 0, and the reason both regions share one base pointer."""
        tight = flash_geometry()
        loose = flash_geometry(plane_rows=60_000)
        for layer_id in (0, 2, 3):
            for slot in range(4):
                assert tight.absolute_row(
                    layer_id, tight.window_index(layer_id, slot, 7)
                ) == loose.absolute_row(layer_id, loose.window_index(layer_id, slot, 7))

    def test_a_plane_too_small_for_the_split_is_rejected(self):
        with pytest.raises(ValueError, match="cannot hold"):
            flash_geometry(plane_rows=100)


class TestRejects:
    def test_unknown_ratio(self):
        with pytest.raises(ValueError, match="unknown V4 compress ratios"):
            UnifiedPoolGeometry([0, 4, 7], 1, 1, ring_slots=8, block_size=256)

    def test_no_layers(self):
        with pytest.raises(ValueError, match="at least one layer"):
            UnifiedPoolGeometry([], 1, 1, ring_slots=8, block_size=256)

    def test_window_out_of_range(self):
        geo = flash_geometry()
        with pytest.raises(ValueError, match="ring position"):
            geo.window_index(0, 0, FLASH_WINDOW)


class TestTheBoundaryIsInvisibleToAddresses:
    """Moving the compress/window split must change no address.

    `swa_write` and `csa_translate_pack` run inside the captured decode graph,
    and a captured launch keeps its scalar arguments by value — so an address
    term that varied with the split would need a re-capture, not just a fresh
    argument. It does not vary: both blocks and slots are addressed by position
    from row 0, so neither region's addressing knows where the other stops, and
    the two counts only gate which ids the allocator may hand out.

    What the split *does* reach is `physical_slot`, since the topmost position
    depends on how many the plane holds. That is why a fixed `plane_rows` is a
    hard requirement rather than a tuning choice — see the tight-plane case.
    """

    PLANE = 60_000
    SPLITS = ((3, 2), (2, 5), (7, 1), (1, 1))

    def geometries(self):
        return [
            flash_geometry(num_blocks=nb, num_slots=ns, plane_rows=self.PLANE)
            for nb, ns in self.SPLITS
        ]

    def test_window_params_are_identical_across_splits(self):
        for ratio in (DENSE_RATIO, HCA_RATIO, CSA_RATIO):
            params = {g.window_params(ratio) for g in self.geometries()}
            assert len(params) == 1, (ratio, params)

    def test_every_address_is_identical_across_splits(self):
        first, *rest = self.geometries()
        for geo in rest:
            for layer_id in range(geo.num_layers):
                for slot, pos in ((0, 0), (0, 130), (1, 7), (1, 64)):
                    assert geo.window_index(layer_id, slot, pos) == (
                        first.window_index(layer_id, slot, pos)
                    ), (layer_id, slot, pos)
                if geo.layer_class(layer_id).ratio == DENSE_RATIO:
                    continue
                assert geo.compress_index(layer_id, 0, 1) == (
                    first.compress_index(layer_id, 0, 1)
                )
                assert geo.layer_base_row(layer_id) == first.layer_base_row(layer_id)

    def test_a_tight_plane_does_move_with_the_split(self):
        """The counter-case, so the requirement above is not mistaken for
        something the formulas guarantee on their own: with `plane_rows` left to
        fit the current split, the split is back inside every window address.
        An elastic pool has to allocate a fixed capacity."""
        tight = [flash_geometry(num_blocks=nb, num_slots=ns) for nb, ns in self.SPLITS]
        assert len({g.physical_slot(0) for g in tight}) > 1
        # The formula itself is innocent — it never mentions the plane.
        assert len({g.window_params(CSA_RATIO) for g in tight}) == 1


# V4-Pro: an all-CSA/HCA trunk whose only ratio-0 layer is the draft slot. The
# shape matters because it is the one FLASH_RATIOS cannot express — Flash has
# two dense layers in the trunk, which keep the dense class alive no matter
# what happens to the draft.
PRO_RATIOS = [128, 128] + [4, 128] * 29 + [4] + [0]


class TestAClassCanBeAbsent:
    """A compress class with no layers is not in the layout at all.

    The dense class used to be in every V4 config, so callers grew a habit of
    asking for it unconditionally. Then a layer that carries its window in a
    state field started leaving the row space entirely, and on a trunk with no
    dense layers of its own the draft was the last one — the class goes with
    it. What must not happen is that this reads as a layout with a dense class
    whose address happens to be wrong.
    """

    def geometry(self, ratios):
        return UnifiedPoolGeometry(
            ratios, num_blocks=4, num_slots=3, ring_slots=FLASH_WINDOW, block_size=256
        )

    def test_a_trunk_dense_layer_keeps_the_class_when_the_draft_leaves(self):
        flash = list(FLASH_RATIOS)
        for i in (43, 44, 45):
            flash[i] = ABSENT_RATIO
        assert DENSE_RATIO in self.geometry(flash).classes

    def test_the_class_goes_with_the_last_layer_that_had_it(self):
        pro = list(PRO_RATIOS)
        assert DENSE_RATIO in self.geometry(pro).classes
        pro[-1] = ABSENT_RATIO
        assert DENSE_RATIO not in self.geometry(pro).classes

    def test_asking_an_absent_class_for_an_address_raises(self):
        pro = list(PRO_RATIOS)
        pro[-1] = ABSENT_RATIO
        geo = self.geometry(pro)
        with pytest.raises(KeyError):
            geo.window_params(DENSE_RATIO)

    def test_the_classes_that_remain_are_laid_out_as_if_it_never_existed(self):
        """The absent class must not leave a hole the others address around."""
        pro = list(PRO_RATIOS)
        pro[-1] = ABSENT_RATIO
        geo = self.geometry(pro)
        without = self.geometry(pro[:-1])
        assert geo.entry_rows == without.entry_rows
        assert geo.envelope_rows == without.envelope_rows
        for ratio in (CSA_RATIO, HCA_RATIO):
            assert geo.window_params(ratio) == without.window_params(ratio)


class TestMergeAbuttingRefusesDisorder:
    """Ordering is a requirement of `merge_abutting`, not a property of callers.

    Each run is compared only against the one before it, so an unordered input
    merges nothing and reads as legal. Nothing downstream can see the result is
    wrong: `checkpoint_image_bytes` over-counts, and both the op validator and
    the sizing cross-check compare against that same number.
    """

    def test_an_overlapping_run_is_refused(self):
        # Returned [(0, 400), (256, 128)] before: 528 bytes claimed for 400.
        with pytest.raises(ValueError, match="ascend and not overlap"):
            merge_abutting([(0, 400), (256, 64), (320, 64)])

    def test_a_descending_run_is_refused(self):
        with pytest.raises(ValueError, match="ascend and not overlap"):
            merge_abutting([(256, 64), (0, 64)])

    def test_a_negative_run_is_refused(self):
        with pytest.raises(ValueError, match="non-negative"):
            merge_abutting([(0, -1)])

    def test_abutting_and_separated_runs_are_unchanged(self):
        assert merge_abutting([(0, 4), (4, 4), (12, 4)]) == [(0, 8), (12, 4)]
        assert merge_abutting([]) == []
