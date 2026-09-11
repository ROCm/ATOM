# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Config-driven compress ratios, and the reuse bias that rides on them.

DeepSeek-V4.1-Flash keeps V4's two compressed classes and moves both ratios
-- 2 and 1 rather than 4 and 128 -- and only four of its forty layers own a
compressor; the rest read one of those four. The layout does not move for
them. `layer_base_row` places a layer's envelope rows and its sliding-window
ring with one number, so pointing a reuse layer at its owner would take the
ring along, and one ring serving two layers is two tokens in one row. A
reuse layer therefore keeps both, its envelope rows simply going unwritten,
and only the compressed half of its index is redirected -- by the additive
term `UnifiedPoolGeometry.compress_bias` gives.

These tests pin that term against the layout it is derived from, and the
ratio-driven layout against the row accounting it has to keep satisfying at
ratio 1, where a block compresses to one row per token rather than fewer.
"""

import pytest
import torch

from atom.model_ops.attentions.pool_layout.v4_pool_geometry import (
    CSA_RATIO,
    DENSE_RATIO,
    HCA_RATIO,
    UnifiedPoolGeometry,
    compress_class_ratios,
    owner_layers,
)
from atom.model_ops.v4_kernels.csa_translate_pack import csa_translate_pack_reference
from atom.models.deepseek_v4 import DeepseekV4Args, _should_skip_v4_index_topk

# V4-Flash-DSpark, as `tests/test_v4_pool_geometry.py` spells it.
V4_RATIOS = [0, 0] + [4, 128] * 20 + [4] + [0, 0, 0]

# V4.1-Flash: 2 dense, 18 at ratio 2, 20 at ratio 1. Four layers own the
# compressed KV and the indexer K; every layer keeps its own 128-wide window.
V41_RATIOS = [0, 0] + [2] * 18 + [1] * 20
V41_KV_SOURCES = (2, 8, 14, 20)
V41_INDEX_SOURCES = (2, 8, 14, 20, 24, 28, 32, 36)
BLOCK_SIZE = 256
WINDOW = 128


def v41_geometry(num_blocks=3, num_slots=2, **kwargs):
    return UnifiedPoolGeometry(
        V41_RATIOS,
        num_blocks=num_blocks,
        num_slots=num_slots,
        ring_slots=WINDOW,
        block_size=BLOCK_SIZE,
        **kwargs,
    )


class TestTheTwoClassesComeFromTheConfig:
    """Which ratios the two compressed slots carry is the config's to say."""

    def test_v4_is_unchanged(self):
        assert compress_class_ratios(V4_RATIOS) == (CSA_RATIO, HCA_RATIO)

    def test_v41_moves_both(self):
        assert compress_class_ratios(V41_RATIOS) == (1, 2)

    def test_one_class_leaves_the_other_where_v4_had_it(self):
        assert compress_class_ratios([0, 4, 4]) == (CSA_RATIO, HCA_RATIO)
        assert compress_class_ratios([0, 128]) == (CSA_RATIO, HCA_RATIO)
        assert compress_class_ratios([0, 0]) == (CSA_RATIO, HCA_RATIO)

    def test_a_third_class_is_refused(self):
        with pytest.raises(ValueError, match="two compressed classes"):
            compress_class_ratios([0, 1, 2, 4])

    def test_the_envelope_orders_the_classes_by_descending_ratio(self):
        """The order has to be a function of the ratios and nothing else, or
        two runs of one model lay their blocks out differently."""
        geo = v41_geometry()
        compressed = [c for c in geo.classes.values() if c.ratio != DENSE_RATIO]
        by_offset = sorted(compressed, key=lambda c: c.envelope_offset)
        assert [c.ratio for c in by_offset] == [2, 1]
        assert geo.classes[2].envelope_offset == 0
        assert geo.classes[1].envelope_offset == 18 * 128

    def test_ratio_one_keeps_one_compressed_row_per_token(self):
        geo = v41_geometry()
        assert geo.classes[1].block_rows == BLOCK_SIZE
        assert geo.classes[2].block_rows == BLOCK_SIZE // 2
        assert geo.classes[DENSE_RATIO].block_rows == 0

    def test_envelope_row_accounting(self):
        geo = v41_geometry()
        assert geo.envelope_rows == 18 * 128 + 20 * 256
        assert geo.envelope_rows == sum(c.envelope_rows for c in geo.classes.values())

    def test_entry_row_accounting(self):
        geo = v41_geometry()
        assert geo.entry_rows == sum(c.entry_rows for c in geo.classes.values())
        # Every layer's whole ring fits inside its own class's part of a slot,
        # one row each -- the property `entry_rows_for` is sized for, at a
        # class whose layer stride is now as wide as a whole block.
        for cls in geo.classes.values():
            reached = {
                cls.ring_row(i, pos)
                for i in range(cls.num_layers)
                for pos in range(WINDOW)
            }
            assert len(reached) == cls.num_layers * WINDOW
            assert max(reached) < cls.entry_rows

    def test_no_two_rows_share_a_plane_row(self):
        """The invariant the whole layout exists for, re-run at the new
        ratios: ratio 1 makes a class's layer stride as wide as a block, which
        is the case a formula tuned to 64-row strides could get wrong."""
        geo = v41_geometry()
        seen = {}
        for layer_id in range(geo.num_layers):
            cls = geo.layer_class(layer_id)
            base = geo.layer_base_row(layer_id)
            for block in range(geo.num_blocks):
                for row in range(cls.block_rows):
                    what = ("compress", layer_id, block, row)
                    at = base + geo.compress_index(layer_id, block, row)
                    assert at not in seen, f"{what} collides with {seen[at]}"
                    seen[at] = what
            for group in range(geo.num_slots):
                slot = geo.physical_slot(group)
                for pos in range(WINDOW):
                    what = ("window", layer_id, slot, pos)
                    at = base + geo.window_index(layer_id, slot, pos)
                    assert at not in seen, f"{what} collides with {seen[at]}"
                    assert 0 <= at < geo.plane_rows, what
                    seen[at] = what


class TestTheOwnerMap:
    def test_v4_layers_own_what_they_read(self):
        assert owner_layers(V4_RATIOS, None) == tuple(
            None if r <= 0 else i for i, r in enumerate(V4_RATIOS)
        )

    def test_the_v41_kv_map(self):
        owners = owner_layers(V41_RATIOS, V41_KV_SOURCES)
        assert owners[:2] == (None, None)
        assert owners[2:8] == (2,) * 6
        assert owners[8:14] == (8,) * 6
        assert owners[14:20] == (14,) * 6
        assert owners[20:40] == (20,) * 20

    def test_an_owner_precedes_every_layer_that_reads_it(self):
        for sources in (V41_KV_SOURCES, V41_INDEX_SOURCES):
            for layer_id, owner in enumerate(owner_layers(V41_RATIOS, sources)):
                assert owner is None or owner <= layer_id

    def test_a_layer_must_share_its_owners_class(self):
        """Layer 20 changes class, so letting layer 19 read it would leave the
        two disagreeing about how many rows a block holds -- and the window
        half of the index, which the bias does not touch, wrong as well."""
        with pytest.raises(ValueError, match="same compress class"):
            owner_layers(V41_RATIOS, (2, 8, 14, 19))

    def test_a_compressed_layer_with_no_source_before_it_is_refused(self):
        with pytest.raises(ValueError, match="no source layer precedes it"):
            owner_layers(V41_RATIOS, (8, 14, 20))

    def test_an_empty_table_reads_as_no_table(self):
        assert owner_layers(V41_RATIOS, ()) == owner_layers(V41_RATIOS, None)

    def test_it_agrees_with_the_v41_model_s_own_map(self):
        """`atom/models/deepseek_v41.py` derives the same table for its own
        use. Two derivations of one map drift; this is where they meet."""
        from atom.models.deepseek_v41 import DeepseekV41Args, V41LayerMap

        layer_map = V41LayerMap.from_args(
            DeepseekV41Args(
                compress_ratios=tuple(V41_RATIOS),
                kv_source_layer_ids=V41_KV_SOURCES,
                index_source_layer_ids=V41_INDEX_SOURCES,
            )
        )
        for sources, theirs in (
            (V41_KV_SOURCES, layer_map.kv_owner),
            (V41_INDEX_SOURCES, layer_map.index_owner),
        ):
            # One derivation: the model's map delegates to `owner_layers`.
            assert owner_layers(V41_RATIOS, sources) == theirs


class TestCompressBias:
    def test_an_owner_biases_by_zero(self):
        geo = v41_geometry()
        for owner in V41_KV_SOURCES:
            assert geo.compress_bias(owner, owner) == 0

    def test_the_v41_bias_values(self):
        """`(owner's place in its class - this layer's) * the class's rows per
        block`. Layers 2-19 are the first 18 of the ratio-2 class in groups of
        six, and 20-39 are the whole ratio-1 class."""
        geo = v41_geometry()
        got = {
            lid: geo.compress_bias(lid, owner)
            for lid, owner in enumerate(owner_layers(V41_RATIOS, V41_KV_SOURCES))
            if owner is not None
        }
        want = {lid: -((lid - 2) % 6) * 128 for lid in range(2, 20)}
        want.update({lid: -(lid - 20) * 256 for lid in range(20, 40)})
        assert got == want

    def test_a_biased_index_lands_on_the_owners_own_plane_row(self):
        """The property the bias exists for, stated in plane rows -- the one
        address space the two layers' views share."""
        geo = v41_geometry()
        for lid, owner in enumerate(owner_layers(V41_RATIOS, V41_KV_SOURCES)):
            if owner is None:
                continue
            bias = geo.compress_bias(lid, owner)
            cls = geo.layer_class(lid)
            for block in range(geo.num_blocks):
                for row in (0, cls.block_rows - 1):
                    reuse = geo.absolute_row(
                        lid, geo.compress_index(lid, block, row) + bias
                    )
                    own = geo.absolute_row(owner, geo.compress_index(owner, block, row))
                    assert reuse == own, (lid, owner, block, row)

    def test_a_reuse_layer_keeps_its_own_window(self):
        """The bias is on the compressed half only. Two layers of one class
        share a window formula but not a window row -- the layer term is in
        the view base -- so the ring must stay where it was."""
        geo = v41_geometry()
        slot = geo.physical_slot(0)
        for lid, owner in enumerate(owner_layers(V41_RATIOS, V41_KV_SOURCES)):
            if owner is None or owner == lid:
                continue
            for pos in (0, 1, WINDOW - 1):
                mine = geo.absolute_row(lid, geo.window_index(lid, slot, pos))
                theirs = geo.absolute_row(owner, geo.window_index(owner, slot, pos))
                assert mine != theirs, (lid, owner, pos)

    def test_a_cross_class_bias_is_refused(self):
        geo = v41_geometry()
        with pytest.raises(ValueError, match="different compress classes"):
            geo.compress_bias(20, 19)

    def test_a_dense_layer_has_no_bias(self):
        geo = v41_geometry()
        with pytest.raises(ValueError, match="dense"):
            geo.compress_bias(0, 0)

    def test_v4_biases_by_zero_everywhere(self):
        """No config field, no redirection: V4's rows are where they were."""
        geo = UnifiedPoolGeometry(
            V4_RATIOS, num_blocks=3, num_slots=2, ring_slots=131, block_size=BLOCK_SIZE
        )
        for lid, owner in enumerate(owner_layers(V4_RATIOS, None)):
            if owner is not None:
                assert geo.compress_bias(lid, owner) == 0


class TestTheTranslatorTakesTheBias:
    """`csa_translate_pack` is the one place a compressed row is stored per
    layer, so it is where the redirection has to land. Stated here on the
    pure-torch reference; `test_the_kernel_and_the_reference_agree_on_a_bias`
    is what keeps the Triton kernel saying the same thing.
    """

    ENVELOPE_ROWS = 10
    CAPACITY = 2

    def _run(self, bias):
        indices = torch.full((6,), -1, dtype=torch.int32)
        csa_translate_pack_reference(
            torch.tensor([[0, 1, 2, -1], [3, 2, 1, -1]], dtype=torch.int32),
            torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.int32),
            torch.tensor([5, 7], dtype=torch.int32),
            torch.tensor([0, 3, 6], dtype=torch.int32),
            torch.tensor([0, 1], dtype=torch.int32),
            torch.zeros(2, dtype=torch.int32),
            indices,
            envelope_rows=self.ENVELOPE_ROWS,
            csa_block_capacity=self.CAPACITY,
            compress_bias=bias,
        )
        return indices.tolist()

    def test_no_bias_is_the_row_it_wrote_before(self):
        # Seq 0's page table is [0, 1, 2]: topk 0, 1, 2 are block 0 row 0,
        # block 0 row 1 and block 1 row 0. Seq 1's is [3, 4, 5].
        assert self._run(0) == [0, 1, 10, 41, 40, 31]

    def test_the_bias_shifts_every_row_by_itself(self):
        base = self._run(0)
        for bias in (-256, -128, 128):
            assert self._run(bias) == [row + bias for row in base]


class TestTheSkipTopkGate:
    """Who reuses an earlier layer's top-k. A table states it; with no table
    the V4 heuristic decides, unchanged."""

    def v41(self, **kwargs):
        return DeepseekV4Args(
            compress_ratios=tuple(V41_RATIOS),
            index_source_layer_ids=V41_INDEX_SOURCES,
            **kwargs,
        )

    def test_the_table_reproduces_the_v41_source_map(self):
        args = self.v41()
        skipped = [lid for lid in range(40) if _should_skip_v4_index_topk(args, lid)]
        assert skipped == [i for i in range(2, 40) if i not in V41_INDEX_SOURCES]

    def test_a_source_layer_refreshes(self):
        args = self.v41()
        for source in V41_INDEX_SOURCES:
            assert not _should_skip_v4_index_topk(args, source)

    def test_a_dense_layer_never_skips(self):
        args = self.v41()
        assert not _should_skip_v4_index_topk(args, 0)
        assert not _should_skip_v4_index_topk(args, 1)

    def test_the_table_does_not_need_the_v4_cache_flag(self):
        """V4 reuse is an optimization behind `use_index_cache`; V4.1's is
        structural -- the consumer has no indexer of its own to fall back
        on."""
        assert self.v41().use_index_cache is False
        assert _should_skip_v4_index_topk(self.v41(), 3)

    def test_a_reuse_layer_with_no_source_before_it_is_refused(self):
        args = DeepseekV4Args(
            compress_ratios=tuple(V41_RATIOS), index_source_layer_ids=(8, 14, 20)
        )
        with pytest.raises(ValueError, match="no source layer"):
            _should_skip_v4_index_topk(args, 3)

    def test_no_table_leaves_the_v4_heuristic_alone(self):
        args = DeepseekV4Args(
            compress_ratios=tuple(V4_RATIOS), use_index_cache=True, index_topk_freq=2
        )
        csa = [i for i, r in enumerate(V4_RATIOS) if r == CSA_RATIO]
        # Every second CSA layer refreshes; the ones between reuse it.
        assert [_should_skip_v4_index_topk(args, lid) for lid in csa[:4]] == [
            False,
            True,
            False,
            True,
        ]

    def test_an_empty_table_falls_back_to_the_v4_heuristic(self):
        args = DeepseekV4Args(
            compress_ratios=tuple(V41_RATIOS), index_source_layer_ids=()
        )
        assert not any(_should_skip_v4_index_topk(args, lid) for lid in range(40))

    def test_no_table_and_no_cache_flag_never_skips(self):
        args = DeepseekV4Args(compress_ratios=tuple(V4_RATIOS), index_topk_freq=2)
        assert not any(
            _should_skip_v4_index_topk(args, lid) for lid in range(len(V4_RATIOS))
        )


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="runs the Triton translate kernel"
)
@pytest.mark.parametrize("bias", [0, -256, 128])
def test_the_kernel_and_the_reference_agree_on_a_bias(bias):
    """The bias is a `constexpr` the kernel folds into its store, so it is
    the one part of the redirection a host-side test cannot see."""
    from atom.model_ops.v4_kernels.csa_translate_pack import csa_translate_pack

    dev = "cuda"
    topk = torch.tensor([[0, 1, 2, -1], [3, 2, 1, -1]], dtype=torch.int32, device=dev)
    block_tables = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.int32, device=dev)
    positions = torch.tensor([5, 7], dtype=torch.int32, device=dev)
    indptr = torch.tensor([0, 3, 6], dtype=torch.int32, device=dev)
    bids = torch.tensor([0, 1], dtype=torch.int32, device=dev)
    skip = torch.zeros(2, dtype=torch.int32, device=dev)

    args = dict(envelope_rows=10, csa_block_capacity=2, compress_bias=bias)
    got = torch.full((6,), -1, dtype=torch.int32, device=dev)
    csa_translate_pack(topk, block_tables, positions, indptr, bids, skip, got, **args)
    want = torch.full((6,), -1, dtype=torch.int32, device=dev)
    csa_translate_pack_reference(
        topk, block_tables, positions, indptr, bids, skip, want, **args
    )
    assert got.tolist() == want.tolist()


def test_hca_is_still_the_coarse_slot():
    """A guard on the naming rather than on a number: `HCA_RATIO` is what the
    coarse slot holds when a config names only one class, and several callers
    still spell V4's pair that way."""
    assert compress_class_ratios([0, CSA_RATIO]) == (CSA_RATIO, HCA_RATIO)
