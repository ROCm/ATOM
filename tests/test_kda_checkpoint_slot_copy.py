# SPDX-License-Identifier: MIT

"""Which bytes of a KDA Active Slot a checkpoint carries, and where they are.

A K3 checkpoint is a byte copy of one slot into ordinary KV blocks. The slot is
not one range: `mamba_k_cache` and `mamba_v_cache` are both
`(num_layers, num_slots, *state)`, so one slot is `num_layers` strided pieces
per plane. This file covers turning that into an ordered byte stream, pricing
it before the tensors exist, and resolving it to real addresses afterwards.

The builder is exercised through unbound methods on a stub rather than a real
one, which would want a ModelRunner, a model and a GPU. What the stub supplies
is exactly what these methods read, so the arithmetic under test is the shipped
arithmetic.
"""

from __future__ import annotations

import math
from itertools import pairwise
from types import SimpleNamespace

import numpy as np
import pytest
import torch

# The module does `from aiter.dist.parallel_state import ...` at load, which a
# non-GPU runner cannot satisfy. `exc_type` is not optional: the module *is*
# found, so a bare `importorskip` would treat the ImportError as the caller's
# mistake and error out on pytest 9.1, which is what CI runs.
Mixin = pytest.importorskip(
    "atom.model_ops.attentions.gdn_attn",
    reason="the GDN builder's module imports aiter at load",
    exc_type=ImportError,
).GDNStateMixin

# Miniature K3: the real thing is 69 layers, conv (3, 4608) bf16 and ssm
# (12, 128, 128) fp32. Kept small enough to assert on every byte, and
# deliberately asymmetric in both shape and dtype, because a plane-agnostic
# bug survives equal planes.
N_LAYERS = 3
SHAPE_K = (3, 8)  # conv
SHAPE_V = (2, 4, 4)  # ssm
DT_K = torch.bfloat16
DT_V = torch.float32
K_BYTES = math.prod(SHAPE_K) * 2
V_BYTES = math.prod(SHAPE_V) * 4


def builder(num_slots: int = 4, allocate: bool = True):
    """A stub carrying exactly what the checkpoint-geometry methods read."""
    runner = SimpleNamespace(num_gdn_attn_state=N_LAYERS)
    if allocate:
        runner.mamba_k_cache = torch.zeros((N_LAYERS, num_slots) + SHAPE_K, dtype=DT_K)
        runner.mamba_v_cache = torch.zeros((N_LAYERS, num_slots) + SHAPE_V, dtype=DT_V)
    else:
        runner.mamba_k_cache = None
        runner.mamba_v_cache = None
    stub = SimpleNamespace(
        model_runner=runner,
        _state_shape_for_runner=lambda: (SHAPE_K, SHAPE_V),
        _state_dtypes=lambda: (DT_K, DT_V),
    )
    # The methods under test call each other through `self`. Bind the real
    # ones, so what runs is the shipped code and only the model/runner reads
    # are stubbed.
    for name in (
        "_checkpoint_plane_shapes",
        "_checkpoint_num_slots",
        "_checkpoint_layer_ranges",
        "_checkpoint_segment_sizes",
    ):
        setattr(stub, name, getattr(Mixin, name).__get__(stub, type(stub)))
    return stub


def call(stub, name: str):
    """Invoke an unbound mixin method against the stub."""
    return getattr(Mixin, name)(stub)


class TestImageIsPricedBeforeTheTensors:
    def test_the_image_is_the_whole_slot_across_both_planes(self):
        stub = builder()
        assert call(stub, "checkpoint_image_bytes") == N_LAYERS * (K_BYTES + V_BYTES)

    def test_pricing_does_not_depend_on_the_slot_count(self):
        """`checkpoint_image_bytes` is called during sizing, before the pool
        exists, and again afterwards. A slot count leaking into it would price
        the image at one shape and cut it at another."""
        sizes = {
            call(builder(num_slots=n), "checkpoint_image_bytes") for n in (1, 4, 927)
        }
        assert len(sizes) == 1

    def test_pricing_works_with_no_tensors_allocated(self):
        """Sizing runs before `allocate_per_req_cache`, so this must answer
        from the shapes alone."""
        stub = builder(allocate=False)
        assert call(stub, "checkpoint_image_bytes") == N_LAYERS * (K_BYTES + V_BYTES)

    def test_segment_sizes_are_conv_then_ssm_all_layers_of_each(self):
        """The order the layout id calls `conv-all-layers,ssm-all-layers`.

        Shapes cannot state it, and a reader assembling the image interleaved
        would get every layer but the first wrong.
        """
        stub = builder()
        assert call(stub, "_checkpoint_segment_sizes") == (
            [K_BYTES] * N_LAYERS + [V_BYTES] * N_LAYERS
        )

    def test_the_segments_sum_to_the_priced_image(self):
        """Two derivations of one number, which is the point of asserting it:
        the planner cuts by the list and the pool reserves by the sum."""
        stub = builder()
        assert sum(call(stub, "_checkpoint_segment_sizes")) == call(
            stub, "checkpoint_image_bytes"
        )


class TestSlotBasesAddressTheRealTensors:
    def test_every_base_is_the_address_torch_gives_that_layer_and_slot(self):
        """Derived from the tensors rather than from the same arithmetic the
        implementation uses, or the test would only prove it is consistent."""
        num_slots = 4
        stub = builder(num_slots=num_slots)
        bases = call(stub, "_checkpoint_slot_bases")
        k, v = stub.model_runner.mamba_k_cache, stub.model_runner.mamba_v_cache

        assert bases.shape == (num_slots, 2 * N_LAYERS)
        for s in range(num_slots):
            for layer in range(N_LAYERS):
                assert bases[s, layer] == k[layer, s].data_ptr()
                assert bases[s, N_LAYERS + layer] == v[layer, s].data_ptr()

    def test_base_order_matches_the_segment_order(self):
        """A plan's segment index addresses a row of this directly, so the two
        have to walk the planes the same way."""
        stub = builder()
        sizes = call(stub, "_checkpoint_segment_sizes")
        bases = call(stub, "_checkpoint_slot_bases")
        assert bases.shape[1] == len(sizes)

    def test_a_slot_s_segments_do_not_overlap_each_other(self):
        stub = builder()
        sizes = np.array(call(stub, "_checkpoint_segment_sizes"), dtype=np.int64)
        row = call(stub, "_checkpoint_slot_bases")[1]
        spans = sorted(zip(row.tolist(), (row + sizes).tolist()))
        assert all(a[1] <= b[0] for a, b in pairwise(spans))

    def test_no_two_slots_share_a_byte(self):
        """The failure this guards is silent: K3's slots are strided by
        `num_slots`, so an off-by-one in `(layer * num_slots + slot)` lands
        inside a neighbouring request's live state rather than off the end of
        the tensor."""
        stub = builder(num_slots=4)
        sizes = np.array(call(stub, "_checkpoint_segment_sizes"), dtype=np.int64)
        bases = call(stub, "_checkpoint_slot_bases")
        covered: set[int] = set()
        for row in bases:
            here = {
                byte
                for start, n in zip(row.tolist(), sizes.tolist())
                for byte in range(start, start + n)
            }
            assert not (here & covered)
            covered |= here

    def test_the_bases_are_cached_but_notice_a_pool_that_moved(self):
        stub = builder()
        first = call(stub, "_checkpoint_slot_bases")
        assert call(stub, "_checkpoint_slot_bases") is first

        stub.model_runner.mamba_k_cache = torch.zeros_like(
            stub.model_runner.mamba_k_cache
        )
        assert call(stub, "_checkpoint_slot_bases") is not first

    def test_a_non_contiguous_plane_is_refused_not_mis_addressed(self):
        """The affine `(layer * num_slots + slot) * nbytes` is only an address
        while the plane is contiguous. Asked once, of the layout."""
        stub = builder()
        stub.model_runner.mamba_v_cache = stub.model_runner.mamba_v_cache.transpose(
            0, 1
        )
        with pytest.raises(RuntimeError, match="contiguous"):
            call(stub, "_checkpoint_slot_bases")


class TestThePageSideIsSomebodyElsesJob:
    def test_the_mixin_refuses_to_guess_where_a_page_unit_is(self):
        """This class owns the state planes; the paged pool belongs to whichever
        builder declared it."""
        with pytest.raises(NotImplementedError, match="PAGE unit"):
            call(builder(), "_page_unit_regions")


# The destination side lives on the concrete builder, because it owns the paged
# pool. Imported separately so the mixin tests above still run if this module
# cannot load.
K3 = pytest.importorskip(
    "atom.model_ops.attentions.kimi_mla_gdn_attn",
    reason="the K3 builder's module imports aiter at load",
    exc_type=ImportError,
)


def hybrid_with_kpool_tail(num_slots: int = 4):
    """Small real hybrid-builder instance with all three checkpoint planes."""
    runner = SimpleNamespace(
        num_gdn_attn_state=N_LAYERS,
        is_deepseek_v32=True,
        config=SimpleNamespace(
            hf_config=SimpleNamespace(index_kpool=4, index_head_dim=2)
        ),
        mamba_k_cache=torch.zeros((N_LAYERS, num_slots) + SHAPE_K, dtype=DT_K),
        mamba_v_cache=torch.zeros((N_LAYERS, num_slots) + SHAPE_V, dtype=DT_V),
        kpool_tail_cache=torch.zeros((2, num_slots, 2, 4, 2), dtype=torch.bfloat16),
    )
    stub = object.__new__(K3._KimiMLAGDNCommon)
    stub.model_runner = runner
    stub._state_shape_for_runner = lambda: (SHAPE_K, SHAPE_V)
    stub._state_dtypes = lambda: (DT_K, DT_V)
    stub._index_cache_layout = lambda: ((3, 7), (3, 7))
    return stub


class TestHybridImageIncludesTheKpoolTail:
    def test_tail_is_priced_and_segmented_after_kda_state(self):
        stub = hybrid_with_kpool_tail()
        tail_per_layer = 2 * 4 * 2 * torch.bfloat16.itemsize

        assert stub._checkpoint_plane_shapes() == [
            (K_BYTES, N_LAYERS),
            (V_BYTES, N_LAYERS),
            (tail_per_layer, 2),
        ]
        assert stub.checkpoint_image_bytes() == (
            N_LAYERS * (K_BYTES + V_BYTES) + 2 * tail_per_layer
        )
        assert stub._checkpoint_segment_sizes() == (
            [K_BYTES] * N_LAYERS + [V_BYTES] * N_LAYERS + [tail_per_layer] * 2
        )

    def test_tail_slot_addresses_follow_the_two_kda_planes(self):
        stub = hybrid_with_kpool_tail()
        bases = stub._checkpoint_slot_bases()
        tail = stub.model_runner.kpool_tail_cache

        assert bases.shape == (tail.shape[1], 2 * N_LAYERS + tail.shape[0])
        for slot in range(tail.shape[1]):
            for layer in range(tail.shape[0]):
                assert bases[slot, 2 * N_LAYERS + layer] == tail[layer, slot].data_ptr()


class TestAStoreRestoreRoundTripMovesExactlyTheImage:
    """Store a slot into PAGE units, gather it back into a different slot.

    Host `memmove` rather than the copy kernel: what is under test is which
    bytes the descriptor names, and running it on the CPU keeps the test off a
    GPU. The same plan serves both directions, as it does in production.
    """

    N_LAYERS = 3
    SLOTS = 4
    ROWS = 2  # MLA layers, i.e. regions per PAGE unit
    ENTRY = 4
    LOGICAL_BS = 8
    N_UNITS = 24  # generous: the image must not need all of them

    def build(self):
        from atom.model_ops.attentions.paged_state_copy import plan_segmented_copy

        k = torch.zeros((self.N_LAYERS, self.SLOTS) + SHAPE_K, dtype=DT_K)
        v = torch.zeros((self.N_LAYERS, self.SLOTS) + SHAPE_V, dtype=DT_V)
        # Every slot a distinct byte, so a stray write into a bystander is
        # visible rather than merely possible.
        for s in range(self.SLOTS):
            k[:, s].view(torch.uint8).fill_(0x10 + s)
            v[:, s].view(torch.uint8).fill_(0x40 + s)

        region = self.LOGICAL_BS * self.ENTRY
        # Shaped like the real `kv_cache`: (rows, physical_blocks,
        # physical_block_size, entry). block_ratio is 1 here, so a physical
        # block is a logical one; the ratio > 1 case is covered above by
        # `test_the_region_is_a_logical_block_not_a_physical_one`.
        pool = torch.zeros(
            self.ROWS, self.N_UNITS, self.LOGICAL_BS, self.ENTRY, dtype=torch.uint8
        )
        runtime = SimpleNamespace(
            checkpoint_spec=SimpleNamespace(page_unit_bytes=self.ROWS * region)
        )
        runner = SimpleNamespace(
            mamba_k_cache=k,
            mamba_v_cache=v,
            kv_cache=pool,
            block_size=self.LOGICAL_BS,
            num_gdn_attn_state=self.N_LAYERS,
            state_runtime=runtime,
            is_deepseek_v32=False,
        )
        stub = SimpleNamespace(
            model_runner=runner,
            _state_shape_for_runner=lambda: (SHAPE_K, SHAPE_V),
            _state_dtypes=lambda: (DT_K, DT_V),
            _page_unit_region_cache=None,
        )
        for name in (
            "_checkpoint_plane_shapes",
            "_checkpoint_num_slots",
            "_checkpoint_layer_ranges",
            "_checkpoint_segment_sizes",
            "_checkpoint_slot_bases",
            "checkpoint_image_bytes",
        ):
            setattr(stub, name, getattr(Mixin, name).__get__(stub, type(stub)))
        for name in (
            "_page_unit_index_cache",
            "_page_unit_regions",
            "_page_unit_bases",
            "_page_unit_stream_sizes",
        ):
            method = getattr(K3._KimiMLAGDNCommon, name)
            setattr(stub, name, method.__get__(stub, type(stub)))
        return stub, k, v, plan_segmented_copy

    def round_trip(self, stub, plan_fn, units, src, dst):
        import ctypes

        image = stub.checkpoint_image_bytes()
        plan = plan_fn(
            stub._checkpoint_segment_sizes(),
            stub._page_unit_stream_sizes(len(units)),
            image,
        )
        bases = stub._checkpoint_slot_bases()
        for slot, forward in ((src, True), (dst, False)):
            desc = np.empty((plan.num_spans, 3), dtype=np.int64)
            plan.write_descriptor(
                desc,
                bases[slot][None],
                stub._page_unit_bases([units]),
                forward=forward,
            )
            for source, destination, nbytes in desc:
                ctypes.memmove(int(destination), int(source), int(nbytes))

    def test_the_image_arrives_intact_in_a_different_slot(self):
        stub, k, v, plan_fn = self.build()
        image = stub.checkpoint_image_bytes()
        assert image == N_LAYERS * (K_BYTES + V_BYTES)
        units = list(range(-(-image // (self.ROWS * self.LOGICAL_BS * self.ENTRY))))

        self.round_trip(stub, plan_fn, units, src=1, dst=3)
        assert torch.equal(k[:, 3], k[:, 1])
        assert torch.equal(v[:, 3], v[:, 1])

    def test_no_bystander_slot_is_touched(self):
        """The failure mode K3 has and V4 does not: slots are strided by
        `num_slots`, so an off-by-one in `(layer * num_slots + slot)` lands
        inside a neighbouring request's live state instead of off the end."""
        stub, k, v, plan_fn = self.build()
        image = stub.checkpoint_image_bytes()
        assert image == N_LAYERS * (K_BYTES + V_BYTES)
        units = list(range(-(-image // (self.ROWS * self.LOGICAL_BS * self.ENTRY))))
        before_k = k.clone()
        before_v = v.clone()

        self.round_trip(stub, plan_fn, units, src=1, dst=3)
        for s in (0, 2):
            assert torch.equal(k[:, s], before_k[:, s]), f"conv slot {s} moved"
            assert torch.equal(v[:, s], before_v[:, s]), f"ssm slot {s} moved"
        # The source is read, never written.
        assert torch.equal(k[:, 1], before_k[:, 1])
        assert torch.equal(v[:, 1], before_v[:, 1])

    def test_a_restore_does_not_read_past_the_image(self):
        """The last unit is only partly filled -- 9,216 spare bytes at the real
        geometry -- so an overrun has somewhere to go."""
        stub, _k, _v, plan_fn = self.build()
        image = stub.checkpoint_image_bytes()
        assert image == N_LAYERS * (K_BYTES + V_BYTES)
        unit_bytes = self.ROWS * self.LOGICAL_BS * self.ENTRY
        units = list(range(-(-image // unit_bytes)))
        assert image % unit_bytes, "the geometry must leave a tail to overrun"

        pool = stub.model_runner.kv_cache
        self.round_trip(stub, plan_fn, units, src=1, dst=3)
        # Beyond the units the image claimed, the pool is untouched.
        for row in range(self.ROWS):
            tail = pool[row].reshape(-1)[len(units) * self.LOGICAL_BS * self.ENTRY :]
            assert int(tail.sum()) == 0


class TestAnIndexerSharesThePageUnit:
    """GLM-5.3-Flash's index cache rides the same paged pool as its MLA rows.

    `sub_pool_specs` adds the index cache to the price of a block, so a unit
    is the MLA regions AND one index region per indexer layer. Naming only the
    MLA side is not a smaller copy — `units_per_checkpoint` is
    `ceil(image / page_unit_bytes)`, priced against the whole block, so the
    tail of every image would have no region to land in.
    """

    ROWS = 2  # MLA layers
    ENTRY = 4
    LOGICAL_BS = 8  # tokens per logical block
    RATIO = 2  # block_ratio: physical blocks per logical block
    # Larger than the six PAGE units this fixture's checkpoint image needs, so
    # the "past the image" assertions below inspect real storage, not empties.
    N_LOGICAL = 8
    INDEX_LAYERS = 3
    INDEX_ROWS = 2  # index rows one block owns, i.e. pooled: block_size // kpool
    INDEX_DIM = 3
    # Production stores fp8 here. Two bytes wide in the test on purpose: at one
    # byte a dropped `element_size()` is an identity, and the region arithmetic
    # is what is under test.
    INDEX_DT = torch.bfloat16

    @property
    def region(self) -> int:
        return self.LOGICAL_BS * self.ENTRY

    @property
    def index_region(self) -> int:
        return self.INDEX_ROWS * self.INDEX_DIM * self.INDEX_DT.itemsize

    @property
    def unit_bytes(self) -> int:
        return self.ROWS * self.region + self.INDEX_LAYERS * self.index_region

    def build(self, spec_bytes=None, indexed=True):
        phys_bs = self.LOGICAL_BS // self.RATIO
        cache = torch.zeros(
            self.ROWS,
            self.N_LOGICAL * self.RATIO,
            phys_bs,
            self.ENTRY,
            dtype=torch.uint8,
        )
        # `(layers, scheduler_blocks, rows_per_block, aligned_dim)`: the block
        # axis is logical already, which is why nothing here divides by
        # `block_ratio` the way the MLA side must.
        index_cache = torch.zeros(
            self.INDEX_LAYERS,
            self.N_LOGICAL,
            self.INDEX_ROWS,
            self.INDEX_DIM,
            dtype=self.INDEX_DT,
        )
        runtime = SimpleNamespace(
            checkpoint_spec=SimpleNamespace(
                page_unit_bytes=self.unit_bytes if spec_bytes is None else spec_bytes
            )
        )
        runner = SimpleNamespace(
            kv_cache=cache,
            index_cache=index_cache,
            block_size=self.LOGICAL_BS,
            state_runtime=runtime,
            is_deepseek_v32=indexed,
        )
        stub = SimpleNamespace(model_runner=runner, _page_unit_region_cache=None)
        for name in (
            "_page_unit_index_cache",
            "_page_unit_regions",
            "_page_unit_bases",
            "_page_unit_stream_sizes",
        ):
            method = getattr(K3._KimiMLAGDNCommon, name)
            setattr(stub, name, method.__get__(stub, type(stub)))
        return stub, cache, index_cache

    def test_a_unit_owns_one_region_per_mla_row_and_one_per_indexer_layer(self):
        stub, _, _ = self.build()
        base, stride = stub._page_unit_regions()
        assert len(base) == len(stride) == self.ROWS + self.INDEX_LAYERS
        assert stride.tolist() == (
            [self.region] * self.ROWS + [self.index_region] * self.INDEX_LAYERS
        )

    def test_the_index_addresses_are_the_ones_slicing_gives(self):
        """The oracle, written out here rather than kept in production."""
        stub, _, index_cache = self.build()
        for unit in range(self.N_LOGICAL):
            addrs = stub._page_unit_bases([[unit]])[0]
            for layer in range(self.INDEX_LAYERS):
                assert addrs[self.ROWS + layer] == index_cache[layer, unit].data_ptr()

    def test_the_regions_sum_to_what_the_unit_was_priced_at(self):
        """The relation the pool and the copy both depend on, and the one that
        broke when the indexer joined the pool: sizing counted the index cache
        into a block's bytes while the copy still described the MLA rows
        alone, and startup aborted on the mismatch."""
        stub, _, _ = self.build()
        spec = stub.model_runner.state_runtime.checkpoint_spec
        assert int(stub._page_unit_regions()[1].sum()) == spec.page_unit_bytes

    def test_a_unit_priced_without_the_index_cache_is_refused(self):
        stub, _, _ = self.build(spec_bytes=self.ROWS * self.region)
        with pytest.raises(RuntimeError, match="index layers"):
            stub._page_unit_regions()

    def test_a_model_without_an_indexer_keeps_the_mla_regions_alone(self):
        """K3 shares this code and prices no index cache into its blocks."""
        stub, _, _ = self.build(spec_bytes=self.ROWS * self.region, indexed=False)
        base, stride = stub._page_unit_regions()
        assert len(base) == len(stride) == self.ROWS

    def test_a_non_contiguous_index_cache_is_refused_not_mis_addressed(self):
        stub, _, index_cache = self.build()
        stub.model_runner.index_cache = index_cache.transpose(0, 1)
        with pytest.raises(RuntimeError, match="contiguous"):
            stub._page_unit_regions()

    def test_the_regions_are_worked_out_once_but_notice_a_moved_index_cache(self):
        """Two pools now, so the key has to be both: an index cache that moved
        under a cached answer scatters into whatever holds that address next."""
        stub, _, index_cache = self.build()
        first = stub._page_unit_regions()
        assert stub._page_unit_regions() is first

        stub.model_runner.index_cache = torch.zeros_like(index_cache)
        assert stub._page_unit_regions() is not first


class TestAnImageSpansBothPoolsIntact:
    """Store a slot into units that reach into the index cache, gather it back.

    Host `memmove`, as in the pure-MLA round trip above: what is under test is
    which bytes the descriptor names.
    """

    SLOTS = 4
    GEO = TestAnIndexerSharesThePageUnit()

    def build(self):
        from atom.model_ops.attentions.paged_state_copy import plan_segmented_copy

        stub, pool, index_cache = self.GEO.build()
        k = torch.zeros((N_LAYERS, self.SLOTS) + SHAPE_K, dtype=DT_K)
        v = torch.zeros((N_LAYERS, self.SLOTS) + SHAPE_V, dtype=DT_V)
        for s in range(self.SLOTS):
            k[:, s].view(torch.uint8).fill_(0x10 + s)
            v[:, s].view(torch.uint8).fill_(0x40 + s)
        runner = stub.model_runner
        runner.mamba_k_cache = k
        runner.mamba_v_cache = v
        runner.num_gdn_attn_state = N_LAYERS
        stub._state_shape_for_runner = lambda: (SHAPE_K, SHAPE_V)
        stub._state_dtypes = lambda: (DT_K, DT_V)
        for name in (
            "_checkpoint_plane_shapes",
            "_checkpoint_num_slots",
            "_checkpoint_layer_ranges",
            "_checkpoint_segment_sizes",
            "_checkpoint_slot_bases",
            "checkpoint_image_bytes",
        ):
            setattr(stub, name, getattr(Mixin, name).__get__(stub, type(stub)))
        return stub, k, v, pool, index_cache, plan_segmented_copy

    def units_for(self, stub):
        image = stub.checkpoint_image_bytes()
        return list(range(-(-image // self.GEO.unit_bytes)))

    def test_the_image_arrives_intact_in_a_different_slot(self):
        stub, k, v, _, _, plan_fn = self.build()
        units = self.units_for(stub)
        assert len(units) <= self.GEO.N_LOGICAL

        round_trip = TestAStoreRestoreRoundTripMovesExactlyTheImage.round_trip
        round_trip(self, stub, plan_fn, units, src=1, dst=3)
        assert torch.equal(k[:, 3], k[:, 1])
        assert torch.equal(v[:, 3], v[:, 1])

    def test_the_image_really_uses_its_index_regions(self):
        """Otherwise the test above would pass on a copy that quietly fit in
        the MLA rows -- which is what makes the priced tail a live question."""
        stub, _, _, _, index_cache, plan_fn = self.build()
        units = self.units_for(stub)
        mla_only = self.GEO.ROWS * self.GEO.region * len(units)
        assert stub.checkpoint_image_bytes() > mla_only

        round_trip = TestAStoreRestoreRoundTripMovesExactlyTheImage.round_trip
        round_trip(self, stub, plan_fn, units, src=1, dst=3)
        assert index_cache.view(torch.uint8).any()

    def test_no_block_beyond_the_image_is_written(self):
        """The units an image claims are pinned; anything past them belongs to
        a live request, in either pool."""
        stub, _, _, pool, index_cache, plan_fn = self.build()
        units = self.units_for(stub)
        pool_before = pool.clone()
        index_before = index_cache.clone()

        round_trip = TestAStoreRestoreRoundTripMovesExactlyTheImage.round_trip
        round_trip(self, stub, plan_fn, units, src=1, dst=3)
        for row in range(self.GEO.ROWS):
            offset = len(units) * self.GEO.region
            tail = pool[row].reshape(-1)[offset:]
            assert tail.numel() > 0
            assert torch.equal(tail, pool_before[row].reshape(-1)[offset:])
        for layer in range(self.GEO.INDEX_LAYERS):
            tail = index_cache[layer, len(units) :]
            assert tail.numel() > 0
            assert torch.equal(tail, index_before[layer, len(units) :])
