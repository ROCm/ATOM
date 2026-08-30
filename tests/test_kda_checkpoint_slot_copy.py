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


class TestPageUnitAddressesAreArithmetic:
    """The addresses `_page_unit_regions` computes are the ones slicing gives.

    Replacing a view per row per unit with one multiplication is only safe if
    it lands on the same bytes, so this asks both and compares. The slicing
    expression is written out here rather than kept in production: it is the
    oracle, not a fallback.
    """

    ROWS = 3  # MLA layers
    ENTRY = 4
    LOGICAL_BS = 2  # tokens per logical block
    RATIO = 4  # block_ratio: physical blocks per logical block
    N_LOGICAL = 5

    def build(self, block_size=None, spec_bytes=None):
        phys_bs = self.LOGICAL_BS // self.RATIO or 1
        cache = torch.arange(
            self.ROWS * self.N_LOGICAL * self.RATIO * phys_bs * self.ENTRY,
            dtype=torch.uint8,
        ).reshape(self.ROWS, self.N_LOGICAL * self.RATIO, phys_bs, self.ENTRY)
        region = self.LOGICAL_BS * self.ENTRY
        runtime = SimpleNamespace(
            checkpoint_spec=SimpleNamespace(
                page_unit_bytes=self.ROWS * region if spec_bytes is None else spec_bytes
            )
        )
        runner = SimpleNamespace(
            kv_cache=cache,
            block_size=self.LOGICAL_BS if block_size is None else block_size,
            state_runtime=runtime,
        )
        stub = SimpleNamespace(model_runner=runner, _page_unit_region_cache=None)
        for name in (
            "_page_unit_regions",
            "_page_unit_bases",
            "_page_unit_stream_sizes",
        ):
            method = getattr(K3._KimiMLAGDNCommon, name)
            setattr(stub, name, method.__get__(stub, type(stub)))
        return stub, cache

    def test_a_unit_addresses_one_region_per_row(self):
        stub, cache = self.build()
        base, stride = stub._page_unit_regions()
        assert len(base) == len(stride) == self.ROWS
        # The oracle: flatten each row and slice the logical block out of it.
        region = self.LOGICAL_BS * self.ENTRY
        for unit in range(self.N_LOGICAL):
            addrs = stub._page_unit_bases([[unit]])[0]
            for row in range(self.ROWS):
                want = cache[row].reshape(-1)[unit * region :].data_ptr()
                assert addrs[row] == want

    def test_the_stream_sizes_match_the_addresses_one_for_one(self):
        stub, _ = self.build()
        units = [0, 3, 1]
        addrs = stub._page_unit_bases([units])[0]
        sizes = stub._page_unit_stream_sizes(len(units))
        assert len(addrs) == len(sizes) == self.ROWS * len(units)

    def test_an_image_covers_its_units_in_order_unit_major_row_minor(self):
        """`_checkpoint_copy_plan` builds the destination stream this way, so
        a plan's segment index has to mean the same thing here."""
        stub, _ = self.build()
        units = [4, 0, 2]  # deliberately not ascending: ids are not a range
        addrs = stub._page_unit_bases([units])[0]
        base, stride = stub._page_unit_regions()
        want = [base[r] + u * stride[r] for u in units for r in range(self.ROWS)]
        assert addrs.tolist() == want

    def test_the_region_is_a_logical_block_not_a_physical_one(self):
        """The `block_ratio` trap: the tensor is shaped in physical blocks, but
        `unit_ids` carries logical ones, and K3's ratio is 128."""
        stub, _ = self.build()
        _, stride = stub._page_unit_regions()
        assert set(stride.tolist()) == {self.LOGICAL_BS * self.ENTRY}

    def test_a_granularity_mismatch_is_refused_at_startup(self):
        """Wrong granularity would not raise on its own -- it would scatter a
        checkpoint across the wrong blocks. The one relation that cannot hold
        if it is wrong is asserted instead."""
        stub, _ = self.build(block_size=self.LOGICAL_BS * 2)
        with pytest.raises(RuntimeError, match="granularity"):
            stub._page_unit_regions()

    def test_a_non_contiguous_pool_is_refused_not_mis_addressed(self):
        stub, _ = self.build()
        stub.model_runner.kv_cache = stub.model_runner.kv_cache.transpose(0, 1)
        with pytest.raises(RuntimeError, match="contiguous"):
            stub._page_unit_regions()

    def test_the_regions_are_worked_out_once_but_notice_a_moved_pool(self):
        stub, _ = self.build()
        first = stub._page_unit_regions()
        assert stub._page_unit_regions() is first

        stub.model_runner.kv_cache = torch.zeros_like(stub.model_runner.kv_cache)
        assert stub._page_unit_regions() is not first


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
