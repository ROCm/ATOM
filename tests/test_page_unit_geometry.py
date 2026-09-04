# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Where a checkpoint image's bytes land in the MLA paged pool.

The CPU tier's only seam onto PAGE units is `page_unit_views`, and its address
counterpart is `_page_unit_regions` / `_page_unit_bases`. Both are pure
arithmetic over the runner's KV pool, so they live in
`atom.model_ops.attentions.pool_layout` -- the package for exactly that, whose
members reach neither aiter nor the rest of atom -- and are tested here,
directly, on the non-GPU runner.

This is deliberately split out of `test_kda_checkpoint_slot_copy.py`: that file
`importorskip`s the K3/GDN builder modules (they import aiter at load), so on a
CI runner without aiter the *whole* module is skipped, and these PAGE-unit tests
-- the ones that decide where every checkpoint byte goes -- never ran there. The
builder mixes `PageUnitGeometryMixin` in over `GDNStateMixin`, so what runs here
is the shipped arithmetic.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from atom.model_ops.attentions.pool_layout.page_unit_geometry import (
    PageUnitGeometryMixin,
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

    def build(self, block_size=None, spec_bytes=None, image_bytes=None, dtype=None):
        phys_bs = self.LOGICAL_BS // self.RATIO or 1
        dtype = dtype or torch.uint8
        cache = (
            torch.arange(
                self.ROWS * self.N_LOGICAL * self.RATIO * phys_bs * self.ENTRY,
                dtype=torch.int64,
            )
            .to(dtype)
            .reshape(self.ROWS, self.N_LOGICAL * self.RATIO, phys_bs, self.ENTRY)
        )
        region = self.LOGICAL_BS * self.ENTRY * cache.element_size()
        spec = SimpleNamespace(
            page_unit_bytes=self.ROWS * region if spec_bytes is None else spec_bytes
        )
        # Left OFF unless a test asks for it: `page_unit_views` trims to
        # `image_bytes` only when the spec carries one, and the addressing tests
        # above are about whole units.
        if image_bytes is not None:
            spec.image_bytes = image_bytes
        runtime = SimpleNamespace(checkpoint_spec=spec)
        runner = SimpleNamespace(
            kv_cache=cache,
            block_size=self.LOGICAL_BS if block_size is None else block_size,
            state_runtime=runtime,
        )
        stub = SimpleNamespace(model_runner=runner, _page_unit_region_cache=None)
        for name in (
            "_page_unit_index_cache",
            "_page_unit_regions",
            "_page_unit_bases",
            "_page_unit_stream_sizes",
        ):
            method = getattr(PageUnitGeometryMixin, name)
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


class TestPageUnitViewsNameTheSameBytes:
    """`page_unit_views` is the CPU tier's seam onto the PAGE units.

    `_page_unit_regions` gives the Triton descriptor raw addresses; the LMCache
    packer wants tensors. Two ways to name one set of bytes is exactly the kind
    of pair that drifts, so every test here asks both and compares.
    """

    H = TestPageUnitAddressesAreArithmetic

    def build(self, **kw):
        stub, cache = self.H.build(self.H(), **kw)
        stub.page_unit_views = PageUnitGeometryMixin.page_unit_views.__get__(
            stub, type(stub)
        )
        return stub, cache

    def test_a_view_starts_exactly_where_the_address_says(self):
        stub, _ = self.build()
        units = [4, 0, 2]  # not ascending: ids are not a range
        views = stub.page_unit_views(units)
        addrs = stub._page_unit_bases([units])[0]
        assert len(views) == len(addrs) == self.H.ROWS * len(units)
        assert [v.data_ptr() for v in views] == addrs.tolist()

    def test_a_view_is_exactly_one_region_long(self):
        stub, _ = self.build()
        views = stub.page_unit_views([1])
        sizes = stub._page_unit_stream_sizes(1)
        assert [v.numel() * v.element_size() for v in views] == sizes.tolist()

    def test_the_order_is_unit_major_row_minor(self):
        """The packer indexes segments positionally, so this order IS the blob
        layout -- and `_checkpoint_copy_plan` built the HBM stream the same
        way. A build that ordered it differently must not read another's blob,
        which is what `layout_id` in the key covers."""
        stub, _ = self.build()
        units = [3, 1]
        base, stride = stub._page_unit_regions()
        want = [base[r] + u * stride[r] for u in units for r in range(self.H.ROWS)]
        assert [v.data_ptr() for v in stub.page_unit_views(units)] == want

    def test_every_view_is_contiguous(self):
        """The Triton packer refuses a strided segment."""
        stub, _ = self.build()
        assert all(v.is_contiguous() for v in stub.page_unit_views([0, 2, 4]))

    def test_a_unit_id_past_the_logical_count_is_refused(self):
        """The `block_ratio` trap in its most dangerous form.

        The bound is computed from the tensor rather than taken from the
        harness's label, because that is the whole point: `unit_ids` counts
        logical blocks and the tensor is shaped in physical ones, so a check
        written against the wrong axis admits ids that read past the end.
        """
        stub, cache = self.build()
        logical = (cache.shape[1] * cache.shape[2]) // self.H.LOGICAL_BS
        assert logical != cache.shape[1], "the harness must exercise a ratio != 1"
        assert stub.page_unit_views([logical - 1])  # the last valid id
        with pytest.raises(IndexError, match="logical blocks"):
            stub.page_unit_views([logical])

    def test_the_views_do_not_reach_into_a_neighbouring_unit(self):
        """Off-by-one here scrambles someone else's checkpoint rather than
        running off the end, so the bound is asserted from the bytes."""
        stub, cache = self.build()
        flat = cache.reshape(-1)
        seen = set()
        logical = (cache.shape[1] * cache.shape[2]) // self.H.LOGICAL_BS
        for unit in range(logical):
            for v in stub.page_unit_views([unit]):
                lo = (v.data_ptr() - flat.data_ptr()) // flat.element_size()
                span = set(range(lo, lo + v.numel()))
                assert not (span & seen), f"unit {unit} overlaps an earlier one"
                seen |= span

    def test_the_granularity_check_runs_before_any_view_is_handed_out(self):
        stub, _ = self.build(
            block_size=TestPageUnitAddressesAreArithmetic.LOGICAL_BS * 2
        )
        with pytest.raises(RuntimeError, match="granularity"):
            stub.page_unit_views([0])


class TestPageUnitViewsStopAtTheImage:
    """An image occupies WHOLE units, so the last one is mostly padding.

    `_checkpoint_copy_plan` already knows this — it hands `plan_segmented_copy`
    the unit stream *and* `spec.image_bytes`, and that function walks both
    streams from offset 0, so the image is by construction the LEADING
    `image_bytes` of the unit stream and the tail belongs to nobody.

    `page_unit_views` is the tensor-view counterpart of the same addressing, so
    it has to stop at the same place. Gathering whole units instead hands the
    packer more bytes than `StateByteCodec.put` allocated, and every store dies
    on "MemoryObj tensor is too small".

    None of the addressing tests above reach this: their spec carries no
    `image_bytes`, so they take the untrimmed path by construction.
    """

    H = TestPageUnitAddressesAreArithmetic

    def build(self, **kw):
        stub, cache = self.H.build(self.H(), **kw)
        stub.page_unit_views = PageUnitGeometryMixin.page_unit_views.__get__(
            stub, type(stub)
        )
        return stub, cache

    @staticmethod
    def total(views):
        return sum(v.numel() * v.element_size() for v in views)

    def test_the_gathered_stream_is_exactly_the_image(self):
        """The invariant the packer depends on, and the one that was broken."""
        stub, _ = self.build(image_bytes=70)  # 3 units x 24 B = 72 B available
        assert self.total(stub.page_unit_views([0, 1, 2])) == 70

    def test_a_whole_unit_past_the_image_is_dropped_entirely(self):
        stub, _ = self.build(image_bytes=20)
        views = stub.page_unit_views([0, 1, 2])
        assert self.total(views) == 20
        assert len(views) == 3, "two whole views plus the one that straddles"

    def test_an_image_that_lands_on_a_boundary_is_not_sliced(self):
        stub, _ = self.build(image_bytes=72)
        untrimmed, _ = self.build()
        assert [v.shape for v in stub.page_unit_views([0, 1, 2])] == [
            v.shape for v in untrimmed.page_unit_views([0, 1, 2])
        ]

    def test_the_kept_bytes_are_the_leading_bytes_and_nothing_else(self):
        """Trimming must not reorder or reslice — it truncates. The blob is
        read back by scattering into the slot in the same order, so a byte that
        moves here lands in the wrong layer there, silently."""
        stub, _ = self.build(image_bytes=70)
        untrimmed, _ = self.build()
        want = torch.cat(
            [
                v.reshape(-1).view(torch.uint8)
                for v in untrimmed.page_unit_views([0, 1, 2])
            ]
        )[:70]
        got = torch.cat(
            [v.reshape(-1).view(torch.uint8) for v in stub.page_unit_views([0, 1, 2])]
        )
        assert torch.equal(got, want)

    def test_a_multi_byte_dtype_is_sliced_by_bytes_not_elements(self):
        """The straddling view is reinterpreted as uint8 before slicing, because
        the image does not end on an element boundary in general. A slice taken
        in elements would silently keep the wrong amount on any dtype wider than
        a byte."""
        stub, cache = self.build(dtype=torch.bfloat16, image_bytes=70)
        assert cache.element_size() == 2
        assert self.total(stub.page_unit_views([0, 1, 2])) == 70

    def test_a_spec_with_no_image_size_keeps_whole_units(self):
        """A fork build carries no spec at all, and there the whole-unit stream
        is the right answer — there is nothing to trim against."""
        stub, _ = self.build()  # no image_bytes on the spec
        assert self.total(stub.page_unit_views([0, 1, 2])) == 3 * self.H.ROWS * (
            self.H.LOGICAL_BS * self.H.ENTRY
        )

    def test_units_that_cannot_cover_the_image_raise(self):
        """A short blob read back as valid is the one outcome worth crashing
        over: it resumes a request onto a truncated state with no exception."""
        stub, _ = self.build(image_bytes=80)  # more than the 72 B on offer
        with pytest.raises(RuntimeError, match="disagree"):
            stub.page_unit_views([0, 1, 2])
