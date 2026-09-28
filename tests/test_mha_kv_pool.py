# SPDX-License-Identifier: MIT
"""The one declaration of what an MHA block holds, and what reads it.

There used to be four copies of these shapes -- the AITER backend's shared
path, its per-module MiMo path, the Triton backend, and the draft's own
builder -- and they agreed on every byte while disagreeing on a *view*: the
draft declared V non-transposed where the others declared it SHUFFLE. Nothing
in the tree could catch that. The byte count is identical either way, and the
only consumer that cares reads `v_cache.ndim` at runtime and quietly relays the
whole pool out when it is not 5-D.

So what is pinned here is the declaration itself and the bytes it lays out.
There is deliberately no four-way comparison: the four copies are gone
structurally, not by assertion -- every one of those sites now takes its views
from a `MhaKvPool`, so a test of the pool is a test of all of them, and a fifth
consumer inherits the same shapes by construction rather than by review.

Runs on a plain runner: the pool is torch and index arithmetic, and
`test_reachable_without_a_gpu_build` is what keeps it that way.
"""

from __future__ import annotations

import pathlib

import pytest
import torch
from test_layout_packages import imported_modules

from atom.model_ops.attentions.mha_kv_pool import (
    MhaKvPool,
    shuffle_pack,
)
from atom.model_ops.attentions.pool_layout.entry_arena import EntryField, plan_regions

# The deployed MiniMax-M3 pair, at tp4: a 60-layer target with 4 KV heads, and
# a 1-layer EAGLE3 draft with 64 -- three heterogeneous contributors to one
# block id space once the indexer cache is counted, which is what makes M3 the
# e2e gate for this.
TARGET = {"layers": 60, "num_kv_heads": 1, "head_dim": 128, "block_size": 128}
DRAFT = {"layers": 1, "num_kv_heads": 16, "head_dim": 128, "block_size": 128}
FP8 = torch.float8_e4m3fnuz
DTYPES = [FP8, torch.bfloat16]
# The backend that pages at 16 while the scheduler blocks at 128, on a shape
# where nothing coincides: `head_dim != block_size` so `hd // x` and
# `block_size // x` differ, and bf16's pack of 8 differs from the block so V's
# outer half is not one. At fp8 every one of those collapses.
SMALL_BLOCK = {"layers": 2, "num_kv_heads": 1, "head_dim": 64, "block_size": 16}
BLOCKS_PER_ENTRY, ENTRIES = 8, 2
# M3's indexer keys: 57 of the 60 layers own one, `sparse_index_dim` wide, at
# the cache dtype (the config names no `index_cache_dtype`).
INDEX_LAYERS, INDEX_DIM, INDEX_DTYPE = 57, 128, torch.float8_e4m3fnuz


def index_field(spec=TARGET, blocks_per_entry=1) -> EntryField:
    """The indexer field as `AiterAttentionMetadataBuilder` declares it.

    Spelled out here, not imported: the backend owns this arithmetic now, and
    a test that shared the expression with it would only restate it.
    """
    per_entry = spec["block_size"] * blocks_per_entry * INDEX_DIM * INDEX_DTYPE.itemsize
    return EntryField("index", INDEX_LAYERS, (per_entry,), torch.uint8)


def index_view(pool: MhaKvPool, layer: int) -> torch.Tensor:
    """And what the binder shapes it back into."""
    return pool.field_view(
        "index", layer, INDEX_DTYPE, (-1, pool.block_size, INDEX_DIM)
    )


def build(spec=TARGET, kv_dtype=torch.float8_e4m3fnuz, entries=8) -> MhaKvPool:
    pool = MhaKvPool(**spec, kv_dtype=kv_dtype)
    pool.allocate(entries, "cpu")
    return pool


def INDEXED(kv_dtype=torch.float8_e4m3fnuz, entries=0) -> MhaKvPool:
    """The target pool as M3 declares it. Unallocated at `entries=0`, which is
    the state sizing asks `entry_bytes` in."""
    pool = MhaKvPool(**TARGET, extra_fields=(index_field(),), kv_dtype=kv_dtype)
    if entries:
        pool.allocate(entries, "cpu")
    return pool


@pytest.mark.parametrize("blocks_per_entry", [1, 8], ids=["one", "eight"])
@pytest.mark.parametrize("kv_dtype", DTYPES, ids=lambda d: str(d).split(".")[-1])
@pytest.mark.parametrize("spec", [TARGET, DRAFT], ids=["target", "draft"])
class TestTheElementOrderABinderGets:
    """K `[blocks, nh, hd//x, bs, x]` and V `[blocks, nh, bs//x, hd, x]`.

    Asserted on `kv_views` and not on the fields, because that is where the
    order is: a field says what an entry costs, which is the same number
    whichever way its tokens lie, and nothing outside this module reads a
    field's shape.

    Not a formatting preference. The fused writer produces this order and
    `cp_mha_gather_cache_kernel` reads it in place; declaring V the other way
    round sends `_gather_prefix_and_concat_kv` down its densifying branch,
    which relaid a 1.4 GiB draft pool once per chunked-prefill forward.

    `blocks_per_entry` moves the block count and nothing else, which is the
    claim: an entry is an accounting unit and no shape is taken at it.
    """

    ENTRIES = 4

    def _views(self, spec, kv_dtype, blocks_per_entry):
        pool = MhaKvPool(**spec, kv_dtype=kv_dtype, blocks_per_entry=blocks_per_entry)
        pool.allocate(self.ENTRIES, "cpu")
        return pool.kv_views(0)

    def test_k_and_v_are_the_shuffle_shapes(self, spec, kv_dtype, blocks_per_entry):
        k, v = self._views(spec, kv_dtype, blocks_per_entry)
        x = shuffle_pack(kv_dtype)
        nh, hd, bs = spec["num_kv_heads"], spec["head_dim"], spec["block_size"]
        blocks = self.ENTRIES * blocks_per_entry

        assert k.shape == (blocks, nh, hd // x, bs, x)
        assert v.shape == (blocks, nh, bs // x, hd, x)

    def test_v_is_five_dimensional(self, spec, kv_dtype, blocks_per_entry):
        """The ndim is load-bearing on its own, which is why it gets its own
        assertion: the reader branches on it and accepts either answer."""
        _, v = self._views(spec, kv_dtype, blocks_per_entry)

        assert v.ndim == 5

    def test_one_pack_serves_both_views(self, spec, kv_dtype, blocks_per_entry):
        """A mismatch would misread V only, and only for some head_dims."""
        k, v = self._views(spec, kv_dtype, blocks_per_entry)

        assert k.shape[-1] == v.shape[-1] == shuffle_pack(kv_dtype)

    def test_k_and_v_are_the_same_size(self, spec, kv_dtype, blocks_per_entry):
        """Whatever the order, a block holds as much V as K."""
        k, v = self._views(spec, kv_dtype, blocks_per_entry)

        assert k.numel() == v.numel()


class TestAnEntryIsSeveralBlocksAndNothingReadsAtIt:
    """The pool holds `entries * blocks_per_entry` blocks, laid end to end.

    An entry is what `page_pool` charges per, because that is the scheduler's
    unit and the index space a draft shares. Nothing reads at it: the block
    tables a kernel gets were expanded by `block_ratio`, so its ids run over
    every block, and `asm_pa.cu` multiplies one by a single `K->stride(0)`.
    Binding a layer per entry instead leaves the ids eight times too large for
    the tensor and the stride eight times too long -- the same bytes, addressed
    as a different number of units, which no byte total distinguishes.

    bf16 throughout: at fp8 the pack is 16 and so is the block, which collapses
    V's `block_size // x` to one and makes K and V look alike.
    """

    def _bound(self, name):
        """`(the layer's region, what a binder gets, the shape it must have)`."""
        pool = MhaKvPool(
            **SMALL_BLOCK, kv_dtype=torch.bfloat16, blocks_per_entry=BLOCKS_PER_ENTRY
        )
        pool.allocate(ENTRIES, "cpu")
        x = shuffle_pack(torch.bfloat16)
        nh, hd, bs = (
            SMALL_BLOCK["num_kv_heads"],
            SMALL_BLOCK["head_dim"],
            SMALL_BLOCK["block_size"],
        )
        per_block = (nh, hd // x, bs, x) if name == "k" else (nh, bs // x, hd, x)
        bound = pool.kv_views(0)[0 if name == "k" else 1]
        return pool._views[name][0], bound, (ENTRIES * BLOCKS_PER_ENTRY, *per_block)

    @pytest.mark.parametrize("name", ["k", "v"])
    def test_a_layer_binds_at_the_block(self, name):
        """Every block of every entry, which is the id space the block tables
        address -- not one unit per entry."""
        _, bound, want = self._bound(name)

        assert bound.shape == want

    @pytest.mark.parametrize("name", ["k", "v"])
    def test_the_view_is_not_a_copy(self, name):
        """The pool is gigabytes; a relayout here would move all of it once per
        bind. Free because an entry's bytes are contiguous, so the blocks in it
        are just the leading dim shaped out."""
        region, bound, _ = self._bound(name)

        assert bound.data_ptr() == region.data_ptr()
        assert bound.is_contiguous()

    @pytest.mark.parametrize("name", ["k", "v"])
    def test_one_stride_reaches_every_block(self, name):
        """`asm_pa.cu` reads `K->stride(0)` and multiplies a block id by it, so
        the step across an entry boundary has to be the step inside one."""
        _, bound, _ = self._bound(name)
        steps = {
            bound[b + 1].data_ptr() - bound[b].data_ptr()
            for b in range(bound.shape[0] - 1)
        }

        assert steps == {bound.stride(0) * bound.element_size()}

    @pytest.mark.parametrize("name", ["k", "v"])
    def test_the_bytes_are_the_layout_this_replaced(self, name):
        """The gate for the change: not "equivalent", identical.

        What shipped before was `[blocks, ...]` dense at the backend's block.
        Marking the region in storage order and reading it back through the
        binder's view has to give exactly that -- otherwise a kernel reads the
        right count of the wrong bytes, which no shape or byte total says
        anything about.
        """
        region, bound, want = self._bound(name)
        # `region` is the field's bytes; two of them make one bf16 element, so
        # the marks are counted in elements.
        marked = region.view(torch.int16).reshape(-1)
        marks = torch.arange(marked.numel(), dtype=torch.int16)
        marked.copy_(marks)

        assert torch.equal(bound.view(torch.int16), marks.reshape(want))


def test_a_pack_that_does_not_divide_the_shapes_is_refused():
    """Silently rounding would give a cache a fraction of a tile per head.

    On the pool and not on the fields: the fields are bytes now, and bytes
    divide by anything. The pack is a property of how they are read.
    """
    with pytest.raises(ValueError, match="head_dim 100"):
        MhaKvPool(
            layers=1,
            block_size=128,
            num_kv_heads=1,
            head_dim=100,
            kv_dtype=torch.bfloat16,
        )


def test_an_entry_of_no_blocks_is_refused():
    """`blocks_per_entry` is `block_ratio`, which comes of a launch flag over
    the backend's own block -- so zero is a bad pairing, not a code change."""
    with pytest.raises(ValueError, match="at least one block"):
        MhaKvPool(**TARGET, kv_dtype=torch.bfloat16, blocks_per_entry=0)


class TestTheBytesItLaysOut:
    """The pool has to reproduce the allocation it replaces, exactly.

    It took over a hand-written `[2, layers, blocks, block_size, kv_heads,
    head_dim]` tensor sliced per layer, so anything it lays out differently is
    a change in the pool's capacity or in what a kernel reads -- neither of
    which this refactor is allowed to do.
    """

    def test_entry_bytes_is_what_the_old_formula_charged(self):
        """Byte-exact, not "about the same": this number times the block count
        is the pool, so a difference is capacity moving.

        fp8 only. The old formula charged the fp32 scale planes at every dtype,
        and a bf16 cache no longer declares them -- deliberately, since nothing
        binds them there; `test_a_bf16_cache_declares_no_scales_at_all` is that
        departure, stated as its own claim rather than folded in here.
        """
        s = TARGET
        cache = 2 * s["layers"] * s["block_size"] * s["num_kv_heads"] * s["head_dim"]
        scale = 2 * s["layers"] * s["num_kv_heads"] * s["block_size"] * 4

        assert MhaKvPool(**s, kv_dtype=FP8).entry_bytes == cache * FP8.itemsize + scale

    def test_k_comes_first_and_v_follows_it(self):
        """The order the `[2, layers, ...]` allocation held them in."""
        pool = build()
        k, v = pool.cache.view("k"), pool.cache.view("v")

        assert k.data_ptr() == pool.cache.buf.data_ptr()
        assert v.data_ptr() == k.data_ptr() + k.numel() * k.element_size()

    def test_a_layer_slice_is_contiguous(self):
        """Several kernels take a layer's base pointer and stride past it.
        Layer-major is the reason they still can -- and the reason an *entry*
        cannot be copied as one, which is what `LayerMajorArena` refuses to
        pretend by having no `entry(i)`."""
        pool = build()

        for name in ("k", "v"):
            assert pool.cache.view(name)[0].is_contiguous()

    def test_the_allocation_is_no_larger_than_the_budget(self):
        """Sizing calls `entry_bytes` before any GPU exists; the allocation
        must come out of the same expression or the two drift."""
        pool = build(entries=8)

        held = pool.cache.buf.numel() + pool.scale.buf.numel()
        assert held == 8 * pool.entry_bytes


class TestWhatABinderGetsBack:

    def test_the_views_alias_the_pool(self):
        """A relayout in the binder reads as a fix and costs the same copy,
        moved earlier -- once per process here, but the pool is gigabytes."""
        pool = build()
        k, v = pool.kv_views(0)
        k_scale, v_scale = pool.scale_views(0)

        for view in (k, v):
            assert view.data_ptr() >= pool.cache.buf.data_ptr()
        assert k_scale.data_ptr() >= pool.scale.buf.data_ptr()
        assert v_scale.data_ptr() > k_scale.data_ptr()

    def test_each_layer_gets_its_own_slice(self):
        """One layer is the shipped draft, which would hide an indexing bug."""
        pool = build(entries=4)
        k = pool.cache.view("k")
        stride = k[1].data_ptr() - k[0].data_ptr()

        assert stride == k[0].numel() * k[0].element_size()
        for layer in range(TARGET["layers"]):
            assert pool.kv_views(layer)[0].data_ptr() == (
                k[0].data_ptr() + layer * stride
            )

    def test_a_bf16_cache_declares_no_scales_at_all(self):
        """Nothing binds them: `build_kv_cache_tensor` sets `k_scale`/`v_scale`
        on a module only for fp8, and every reader is guarded on that. So the
        two fp32 planes a bf16 entry used to buy per layer were unreachable --
        charged, allocated, and registered for P/D transfer with no consumer.

        Asserted on the declaration and not just the allocation because the
        declaration is what sizing charges: leaving the fields in and skipping
        the `torch.zeros` would still take the entries out of the budget.
        """
        pool = build(kv_dtype=torch.bfloat16)

        assert pool.scale_fields == []
        assert pool.scale is None
        assert not any(name.endswith("_scale") for name in pool._views)

    def test_dropping_them_is_worth_a_block_in_sixty_four(self):
        """The size of the thing, so a later change that quietly puts them back
        fails here rather than surfacing as a capacity number in a server log.
        """
        rows = 2 * TARGET["layers"] * TARGET["num_kv_heads"] * TARGET["block_size"]
        cache = rows * TARGET["head_dim"] * torch.bfloat16.itemsize
        scales = rows * 4  # what the fp32 planes cost when they were declared

        assert MhaKvPool(**TARGET, kv_dtype=torch.bfloat16).entry_bytes == cache
        assert scales * 64 == cache


class TestTheTransferGranularity:

    def test_one_region_per_layer_and_field(self):
        """Not one per block, and it cannot be while the pool is layer-major:
        a block's bytes are `blocks` apart. That collapses when the pool turns
        block-major, and not before."""
        pool = build()
        regions = pool.region_tensors()

        assert len(regions) == 4 * TARGET["layers"]
        # Bytes, which is what a transfer moves and all it needs to know: a
        # region carries a base, a span and a per-block stride, and none of
        # those is a dtype.
        assert {t.dtype for _, t in regions} == {torch.uint8}

    def test_every_region_is_named_and_no_two_alike(self):
        """Position is the only other identity a region has to the far end, and
        folding the fields into a pool changed it: `K0,V0,K1,V1…` became all of
        K then all of V, at the same count and the same bytes. A name is what
        lets a mismatched pair notice instead of transferring K into V."""
        pool = INDEXED(entries=4)
        roles = [role for role, _ in pool.region_tensors()]

        assert all(roles)
        assert len(set(roles)) == len(roles)
        assert roles[0] == "k.layer_0"

    def test_regions_are_empty_before_allocation(self):
        """Declared and allocated are two steps — sizing runs in between."""
        assert MhaKvPool(**TARGET, kv_dtype=torch.bfloat16).region_tensors() == []


class TestTheIndexerCacheIsAField:
    """MiniMax-M3's indexer keys, which used to be declared twice.

    Sizing added `sparse_layers * page * dim * itemsize` to the block cost
    while allocation wrote `torch.zeros(sparse_layers, blocks, page, dim)` --
    and the two read *different* block-count attributes off the runner, which
    is exactly how the sparse index cache ended up sized against one rank's own
    estimate while the pool was built at the broadcast one. Being a field is
    what makes that unrepresentable: one expression answers both.
    """

    def test_a_model_without_an_indexer_declares_none_and_pays_nothing(self):
        """The field is opt-in, so every other MHA model is untouched."""
        plain = MhaKvPool(**TARGET, kv_dtype=torch.float8_e4m3fnuz)

        assert plain.extra_fields == []
        assert plain.entry_bytes < INDEXED(torch.float8_e4m3fnuz).entry_bytes

    @pytest.mark.parametrize("kv_dtype", DTYPES, ids=lambda d: str(d).split(".")[-1])
    def test_entry_bytes_is_what_the_two_old_declarations_charged(self, kv_dtype):
        """The gate for the fold. `entry_bytes_for` aligns and the old formula
        did not, so equality here is a fact about M3's numbers, not a
        tautology -- a shape whose indexer segment is not a multiple of the
        arena's alignment would move the pool's capacity, and has to be caught
        before it ships rather than in a server log."""
        raw = INDEX_LAYERS * TARGET["block_size"] * INDEX_DIM

        assert (
            INDEXED(kv_dtype).entry_bytes
            == MhaKvPool(**TARGET, kv_dtype=kv_dtype).entry_bytes
            + raw * INDEX_DTYPE.itemsize
        )

    def test_the_indexer_is_its_own_allocation(self):
        """Not carved out of the K/V buffer: it has its own dtype and its own
        layer count, and the P/D export ships the K/V one by itself."""
        pool = INDEXED(entries=4)

        assert pool.extra is not None
        assert pool.extra.buf.data_ptr() != pool.cache.buf.data_ptr()
        assert pool.extra.buf.numel() == 4 * (
            pool.entry_bytes
            - MhaKvPool(**TARGET, kv_dtype=torch.float8_e4m3fnuz).entry_bytes
        )

    def test_a_layer_gets_the_scheduler_blocks_rows(self):
        """Shaped in scheduler blocks, which is the entry class `page_pool`
        charges. The backend reshapes to its own page when it binds -- sizing
        it at that page instead is what undercharged by `block_ratio`."""
        pool = INDEXED(entries=4)
        view = index_view(pool, 0)

        assert view.shape == (4, TARGET["block_size"], INDEX_DIM)
        assert view.is_contiguous()

    def test_each_indexer_layer_gets_its_own_slice(self):
        """Bind order assigns these, so an off-by-one is a layer reading
        another layer's keys — silently, with plausible top-k output."""
        pool = INDEXED(entries=4)
        first = index_view(pool, 0)
        stride = index_view(pool, 1).data_ptr() - first.data_ptr()

        assert stride == first.numel() * first.element_size()
        for layer in range(INDEX_LAYERS):
            assert index_view(pool, layer).data_ptr() == (
                first.data_ptr() + layer * stride
            )

    def test_indexer_regions_come_after_the_kv_ones(self):
        """The order a P/D peer reconstructs regions in. It used to be spelled
        out at the call site (`region_tensors()` and then a loop over the index
        cache); now it falls out of field order, so it is pinned here."""
        pool = INDEXED(entries=4)
        regions = pool.region_tensors()
        kv = 4 * TARGET["layers"]

        assert len(regions) == kv + INDEX_LAYERS
        assert [t.data_ptr() for _, t in regions[kv:]] == [
            index_view(pool, i).data_ptr() for i in range(INDEX_LAYERS)
        ]

    def test_an_indexer_region_is_one_scheduler_block_wide(self):
        """`unit_bytes` is what a peer multiplies a block id by, and the ids
        are the scheduler's, so a region's unit has to be the scheduler's block.

        This is the one place the fold is *not* byte-for-byte with what it
        replaced. The old allocation was shaped in the backend's own page
        (`[layers, blocks * block_ratio, backend_page, dim]`), so its per-unit
        stride was `backend_page * dim` -- the same total bytes divided into
        `block_ratio` times as many units. No byte count distinguishes the two,
        which is why `KVTransferTensors.set_block_count` checks the unit and
        not the size. Where the two block sizes are equal the numbers coincide;
        that is a property of those configs, not of the code.
        """
        pool = INDEXED(entries=4)
        _, region = pool.region_tensors()[-1]
        stride = region.stride(0) * region.element_size()

        assert stride == TARGET["block_size"] * INDEX_DIM
        assert region.numel() * region.element_size() // stride == 4

    def test_release_drops_the_indexer_too(self):
        """Nothing outside the pool holds this buffer, so the pool is what
        frees it across a rollout sleep."""
        pool = INDEXED(entries=4)
        pool.release()

        assert pool.extra is None
        assert pool.region_tensors() == []


class TestAdoptingSomeoneElsesAllocation:
    """The decode side of a P/D pair gets the pool as an IPC handle.

    It has to read those bytes at exactly the layout the prefill side wrote
    them at, which is why the same declaration backs both -- and why handing
    the buffers in is also the check that they fit it.
    """

    def test_an_imported_buffer_is_read_at_the_same_layout(self):
        """Every field, not just the cache: the scales and the indexer keys are
        regions of the one buffer now, so where they land is decided by the
        declaration on each side rather than by a handle per tensor."""
        donor = INDEXED()
        buf = torch.zeros(donor.pool_bytes(8), dtype=torch.uint8)
        donor.allocate(8, "cpu", buf=buf)
        adopter = INDEXED()

        adopter.allocate(8, "cpu", buf=buf)

        for name in ("k", "v", "k_scale", "v_scale", "index"):
            assert adopter._views[name].data_ptr() == donor._views[name].data_ptr()

    def test_a_buffer_that_does_not_fit_the_declaration_is_refused(self):
        """Refusing is the point: the alternative is addressing past the end
        of someone else's allocation, which no shape says anything about.

        Raised where the buffer is cut, quoting what the pool asked for. It
        used to slip past that -- an over-long slice truncates instead of
        raising -- and land on whichever arena got the short region, quoting
        that arena's own field bytes, a number the caller never named.
        """
        adopter = MhaKvPool(**TARGET, kv_dtype=torch.float8_e4m3fnuz)
        short = torch.zeros(adopter.pool_bytes(4), dtype=torch.uint8)

        with pytest.raises(ValueError, match="regions needing"):
            adopter.allocate(8, "cpu", buf=short)

    def test_release_drops_the_backing_and_keeps_the_declaration(self):
        """The rollout sleep path frees the pool by dropping the runner's
        buffers; a pool holding views of them would keep the allocation
        alive."""
        pool = build()
        entry_bytes = pool.entry_bytes

        pool.release()

        assert pool.cache is None and pool.scale is None
        assert pool.region_tensors() == [], "the views would hold it alive too"
        assert pool.entry_bytes == entry_bytes


class TestSharingOneAllocationWithADraft:
    """M3's two pools in one buffer, the way ModelRunner carves it.

    A draft riding the target's block ids was already how the *budget* worked
    -- two specs naming one entry class sum -- while the memory was two
    allocations reached under two names. Here it is one, so what a block costs
    and what a block occupies are the same walk, and neither pool can be built
    somewhere the other does not expect.
    """

    ENTRIES = 8

    def _carve(self):
        target = INDEXED()
        draft = MhaKvPool(**DRAFT, kv_dtype=torch.float8_e4m3fnuz)
        sizes = [p.pool_bytes(self.ENTRIES) for p in (target, draft)]
        offsets, total = plan_regions(sizes)
        buf = torch.zeros(total, dtype=torch.uint8)
        for pool, start, size in zip((target, draft), offsets, sizes):
            pool.allocate(self.ENTRIES, "cpu", buf=buf[start : start + size])
        return buf, target, draft, list(zip(offsets, sizes))

    @staticmethod
    def _span(view):
        """`[first, last)` byte of a field's region, relative to nothing."""
        return view.data_ptr(), view.data_ptr() + view.numel() * view.element_size()

    def test_no_view_leaves_the_region_its_pool_was_given(self):
        buf, target, draft, extents = self._carve()

        for pool, (start, size) in zip((target, draft), extents):
            lo = buf.data_ptr() + start
            for _, view in pool.region_tensors():
                first, last = self._span(view)
                assert lo <= first and last <= lo + size

    def test_the_two_together_cost_what_a_block_was_charged(self):
        """`sub_pool_specs` adds the draft's `entry_bytes` to the target's and
        the budget buys blocks of the sum. Packing them adds no padding, so the
        buffer is that price exactly -- which is what lets the sizing number and
        the allocation be read as one."""
        buf, target, draft, _ = self._carve()

        charged = target.entry_bytes + draft.entry_bytes
        assert buf.numel() == charged * self.ENTRIES

    def test_every_region_keeps_what_was_written_to_it(self):
        """The check the byte counts cannot do, and the reason placement is not
        a matter of taste: regions sized right and placed wrong agree on every
        total while writing through each other. Distinct marks over both pools
        at once, so an overlap between two fields of one pool fails it as
        surely as one between the two pools."""
        _, target, draft, _ = self._carve()
        regions = [t for _, t in target.region_tensors() + draft.region_tensors()]
        marks = [1 + i % 251 for i in range(len(regions))]
        for view, mark in zip(regions, marks):
            view.view(torch.uint8).fill_(mark)

        for view, mark in zip(regions, marks):
            assert view.view(torch.uint8).eq(mark).all()


class TestFromHfConfig:

    class _Cfg:
        num_hidden_layers = 60
        num_key_value_heads = 4
        head_dim = 128

    def _pool(self, world_size: int, layers: int = 2) -> MhaKvPool:
        return MhaKvPool.from_hf_config(
            self._Cfg(),
            world_size=world_size,
            block_size=128,
            layers=layers,
            kv_dtype=torch.bfloat16,
        )

    def test_kv_heads_are_sharded_by_world_size(self):
        # One KV head's worth of bytes per block, at bf16.
        assert self._pool(world_size=4).cache_fields[0].shape == (1 * 128 * 128 * 2,)

    def test_a_head_per_rank_is_the_floor(self):
        """More ranks than KV heads replicates rather than allocating none --
        the rule `ModelRunner._get_num_kv_heads` has always applied."""
        assert self._pool(world_size=8).cache_fields[0].shape == (1 * 128 * 128 * 2,)

    def test_the_caller_says_how_many_rows_not_the_config(self):
        """`num_hidden_layers` is 60 here and is not what comes back. Only the
        walk over the built model knows how many rows a pool has -- reading the
        config instead sizes it off one count and addresses it by another."""
        assert self._Cfg.num_hidden_layers == 60
        assert self._pool(world_size=4, layers=3).cache_fields[0].layers == 3


def test_reachable_without_a_gpu_build():
    """The pool is what a draft's builder and CI both reach for.

    Transitively, not just directly: it imports `pool_layout`, which is held to
    the same rule by `test_layout_packages`, and this walks the closure so a
    new hop cannot smuggle AITER in. CI has no AITER build, and one import
    failure during collection aborts the whole run rather than one test.
    """
    root = pathlib.Path(__file__).resolve().parent.parent
    seen: set[str] = set()
    frontier = ["atom.model_ops.attentions.mha_kv_pool"]
    while frontier:
        module = frontier.pop()
        if module in seen:
            continue
        seen.add(module)
        path = root / (module.replace(".", "/") + ".py")
        assert path.is_file(), f"{module} is not a plain module"
        for imported in imported_modules(path):
            assert not imported.startswith(("aiter", "triton")), (
                f"{module} reaches {imported}, which a runner with no AITER "
                "build does not have -- and one collection error there aborts "
                "every test, not this one"
            )
            if imported.startswith("atom."):
                frontier.append(imported)
