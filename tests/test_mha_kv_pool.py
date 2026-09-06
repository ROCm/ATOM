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
    mha_kv_fields,
    shuffle_pack,
)

# The deployed MiniMax-M3 pair, at tp4: a 60-layer target with 4 KV heads, and
# a 1-layer EAGLE3 draft with 64 -- three heterogeneous contributors to one
# block id space once the indexer cache is counted, which is what makes M3 the
# e2e gate for this.
TARGET = {"layers": 60, "num_kv_heads": 1, "head_dim": 128, "block_size": 128}
DRAFT = {"layers": 1, "num_kv_heads": 16, "head_dim": 128, "block_size": 128}
DTYPES = [torch.float8_e4m3fnuz, torch.bfloat16]


def build(spec=TARGET, kv_dtype=torch.float8_e4m3fnuz, blocks=8) -> MhaKvPool:
    pool = MhaKvPool(**spec, kv_dtype=kv_dtype)
    pool.allocate(blocks, "cpu")
    return pool


@pytest.mark.parametrize("kv_dtype", DTYPES, ids=lambda d: str(d).split(".")[-1])
@pytest.mark.parametrize("spec", [TARGET, DRAFT], ids=["target", "draft"])
class TestTheDeclaredElementOrder:
    """K `[nh, hd//x, bs, x]` and V `[nh, bs//x, hd, x]` — the SHUFFLE layout.

    Not a formatting preference. The fused writer produces this order and
    `cp_mha_gather_cache_kernel` reads it in place; declaring V the other way
    round sends `_gather_prefix_and_concat_kv` down its densifying branch,
    which relaid a 1.4 GiB draft pool once per chunked-prefill forward.
    """

    def test_k_and_v_are_the_shuffle_shapes(self, spec, kv_dtype):
        k, v = mha_kv_fields(**spec, kv_dtype=kv_dtype)
        x = shuffle_pack(kv_dtype)
        nh, hd, bs = spec["num_kv_heads"], spec["head_dim"], spec["block_size"]

        assert k.shape == (nh, hd // x, bs, x)
        assert v.shape == (nh, bs // x, hd, x)

    def test_v_is_five_dimensional(self, spec, kv_dtype):
        """The ndim is load-bearing on its own, which is why it gets its own
        assertion: the reader branches on it and accepts either answer."""
        _, v = mha_kv_fields(**spec, kv_dtype=kv_dtype)

        assert len(v.shape) == 4, "plus the leading (layer, block) pair = 5-D"

    def test_one_pack_serves_both_views(self, spec, kv_dtype):
        """A mismatch would misread V only, and only for some head_dims."""
        k, v = mha_kv_fields(**spec, kv_dtype=kv_dtype)

        assert k.shape[-1] == v.shape[-1] == shuffle_pack(kv_dtype)

    def test_k_and_v_are_the_same_size(self, spec, kv_dtype):
        """Whatever the order, a block holds as much V as K."""
        k, v = mha_kv_fields(**spec, kv_dtype=kv_dtype)

        assert k.per_layer_numel == v.per_layer_numel


def test_a_pack_that_does_not_divide_the_shapes_is_refused():
    """Silently rounding would give a cache a fraction of a tile per head."""
    with pytest.raises(ValueError, match="head_dim 100"):
        mha_kv_fields(
            layers=1,
            block_size=128,
            num_kv_heads=1,
            head_dim=100,
            kv_dtype=torch.bfloat16,
        )


class TestTheBytesItLaysOut:
    """The pool has to reproduce the allocation it replaces, exactly.

    It took over a hand-written `[2, layers, blocks, block_size, kv_heads,
    head_dim]` tensor sliced per layer, so anything it lays out differently is
    a change in the pool's capacity or in what a kernel reads -- neither of
    which this refactor is allowed to do.
    """

    @pytest.mark.parametrize("kv_dtype", DTYPES, ids=lambda d: str(d).split(".")[-1])
    def test_entry_bytes_is_what_the_old_formula_charged(self, kv_dtype):
        """Byte-exact, not "about the same": this number times the block count
        is the pool, so a difference is capacity moving."""
        s, itemsize = TARGET, kv_dtype.itemsize
        cache = 2 * s["layers"] * s["block_size"] * s["num_kv_heads"] * s["head_dim"]
        scale = 2 * s["layers"] * s["num_kv_heads"] * s["block_size"] * 4

        assert MhaKvPool(**s, kv_dtype=kv_dtype).entry_bytes == cache * itemsize + scale

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
        pool = build(blocks=8)

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
        pool = build(blocks=4)
        k = pool.cache.view("k")
        stride = k[1].data_ptr() - k[0].data_ptr()

        assert stride == k[0].numel() * k[0].element_size()
        for layer in range(TARGET["layers"]):
            assert pool.kv_views(layer)[0].data_ptr() == (
                k[0].data_ptr() + layer * stride
            )

    def test_the_scales_are_allocated_for_a_bf16_cache_too(self):
        """The byte budget has always priced them; only an fp8 cache reads
        through them, which is the binder's call and not the pool's."""
        pool = build(kv_dtype=torch.bfloat16)

        assert pool.scale_views(0)[0].numel() > 0


class TestTheTransferGranularity:

    def test_one_region_per_layer_and_field(self):
        """Not one per block, and it cannot be while the pool is layer-major:
        a block's bytes are `blocks` apart. That collapses when the pool turns
        block-major, and not before."""
        pool = build()
        regions = pool.region_tensors()

        assert len(regions) == 4 * TARGET["layers"]
        assert [r.dtype for r in regions[: 2 * TARGET["layers"]]] == [
            torch.float8_e4m3fnuz
        ] * (2 * TARGET["layers"])

    def test_regions_are_empty_before_allocation(self):
        """Declared and allocated are two steps — sizing runs in between."""
        assert MhaKvPool(**TARGET, kv_dtype=torch.bfloat16).region_tensors() == []


class TestAdoptingSomeoneElsesAllocation:
    """The decode side of a P/D pair gets the pool as an IPC handle.

    It has to read those bytes at exactly the layout the prefill side wrote
    them at, which is why the same declaration backs both -- and why handing
    the buffers in is also the check that they fit it.
    """

    def test_an_imported_buffer_is_read_at_the_same_layout(self):
        donor = build()
        adopter = MhaKvPool(**TARGET, kv_dtype=torch.float8_e4m3fnuz)

        adopter.allocate(8, "cpu", cache_buf=donor.cache.buf, scale_buf=donor.scale.buf)

        for name in ("k", "v"):
            assert (
                adopter.cache.view(name).data_ptr() == donor.cache.view(name).data_ptr()
            )

    def test_a_buffer_that_does_not_fit_the_declaration_is_refused(self):
        """Refusing is the point: the alternative is addressing past the end
        of someone else's allocation, which no shape says anything about."""
        small = build(blocks=4)
        adopter = MhaKvPool(**TARGET, kv_dtype=torch.float8_e4m3fnuz)

        with pytest.raises(ValueError, match="at least"):
            adopter.allocate(8, "cpu", cache_buf=small.cache.buf)

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


class TestFromHfConfig:

    class _Cfg:
        num_hidden_layers = 60
        num_key_value_heads = 4
        head_dim = 128

    def test_kv_heads_are_sharded_by_world_size(self):
        pool = MhaKvPool.from_hf_config(
            self._Cfg(), world_size=4, block_size=128, kv_dtype=torch.bfloat16
        )

        assert pool.cache_fields[0].shape[0] == 1

    def test_a_head_per_rank_is_the_floor(self):
        """More ranks than KV heads replicates rather than allocating none --
        the rule `ModelRunner._get_num_kv_heads` has always applied."""
        pool = MhaKvPool.from_hf_config(
            self._Cfg(), world_size=8, block_size=128, kv_dtype=torch.bfloat16
        )

        assert pool.cache_fields[0].shape[0] == 1


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
        for imported in imported_modules(path, package=module.rsplit(".", 1)[0]):
            assert not imported.startswith(("aiter", "triton")), (
                f"{module} reaches {imported}, which a runner with no AITER "
                "build does not have -- and one collection error there aborts "
                "every test, not this one"
            )
            if imported.startswith("atom."):
                frontier.append(imported)
