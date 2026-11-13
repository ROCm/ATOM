# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Where a draft's pool learns how many rows it has, and who owns one at all.

The rows a draft occupies and the rows its pool was built for are the same
number asked twice, and `PoolRowsMixin` exists so it is asked once. The pool
used to take `hf_config.num_hidden_layers` while every row index came from the
walk: one layer too many is a whole layer of the pool paid for and never
addressed, one too few is a write past the end, and the startup byte check
sees neither -- both sides read the same inflated `entry_bytes`.

Imports cleanly without aiter (the backend lookup is function-local), so this
runs on CI, which is where such a regression would land.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from atom.model_ops.attentions.mha_kv_pool import MhaKvPool
from atom.spec_decode.draft_kv import (
    DRAFT_KV_ROWS,
    DraftKvBuilder,
    DraftKvPoolSpec,
    draft_kv_builder,
    draft_owns_kv_pool,
)


class _PagedAttention(nn.Module):
    """What `_module_kinds` counts: paged, and not MLA."""

    def __init__(self):
        super().__init__()
        self.base_attention = True
        self.use_mla = False
        self.impl = SimpleNamespace(kv_layout=None)


class _Norm(nn.Module):
    """A sibling the walk must not count."""


# The target builder's block. A draft's pool takes it rather than picking one,
# because the draft indexes that pool with the target's block tables.
TARGET_BLOCK = 128


def _runner(
    draft_layers: int,
    block_size: int = TARGET_BLOCK,
    builder_block_size: int | None = None,
):
    """A stand-in runner.

    `block_size` is the scheduler's; `builder_block_size` is the target
    builder's, which defaults to the same number because a paged target pages at
    the block it is scheduled in. They are separate arguments because an MLA
    target's builder does NOT: it pages at ATOM_MLA_PAGE_SIZE, a kernel page
    that is 1 by default.
    """
    model = nn.Sequential(
        *[m for _ in range(draft_layers) for m in (_PagedAttention(), _Norm())]
    )
    return SimpleNamespace(
        block_size=block_size,
        drafter=SimpleNamespace(model=model),
        attn_metadata_builder=SimpleNamespace(
            block_size=block_size if builder_block_size is None else builder_block_size
        ),
    )


def _recording_factory(pool):
    """A stand-in for the backend's `make_kv_pool`, keeping what it was asked."""
    asked: list[tuple[int, int]] = []

    def make(*, layers: int, target_block_size: int):
        asked.append((layers, target_block_size))
        return pool

    return make, asked


def test_the_pool_is_built_at_the_row_count_the_walk_found():
    """Not at `num_hidden_layers`. The double carries neither, so the only way
    to get 3 is to have counted the modules."""
    pool = object()
    make, asked = _recording_factory(pool)
    builder = DraftKvBuilder(_runner(draft_layers=3), make)

    assert builder.kv_pool is pool
    assert asked == [(3, TARGET_BLOCK)]


def test_the_pool_is_built_at_the_target_builders_block():
    """Not at one the draft's own backend would pick: `propose` hands the draft
    the target's block tables, and its kernels read the block off the cache it
    was bound to, so the two are the same number or it reads another page.

    An off-default value, because a number every side agrees on by accident
    proves nothing about which side it came from."""
    make, asked = _recording_factory(object())
    builder = DraftKvBuilder(_runner(draft_layers=2, block_size=256), make)

    assert builder.kv_pool is not None
    assert asked == [(2, 256)]


def test_the_gqa_draft_pool_takes_the_schedulers_block():
    """The DSpark GQA draft's pool sits beside an MLA target, whose builder pages at
    ATOM_MLA_PAGE_SIZE -- 1 by default -- a kernel page, not a block id. Its
    block tables count SCHEDULER blocks, so that is the number to size at; at
    the builder's, MhaKvPool's SHUFFLE packing would not even construct.
    """
    make, asked = _recording_factory(object())
    builder = DraftKvBuilder(
        _runner(draft_layers=2, block_size=128, builder_block_size=1),
        make,
        block_size=128,
    )

    assert builder.kv_pool is not None
    assert asked == [(2, 128)]


def test_other_pools_keep_the_target_builders_block():
    """MiniMax-M3.1 DSpark and Eagle3 page at the builder's block, and
    `allocate_kv_cache_tensors` refuses a scheduler block that differs."""
    make, asked = _recording_factory(object())
    builder = DraftKvBuilder(
        _runner(draft_layers=2, block_size=256, builder_block_size=128), make
    )

    assert builder.kv_pool is not None
    assert asked == [(2, 128)]


def test_siblings_are_not_rows():
    """The walk's predicate, not the module count: a draft layer is more than
    one `nn.Module`, and charging for the norms would double the pool."""
    make, asked = _recording_factory(object())
    builder = DraftKvBuilder(_runner(draft_layers=4), make)

    assert len(list(builder.model_runner.drafter.model.modules())) > 4 + 1
    assert builder.kv_pool is not None
    assert asked == [(4, TARGET_BLOCK)]


def test_a_draft_with_no_rows_is_a_contradiction_not_an_empty_pool():
    """This builder exists because the flavor owns a pool, so finding no rows
    means the walk's predicate missed the draft's modules. Defaulted to zero
    that prices at nothing and binds nothing -- a draft running on no KV, with
    no error and no log. `pool_rows` promises a `KeyError` for exactly this."""
    make, asked = _recording_factory(object())
    builder = DraftKvBuilder(
        SimpleNamespace(drafter=SimpleNamespace(model=_Norm())), make
    )

    with pytest.raises(KeyError, match=DRAFT_KV_ROWS):
        _ = builder.kv_pool

    assert asked == []


def test_the_pool_is_built_once():
    """Every hook reads `kv_pool`; a second build would be a second pool, and
    the one already bound to modules would be the one nobody allocated."""
    make, asked = _recording_factory(object())
    builder = DraftKvBuilder(_runner(draft_layers=2), make)

    assert builder.kv_pool is builder.kv_pool is builder.kv_pool
    assert asked == [(2, TARGET_BLOCK)]


def test_it_is_not_built_before_the_draft_model_exists():
    """`draft_kv_builder` runs from the proposer's `__init__`, which is
    `build_drafter` still running -- `runner.drafter` is not set yet. Touching
    the pool there is the bug this laziness prevents, so the double has no
    drafter at all and construction still has to succeed."""
    make, asked = _recording_factory(object())

    DraftKvBuilder(SimpleNamespace(), make)

    assert asked == []


def test_invalidating_the_walk_drops_the_pool_that_came_from_it():
    """They are one answer. A pool surviving its walk is the split this whole
    class was rewritten to close, just deferred by one rebind."""
    first, second = object(), object()
    pools = iter((first, second))
    builder = DraftKvBuilder(
        _runner(draft_layers=2), lambda *, layers, target_block_size: next(pools)
    )

    assert builder.kv_pool is first
    builder.invalidate_pool_rows()
    assert builder.kv_pool is second


def test_releasing_before_the_first_build_builds_nothing():
    """A rollout sleep can land before any allocation, and `release` reaching
    through the property would build a pool in order to free it."""
    make, asked = _recording_factory(object())
    builder = DraftKvBuilder(_runner(draft_layers=2), make)

    builder.release_kv_pools()

    assert asked == []


def test_draft_exports_zero_copy_mp_views_and_disables_tp_collapse():
    pool = MhaKvPool(
        layers=1,
        block_size=128,
        num_kv_heads=4,
        head_dim=128,
        kv_dtype=torch.bfloat16,
    )
    pool.allocate(2, "cpu")
    builder = DraftKvBuilder.__new__(DraftKvBuilder)
    builder._kv_pool = pool
    layer = type("Layer", (), {"layer_num": 9})()
    builder._pool_rows = {DRAFT_KV_ROWS: {layer: 0}}

    transfer = builder.get_kv_transfer_tensors()
    transfer.set_block_count(2)

    assert transfer.tp_replication_factor == 1
    assert [region.semantic_role for region in transfer.block_regions] == [
        "draft.k.layer_0",
        "draft.v.layer_0",
    ]
    # Named by its layer under its own namespace, split by head like a target.
    assert [(p.unit_spec.namespace, p.unit_spec.layer) for p in transfer.pages] == [
        ("draft", 9)
    ] * 2
    assert {p.head_major_runs for p in transfer.pages} == {(1, 4)}
    assert [tuple(view.shape[:2]) for view in transfer.block_tensor_views] == [
        (2, 1),
        (2, 1),
    ]
    assert all(
        region.base_addr == view.data_ptr()
        for region, view in zip(
            transfer.block_regions, transfer.block_tensor_views, strict=True
        )
    )


def test_rows_are_the_pool_layers_the_modules_index():
    """The two answers this class is about, taken from one walk: every row
    index a module gets has to fall inside the pool built for them."""
    make, _ = _recording_factory(object())
    builder = DraftKvBuilder(_runner(draft_layers=5), make)

    rows = builder.pool_rows[DRAFT_KV_ROWS]

    assert sorted(rows.values()) == list(range(5))
    assert builder.row_counts()[DRAFT_KV_ROWS] == 5


@pytest.mark.parametrize("draft_layers", [1, 3, 8])
def test_the_count_tracks_the_model_it_walked(draft_layers):
    """A fixed number would pass the tests above on one shape only."""
    make, asked = _recording_factory(object())
    builder = DraftKvBuilder(_runner(draft_layers), make)

    assert builder.kv_pool is not None

    assert asked == [(draft_layers, TARGET_BLOCK)]


def test_a_plugin_config_answers_no_without_reading_native_draft_fields(monkeypatch):
    """The vLLM offload adapter hands over vLLM's own SpeculativeConfig: `method`, no
    `draft_model_hf_config`; and the selector has no backend for every draft flavor
    there. Neither is consulted under a plugin."""
    from atom.plugin import prepare
    from atom.utils import selector

    def refuse(*_a):
        raise AssertionError("the selector was asked")

    monkeypatch.setattr(prepare, "_CURRENT_FRAMEWORK", "vllm")
    monkeypatch.setattr(selector, "attn_family", refuse)
    vllm_spec = SimpleNamespace(method="eagle3", draft_model_config=object())
    assert draft_owns_kv_pool(SimpleNamespace(speculative_config=vllm_spec)) is False


@pytest.mark.parametrize(
    "target, pool_dtype", [("fp4", "bf16"), ("fp8", "fp8")]  # fp4 has no draft layout
)
def test_an_eagle3_draft_attention_is_built_at_the_dtype_of_its_pool(
    target, pool_dtype
):
    """The draft's attention reads its dtype off the config it is built with and its
    pool is built at `draft_kv_dtype`: one value; the target's config keeps its own."""
    # eagle_proposer pulls aiter in through atom.distributed.pcp_utils
    pytest.importorskip("aiter", reason="the proposer imports aiter at module load")
    import atom.spec_decode.eagle_proposer as mod
    from atom.config import CompilationConfig, resolve_draft_kv_dtype

    spec = SimpleNamespace(method="eagle3", draft_model_hf_config=SimpleNamespace())
    spec.draft_kv_dtype = resolve_draft_kv_dtype(target, own_pool=True)
    assert spec.draft_kv_dtype == pool_dtype
    p = mod.EagleProposer.__new__(mod.EagleProposer)
    p.speculative_config, p.runner = spec, SimpleNamespace()
    p.config = SimpleNamespace(
        kv_cache_dtype=target,
        hf_config=SimpleNamespace(num_hidden_layers=3),
        compilation_config=CompilationConfig(),
    )
    seen = {}

    def model_class(cfg, layer_offset):
        seen["dtype"] = cfg.kv_cache_dtype
        return object()

    orig, mod.draft_kv_builder = mod.draft_kv_builder, lambda runner, hf, model: None
    try:
        p._build_draft_model(model_class)
    finally:
        mod.draft_kv_builder = orig

    assert seen == {"dtype": pool_dtype} and p.config.kv_cache_dtype == target


def test_gqa_draft_layers_share_one_bound_unit_scale_tensor():
    """The DSpark GQA draft's pool binds every layer the same one-element tensor
    at the scale its spec declares, at bind time: both its write and its read
    take the scale as a device pointer, and nothing may be built for it inside
    a captured forward."""
    pool = MhaKvPool(
        layers=2,
        block_size=128,
        num_kv_heads=2,
        head_dim=64,
        kv_dtype=torch.bfloat16,
        flash_layout=True,
    )
    pool.allocate(2, "cpu")
    runner = _runner(draft_layers=2)
    runner.config = SimpleNamespace(max_model_len=4096, kv_cache_dtype="fp8")
    builder = DraftKvBuilder(
        runner, lambda **_: pool, kv_scale=DraftKvPoolSpec().kv_scale
    )
    builder._kv_pool = pool

    layers = [m for m in runner.drafter.model if isinstance(m, _PagedAttention)]
    for m in layers:
        m.layer_num = 0
        builder.build_kv_cache_tensor(m)

    scale = layers[0].k_scale
    assert torch.is_tensor(scale) and scale.numel() == 1 and float(scale) == 1.0
    assert all(m.k_scale is scale and m.v_scale is scale for m in layers)


def _bind_first_layer(pool, kv_cache_dtype, kv_scale=None):
    """Bind one paged layer through the builder; return the module."""
    pool.allocate(2, "cpu")
    runner = _runner(draft_layers=1)
    runner.config = SimpleNamespace(max_model_len=4096, kv_cache_dtype=kv_cache_dtype)
    builder = DraftKvBuilder(runner, lambda **_: pool, kv_scale=kv_scale)
    builder._kv_pool = pool
    module = next(m for m in runner.drafter.model if isinstance(m, _PagedAttention))
    module.layer_num = 0
    builder.build_kv_cache_tensor(module)
    return module


def test_a_default_pool_binds_shuffle_views_and_its_own_scale_planes():
    """MiniMax-M3.1 DSpark and Eagle3: the pool's defaults bind what they
    always did, `kv_views` and the per-token `scale_views`."""
    pool = MhaKvPool(
        layers=1,
        block_size=128,
        num_kv_heads=2,
        head_dim=64,
        kv_dtype=torch.float8_e4m3fnuz,
    )
    module = _bind_first_layer(pool, "fp8")

    k, v = pool.kv_views(0)
    k_scale, v_scale = pool.scale_views(0)
    assert module.k_cache.shape == k.shape and module.v_cache.shape == v.shape
    assert module.k_cache.data_ptr() == k.data_ptr()
    assert module.k_scale.shape == k_scale.shape
    assert module.k_scale.data_ptr() == k_scale.data_ptr()
    assert module.v_scale.data_ptr() == v_scale.data_ptr()


@pytest.mark.parametrize("blocks_per_entry", [1, 8])
def test_a_flash_pool_binds_entry_wide_flash_views_and_the_declared_scale(
    blocks_per_entry,
):
    """The FLASH order is the pool's own declaration, not inferred from the
    entry width: at one block an entry it still binds `fa_views`. The scale is
    the spec's, which the builder is handed, not one the layout implies."""
    pool = MhaKvPool(
        layers=1,
        block_size=128,
        num_kv_heads=2,
        head_dim=64,
        kv_dtype=torch.float8_e4m3fnuz,
        blocks_per_entry=blocks_per_entry,
        flash_layout=True,
    )
    module = _bind_first_layer(pool, "fp8", kv_scale=DraftKvPoolSpec().kv_scale)

    assert module.k_cache.shape == (2, 128 * blocks_per_entry, 2, 64)
    assert module.k_cache.shape[1] == pool.entry_tokens
    assert module.v_cache.shape == pool.fa_views(0)[1].shape
    assert module.k_scale.numel() == 1 and float(module.k_scale) == 1.0


@pytest.mark.parametrize("dcp", [1, 8])
def test_a_declared_entry_is_the_span_of_the_pool_the_spec_builds(dcp):
    """`entry_tokens` is what a draft builds its reader for; it has to be the
    width of the pool built from the same spec's block and kwargs."""
    spec = DraftKvPoolSpec()
    config = SimpleNamespace(kv_cache_block_size=128, decode_context_parallel_size=dcp)
    pool = MhaKvPool(
        layers=1,
        block_size=spec.block_size(config),
        num_kv_heads=2,
        head_dim=64,
        kv_dtype=torch.bfloat16,
        **spec.pool_kwargs(dcp),
    )
    assert spec.entry_tokens(config) == pool.entry_tokens == 128 * dcp


@pytest.mark.parametrize("dcp", [1, 8])
def test_a_pooled_draft_has_slots_of_its_own_only_under_dcp(dcp):
    """At DCP 1 an entry is one target block, so the target's slots address
    the draft's pool; under DCP they don't."""
    config = SimpleNamespace(kv_cache_block_size=128, decode_context_parallel_size=dcp)
    assert DraftKvPoolSpec().needs_own_ctx_slots(config) is (dcp > 1)


def test_a_layer_refuses_a_cache_its_reader_was_not_built_for():
    """A layer's `check_cache` sees the views before they are bound, and its
    error stops the bind: the layer keeps no cache it would misread."""
    pool = MhaKvPool(
        layers=1,
        block_size=128,
        num_kv_heads=2,
        head_dim=64,
        kv_dtype=torch.bfloat16,
        blocks_per_entry=2,
        flash_layout=True,
    )
    pool.allocate(2, "cpu")
    runner = _runner(draft_layers=1)
    runner.config = SimpleNamespace(max_model_len=4096, kv_cache_dtype="bf16")
    builder = DraftKvBuilder(runner, lambda **_: pool, kv_scale=1.0)
    builder._kv_pool = pool
    module = next(m for m in runner.drafter.model if isinstance(m, _PagedAttention))
    module.layer_num = 0
    seen = []

    def check_cache(k_cache, v_cache):
        seen.append((k_cache.shape, v_cache.shape))
        if k_cache.shape[1] != 128:
            raise ValueError("entry width")

    module.check_cache = check_cache
    with pytest.raises(ValueError, match="entry width"):
        builder.build_kv_cache_tensor(module)
    assert seen == [((2, 256, 2, 64), (2, 256, 2, 64))]
    assert not hasattr(module, "k_cache")


@pytest.mark.parametrize("declared", [True, False])
def test_a_declared_pool_on_a_backend_that_cannot_own_one_is_refused(
    monkeypatch, declared
):
    """A draft that declares a pool of its own on a backend that owns none is
    refused here, not dropped: the target would still stage its slots and
    fail later without a builder. A draft that declares none gets None."""
    pytest.importorskip("aiter")
    from atom.utils import selector

    class _Backend:
        pass

    class _Draft:
        kv_pool_spec = DraftKvPoolSpec() if declared else None

    monkeypatch.setattr(selector, "get_attn_backend", lambda _family: _Backend)
    monkeypatch.setattr(selector, "attn_family", lambda _cfg: selector.Family.MLA)
    runner = SimpleNamespace(config=SimpleNamespace(kv_cache_dtype="bf16"))
    if declared:
        with pytest.raises(ValueError, match="cannot own a draft pool"):
            draft_kv_builder(runner, None, _Draft())
    else:
        assert draft_kv_builder(runner, None, _Draft()) is None


def test_a_declared_scale_other_than_unit_is_refused():
    """The pool has no scale planes and nothing loads a scale into it, so a
    spec declaring any other scale is refused when its builder is asked for,
    before a backend is resolved, rather than bound as if it were 1.0."""

    class _Draft:
        kv_pool_spec = DraftKvPoolSpec(kv_scale=0.5)

    with pytest.raises(ValueError, match="declares a KV scale of 0.5"):
        draft_kv_builder(SimpleNamespace(), None, _Draft())
