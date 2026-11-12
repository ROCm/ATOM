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
from atom.spec_decode.draft_kv import DRAFT_KV_ROWS, DraftKvBuilder, draft_owns_kv_pool
from atom.utils.selector import Family


class _PagedAttention(nn.Module):
    """What `_module_kinds` counts: paged, and not MLA."""

    def __init__(self):
        super().__init__()
        self.base_attention = True
        self.use_mla = False


class _Norm(nn.Module):
    """A sibling the walk must not count."""


# The target builder's block. A draft's pool takes it rather than picking one,
# because the draft indexes that pool with the target's block tables.
TARGET_BLOCK = 128


def _runner(draft_layers: int, block_size: int = TARGET_BLOCK):
    model = nn.Sequential(
        *[m for _ in range(draft_layers) for m in (_PagedAttention(), _Norm())]
    )
    return SimpleNamespace(
        drafter=SimpleNamespace(model=model),
        attn_metadata_builder=SimpleNamespace(block_size=block_size),
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


class _FakePool:
    def __init__(self, scale_fields):
        self.scale_fields = scale_fields

    def kv_views(self, idx):
        return torch.zeros(1), torch.zeros(1)

    def scale_views(self, idx):
        return torch.ones(1), torch.ones(1)


@pytest.mark.parametrize("scale_fields", [[object()], []])
def test_scales_are_bound_when_the_pools_layout_has_them(scale_fields):
    """The layout decides, as the target binder asks it: not a dtype string."""
    runner = _runner(draft_layers=1)
    runner.config = SimpleNamespace(max_model_len=64)
    builder = DraftKvBuilder(runner, None)
    builder._kv_pool = _FakePool(scale_fields)
    module = next(iter(builder.pool_rows[DRAFT_KV_ROWS]))
    module.layer_num = 0

    tensor = builder.build_kv_cache_tensor(module)

    assert (tensor.k_scale is not None) is bool(scale_fields)


def _spec_config(method="dspark", draft_hf="draft-hf"):
    return SimpleNamespace(
        speculative_config=SimpleNamespace(
            method=method, draft_model_hf_config=draft_hf
        )
    )


@pytest.mark.parametrize(
    "config, owns, want",
    [
        (_spec_config(), True, True),
        (_spec_config(), False, False),  # the draft's backend shares the target's pool
        (_spec_config("eagle3"), True, True),  # a draft model of its own, MHA: a pool
        (
            _spec_config("eagle3"),
            False,
            False,
        ),  # ... an MLA one binds into the target's
        (_spec_config("mtp"), True, False),  # MTP runs the target's layers
        (_spec_config(draft_hf=None), True, False),
        (SimpleNamespace(speculative_config=None), True, False),
    ],
)
def test_a_native_draft_owns_a_pool_when_its_backend_says_so(
    monkeypatch, config, owns, want
):
    from atom.utils import selector

    monkeypatch.setattr(
        selector, "attn_family", lambda _hf: Family.MHA if owns else Family.MLA
    )
    assert draft_owns_kv_pool(config) is want


def test_a_plugin_draft_owns_no_pool_and_the_selector_is_not_asked(monkeypatch):
    """The native runner builds a draft's pool; under vLLM / SGLang the selector has
    no backend for every draft flavor (CSA2 raises), so it is not consulted."""
    from atom.plugin import prepare
    from atom.utils import selector

    def refuse(*_a):
        raise AssertionError("the selector was asked")

    monkeypatch.setattr(prepare, "_CURRENT_FRAMEWORK", "vllm")
    monkeypatch.setattr(selector, "attn_family", refuse)
    assert draft_owns_kv_pool(_spec_config()) is False


def test_a_plugin_config_without_native_draft_fields_answers_no(monkeypatch):
    """The vLLM offload adapter passes vLLM's SpeculativeConfig through unchanged: it
    has `method` but no `draft_model_hf_config`, and the answer must not read it."""
    from atom.plugin import prepare

    monkeypatch.setattr(prepare, "_CURRENT_FRAMEWORK", "vllm")
    vllm_spec = SimpleNamespace(method="eagle3", draft_model_config=object())
    assert draft_owns_kv_pool(SimpleNamespace(speculative_config=vllm_spec)) is False


@pytest.mark.parametrize(
    "target, asked, pool_dtype",
    [
        ("fp8", "bf16", "bf16"),  # the inverse overrides mismatch too
        ("bf16", "fp8", "fp8"),
        ("fp4", None, "bf16"),  # an fp4 target has no draft layout
        ("fp8", None, "fp8"),
    ],
)
def test_an_eagle3_draft_attention_is_built_at_the_dtype_of_its_pool(
    target, asked, pool_dtype
):
    """The draft's attention reads its cache dtype off the config it is built
    with, and its pool is built at `speculative_config.kv_cache_dtype` (what
    `draft_kv_builder` hands `make_kv_pool`). They are one value; the target's
    config keeps its own."""
    # eagle_proposer pulls aiter in through atom.distributed.pcp_utils
    pytest.importorskip("aiter", reason="the proposer imports aiter at module load")
    from atom.config import CompilationConfig, resolve_draft_kv_dtype
    from atom.spec_decode.eagle_proposer import EagleProposer

    spec = SimpleNamespace(
        method="eagle3",
        draft_model_hf_config=SimpleNamespace(),
        kv_cache_dtype=asked,
    )
    spec.kv_cache_dtype = resolve_draft_kv_dtype(spec, target, own_pool=True)
    assert spec.kv_cache_dtype == pool_dtype

    p = EagleProposer.__new__(EagleProposer)
    p.speculative_config = spec
    p.runner = SimpleNamespace()
    p.config = SimpleNamespace(
        kv_cache_dtype=target,
        hf_config=SimpleNamespace(num_hidden_layers=3),
        compilation_config=CompilationConfig(),
    )
    seen = {}

    def model_class(cfg, layer_offset):
        seen["dtype"] = cfg.kv_cache_dtype
        return object()

    import atom.spec_decode.eagle_proposer as mod

    orig, mod.draft_kv_builder = mod.draft_kv_builder, lambda runner, hf: None
    try:
        p._build_draft_model(model_class)
    finally:
        mod.draft_kv_builder = orig

    assert seen == {"dtype": pool_dtype}
    assert p.config.kv_cache_dtype == target
