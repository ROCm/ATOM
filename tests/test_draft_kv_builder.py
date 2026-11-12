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

from atom.kv_transfer.disaggregation.sharded_transfer import consumer_head_slices
from atom.model_ops.attentions.mha_kv_pool import MhaKvPool
from atom.spec_decode.draft_kv import DRAFT_KV_ROWS, DraftKvBuilder


class _PagedAttention(nn.Module):
    """What `_module_kinds` counts: paged, and not MLA."""

    def __init__(self):
        super().__init__()
        self.base_attention = True
        self.use_mla = False
        self.page_tokens = None


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
    layer = type("Layer", (), {"layer_num": 9, "page_tokens": None})()
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


def _paged(page):
    """A binding's K, V and scale views of a pool row, in `page`-token pages."""

    def views(pool, row):
        (k, v), scales = pool.kv_views(row), pool.scale_views(row)
        n, heads, x = k.shape[0] * pool.block_size // page, k.shape[1], k.shape[-1]
        return (
            k.view(n, heads, -1, page, x),
            v.view(n, heads, page // x, -1, x),
            *(s.view(n, heads, page) for s in scales),
        )

    return views


def _cut_into_tp4(binding, views):
    """Fill a TP1 draft per head through `views`, cut it for four TP4 ranks,
    and return the values each rank reads back through `views`."""

    def draft(heads):
        pool = MhaKvPool(
            layers=1,
            block_size=128,
            num_kv_heads=heads,
            head_dim=128,
            kv_dtype=torch.float8_e4m3fn,
        )
        pool.allocate(2, "cpu")
        builder = DraftKvBuilder.__new__(DraftKvBuilder)
        builder._kv_pool = pool
        builder._pool_rows = {DRAFT_KV_ROWS: {binding: 0}}
        return pool, builder.get_kv_transfer_tensors().pages

    src, src_pages = draft(4)
    for t in views(src, 0):
        for head in range(4):
            t[:, head] = torch.tensor(head + 1.0).to(t.dtype)
    received = []
    for rank in range(4):
        dst, dst_pages = draft(1)
        for s, d in zip(src_pages, dst_pages, strict=True):
            cuts = consumer_head_slices(
                s.region.unit_bytes,
                d.region.unit_bytes,
                s.head_major_runs,
                d.head_major_runs,
                rank,
                4,
                str(s.unit_spec),
            )
            for block in range(2):
                for so, do, n in cuts.tolist():
                    d.view[block].view(-1)[do : do + n] = s.view[block].view(-1)[
                        so : so + n
                    ]
        received.append({v for t in views(dst, 0) for v in t.float().unique().tolist()})
    return received


@pytest.mark.parametrize("page", [128, 16])
def test_each_tp4_rank_gets_its_head_in_the_pages_its_binding_views(page):
    binding = type("Binding", (), {"layer_num": 60, "page_tokens": page})()
    assert _cut_into_tp4(binding, _paged(page)) == [{1.0}, {2.0}, {3.0}, {4.0}]


def test_the_m31_dspark_draft_is_cut_as_its_attention_views_it():
    dspark = pytest.importorskip("atom.models.minimax_m3.dspark", exc_type=ImportError)
    from atom.model_ops.minimax_m3.sparse_attn import to_page16_shuffle

    def views(pool, row):
        return to_page16_shuffle(*pool.kv_views(row), *pool.scale_views(row))

    received = _cut_into_tp4(dspark.KVCacheBinding(60), views)
    assert received == [{1.0}, {2.0}, {3.0}, {4.0}]


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
