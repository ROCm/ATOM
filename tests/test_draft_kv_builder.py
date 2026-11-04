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
from atom.spec_decode.draft_kv import DRAFT_KV_ROWS, DraftKvBuilder


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


def test_the_dflash_pool_takes_the_schedulers_block():
    """The DFlash pool sits beside an MLA target, whose builder pages at
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

    transfer = builder.get_kv_transfer_tensors()
    transfer.set_block_count(2)

    assert transfer.tp_replication_factor == 1
    assert [region.semantic_role for region in transfer.block_regions] == [
        "draft.k.layer_0",
        "draft.v.layer_0",
    ]
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


def test_dflash_layers_share_one_bound_unit_scale_tensor():
    """The DFlash pool binds every layer the same one-element 1.0 tensor, at
    bind time: both its write and its read take the scale as a device pointer,
    and nothing may be built for it inside a captured forward."""
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
    builder = DraftKvBuilder(runner, lambda **_: pool)
    builder._kv_pool = pool

    layers = [m for m in runner.drafter.model if isinstance(m, _PagedAttention)]
    for m in layers:
        m.layer_num = 0
        builder.build_kv_cache_tensor(m)

    scale = layers[0].k_scale
    assert torch.is_tensor(scale) and scale.numel() == 1 and float(scale) == 1.0
    assert all(m.k_scale is scale and m.v_scale is scale for m in layers)


def _bind_first_layer(pool, kv_cache_dtype):
    """Bind one paged layer through the builder; return the module."""
    pool.allocate(2, "cpu")
    runner = _runner(draft_layers=1)
    runner.config = SimpleNamespace(max_model_len=4096, kv_cache_dtype=kv_cache_dtype)
    builder = DraftKvBuilder(runner, lambda **_: pool)
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
def test_a_flash_pool_binds_entry_wide_flash_views_and_a_unit_scale(
    blocks_per_entry,
):
    """The FLASH order is the pool's own declaration, not inferred from the
    entry width: at one block an entry it still binds `fa_views`."""
    pool = MhaKvPool(
        layers=1,
        block_size=128,
        num_kv_heads=2,
        head_dim=64,
        kv_dtype=torch.float8_e4m3fnuz,
        blocks_per_entry=blocks_per_entry,
        flash_layout=True,
    )
    module = _bind_first_layer(pool, "fp8")

    assert module.k_cache.shape == (2, 128 * blocks_per_entry, 2, 64)
    assert module.k_cache.shape[1] == pool.entry_tokens
    assert module.v_cache.shape == pool.fa_views(0)[1].shape
    assert module.k_scale.numel() == 1 and float(module.k_scale) == 1.0


def test_kv_views_refuses_a_flash_pool():
    """Both views cover the same bytes, so the SHUFFLE one would read a FLASH
    pool in the wrong order without any size check noticing."""
    pool = MhaKvPool(
        layers=1,
        block_size=128,
        num_kv_heads=2,
        head_dim=64,
        kv_dtype=torch.float8_e4m3fnuz,
        flash_layout=True,
    )
    pool.allocate(2, "cpu")
    with pytest.raises(ValueError, match="fa_views"):
        pool.kv_views(0)
    assert pool.scale_fields == []
