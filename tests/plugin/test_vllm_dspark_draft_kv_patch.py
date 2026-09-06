# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The DSpark draft's SWA group must abstain from the prefix cache.

`HybridKVCacheCoordinator` reconciles every group's hit down to the minimum, so
the draft group -- whose blocks are evicted by `remove_skipped_blocks` long
before the next request looks them up -- caps the target's hit. These tests pin
the abstention and, just as importantly, pin everything the retype must leave
alone: field values, page size, grouping, and picklability.
"""

import dataclasses
import pickle

import pytest
import torch

from atom.plugin.vllm.dspark_draft_kv_patch import (
    convert_draft_specs,
    ensure_registered,
)

pytest.importorskip("vllm")

from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    SlidingWindowMLASpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry

BLOCK = 64
LAYERS = ("model.layers.43.attn.swa_cache", "model.layers.44.attn.swa_cache")


class _FakeBlockPool:
    null_block = "NULL"


def _swa_spec():
    # DSpark's draft geometry: block 64, window 128, MLA head.
    return SlidingWindowMLASpec(
        block_size=BLOCK,
        num_kv_heads=1,
        head_size=576,
        dtype=torch.bfloat16,
        sliding_window=128,
    )


def _draft_specs():
    return {name: _swa_spec() for name in LAYERS}


def _converted_spec():
    return next(iter(convert_draft_specs(_draft_specs()).values()))


def _hit(max_length, *, drop_eagle_block=False, alignment_tokens=0, groups=(0,)):
    spec = _converted_spec()
    manager = KVCacheSpecRegistry.get_manager_class(spec)
    return manager.find_longest_cache_hit(
        block_hashes=[],
        max_length=max_length,
        kv_cache_group_ids=groups,
        block_pool=_FakeBlockPool(),
        kv_cache_spec=spec,
        drop_eagle_block=drop_eagle_block,
        alignment_tokens=alignment_tokens,
    )


# --------------------------------------------------------------------------
# The retype must change the type and nothing else.
# --------------------------------------------------------------------------


def test_retype_preserves_every_spec_field():
    original = _swa_spec()
    converted = _converted_spec()

    assert type(converted) is not SlidingWindowMLASpec
    assert isinstance(converted, SlidingWindowMLASpec)
    for field in dataclasses.fields(original):
        assert getattr(converted, field.name) == getattr(original, field.name)


def test_retype_leaves_kv_pool_sizing_untouched():
    # The subclass adds no fields, so `__post_init__` padding and the page size
    # the memory profiler budgets against must be identical.
    assert _converted_spec().page_size_bytes == _swa_spec().page_size_bytes


def test_retyped_specs_still_group_as_one_uniform_type():
    # The draft's three layers must stay a single KV cache group; splitting them
    # would hand vLLM more groups than the proxy layout expects.
    converted = convert_draft_specs(_draft_specs())

    assert UniformTypeKVCacheSpecs.is_uniform_type(converted)
    assert (
        KVCacheSpecRegistry.get_uniform_type_base_spec(next(iter(converted.values())))
        is SlidingWindowMLASpec
    )


def test_retyped_spec_survives_the_broadcast_to_workers():
    # KVCacheConfig is pickled to every worker, and pickle resolves a class by
    # __module__ + __qualname__. A class built inside a function keeps a
    # `<locals>` qualname that resolves nowhere.
    spec = _converted_spec()

    assert type(pickle.loads(pickle.dumps(spec))) is type(spec)


def test_registering_the_draft_spec_keeps_vllm_builtins_registered():
    # `_ensure_registered` populates the built-ins only if the registry is
    # empty, so registering into an empty registry would starve all of them.
    ensure_registered()

    assert KVCacheSpecRegistry.get_manager_class(_swa_spec()) is not None
    assert KVCacheSpecRegistry.get_manager_class(_converted_spec()) is not None


def test_ensure_registered_returns_the_same_class_every_time():
    # Two unequal types would make pickle resolve to whichever was cached last.
    assert ensure_registered() is ensure_registered()


# --------------------------------------------------------------------------
# The abstention itself.
# --------------------------------------------------------------------------


def test_draft_group_never_shortens_the_reconciled_hit():
    _, hit = _hit(10 * BLOCK)

    assert hit == 10 * BLOCK


def test_hit_is_block_aligned_down():
    # The coordinator rejects a hit length that is not a whole number of blocks.
    _, hit = _hit(10 * BLOCK + 1)

    assert hit == 10 * BLOCK


def test_hit_honours_the_alignment_the_coordinator_asks_for():
    _, hit = _hit(1000, alignment_tokens=256)

    assert hit == 768


def test_drop_eagle_block_gives_one_block_back():
    # The coordinator hands us `candidate + block_size` when it is set and
    # expects to get the candidate back.
    _, hit = _hit(10 * BLOCK, drop_eagle_block=True)

    assert hit == 9 * BLOCK


def test_drop_eagle_block_cannot_drive_the_hit_negative():
    _, hit = _hit(0, drop_eagle_block=True)

    assert hit == 0


def test_the_hit_claims_no_real_storage():
    # Tokens are declared computed, but every block is the null block: nothing
    # reads that region, and `_roll_back_prefix_hit` re-forwards the last 512
    # tokens -- more than the 128-token window -- into fresh blocks.
    blocks, hit = _hit(10 * BLOCK, groups=(0, 1))

    assert len(blocks) == 2
    for group in blocks:
        assert len(group) == hit // BLOCK
        assert set(group) == {_FakeBlockPool.null_block}


def test_the_draft_publishes_nothing_to_the_prefix_cache():
    spec = _converted_spec()
    manager = KVCacheSpecRegistry.get_manager_class(spec)

    # An entry here could only keep a draft block hash-resident for a hit the
    # lookup above will never take. Called unbound -- the override touches no
    # instance state, and a real manager needs a live block pool to construct.
    assert manager.cache_blocks(None, request=object(), num_tokens=1024) is None


# --------------------------------------------------------------------------
# When the conversion must not happen.
# --------------------------------------------------------------------------


def test_non_sliding_window_drafts_keep_vllm_stock_behaviour():
    # An MTP draft is full attention and does contribute real cache hits.
    specs = {
        "model.layers.61.attn": FullAttentionSpec(
            block_size=128, num_kv_heads=1, head_size=576, dtype=torch.bfloat16
        )
    }

    assert convert_draft_specs(specs) is specs


def test_mixed_draft_specs_are_left_alone_wholesale():
    specs = dict(_draft_specs())
    specs["model.layers.61.attn"] = FullAttentionSpec(
        block_size=128, num_kv_heads=1, head_size=576, dtype=torch.bfloat16
    )

    assert convert_draft_specs(specs) is specs


def test_no_draft_specs_is_a_no_op():
    assert convert_draft_specs({}) == {}


def test_a_broken_conversion_never_blocks_startup(monkeypatch):
    # A prefix-cache optimisation must not be the reason a server fails to boot.
    monkeypatch.setattr(
        "atom.plugin.vllm.dspark_draft_kv_patch.ensure_registered",
        lambda: (_ for _ in ()).throw(RuntimeError("registry moved")),
    )
    specs = _draft_specs()

    assert convert_draft_specs(specs) is specs
