# SPDX-License-Identifier: MIT
"""Hybrid KV-load-failure recovery: the path whose crash is EngineDeadError
and whose wrong truncation is silent bad tokens.

vLLM's own ``_update_requests_with_invalid_blocks`` unpacks
``get_block_ids`` as a 1-tuple. Kimi-K3 has four groups, so the first failed
tier load kills the engine. ATOM replaces that with a group-aware scan in
``atom.plugin.vllm.scheduler``. These tests pin the accuracy contracts of
that scan, not just that the unpack no longer raises:

* a mamba group's block index does not convert to a token offset -- using it
  would silently move ``num_computed_tokens`` to the wrong place;
* when two attention groups disagree, the prefix is truncated at the
  earliest valid point;
* a shared invalid block already claimed by another request does not
  re-truncate this one into a prefix it does not own.

The CI unit job has no vLLM. The stubs below are enough for the mixin and
for ``select_scheduler_cls``; they are installed only when the real package
is absent, so a job that does have vLLM still exercises the live module.
"""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest


def _install_vllm_stubs() -> None:
    if "vllm" in sys.modules:
        return

    def _pkg(name: str) -> types.ModuleType:
        mod = types.ModuleType(name)
        mod.__path__ = []  # type: ignore[attr-defined]
        sys.modules[name] = mod
        return mod

    _pkg("vllm")
    _pkg("vllm.v1")
    _pkg("vllm.v1.core")
    _pkg("vllm.v1.core.sched")
    kvci = _pkg("vllm.v1.kv_cache_interface")

    class AttentionSpec:
        pass

    class CrossAttentionSpec(AttentionSpec):
        pass

    class EncoderOnlyAttentionSpec(AttentionSpec):
        pass

    kvci.AttentionSpec = AttentionSpec
    kvci.CrossAttentionSpec = CrossAttentionSpec
    kvci.EncoderOnlyAttentionSpec = EncoderOnlyAttentionSpec

    class Scheduler:
        def _update_requests_with_invalid_blocks(self, *args, **kwargs):
            # The sentinel ATOM greps for. Keep the spelling identical.
            (req_block_ids,) = self.kv_cache_manager.get_block_ids("x")
            return req_block_ids

    class AsyncScheduler(Scheduler):
        pass

    sched = types.ModuleType("vllm.v1.core.sched.scheduler")
    sched.Scheduler = Scheduler
    sys.modules[sched.__name__] = sched
    async_sched = types.ModuleType("vllm.v1.core.sched.async_scheduler")
    async_sched.AsyncScheduler = AsyncScheduler
    sys.modules[async_sched.__name__] = async_sched


_install_vllm_stubs()

from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    CrossAttentionSpec,
    EncoderOnlyAttentionSpec,
)

from atom.plugin.vllm.scheduler import (
    _HybridKVLoadFailureMixin,
    select_scheduler_cls,
    vllm_needs_hybrid_kv_load_fix,
)


class _MambaSpec:
    """Not an AttentionSpec -- KDA / GDN groups look like this to the mixin."""


class _Group:
    def __init__(self, spec):
        self.kv_cache_spec = spec


def _spec(cls, **attrs):
    """Build a spec without running its dataclass ``__init__``.

    The live vLLM ``AttentionSpec`` requires a dozen fields this scan never
    reads. ``__new__`` keeps ``isinstance`` checks honest on both the stub
    used when vLLM is absent and the real class.
    """
    obj = cls.__new__(cls)
    for name, value in attrs.items():
        object.__setattr__(obj, name, value)
    return obj


def _attn(block_size: int):
    return _spec(AttentionSpec, block_size=block_size)


class _Engine(_HybridKVLoadFailureMixin):
    def __init__(self, groups, block_ids_by_req):
        self.kv_cache_manager = SimpleNamespace(
            kv_cache_config=SimpleNamespace(kv_cache_groups=list(groups)),
            get_block_ids=lambda req_id: block_ids_by_req[req_id],
        )


def _req(req_id: str, num_computed_tokens: int):
    return SimpleNamespace(request_id=req_id, num_computed_tokens=num_computed_tokens)


def test_four_groups_do_not_raise_on_the_unpack_that_kills_hybrid_engines():
    """The bug this module exists for: ``(ids,) = get_block_ids`` with 4 groups."""
    groups = [
        _Group(_attn(64)),
        _Group(_MambaSpec()),
        _Group(_MambaSpec()),
        _Group(_MambaSpec()),
    ]
    req = _req("r0", 256)
    engine = _Engine(groups, {"r0": ([10, 11, 12, 13], [1], [2], [3])})

    affected, tokens, evict = engine._update_requests_with_invalid_blocks(
        [req], invalid_block_ids={11}, num_scheduled_tokens={"r0": 0}
    )

    assert affected == {"r0"}
    assert req.num_computed_tokens == 64
    assert tokens == 256 - 64
    assert 11 in evict


def test_mamba_invalid_ids_are_ignored_they_are_not_token_offsets():
    """Using a mamba block index as ``idx * block_size`` would silently
    rewrite the computed frontier to a token the forward never had."""
    groups = [_Group(_attn(64)), _Group(_MambaSpec())]
    # Attention blocks are all valid. The "invalid" id lives only in the
    # mamba group. Truncating from it is wrong output, not a crash.
    req = _req("r0", 192)
    engine = _Engine(groups, {"r0": ([10, 11, 12], [99, 98, 97])})

    affected, tokens, evict = engine._update_requests_with_invalid_blocks(
        [req], invalid_block_ids={99, 98}, num_scheduled_tokens={"r0": 0}
    )

    assert affected == set()
    assert req.num_computed_tokens == 192
    assert tokens == 0
    assert evict == set()


def test_earliest_attention_group_wins_when_groups_disagree():
    """A prefix is valid only where it is valid in every attention group."""
    groups = [_Group(_attn(64)), _Group(_attn(64)), _Group(_MambaSpec())]
    req = _req("r0", 256)
    # Group 0 first invalid at idx 2 -> 128; group 1 first invalid at idx 1 -> 64.
    engine = _Engine(
        groups,
        {"r0": ([10, 11, 12, 13], [20, 21, 22, 23], [1])},
    )

    affected, _, _ = engine._update_requests_with_invalid_blocks(
        [req], invalid_block_ids={12, 21}, num_scheduled_tokens={"r0": 0}
    )

    assert affected == {"r0"}
    assert req.num_computed_tokens == 64


def test_cross_and_encoder_attention_are_skipped_like_mamba():
    groups = [
        _Group(_attn(32)),
        _Group(_spec(CrossAttentionSpec)),
        _Group(_spec(EncoderOnlyAttentionSpec)),
    ]
    req = _req("r0", 96)
    engine = _Engine(groups, {"r0": ([10, 11, 12], [50], [60])})

    engine._update_requests_with_invalid_blocks(
        [req], invalid_block_ids={50, 60, 11}, num_scheduled_tokens={"r0": 0}
    )

    assert req.num_computed_tokens == 32


def test_shared_invalid_block_does_not_retruncate_the_second_request():
    groups = [_Group(_attn(64))]
    first = _req("a", 192)
    second = _req("b", 192)
    engine = _Engine(
        groups,
        {"a": ([10, 11, 12],), "b": ([10, 11, 12],)},
    )

    engine._update_requests_with_invalid_blocks(
        [first, second], invalid_block_ids={11}, num_scheduled_tokens={}
    )

    assert first.num_computed_tokens == 64
    # Shared with `a`, already marked. `b` stays at the cached-only floor
    # (num_computed - scheduled), which here is 192.
    assert second.num_computed_tokens == 192


def test_select_scheduler_cls_does_not_override_an_explicit_choice():
    cfg = SimpleNamespace(scheduler_cls="someone.Else", async_scheduling=False)
    assert select_scheduler_cls(cfg) is None


def test_select_scheduler_cls_picks_async_when_resolved_true():
    if not vllm_needs_hybrid_kv_load_fix():
        pytest.skip("live vLLM already has hybrid recovery")
    cfg = SimpleNamespace(scheduler_cls=None, async_scheduling=True)
    assert select_scheduler_cls(cfg) == (
        "atom.plugin.vllm.scheduler.VllmAtomAsyncScheduler"
    )
    cfg.async_scheduling = False
    assert select_scheduler_cls(cfg) == "atom.plugin.vllm.scheduler.VllmAtomScheduler"
