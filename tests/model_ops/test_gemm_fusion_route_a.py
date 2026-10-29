# SPDX-License-Identifier: MIT
"""Route A (aiter large/r19) tag-width support in ATOM's gemm_fusion wrapper.

CPU-only: no aiter or GPU is required. Route A's extended large band runs the ``r19``
source at LARGE_ABI generation 6, whose handshake tag repartitions the r13 word from
13-bit m_real / 19-bit call_id to 14-bit m_real / 18-bit call_id. ATOM must therefore
assign call ids whose low **18** bits are distinct per token count (a ``decision_key``
would collide there), while every SMALL plan -- including the SMALL ``r19`` source at
generation 3 -- keeps the full call_id in its tag.
"""

from types import SimpleNamespace

from atom.model_ops.gemm_fusion import hooks
from atom.model_ops.gemm_fusion.resources import (
    FusionResourcesWrapper,
    large_plan_tag_bits,
    large_tag_bits,
)

LARGE = "sp_qkv.large.v1"
SMALL = "sp_qkv.small.v1"


def _plan(src, abi, vid=None):
    return SimpleNamespace(src=src, abi=abi, **({} if vid is None else {"vid": vid}))


def test_route_a_large_r19_uses_the_18_bit_tag():
    assert large_plan_tag_bits(_plan("r19", 6, LARGE)) == 18
    assert large_tag_bits("r19") == 18
    # The SMALL r19 source is ABI generation 3 and tags the full call_id.
    assert large_plan_tag_bits(_plan("r19", 3, SMALL)) is None


def test_shipped_large_sources_keep_the_19_bit_tag():
    for src in ("r13", "r18"):
        assert large_plan_tag_bits(_plan(src, 4, LARGE)) == 19
    # r14's split-K generation keeps the same 19-bit tag word.
    assert large_plan_tag_bits(_plan("r14", 5, LARGE)) == 19
    # The SMALL r18 plan is the pre-R12 SMALL_ABI 2 plan: full call_id.
    assert large_plan_tag_bits(_plan("r18", 2, SMALL)) is None
    assert large_plan_tag_bits(_plan("r12", 3, SMALL)) is None


def test_missing_or_unknown_plan_fields_fail_safe():
    assert large_plan_tag_bits(_plan("r13", None)) is None
    assert large_plan_tag_bits(SimpleNamespace(abi=4)) is None  # no src
    assert large_plan_tag_bits(SimpleNamespace(src="r19")) is None  # no ABI
    assert large_plan_tag_bits(None) is None
    # Test doubles/older aiter plans may omit vid: the ABI still classifies them.
    assert large_plan_tag_bits(_plan("r13", 4)) == 19
    assert large_plan_tag_bits(_plan("r19", 3)) is None


def test_wrapper_plan_tag_bits_delegates_to_the_single_helper():
    assert FusionResourcesWrapper.plan_tag_bits(_plan("r19", 6, LARGE)) == 18
    assert FusionResourcesWrapper.plan_tag_bits(_plan("r18", 4, LARGE)) == 19
    assert FusionResourcesWrapper.plan_tag_bits(_plan("r19", 3, SMALL)) is None


def _wrapper_with_callid(monkeypatch, module):
    wrapper = object.__new__(FusionResourcesWrapper)
    monkeypatch.setattr(
        FusionResourcesWrapper, "_callid_module", staticmethod(lambda: module)
    )
    return wrapper


def test_wrapper_tag_unique_ids_passes_the_requested_width(monkeypatch):
    calls = []

    def tag_unique_ids(keys, tag_bits=19):
        calls.append(tag_bits)
        return {k: 0x12300000 | (i + 1) for i, k in enumerate(sorted(set(keys)))}

    module = SimpleNamespace(tag_unique_ids=tag_unique_ids)
    wrapper = _wrapper_with_callid(monkeypatch, module)
    keys = [("layer.0", 65536, LARGE, 1)]
    assert wrapper.tag_unique_ids(keys, tag_bits=18) is not None
    assert wrapper.tag_unique_ids(keys) is not None
    assert wrapper.tag_unique_ids(keys, tag_bits=19) is not None
    assert calls == [18, 19, 19]


def test_wrapper_refuses_a_width_an_older_aiter_cannot_assign(monkeypatch):
    # An aiter without the width argument can only assign 19-bit ids. Handing those to a
    # plan whose kernel keeps 18 bits would collide, so the wrapper reports "no ids".
    module = SimpleNamespace(tag_unique_ids=lambda keys: {k: 7 for k in keys})
    wrapper = _wrapper_with_callid(monkeypatch, module)
    keys = [("layer.0", 65536, LARGE, 1)]
    assert wrapper.tag_unique_ids(keys, tag_bits=18) is None
    assert wrapper.tag_unique_ids(keys) == {keys[0]: 7}


class _Site:
    def __init__(self, name, plan):
        self.name = name
        self.plan = plan

    def raw_resolution(self, tokens, mode):
        return SimpleNamespace(
            plan=self.plan,
            vid=self.plan.vid,
            n_fused=1,
            bucket_tokens=0,
        )


class _FakeResources:
    world = 4

    def __init__(self, *, tag_ids=None, refused=()):
        self.tag_calls = []
        self.check_calls = []
        self._tag_ids = tag_ids
        self._refused = set(refused)

    def plan_tag_bits(self, plan):
        return large_plan_tag_bits(plan)

    def tag_unique_ids(self, keys, tag_bits=None):
        self.tag_calls.append(tag_bits)
        if self._tag_ids is not None:
            return self._tag_ids(set(keys))
        return {k: 0x100000 | (i + 1) for i, k in enumerate(sorted(set(keys)))}

    def check_call_ids(self, keys, **kwargs):
        self.check_calls.append(kwargs)
        return SimpleNamespace(ok=not self._refused, refused_keys=set(self._refused))


def _install(monkeypatch, resources, sites):
    monkeypatch.setattr(hooks, "_RES", resources)
    monkeypatch.setattr(hooks, "_SITES", sites)
    monkeypatch.setattr(hooks, "_TOKEN_IDS", {})


def test_build_token_call_ids_assigns_18_bit_ids_for_route_a(monkeypatch):
    resources = _FakeResources()
    sites = {
        f"layer.{i}": _Site(f"layer.{i}", _plan("r19", 6, LARGE)) for i in range(4)
    }
    _install(monkeypatch, resources, sites)

    result = hooks.token_call_ids(65536, "eager")

    assert resources.tag_calls == [18]
    assert resources.check_calls and resources.check_calls[0]["tag_bits"] == 18
    assert len(result.ids) == 4
    assert not result.refused


def test_build_token_call_ids_refuses_a_low_bit_collision(monkeypatch):
    resources = _FakeResources()
    sites = {
        f"layer.{i}": _Site(f"layer.{i}", _plan("r19", 6, LARGE)) for i in range(4)
    }
    _install(monkeypatch, resources, sites)
    collided = ("layer.0", 65536, LARGE, 1)
    resources._refused = {collided}

    result = hooks.token_call_ids(65536, "eager")

    assert collided in result.refused
    assert resources.check_calls[0]["tag_bits"] == 18


def test_build_token_call_ids_refuses_when_no_ids_can_be_assigned(monkeypatch):
    def fail(_keys):
        raise ValueError("more (site, vid, n_fused) triples than tag ids")

    resources = _FakeResources(tag_ids=fail)
    sites = {
        f"layer.{i}": _Site(f"layer.{i}", _plan("r19", 6, LARGE)) for i in range(4)
    }
    _install(monkeypatch, resources, sites)

    result = hooks.token_call_ids(65536, "eager")

    assert len(result.refused) == 4
    assert not result.ids


def test_mixed_tag_widths_at_one_token_count_run_native(monkeypatch):
    # Route A keeps one tag family per token count; a mixture cannot be checked with one
    # width, so every fused launch of that token count must go native (never a guess).
    resources = _FakeResources()
    sites = {
        "layer.0": _Site("layer.0", _plan("r19", 6, LARGE)),
        "layer.1": _Site("layer.1", _plan("r13", 4, LARGE)),
    }
    _install(monkeypatch, resources, sites)

    result = hooks.token_call_ids(65536, "eager")

    assert resources.tag_calls == []
    assert len(result.refused) == 2
    assert not result.ids
