"""Forward-level state of the SGLang GDN wrapper (`_GDNStep`) and the reuse of
SGLang's mamba pool layer views (`_mamba_layer_cache`).

The per-layer path re-derived the forward mode, the SGLang backend, the track
metadata and the radix-track launch arguments, and rebuilt a
`MambaPool.State` of views per call. These tests pin when the forward-level
step is used, that the views and KV tensors handed out are the ones SGLang
would build, and when caching is off.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from atom.plugin.sglang.attention_backend import attention_gdn
from atom.plugin.sglang.attention_backend.attention_gdn import (
    SGLangGatedDeltaNet,
    SGLangGDNForwardContext,
    _GDNStep,
    _mamba_layer_cache,
)

memory_pool = pytest.importorskip("sglang.srt.mem_cache.memory_pool")
HybridReqToTokenPool = memory_pool.HybridReqToTokenPool
MambaPool = memory_pool.MambaPool

LAYERS = (0, 1, 2, 4, 5, 6)


def _pool(slots=5, dim=12, width=3, heads=2, k=4, v=4):
    conv = torch.randn(len(LAYERS), slots, dim, width)
    temporal = torch.randn(len(LAYERS), slots, heads, k, v)
    mamba_pool = MambaPool.__new__(MambaPool)
    mamba_pool.mamba_cache = MambaPool.State(conv=[conv], temporal=temporal)
    pool = HybridReqToTokenPool.__new__(HybridReqToTokenPool)
    pool.mamba_pool = mamba_pool
    pool.mamba_map = {layer: i for i, layer in enumerate(LAYERS)}
    pool.layer_transfer_counter = None
    pool.start_layer = 0
    return pool


def _same_view(a, b):
    tensors = [(x, y) for x, y in zip(a.conv, b.conv)] + [(a.temporal, b.temporal)]
    return all(
        x.data_ptr() == y.data_ptr()
        and x.shape == y.shape
        and x.stride() == y.stride()
        and x.dtype == y.dtype
        for x, y in tensors
    )


def test_layer_views_are_reused_while_the_pool_state_is_the_same():
    pool = _pool()
    first = _mamba_layer_cache(pool, 4)
    assert _mamba_layer_cache(pool, 4) is first
    # What SGLang's accessor builds right now, field for field.
    assert _same_view(first, pool.mamba2_layer_cache(4))
    assert _mamba_layer_cache(pool, 5) is not first

    # A new state object (a reallocated pool) invalidates every layer.
    state = pool.mamba_pool.mamba_cache
    pool.mamba_pool.mamba_cache = MambaPool.State(
        conv=[c.clone() for c in state.conv], temporal=state.temporal.clone()
    )
    fresh = _mamba_layer_cache(pool, 4)
    assert fresh is not first
    assert _same_view(fresh, pool.mamba2_layer_cache(4))


def test_layer_views_follow_a_hicache_transfer_counter():
    """The per-layer transfer wait lives in `mamba2_layer_cache`; with a
    counter attached every call goes through it."""
    waits = []
    pool = _pool()
    pool.layer_transfer_counter = SimpleNamespace(wait_until=waits.append)
    a = _mamba_layer_cache(pool, 4)
    b = _mamba_layer_cache(pool, 4)
    assert a is not b and _same_view(a, b)
    assert waits == [4, 4]


def test_other_pools_are_asked_every_time():
    calls = []

    class _Pool:
        def __init__(self):
            self.mamba_map = {3: 0}

        def mamba2_layer_cache(self, layer_id):
            calls.append(layer_id)
            return SimpleNamespace(conv=[torch.zeros(1)], temporal=torch.zeros(1))

    pool = _Pool()
    _mamba_layer_cache(pool, 3)
    _mamba_layer_cache(pool, 3)
    assert calls == [3, 3]

    class _Subclass(HybridReqToTokenPool):
        def mamba2_layer_cache(self, layer_id):
            calls.append(("sub", layer_id))
            return super().mamba2_layer_cache(layer_id)

    sub = _pool()
    sub.__class__ = _Subclass
    _mamba_layer_cache(sub, 4)
    _mamba_layer_cache(sub, 4)
    assert calls[-2:] == [("sub", 4), ("sub", 4)]


def test_kv_cache_tensors_are_reused_with_the_same_views():
    pool = _pool()
    linear = SimpleNamespace(req_to_token_pool=pool)
    fb = SimpleNamespace(token_to_kv_pool=None)
    first = SGLangGDNForwardContext._build_kv_cache_tensors(fb, linear)
    second = SGLangGDNForwardContext._build_kv_cache_tensors(fb, linear)
    assert first is not second  # callers may extend their dict
    assert first.keys() == second.keys() == {f"layer_{i}" for i in LAYERS}
    for name, tensor in first.items():
        assert second[name] is tensor
        view = pool.mamba2_layer_cache(tensor.layer_num)
        assert tensor.k_cache.data_ptr() == view.conv[0].data_ptr()
        assert tensor.k_cache.stride() == view.conv[0].stride()
        assert tensor.v_cache.data_ptr() == view.temporal.data_ptr()
        assert tensor.v_cache.stride() == view.temporal.stride()
        assert tensor.per_request_state


class _Mode:
    def __init__(self, kind):
        self.kind = kind

    def is_target_verify(self):
        return self.kind == "verify"

    def is_decode_or_idle(self):
        return self.kind == "decode"

    def is_extend(self):
        return self.kind in ("extend", "verify")


def _metadata(**overrides):
    md = SimpleNamespace(
        has_mamba_track_mask=True,
        track_conv_indices=torch.tensor([[1, 2, 3]]),
        conv_states_mask_indices=torch.tensor([2]),
        track_ssm_h_src=torch.tensor([0]),
        track_ssm_h_dst=torch.tensor([2]),
        track_ssm_final_src=torch.tensor([1]),
        track_ssm_final_dst=torch.tensor([3]),
        track_ssm_recompute_dst=None,
    )
    for key, value in overrides.items():
        setattr(md, key, value)
    return md


def _step(kind="extend", mask=True, **md):
    fb = SimpleNamespace(
        forward_mode=_Mode(kind),
        mamba_track_mask=torch.ones(1, dtype=torch.bool) if mask else None,
    )
    linear = SimpleNamespace(
        forward_metadata=_metadata(**md), req_to_token_pool=_pool()
    )
    return _GDNStep(fb, linear)


def test_step_extend_with_track_rows_precomputes_the_plan():
    step = _step()
    assert step.track == _GDNStep.EXTEND and not step.verify
    plan = step.plan
    assert (plan.n_conv, plan.n_h, plan.n_f, plan.state_len) == (1, 1, 1, 3)
    assert plan.conv_slots is step.metadata.conv_states_mask_indices
    assert torch.equal(plan.conv_idx, step.metadata.track_conv_indices)


@pytest.mark.parametrize(
    "kwargs, track, plan",
    [
        ({"kind": "verify"}, _GDNStep.NONE, False),
        ({"mask": False}, _GDNStep.NONE, False),
        ({"kind": "decode"}, _GDNStep.LAYER, False),
        ({"kind": "idle-other"}, _GDNStep.NONE, False),
        ({"has_mamba_track_mask": False}, _GDNStep.EXTEND, False),
        ({"track_ssm_recompute_dst": torch.tensor([1])}, _GDNStep.LAYER, False),
        ({"track_ssm_h_src": None}, _GDNStep.LAYER, False),
        ({"conv_states_mask_indices": None}, _GDNStep.LAYER, False),
        (
            {"track_conv_indices": torch.zeros(0, 3, dtype=torch.long)},
            _GDNStep.EXTEND,
            True,
        ),
    ],
    ids=[
        "verify",
        "no-track-mask",
        "decode",
        "other-mode",
        "nothing-tracked",
        "recompute-rows",
        "missing-ssm-index",
        "missing-conv-slots",
        "no-conv-rows",
    ],
)
def test_step_keeps_unusual_forwards_on_the_per_layer_path(kwargs, track, plan):
    step = _step(**kwargs)
    assert step.track == track
    assert (step.plan is not None) == plan
    assert step.verify == (kwargs.get("kind") == "verify")


def _layer(monkeypatch):
    calls = []
    monkeypatch.setattr(
        attention_gdn.GatedDeltaNet, "forward", lambda self, *a: calls.append("core")
    )
    impl = SGLangGatedDeltaNet.__new__(SGLangGatedDeltaNet)
    torch.nn.Module.__init__(impl)
    impl.layer_num = 4
    impl.dt_bias = torch.nn.Parameter(torch.zeros(2), requires_grad=False)
    impl._track_mamba_states = lambda fb, x: calls.append(("layer", fb))
    impl._track_extend = lambda step, x: calls.append(("step", step))
    return impl, calls


def _run(impl, fb):
    from atom.plugin.sglang.runtime import context

    token = context._current_forward_batch.set(fb)
    try:
        x = torch.zeros(4, 3)
        impl.forward(x, x[:, :2], x[:, :2], x, "layer")
    finally:
        context._current_forward_batch.reset(token)


def test_layer_uses_the_step_of_its_own_forward(monkeypatch):
    impl, calls = _layer(monkeypatch)
    step = _step()
    token = attention_gdn._GDN_STEP.set(step)
    try:
        _run(impl, step.forward_batch)
        # A step bound for another forward batch is not used.
        other = SimpleNamespace(forward_mode=_Mode("extend"), mamba_track_mask=None)
        _run(impl, other)
    finally:
        attention_gdn._GDN_STEP.reset(token)
    _run(impl, step.forward_batch)  # nothing bound
    assert calls == [
        "core",
        ("step", step),
        "core",
        ("layer", other),
        "core",
        ("layer", step.forward_batch),
    ]


def test_layer_step_kinds(monkeypatch):
    """Decode keeps the per-layer path; a forward with nothing to track does
    nothing after the core."""
    impl, calls = _layer(monkeypatch)
    decode, untracked = _step(kind="decode"), _step(mask=False)
    for step in (decode, untracked):
        token = attention_gdn._GDN_STEP.set(step)
        try:
            _run(impl, step.forward_batch)
        finally:
            attention_gdn._GDN_STEP.reset(token)
    assert calls == ["core", ("layer", decode.forward_batch), "core"]


def test_bind_publishes_the_step_only_inside_the_forward(monkeypatch):
    step = object()
    current = SimpleNamespace(context=object(), attn_metadata=None, kv_cache_data={})
    forward_context = SimpleNamespace(gdn_metadata=None, kv_cache_data={}, step=step)
    monkeypatch.setattr(
        SGLangGDNForwardContext, "build", classmethod(lambda cls, _m: forward_context)
    )
    monkeypatch.setattr(attention_gdn, "get_forward_context", lambda: current)
    monkeypatch.setattr(attention_gdn, "set_kv_cache_data", lambda kv: None)
    assert attention_gdn._GDN_STEP.get() is None
    with SGLangGDNForwardContext.bind(object()):
        assert attention_gdn._GDN_STEP.get() is step
    assert attention_gdn._GDN_STEP.get() is None
    with pytest.raises(RuntimeError), SGLangGDNForwardContext.bind(object()):
        raise RuntimeError("layer failed")
    assert attention_gdn._GDN_STEP.get() is None
