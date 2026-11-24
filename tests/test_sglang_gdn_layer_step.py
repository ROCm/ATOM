# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Forward-level state of the SGLang GDN wrapper (`_GDNStep`) and the mamba
pool layer views it reads (`_layer_states`).

Every GDN layer of a forward tracks SGLang's radix cache through the step
bound for that forward, the track writes land in each layer's own slice of
the pool, and the views of those slices are taken from the pool once.
"""

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from aiter_stub import atom_without_aiter

LAYERS = (0, 1, 2, 4, 5, 6)


@pytest.fixture(scope="module")
def gdn():
    with atom_without_aiter() as load:
        yield SimpleNamespace(
            attention_gdn=load("atom.plugin.sglang.attention_backend.attention_gdn"),
            chunk=load("atom.model_ops.fla_ops.chunk"),
        )


class _Pool:
    """The parts of SGLang's `HybridReqToTokenPool` the wrapper reads."""

    def __init__(self, slots=5, dim=12, width=3, heads=2, k=4, v=4):
        self.mamba_map = {layer: i for i, layer in enumerate(LAYERS)}
        self.mamba_pool = SimpleNamespace(
            mamba_cache=SimpleNamespace(
                conv=[torch.randn(len(LAYERS), slots, dim, width)],
                temporal=torch.randn(len(LAYERS), slots, heads, k, v),
            )
        )
        self.layer_transfer_counter = None
        self.start_layer = 0

    @staticmethod
    def get_mamba_indices(req_pool_indices):
        return req_pool_indices

    def mamba2_layer_index(self, layer_id):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        return self.mamba_map[layer_id]


def _is_slice(view, tensor, index):
    """`view` is `tensor[index]`: same storage offset, shape, strides, dtype."""
    offset = index * tensor.stride(0) * tensor.element_size()
    return (
        view.data_ptr() == tensor.data_ptr() + offset
        and view.shape == tensor.shape[1:]
        and view.stride() == tensor.stride()[1:]
        and view.dtype == tensor.dtype
    )


def test_layer_states_are_each_layers_slice_of_the_pool(gdn):
    pool = _Pool()
    state = pool.mamba_pool.mamba_cache
    for layer, index in pool.mamba_map.items():
        conv, temporal = gdn.attention_gdn._layer_states(pool, layer)
        assert _is_slice(conv, state.conv[0], index)
        assert _is_slice(temporal, state.temporal, index)


def test_layer_views_are_taken_once_per_pool(gdn):
    ag = gdn.attention_gdn
    pool = _Pool()
    conv, temporal = ag._layer_states(pool, 4)
    for _ in range(3):
        again = ag._layer_states(pool, 4)
        assert again[0] is conv and again[1] is temporal

    other = _Pool()
    conv, temporal = ag._layer_states(other, 4)
    state = other.mamba_pool.mamba_cache
    assert _is_slice(conv, state.conv[0], 3)
    assert _is_slice(temporal, state.temporal, 3)


def test_layer_views_follow_a_rebound_mamba_cache(gdn):
    ag = gdn.attention_gdn
    pool = _Pool()
    ag._layer_states(pool, 4)
    pool.mamba_pool.mamba_cache = _Pool().mamba_pool.mamba_cache
    conv, temporal = ag._layer_states(pool, 4)
    state = pool.mamba_pool.mamba_cache
    assert _is_slice(conv, state.conv[0], 3)
    assert _is_slice(temporal, state.temporal, 3)


def test_layer_views_follow_a_new_pool_on_the_same_mamba_cache(gdn):
    ag = gdn.attention_gdn
    old = _Pool()
    old.mamba_map = {layer: i for i, layer in enumerate(LAYERS[:3])}
    ag._layer_states(old, LAYERS[0])
    new = _Pool()
    new.mamba_pool = old.mamba_pool
    new.mamba_map = {layer: i for i, layer in enumerate(LAYERS[:4])}
    conv, temporal = ag._layer_states(new, LAYERS[3])
    state = new.mamba_pool.mamba_cache
    assert _is_slice(conv, state.conv[0], 3)
    assert _is_slice(temporal, state.temporal, 3)


def test_every_lookup_waits_for_the_layer_transfer(gdn):
    """The HiCache per-layer wait lives in `mamba2_layer_index`; taking the
    views once must not skip it."""
    waits = []
    pool = _Pool()
    pool.start_layer = 1
    pool.layer_transfer_counter = SimpleNamespace(wait_until=waits.append)
    for layer in (4, 4, 6):
        gdn.attention_gdn._layer_states(pool, layer)
    assert waits == [3, 3, 5]


def _same_view(a, b):
    return (
        a.data_ptr() == b.data_ptr()
        and a.shape == b.shape
        and a.stride() == b.stride()
        and a.dtype == b.dtype
    )


def test_layer_states_match_sglangs_own_accessor(gdn):
    memory_pool = pytest.importorskip("sglang.srt.mem_cache.memory_pool")
    state = _Pool().mamba_pool.mamba_cache
    mamba_pool = memory_pool.MambaPool.__new__(memory_pool.MambaPool)
    mamba_pool.mamba_cache = memory_pool.MambaPool.State(
        conv=state.conv, temporal=state.temporal
    )
    pool = memory_pool.HybridReqToTokenPool.__new__(memory_pool.HybridReqToTokenPool)
    pool.mamba_pool = mamba_pool
    pool.mamba_map = {layer: i for i, layer in enumerate(LAYERS)}
    pool.start_layer = 0
    waits = []
    pool.layer_transfer_counter = SimpleNamespace(wait_until=waits.append)
    for layer in LAYERS:
        conv, temporal = gdn.attention_gdn._layer_states(pool, layer)
        expected = pool.mamba2_layer_cache(layer)
        assert _same_view(conv, expected.conv[0])
        assert _same_view(temporal, expected.temporal)
    assert waits == [layer for layer in LAYERS for _ in range(2)]


@pytest.mark.parametrize("decode_view", [False, True], ids=["kv", "vk"])
def test_kv_cache_tensors_hand_out_the_pool_views(gdn, monkeypatch, decode_view):
    ag = gdn.attention_gdn
    impl = SimpleNamespace(
        allow_aiter_flydsl=True,
        gdn_flydsl_policy=SimpleNamespace(decode=decode_view),
    )
    config = SimpleNamespace(
        compilation_config=SimpleNamespace(
            static_forward_context={"gdn": SimpleNamespace(impl=impl)}
        )
    )
    monkeypatch.setattr(ag, "get_current_atom_config", lambda: config)
    pool = _Pool()
    state = pool.mamba_pool.mamba_cache
    fb = SimpleNamespace(token_to_kv_pool=None, req_to_token_pool=pool)
    linear = SimpleNamespace(req_to_token_pool=pool)
    first = ag.SGLangGDNForwardContext._build_kv_cache_tensors(fb, linear)
    second = ag.SGLangGDNForwardContext._build_kv_cache_tensors(fb, linear)
    assert first.keys() == second.keys() == {f"layer_{i}" for i in LAYERS}
    for layer, index in pool.mamba_map.items():
        tensor = first[f"layer_{layer}"]
        assert tensor.layer_num == layer and tensor.per_request_state
        assert _is_slice(tensor.k_cache, state.conv[0], index)
        ssm = tensor.v_cache.transpose(-1, -2) if decode_view else tensor.v_cache
        assert _is_slice(ssm, state.temporal, index)
        assert second[f"layer_{layer}"].k_cache is tensor.k_cache


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
        mamba_track_indices=torch.tensor([3, 4]),
        track_ssm_h_src=torch.tensor([0]),
    )
    for key, value in overrides.items():
        setattr(md, key, value)
    return md


class _Linear:
    """SGLang's linear backend: forward metadata, pool, extend track helper."""

    def __init__(self, metadata, pool=None):
        self.forward_metadata = metadata
        self.req_to_token_pool = _Pool() if pool is None else pool
        self.extend_calls = []

    def _track_mamba_state_extend(self, fb, h, ssm_states, metadata):
        self.extend_calls.append((fb, h, ssm_states, metadata))


def _batch(linear, kind="extend", mask=(True,)):
    return SimpleNamespace(
        forward_mode=_Mode(kind),
        mamba_track_mask=None if mask is None else torch.tensor(mask),
        attn_backend=linear,
    )


def _layer(gdn, monkeypatch, layer_num=4, decode_view=False):
    ag = gdn.attention_gdn
    monkeypatch.setattr(ag.GatedDeltaNet, "forward", lambda self, *a: None)
    impl = ag.SGLangGatedDeltaNet.__new__(ag.SGLangGatedDeltaNet)
    torch.nn.Module.__init__(impl)
    impl.layer_num = layer_num
    impl.dt_bias = torch.nn.Parameter(torch.zeros(2), requires_grad=False)
    impl.gdn_flydsl_policy = SimpleNamespace(decode=decode_view)
    return impl


@pytest.mark.parametrize(
    "kind, mask, metadata, conv_written, ssm_tracked, h_taken",
    [
        ("extend", (True,), _metadata(), True, True, True),
        ("extend", None, _metadata(), False, False, False),
        ("other", (True,), _metadata(), False, False, False),
        ("extend", (True,), None, False, False, False),
        (
            "extend",
            (True,),
            _metadata(has_mamba_track_mask=False),
            False,
            False,
            True,
        ),
        (
            "extend",
            (True,),
            _metadata(track_conv_indices=torch.zeros(0, 3, dtype=torch.long)),
            False,
            True,
            True,
        ),
    ],
    ids=[
        "tracked",
        "no-track-mask",
        "other-mode",
        "no-metadata",
        "nothing-tracked",
        "no-conv-rows",
    ],
)
def test_what_a_layer_tracks(
    gdn, monkeypatch, kind, mask, metadata, conv_written, ssm_tracked, h_taken
):
    pool = _Pool()
    state = pool.mamba_pool.mamba_cache
    conv_before, ssm_before = state.conv[0].clone(), state.temporal.clone()
    linear = _Linear(metadata, pool)
    h = torch.randn(1, 3, 2, 4, 4)
    gdn.chunk.ChunkGatedDeltaRuleFunction.last_h = h
    x = torch.randn(6, 12)
    step = gdn.attention_gdn._GDNStep(_batch(linear, kind, mask))
    _layer(gdn, monkeypatch)._track_mamba_states(step, x)

    expected_conv = conv_before.clone()
    if conv_written:
        expected_conv[pool.mamba_map[4], 2] = x[[1, 2, 3]].T
    assert torch.equal(state.conv[0], expected_conv)
    assert torch.equal(state.temporal, ssm_before)
    assert len(linear.extend_calls) == int(ssm_tracked)
    if ssm_tracked:
        assert linear.extend_calls[0][1] is h
    taken = gdn.chunk.ChunkGatedDeltaRuleFunction.last_h is None
    gdn.chunk.ChunkGatedDeltaRuleFunction.last_h = None
    assert taken == h_taken


def test_conv_window_goes_to_the_slot_the_backend_did_not_cache(gdn, monkeypatch):
    pool = _Pool()
    conv = pool.mamba_pool.mamba_cache.conv[0]
    expected = conv.clone()
    linear = _Linear(_metadata(conv_states_mask_indices=None), pool)
    step = gdn.attention_gdn._GDNStep(_batch(linear, mask=(False, True)))
    gdn.chunk.ChunkGatedDeltaRuleFunction.last_h = torch.randn(1, 3, 2, 4, 4)
    x = torch.randn(6, 12)
    _layer(gdn, monkeypatch)._track_mamba_states(step, x)
    # Row 1 of the batch is the tracked one: its slot is mamba_track_indices[1].
    expected[pool.mamba_map[4], 4] = x[[1, 2, 3]].T
    assert torch.equal(conv, expected)


@pytest.fixture
def run(monkeypatch):
    """Run a layer's forward for a forward batch. The plugin runtime, which
    imports SGLang, is replaced by its current-batch lookup."""
    current = []
    runtime = ModuleType("atom.plugin.sglang.runtime")
    runtime.get_current_forward_batch = lambda: current[-1]
    monkeypatch.setitem(sys.modules, runtime.__name__, runtime)

    def forward(impl, fb):
        x = torch.zeros(4, 3)
        current.append(fb)
        impl.forward(x, x[:, :2], x[:, :2], x, "layer")

    return forward


def test_layer_uses_the_step_of_its_own_forward(gdn, monkeypatch, run):
    ag = gdn.attention_gdn
    impl = _layer(gdn, monkeypatch)
    seen = []
    impl._track_mamba_states = lambda step, x: seen.append(step)
    linear = _Linear(_metadata())
    step = ag._GDNStep(_batch(linear))
    other = _batch(linear, mask=None)
    token = ag._GDN_STEP.set(step)
    try:
        run(impl, step.forward_batch)
        run(impl, step.forward_batch)
        run(impl, other)  # a step bound for another batch is not used
    finally:
        ag._GDN_STEP.reset(token)
    run(impl, step.forward_batch)  # nothing bound
    assert seen[:2] == [step, step]
    assert seen[2] is not step and seen[2].forward_batch is other
    assert seen[3] is not step and seen[3].forward_batch is step.forward_batch
    assert seen[3].conv_slots is step.conv_slots


def test_verify_forward_takes_the_verify_path(gdn, monkeypatch, run):
    impl = _layer(gdn, monkeypatch)
    with pytest.raises(RuntimeError, match="TARGET_VERIFY requires an active"):
        run(impl, _batch(None, "verify"))


def test_bind_publishes_the_step_only_inside_the_forward(gdn, monkeypatch):
    ag = gdn.attention_gdn
    step = object()
    current = SimpleNamespace(context=object(), attn_metadata=None, kv_cache_data={})
    forward_context = SimpleNamespace(gdn_metadata=None, kv_cache_data={}, step=step)
    monkeypatch.setattr(
        ag.SGLangGDNForwardContext,
        "build",
        classmethod(lambda cls, _m: forward_context),
    )
    monkeypatch.setattr(ag, "get_forward_context", lambda: current)
    monkeypatch.setattr(ag, "set_kv_cache_data", lambda kv: None)
    assert ag._GDN_STEP.get() is None
    with ag.SGLangGDNForwardContext.bind(object()):
        assert ag._GDN_STEP.get() is step
    assert ag._GDN_STEP.get() is None
    with pytest.raises(RuntimeError), ag.SGLangGDNForwardContext.bind(object()):
        raise RuntimeError("layer failed")
    assert ag._GDN_STEP.get() is None


@pytest.mark.parametrize("decode_view", [False, True], ids=["kv", "vk"])
def test_every_layer_tracks_into_its_own_pool_views(gdn, monkeypatch, decode_view):
    """Through one step, each layer writes its conv window into its own conv
    slice and hands its own SSM slice and `h` to SGLang's extend helper."""
    pool = _Pool()
    linear = _Linear(_metadata(), pool)
    step = gdn.attention_gdn._GDNStep(_batch(linear))
    x = torch.randn(6, 12)
    hs = []
    for layer in LAYERS:
        h = torch.randn(1, 3, 2, 4, 4)
        hs.append(h)
        gdn.chunk.ChunkGatedDeltaRuleFunction.last_h = h
        _layer(gdn, monkeypatch, layer, decode_view)._track_mamba_states(step, x)
        assert gdn.chunk.ChunkGatedDeltaRuleFunction.last_h is None
    state = pool.mamba_pool.mamba_cache
    for (layer, i), h, call in zip(pool.mamba_map.items(), hs, linear.extend_calls):
        assert torch.equal(state.conv[0][i, 2], x[[1, 2, 3]].T)
        fb, got_h, ssm, metadata = call
        assert fb is step.forward_batch and got_h is h and metadata is step.metadata
        ssm = ssm.transpose(-1, -2) if decode_view else ssm
        assert _is_slice(ssm, state.temporal, i)
    assert len(linear.extend_calls) == len(LAYERS)


def test_track_without_kept_states_fails_closed(gdn, monkeypatch):
    step = gdn.attention_gdn._GDNStep(_batch(_Linear(_metadata())))
    gdn.chunk.ChunkGatedDeltaRuleFunction.last_h = None
    with pytest.raises(RuntimeError, match="intermediate states"):
        _layer(gdn, monkeypatch)._track_mamba_states(step, torch.zeros(6, 12))


def test_decode_tracks_through_the_radix_slot_helper(gdn, monkeypatch):
    ag = gdn.attention_gdn
    seen = []
    monkeypatch.setattr(
        ag, "_track_mamba_radix_slots", lambda fb, layer: seen.append((fb, layer))
    )
    step = ag._GDNStep(_batch(_Linear(_metadata()), "decode"))
    _layer(gdn, monkeypatch, 5)._track_mamba_states(step, torch.zeros(2, 12))
    assert seen == [(step.forward_batch, 5)]
