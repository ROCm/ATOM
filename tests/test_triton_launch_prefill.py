# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The Qwen3.8-Flash-Next prefill wrappers that launch through
`CachedLauncher`: each launch's key against Triton's own (from the arguments
the wrapper really passes, recorded on CPU), and each wrapper on the GPU
against the same launches through plain `kernel[grid]`."""

from types import SimpleNamespace

import pytest
import torch

triton = pytest.importorskip("triton")

from atom.utils.triton_launch import CachedLauncher
from tests.test_triton_launch import (
    _assert_key_is_at_least_tritons,
    _bitwise_equal,
)


class _Recorder:
    """Stands in for a module's launcher and keeps the arguments it gets."""

    def __init__(self, launcher):
        self.launcher = launcher
        self.calls = []

    def __call__(self, grid, *args, **kwargs):
        assert type(grid) is tuple
        self.calls.append(list(args) + list(kwargs.values()))


class _Plain:
    """The launch as it was: `kernel[grid](*args, **options)` every time."""

    def __init__(self, launcher):
        self.kernel, self.options = launcher.kernel, launcher.options

    def __call__(self, grid, *args, **kwargs):
        self.kernel[grid](*args, **kwargs, **self.options)


def _record(monkeypatch, module, name, run):
    """The launcher `module.<name>` and the argument lists `run()` passes it."""
    recorder = _Recorder(getattr(module, name))
    monkeypatch.setattr(module, name, recorder)
    run()
    monkeypatch.setattr(module, name, recorder.launcher)
    assert recorder.calls
    return recorder.launcher, recorder.calls


def _check_keys(launcher, calls):
    assert isinstance(launcher, CachedLauncher) and launcher._enabled
    for args in calls:
        assert len(args) == len(launcher._names)
        _assert_key_is_at_least_tritons(launcher, args)


def _plain_vs_cached(monkeypatch, launchers, run):
    """`run()` with `launchers` (module, name) on plain `kernel[grid]`, then
    twice through the cached launchers, the second time from the cache."""
    cached = [getattr(module, name) for module, name in launchers]
    for module, name in launchers:
        monkeypatch.setattr(module, name, _Plain(getattr(module, name)))
    ref = [t.clone() for t in run()]
    for (module, name), launcher in zip(launchers, cached):
        monkeypatch.setattr(module, name, launcher)
    first = [t.clone() for t in run()]
    second = run()
    torch.cuda.synchronize()
    assert _bitwise_equal(first, ref)
    assert _bitwise_equal(second, ref)
    assert all(launcher._kernels for launcher in cached)


def _on_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)


gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")


def _randn(*shape, dtype=torch.bfloat16, offset=0, scale=1.0, seed=0, device="cuda"):
    gen = torch.Generator(device=device).manual_seed(seed)
    numel = 1
    for size in shape:
        numel *= size
    flat = torch.randn(numel + offset, generator=gen, device=device)
    return (flat[offset:].view(shape) * scale).to(dtype)


# --- GDN: q/k L2 norm, causal conv1d, gating -------------------------------

# Qwen3.8-Flash-Next linear attention at TP2: 8 key heads, 24 value heads.
_K_HEADS, _V_HEADS, _HEAD_DIM, _CONV_WIDTH = 8, 24, 128, 4
_K_DIM, _V_DIM = _K_HEADS * _HEAD_DIM, _V_HEADS * _HEAD_DIM
_CONV_DIM = 2 * _K_DIM + _V_DIM


@pytest.fixture
def gdn():
    pytest.importorskip("aiter.dist.parallel_state", exc_type=ImportError)
    from atom.model_ops import attention_gdn
    from atom.model_ops.fla_ops import l2norm
    from atom.model_ops.mamba_ops import causal_conv1d

    return SimpleNamespace(l2norm=l2norm, conv=causal_conv1d, gating=attention_gdn)


def _conv_inputs(lens, offset=0, native_layout=False, initial=True, device="cuda"):
    tokens = sum(lens)
    mixed = _randn(tokens, _CONV_DIM, offset=offset, seed=tokens, device=device)
    weight = _randn(_CONV_DIM, _CONV_WIDTH, scale=0.3, seed=1, device=device)
    slots = 8
    if native_layout:  # ModelRunner's [slot, state_len, conv_dim], transposed
        states = _randn(slots, _CONV_WIDTH - 1, _CONV_DIM, device=device)
        states = states.transpose(-1, -2)
    else:  # SGLang's [slot, conv_dim, state_len]
        states = _randn(slots, _CONV_DIM, _CONV_WIDTH - 1, device=device)
    starts = [0]
    for n in lens:
        starts.append(starts[-1] + n)
    return SimpleNamespace(
        x=mixed.transpose(0, 1),
        weight=weight,
        states=states,
        query_start_loc=torch.tensor(starts, dtype=torch.int32, device=device),
        cache_indices=torch.arange(1, len(lens) + 1, dtype=torch.int32, device=device),
        has_initial_state=torch.full(
            (len(lens),), initial, dtype=torch.bool, device=device
        ),
        tokens=tokens,
    )


def _conv1d(gdn, t, states, metadata):
    return gdn.conv.causal_conv1d_fn(
        t.x,
        t.weight,
        None,
        states,
        t.query_start_loc,
        _K_DIM,
        _V_DIM,
        cache_indices=t.cache_indices,
        has_initial_state=t.has_initial_state,
        activation="silu",
        metadata=metadata,
    )


def _conv_metadata(gdn, t):
    nums, batch, offsets = gdn.conv.compute_causal_conv1d_metadata(
        t.query_start_loc, t.tokens
    )
    return SimpleNamespace(
        nums_dict=nums, batch_ptr=batch, token_chunk_offset_ptr=offsets
    )


def _gating_inputs(tokens, offset=0, device="cuda"):
    ba = _randn(tokens, 2 * _V_HEADS, offset=offset, seed=tokens, device=device)
    params = [
        torch.nn.Parameter(_randn(_V_HEADS, seed=s, device=device), False)
        for s in (2, 3)
    ]
    # `in_proj_ba` hands over strided views of one projection.
    return params[0], ba[:, _V_HEADS:], ba[:, :_V_HEADS], params[1]


@pytest.mark.parametrize("native_layout", [False, True], ids=["sglang", "native"])
@pytest.mark.parametrize("buffer_ops", ["1", "0"])
def test_gdn_keys_tell_apart_what_triton_tells_apart(
    monkeypatch, gdn, buffer_ops, native_layout
):
    _on_cpu(monkeypatch)
    monkeypatch.setenv("AMDGCN_USE_BUFFER_OPS", buffer_ops)
    x = _randn(1, 300, _K_HEADS, _HEAD_DIM, offset=1, device="cpu")
    _check_keys(
        *_record(
            monkeypatch, gdn.l2norm, "_l2norm_fwd2", lambda: gdn.l2norm.l2norm_fwd(x)
        )
    )
    t = _conv_inputs((7, 290, 3), offset=1, native_layout=native_layout, device="cpu")
    metadata = _record(
        monkeypatch,
        gdn.conv,
        "_causal_conv1d_metadata",
        lambda: gdn.conv.compute_causal_conv1d_metadata(t.query_start_loc, t.tokens),
    )
    _check_keys(*metadata)
    # The program map lives on the GPU; on CPU any map of the right size does.
    cpu_map = torch.zeros(40, dtype=torch.int32)
    meta = SimpleNamespace(
        nums_dict={8: {"tot": 40, "num_tokens": t.tokens}},
        batch_ptr=cpu_map,
        token_chunk_offset_ptr=cpu_map,
    )
    _check_keys(
        *_record(
            monkeypatch,
            gdn.conv,
            "_causal_conv1d_fwd",
            lambda: _conv1d(gdn, t, t.states, meta),
        )
    )
    gating = _gating_inputs(300, offset=1, device="cpu")
    _check_keys(
        *_record(
            monkeypatch,
            gdn.gating,
            "_fused_gdn_gating",
            lambda: gdn.gating.fused_gdn_gating(*gating),
        )
    )


@gpu
@pytest.mark.parametrize("offset", [0, 1], ids=["aligned", "misaligned"])
@pytest.mark.parametrize("tokens", [2112, 2096, 16, 2111, 3, 1])
@torch.inference_mode()
def test_l2norm_and_gating_match_plain_triton_bitwise(monkeypatch, gdn, tokens, offset):
    x = _randn(1, tokens, _K_HEADS, _HEAD_DIM, offset=offset, seed=tokens)
    _plain_vs_cached(
        monkeypatch,
        [(gdn.l2norm, "_l2norm_fwd2")],
        lambda: [gdn.l2norm.l2norm_fwd(x)],
    )
    gating = _gating_inputs(tokens, offset)
    _plain_vs_cached(
        monkeypatch,
        [(gdn.gating, "_fused_gdn_gating")],
        lambda: list(gdn.gating.fused_gdn_gating(*gating)),
    )


@gpu
@pytest.mark.parametrize("initial", [True, False], ids=["resume", "fresh"])
@pytest.mark.parametrize("native_layout", [False, True], ids=["sglang", "native"])
@pytest.mark.parametrize("offset", [0, 1], ids=["aligned", "misaligned"])
@pytest.mark.parametrize("lens", [(2112,), (16,), (2111,), (7, 1000, 1, 64)])
@torch.inference_mode()
def test_causal_conv1d_matches_plain_triton_bitwise(
    monkeypatch, gdn, lens, offset, native_layout, initial
):
    """The conv state is updated in place, so each run starts from a copy."""
    t = _conv_inputs(lens, offset, native_layout, initial)
    metadata = _conv_metadata(gdn, t)

    def run():
        if native_layout:
            storage = t.states.transpose(-1, -2).clone()
            states = storage.transpose(-1, -2)
        else:
            storage = states = t.states.clone()
        return [*_conv1d(gdn, t, states, metadata), storage]

    _plain_vs_cached(monkeypatch, [(gdn.conv, "_causal_conv1d_fwd")], run)
    _plain_vs_cached(
        monkeypatch,
        [(gdn.conv, "_causal_conv1d_metadata")],
        lambda: list(_conv_metadata(gdn, t).__dict__.values())[1:],
    )


@gpu
@torch.inference_mode()
def test_gating_and_l2norm_replay_in_a_cuda_graph(gdn):
    """`fused_gdn_gating` also runs in the decode graph."""
    gating = _gating_inputs(16)
    x = _randn(1, 16, _K_HEADS, _HEAD_DIM)
    ref = (*gdn.gating.fused_gdn_gating(*gating), gdn.l2norm.l2norm_fwd(x))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = (*gdn.gating.fused_gdn_gating(*gating), gdn.l2norm.l2norm_fwd(x))
    graph.replay()
    torch.cuda.synchronize()
    assert _bitwise_equal(out, ref)
