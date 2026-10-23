# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""K3 mono checks the KDA state pools of EVERY mono layer, at bind.

`_conv_state` / `_recurrent_state` are the only validation those pools get.
Everything they refuse -- the channel axis, the conv dtype, the recurrent
dtype, the conv window -- is a build-time fact, the same on every step. The
bind used to check only the first mono layer, which left the rest to be caught
by `_k1`, i.e. part-way through a forward pass. Neither way of recovering from
there works:

  - replaying the step on the production path double-advances state. `_k1`
    steps the KDA conv window and the recurrence IN PLACE, in the KV cache
    itself, so by the time layer N refuses, layers 0..N-1 have already
    stepped -- a crash traded for silent numerical corruption;
  - falling back layer by layer deadlocks. These criteria are read per-rank
    off that rank's own KV cache, and the kernels wait in-kernel on every peer
    (`mono/dispatch.py`), so one rank dropping to production at layer N leaves
    its peers waiting for ever -- a crash traded for a hang.

So an unsupported layer refuses the WHOLE runner, at bind, inside
`bind_agreed`, which makes the verdict TP-uniform for free; `LazyRunner`
catches it and the width runs production end to end, with no mono kernel
launched at all.
"""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("flydsl", reason="importing the K3 mono runner needs FlyDSL")
pytest.importorskip("aiter", reason="the K3 mono runner imports aiter.dist")

import atom.models.kimi_k3.mono.runner as runner_mod
from atom.models.kimi_k3.mono.runner import K3MonoDecodeRunner
from atom.mono.runtime.consensus import MonoUnsupported

QKV, WINDOW = 8, 5


def _pools(window=WINDOW, *, channels=QKV, conv=torch.bfloat16, rec=torch.float32):
    return SimpleNamespace(
        k_cache=torch.zeros(2, channels, window, dtype=conv),
        v_cache=torch.zeros(2, channels, 4, dtype=rec),
    )


def _runner(monkeypatch, caches, *, mono_layers=(0, 1, 2)):
    """A runner stubbed down to what `_state_len` reads."""
    runner = object.__new__(K3MonoDecodeRunner)
    runner.dims = SimpleNamespace(qkv=QKV)
    runner.plans = [
        SimpleNamespace(layer_id=lid, mono=lid in mono_layers) for lid in range(4)
    ]
    monkeypatch.setattr(
        runner_mod,
        "get_forward_context",
        lambda: SimpleNamespace(
            kv_cache_data={f"layer_{lid}": c for lid, c in caches.items()}
        ),
    )
    return runner


def test_matching_pools_give_the_build_constant(monkeypatch):
    runner = _runner(monkeypatch, {lid: _pools() for lid in range(4)})
    assert runner._state_len() == WINDOW


@pytest.mark.parametrize(
    ("broken", "match"),
    [
        ({"channels": QKV + 1}, "no axis of"),
        ({"conv": torch.float16}, "conv state"),
        ({"rec": torch.bfloat16}, "recurrent state"),
    ],
)
def test_a_later_mono_layers_bad_pool_is_refused_at_bind(monkeypatch, broken, match):
    # layer 0 is fine, so a bind that only checks the first mono layer passes
    # this deployment through and lets `_k1` raise at layer 2 instead --
    # mid-forward, with layers 0 and 1 already advanced.
    caches = {lid: _pools() for lid in range(4)}
    caches[2] = _pools(**broken)
    runner = _runner(monkeypatch, caches)
    with pytest.raises(MonoUnsupported, match=match):
        runner._state_len()


def test_mono_layers_that_disagree_on_the_window_are_refused(monkeypatch):
    # `state_len` is ONE build constant baked into every layer's K1 key, so
    # pools of two different widths cannot both be served -- picking the first
    # layer's width would mis-specialize the others.
    caches = {lid: _pools() for lid in range(4)}
    caches[2] = _pools(window=WINDOW + 1)
    runner = _runner(monkeypatch, caches)
    with pytest.raises(MonoUnsupported, match="disagree on the conv window"):
        runner._state_len()


def test_a_non_mono_layers_pool_is_not_checked(monkeypatch):
    # layer 3 runs the production path, which owns its own state entirely.
    caches = {lid: _pools() for lid in range(4)}
    caches[3] = _pools(WINDOW + 7, conv=torch.float16)
    runner = _runner(monkeypatch, caches)
    assert runner._state_len() == WINDOW
