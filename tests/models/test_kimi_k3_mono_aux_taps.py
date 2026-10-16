# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""K3 mono owes the DSpark drafter its aux hidden states.

The drafter taps target layers with forward hooks on ``layers[lid]``. A mono
step calls that module on NEITHER path -- the kernels replace it, and the
fallback reruns its forward from the submodules -- so no hook fires and the
drafter would draft from whatever the last production step left behind. K3 mono
requires DSpark, so that is every armed deployment.
"""

from types import SimpleNamespace

import pytest
import torch

from atom.mono.runtime.aux_taps import AuxTaps

pytest.importorskip("flydsl", reason="importing the K3 mono runner needs FlyDSL")
# the runner imports `aiter.dist.parallel_state` at module scope, so FlyDSL
# alone is not enough to collect this file -- the same guard
# `test_minimax_m3_mono_dispatch.py` puts before its own runner import.
pytest.importorskip("aiter", reason="the K3 mono runner imports aiter.dist")

from atom.models.kimi_k3.mono.runner import K3MonoDecodeRunner

HIDDEN, ROWS = 8, 4


class _Layer:
    """Only what the tap needs: K3's own reconstruction of the post-layer
    hidden state, which is `KimiDecoderLayer.aux_hidden_state` verbatim."""

    @staticmethod
    def aux_hidden_state(output):
        prefix_sum, *pendings, _block_residual = output
        if prefix_sum is None:
            return None
        for pending in pendings:
            if pending is not None:
                prefix_sum = prefix_sum + pending
        return prefix_sum


def _runner(tapped, *, mono):
    """A runner stubbed down to `_layer` and the tap it owes."""
    runner = object.__new__(K3MonoDecodeRunner)
    runner.aux = AuxTaps(
        tuple(tapped), tuple(torch.zeros(ROWS, HIDDEN) for _ in tapped)
    )
    runner.inner = SimpleNamespace(layers={i: _Layer() for i in range(4)})
    out = (torch.full((ROWS, HIDDEN), 2.0), torch.full((ROWS, HIDDEN), 3.0), None)
    if mono:
        runner._k1 = lambda *a, **k: None
        runner._k2 = lambda build, plan, core, prefix, blocks: out[1]
    else:
        runner._fallback = lambda *a, **k: out
    return runner, out


def _drive(runner, *, layer_id, mono, prefix):
    plan = SimpleNamespace(mono=mono, layer_id=layer_id)
    return runner._layer(None, plan, None, prefix, None, None, None, None)


@pytest.mark.parametrize("mono", [True, False])
def test_a_tapped_layer_records_its_settled_hidden_state(mono):
    """`prefix + add`, the same tensor the drafter's hook would have recorded."""
    runner, out = _runner([1], mono=mono)
    _drive(runner, layer_id=1, mono=mono, prefix=out[0])
    assert torch.equal(runner.aux.buffers[0], torch.full((ROWS, HIDDEN), 5.0))


@pytest.mark.parametrize("mono", [True, False])
def test_an_untapped_layer_leaves_every_buffer_alone(mono):
    runner, out = _runner([1], mono=mono)
    _drive(runner, layer_id=2, mono=mono, prefix=out[0])
    assert torch.equal(runner.aux.buffers[0], torch.zeros(ROWS, HIDDEN))


def test_the_right_buffer_is_written():
    """Buffers are positional: tap order, not layer id."""
    runner, out = _runner([0, 3], mono=True)
    _drive(runner, layer_id=3, mono=True, prefix=out[0])
    assert torch.equal(runner.aux.buffers[0], torch.zeros(ROWS, HIDDEN))
    assert torch.equal(runner.aux.buffers[1], torch.full((ROWS, HIDDEN), 5.0))


def test_a_block_write_layer_records_nothing():
    """`_fallback` returns `prefix=None` where the layer banked its prefix into
    a block slot. The drafter's hook treats a None reconstruction as nothing to
    record, so mono must leave the buffer as it was rather than write zeros."""
    runner, _ = _runner([1], mono=False)
    runner.aux.buffers[0].fill_(7.0)
    runner._fallback = lambda *a, **k: (None, torch.ones(ROWS, HIDDEN), None)
    _drive(runner, layer_id=1, mono=False, prefix=torch.zeros(ROWS, HIDDEN))
    assert torch.equal(runner.aux.buffers[0], torch.full((ROWS, HIDDEN), 7.0))
