# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""K1's ``delta`` after a row-sharded fallback layer.

A layer that takes the row-sharded MoE tail hands its addends back as folded
zeros, left unwritten. K1 loads ``delta`` and the runner sums two addends in
torch, so both must read them as zero."""

from types import SimpleNamespace

import pytest
import torch

from atom.mono.runtime.aux_taps import AuxTaps

pytest.importorskip("flydsl", reason="importing the K3 mono runner needs FlyDSL")
pytest.importorskip("aiter", reason="the K3 mono runner imports aiter.dist")

from atom.model_ops.kimi_k3 import attention_residual
from atom.models.kimi_k3.mono.runner import K3MonoDecodeRunner

ROWS, HIDDEN = 4, 8


def _unwritten(like):
    return attention_residual.folded_zero_like(like).fill_(float("nan"))


@pytest.mark.parametrize("routed_folded", [True, False])
def test_k1_reads_folded_zeros_as_zero(routed_folded):
    """The layer's own row-sharded tail folds both addends; the MoE block's
    folds only ``shared``."""
    prefix = torch.ones(ROWS, HIDDEN)
    routed = _unwritten(prefix) if routed_folded else torch.full_like(prefix, 3.0)
    expected = torch.zeros_like(prefix) if routed_folded else routed.clone()
    deltas = []
    runner = object.__new__(K3MonoDecodeRunner)
    runner.aux = AuxTaps((), ())
    runner._k1 = lambda build, plan, prefix, delta, *rest: deltas.append(delta)
    runner._k2 = lambda build, plan, core, prefix, blocks: torch.zeros_like(prefix)
    plan = SimpleNamespace(mono=True, ffn_only=False, layer_id=1)
    runner._layer(None, plan, None, prefix, None, routed, _unwritten(prefix), None)
    assert torch.equal(deltas[0], expected)
