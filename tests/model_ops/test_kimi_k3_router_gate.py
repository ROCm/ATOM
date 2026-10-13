# SPDX-License-Identifier: MIT

import pytest
import torch
from torch import nn

pytest.importorskip("aiter")

from atom.models.kimi_k3 import KimiSparseMoeBlock


class _RecordingGate(nn.Module):
    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.calls = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return torch.full((x.shape[0], self.weight.shape[0]), -1.0, dtype=x.dtype)


def _block(gate_via_linear: bool) -> KimiSparseMoeBlock:
    block = KimiSparseMoeBlock.__new__(KimiSparseMoeBlock)
    nn.Module.__init__(block)
    torch.manual_seed(0)
    block.gate = _RecordingGate(torch.randn(16, 32, dtype=torch.bfloat16))
    block._gate_via_linear = gate_via_linear
    return block


def test_an_eager_run_computes_router_logits_without_the_tuned_gate():
    block = _block(gate_via_linear=True)
    x = torch.randn(11, 32, dtype=torch.bfloat16)

    logits = block.router_logits(x)

    assert block.gate.calls == 0
    torch.testing.assert_close(logits, x @ block.gate.weight.t())


def test_a_graph_run_keeps_the_tuned_gate():
    block = _block(gate_via_linear=False)
    x = torch.randn(11, 32, dtype=torch.bfloat16)

    logits = block.router_logits(x)

    assert block.gate.calls == 1
    assert torch.all(logits == -1.0)
