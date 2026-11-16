# SPDX-License-Identifier: MIT
"""CPU coverage for the FFN seam's quantized shared-expert input.

`maybe_dual_stream_forward` is the op V2, V3, V3.2, V4 and K3 all call.
V2/V3/V3.2 and K3 take only `hidden_states`; passing `shared_quantized`
even as None is a second positional argument and TypeErrors. The seam pair
is forwarded only when the caller supplies it.

The fuse-quant gate (`takes_quantized_shared_input`, `Block.hc_ffn_fuse_quant`)
is pure Python. It is exercised here against AITER stand-ins so a CPU job
runs it; the GPU file only checks the kernel and the dispatch paths.
"""

from types import SimpleNamespace

import pytest
import torch
from aiter_stub import atom_under_stubs
from torch import nn


def _bind(monkeypatch, module):
    import atom.model_ops.module_dispatch_ops as ops

    monkeypatch.setattr(
        ops,
        "get_current_atom_config",
        lambda: SimpleNamespace(
            compilation_config=SimpleNamespace(static_forward_context={"moe": module})
        ),
    )
    return ops


class _HiddenOnly:
    """V2/V3/V3.2 and K3: both stream methods take hidden_states alone."""

    def __init__(self, dual):
        self._use_dual_stream = dual
        self.calls = []

    def dual_stream_moe_forward(self, hidden_states):
        self.calls.append(("dual", hidden_states))
        return hidden_states

    def single_stream_moe_forward(self, hidden_states):
        self.calls.append(("single", hidden_states))
        return hidden_states


class _AcceptsPair:
    """V4 / V4.1: the seam pair is an optional second argument."""

    def __init__(self, dual):
        self._use_dual_stream = dual
        self.calls = []

    def dual_stream_moe_forward(self, hidden_states, shared_quantized=None):
        self.calls.append(("dual", hidden_states, shared_quantized))
        return hidden_states

    def single_stream_moe_forward(self, hidden_states, shared_quantized=None):
        self.calls.append(("single", hidden_states, shared_quantized))
        return hidden_states


@pytest.mark.parametrize("dual", [False, True])
def test_v2_style_signature_is_called_with_hidden_states_only(monkeypatch, dual):
    module = _HiddenOnly(dual)
    ops = _bind(monkeypatch, module)
    hidden = torch.zeros(4, 8)
    out = ops.maybe_dual_stream_forward(hidden, "moe")
    assert out is hidden
    which = "dual" if dual else "single"
    assert module.calls == [(which, hidden)]


@pytest.mark.parametrize("dual", [False, True])
def test_seam_pair_is_forwarded_when_present(monkeypatch, dual):
    module = _AcceptsPair(dual)
    ops = _bind(monkeypatch, module)
    hidden = torch.zeros(4, 8)
    fp8 = torch.zeros(4, 8)
    scale = torch.zeros(4, 1)
    ops.maybe_dual_stream_forward(hidden, "moe", fp8, scale)
    which = "dual" if dual else "single"
    assert module.calls == [(which, hidden, (fp8, scale))]


def test_takes_quantized_shared_input_matches_native_mx_full_dim():
    with atom_under_stubs() as load:
        moe_cls = load("atom.models.deepseek_v41.moe").MoE
        moe = moe_cls.__new__(moe_cls)
        moe.dim = 128
        gate = SimpleNamespace(native_a8_group_rows=32, input_size=128)
        moe.shared_experts = SimpleNamespace(gate_up_proj=gate)
        assert moe.takes_quantized_shared_input()
        gate.input_size = 64
        assert not moe.takes_quantized_shared_input()
        gate.input_size = 128
        gate.native_a8_group_rows = None
        assert not moe.takes_quantized_shared_input()
        moe.shared_experts = None
        assert not moe.takes_quantized_shared_input()


def test_hc_ffn_fuse_quant_needs_pack_support_and_a_native_shared_gate():
    with atom_under_stubs() as load:
        model = load("atom.models.deepseek_v41.model")
        block = model.Block.__new__(model.Block)
        nn.Module.__init__(block)

        class FFN:
            def __init__(self, takes):
                self.takes = takes

            def takes_quantized_shared_input(self):
                return self.takes

        block.ffn = FFN(True)
        block.hc_attn_fn = object()
        block.hc_ffn_fn = object()

        def pack(fn):
            return None if fn is block.hc_ffn_fn else "packed"

        model.pack_mhc_fn = pack
        model.fuse_quant_supported = lambda: True
        block.process_weights_after_loading()
        assert block.hc_ffn_fuse_quant is False
        assert block.hc_fuse_quant is True

        model.pack_mhc_fn = lambda fn: "packed"
        block.ffn = FFN(False)
        block.process_weights_after_loading()
        assert block.hc_ffn_fuse_quant is False

        block.ffn = FFN(True)
        model.fuse_quant_supported = lambda: False
        block.process_weights_after_loading()
        assert block.hc_ffn_fuse_quant is False

        model.fuse_quant_supported = lambda: True
        block.process_weights_after_loading()
        assert block.hc_ffn_fuse_quant is True
