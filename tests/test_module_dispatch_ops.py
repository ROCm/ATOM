import sys
import types
from enum import Enum
from types import SimpleNamespace

import pytest
import torch

from atom.config import CUDAGraphMode
from atom.model_ops import module_dispatch_ops
from atom.utils import forward_context


class _FrontendCUDAGraphMode(Enum):
    NONE = 10
    PIECEWISE = 20
    FULL = 30
    FULL_AND_PIECEWISE = (30, 20)


class _RecordingMoE:
    def __init__(self, use_dual_stream=True):
        self._use_dual_stream = use_dual_stream
        self.calls = []

    def single_stream_moe_forward(self, hidden_states):
        self.calls.append("single")
        return hidden_states + 1

    def dual_stream_moe_forward(self, hidden_states):
        self.calls.append("dual")
        return hidden_states + 2


def _patch_vllm_forward_context(monkeypatch, mode):
    vllm = types.ModuleType("vllm")
    vllm.__path__ = []
    vllm_forward_context = types.ModuleType("vllm.forward_context")
    vllm_forward_context.is_forward_context_available = lambda: True
    vllm_forward_context.get_forward_context = lambda: SimpleNamespace(
        cudagraph_runtime_mode=mode
    )
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.forward_context", vllm_forward_context)


def _patch_dispatch(monkeypatch, moe, runtime_mode, *, threshold=4, tbo=False):
    config = SimpleNamespace(
        compilation_config=SimpleNamespace(
            cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
            static_forward_context={"moe": moe},
        )
    )
    monkeypatch.setattr(module_dispatch_ops, "get_current_atom_config", lambda: config)
    monkeypatch.setattr(
        module_dispatch_ops,
        "get_current_cudagraph_runtime_mode",
        lambda: runtime_mode,
    )
    monkeypatch.setattr("atom.utils.tbo.ubatching.tbo_active", lambda: tbo)
    monkeypatch.setattr(
        module_dispatch_ops.envs,
        "ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD",
        threshold,
    )


@pytest.mark.parametrize(
    ("frontend_mode", "atom_mode"),
    [
        (_FrontendCUDAGraphMode.NONE, CUDAGraphMode.NONE),
        (_FrontendCUDAGraphMode.PIECEWISE, CUDAGraphMode.PIECEWISE),
        (_FrontendCUDAGraphMode.FULL, CUDAGraphMode.FULL),
    ],
)
def test_normalize_cudagraph_runtime_mode_by_name(frontend_mode, atom_mode):
    assert forward_context._normalize_cudagraph_runtime_mode(frontend_mode) == atom_mode


def test_normalize_cudagraph_runtime_mode_rejects_composite_mode():
    assert (
        forward_context._normalize_cudagraph_runtime_mode(
            _FrontendCUDAGraphMode.FULL_AND_PIECEWISE
        )
        is None
    )


@pytest.mark.parametrize(
    "mode",
    [
        _FrontendCUDAGraphMode.NONE,
        _FrontendCUDAGraphMode.PIECEWISE,
        _FrontendCUDAGraphMode.FULL,
    ],
)
def test_current_cudagraph_runtime_mode_uses_vllm_context(monkeypatch, mode):
    import atom.plugin

    monkeypatch.setattr(atom.plugin, "is_vllm", lambda: True)
    _patch_vllm_forward_context(monkeypatch, mode)

    assert (
        forward_context.get_current_cudagraph_runtime_mode().name == mode.name
    )


@pytest.mark.parametrize(
    ("context_mode", "expected"),
    [
        (CUDAGraphMode.FULL, CUDAGraphMode.FULL),
        (None, CUDAGraphMode.NONE),
    ],
)
def test_current_cudagraph_runtime_mode_uses_atom_fallback(
    monkeypatch, context_mode, expected
):
    import atom.plugin

    monkeypatch.setattr(atom.plugin, "is_vllm", lambda: False)
    monkeypatch.setattr(
        forward_context,
        "get_forward_context",
        lambda: SimpleNamespace(cudagraph_runtime_mode=context_mode),
    )

    assert forward_context.get_current_cudagraph_runtime_mode() == expected


@pytest.mark.parametrize(
    ("runtime_mode", "expected_call", "expected_offset"),
    [
        (CUDAGraphMode.NONE, "dual", 2),
        (CUDAGraphMode.FULL, "dual", 2),
        (CUDAGraphMode.PIECEWISE, "single", 1),
    ],
)
def test_dual_stream_dispatch_uses_runtime_mode(
    monkeypatch, runtime_mode, expected_call, expected_offset
):
    moe = _RecordingMoE()
    _patch_dispatch(monkeypatch, moe, runtime_mode)
    hidden_states = torch.zeros(4, 2)

    output = module_dispatch_ops.maybe_dual_stream_forward(hidden_states, "moe")

    assert moe.calls == [expected_call]
    torch.testing.assert_close(output, hidden_states + expected_offset)


@pytest.mark.parametrize(
    ("num_tokens", "tbo_active", "use_dual_stream"),
    [
        (5, False, True),
        (4, True, True),
        (4, False, False),
    ],
)
def test_dual_stream_dispatch_preserves_other_gates(
    monkeypatch, num_tokens, tbo_active, use_dual_stream
):
    moe = _RecordingMoE(use_dual_stream=use_dual_stream)
    _patch_dispatch(
        monkeypatch,
        moe,
        CUDAGraphMode.FULL,
        tbo=tbo_active,
    )

    module_dispatch_ops.maybe_dual_stream_forward(
        torch.zeros(num_tokens, 2), "moe"
    )

    assert moe.calls == ["single"]
