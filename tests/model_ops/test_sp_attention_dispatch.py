# SPDX-License-Identifier: MIT
"""SP FP8 attention transport and head exchange fall back off their contract."""

from types import SimpleNamespace

import pytest
import torch

aiter = pytest.importorskip("aiter")

from aiter.jit.utils import chip_info

from atom import config
from atom.distributed import fp8_head_exchange, token_transport
from atom.model_ops import gemma_norm_fp8
from atom.plugin import prepare


@pytest.fixture
def supported(monkeypatch):
    settings = SimpleNamespace(torch_dtype=torch.bfloat16)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(chip_info, "get_gfx_runtime", lambda: "gfx950")
    monkeypatch.setattr(prepare, "_CURRENT_FRAMEWORK", "atom")
    monkeypatch.setattr(fp8_head_exchange.dtypes, "fp8", torch.float8_e4m3fn)
    monkeypatch.setattr(gemma_norm_fp8, "_fused_per_token_quant", lambda *a, **k: None)
    monkeypatch.setattr(token_transport, "_head_exchange_kernel", lambda: object())
    monkeypatch.setenv("ATOM_USE_CUSTOM_ALL_GATHER", "1")
    with config.use_custom_atom_config(settings):
        yield settings


def _head_input(**overrides):
    fields = {
        "is_cuda": True,
        "ndim": 2,
        "shape": (8192, 2048),
        "dtype": torch.bfloat16,
        "element_size": lambda: 2,
        "is_contiguous": lambda: True,
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def test_validated_layout_selects_fp8_without_opt_in(supported):
    assert fp8_head_exchange.token_fp8_supported()
    assert gemma_norm_fp8.gemma_rms_norm_fp8_supported(6144)
    assert not gemma_norm_fp8.gemma_rms_norm_fp8_supported(8193)


@pytest.mark.parametrize("case", ["cpu", "dtype", "gfx", "plugin"])
def test_unsupported_runtime_keeps_existing_kernels(monkeypatch, supported, case):
    if case == "cpu":
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(config, "_current_atom_config", None)
        group = SimpleNamespace(world_size=4)
        head = _head_input(is_cuda=False)
        assert token_transport.head_exchange_communicator(head, group) is None
    elif case == "dtype":
        supported.torch_dtype = torch.float16
    elif case == "gfx":
        monkeypatch.setattr(chip_info, "get_gfx_runtime", lambda: "gfx942")
    else:
        monkeypatch.setattr(prepare, "_CURRENT_FRAMEWORK", "vllm")
    assert not fp8_head_exchange.token_fp8_supported()


def test_head_exchange_requires_usable_collective(supported):
    ca = SimpleNamespace(disabled=False, _pool={}, should_custom_ag=lambda x: True)
    group = SimpleNamespace(
        world_size=4, device_communicator=SimpleNamespace(ca_comm=ca)
    )

    def communicator(**overrides):
        return token_transport.head_exchange_communicator(
            _head_input(**overrides), group
        )

    assert communicator() is ca
    assert communicator(shape=(8192, 1024)) is ca
    for overrides in (
        {"is_cuda": False},
        {"shape": (4, 2048)},
        {"shape": (8193, 2048)},
        {"shape": (8192, 4)},
        {"dtype": torch.float16},
        {"is_contiguous": lambda: False},
    ):
        assert communicator(**overrides) is None
    ca.disabled = True
    assert communicator() is None
    ca.disabled = False
    ca.should_custom_ag = lambda x: False
    assert communicator() is None
    ca.should_custom_ag = lambda x: True
    group.world_size = 2
    assert communicator() is None


class _Norm:
    weight = torch.ones(8, dtype=torch.bfloat16)
    variance_epsilon = 1e-6

    def __call__(self, hidden_states, residual):
        return hidden_states, residual


def test_norm_quant_fuses_only_when_asked(monkeypatch):
    """Plain TP keeps the BF16-rounded quantization that mono matches."""
    from atom.model_ops import layernorm  # its Triton import needs a GPU

    calls = []
    monkeypatch.setattr(layernorm, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(layernorm, "gemma_rms_norm_fp8_supported", lambda width: True)
    monkeypatch.setattr(
        layernorm, "gemma_rms_norm_fp8", lambda *args: calls.append("fused")
    )

    def quant(x, **kwargs):
        calls.append("bf16")
        return x, x

    monkeypatch.setattr(aiter, "get_hip_quant", lambda quant_type: quant)
    x = torch.ones(2, 8, dtype=torch.bfloat16)
    for fused in (False, True):
        layernorm.gemma_rms_norm_quant(x, x, _Norm(), fused=fused)
        layernorm.fused_allreduce_gemma_rms_norm_quant(x, x, _Norm(), fused=fused)
    assert calls == ["bf16", "bf16", "fused", "fused"]
