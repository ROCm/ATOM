# SPDX-License-Identifier: MIT
"""SP FP8 attention transport and head exchange fall back off their contract."""

from types import SimpleNamespace

import pytest
import torch

aiter = pytest.importorskip("aiter")

from aiter.jit.utils import chip_info

from atom import config
from atom.distributed import fp8_head_exchange, token_transport
from atom.plugin import prepare


@pytest.fixture
def supported(monkeypatch):
    settings = SimpleNamespace(torch_dtype=torch.bfloat16)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(chip_info, "get_gfx_runtime", lambda: "gfx950")
    monkeypatch.setattr(prepare, "_CURRENT_FRAMEWORK", "atom")
    monkeypatch.setattr(fp8_head_exchange.dtypes, "fp8", torch.float8_e4m3fn)
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
    ca = SimpleNamespace(
        disabled=False,
        should_custom_ag=lambda x: True,
        should_head_exchange=lambda x: True,
    )
    group = SimpleNamespace(
        world_size=4, device_communicator=SimpleNamespace(ca_comm=ca)
    )

    def communicator(**overrides):
        return token_transport.head_exchange_communicator(
            _head_input(**overrides), group
        )

    assert communicator() is ca
    for overrides in (
        {"is_cuda": False},
        {"shape": (4, 2048)},
        {"dtype": torch.float16},
    ):
        assert communicator(**overrides) is None
    ca.disabled = True
    assert communicator() is None
    ca.disabled = False
    for check in ("should_custom_ag", "should_head_exchange"):
        setattr(ca, check, lambda x: False)
        assert communicator() is None
        setattr(ca, check, lambda x: True)


class _Norm:
    weight = torch.ones(8, dtype=torch.bfloat16)
    variance_epsilon = 1e-6

    def __call__(self, hidden_states, residual):
        return hidden_states, residual


def test_fp32_norm_quant_only_when_asked(monkeypatch):
    """Plain TP keeps the BF16-rounded quantization that mono matches."""
    from aiter.ops import fused_qk_rmsnorm_group_quant as norm_ops
    from aiter.utility import dtypes

    from atom.model_ops import layernorm  # its Triton import needs a GPU

    calls = []
    monkeypatch.setattr(layernorm, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(dtypes, "fp8", torch.float8_e4m3fn)
    monkeypatch.setattr(
        norm_ops,
        "fused_qk_rmsnorm_per_token_quant",
        lambda *args, **kwargs: calls.append("fp32"),
    )

    def quant(x, **kwargs):
        calls.append("bf16")
        return x, x

    monkeypatch.setattr(aiter, "get_hip_quant", lambda quant_type: quant)
    x = torch.ones(2, 8, dtype=torch.bfloat16)
    layernorm.fused_allreduce_gemma_rms_norm_quant(x, x, _Norm())
    layernorm.gemma_rms_norm_quant(x, x, _Norm())
    layernorm.gemma_rms_norm_quant(x, x, _Norm(), quantize_fp32_norm=True)
    assert calls == ["bf16", "bf16", "fp32"]
    # No silent fallback to the BF16-rounded quantization.
    with pytest.raises(ValueError, match="quantize_fp32_norm"):
        layernorm.gemma_rms_norm_quant(
            x.float(), None, _Norm(), quantize_fp32_norm=True
        )


@pytest.mark.parametrize("rows", [0, 2])
def test_empty_rows_launch_no_norm_kernel(monkeypatch, rows):
    """An SP rank can hold no rows; an empty grid fails the next launch."""
    from aiter.ops import fused_qk_rmsnorm_group_quant as norm_ops
    from aiter.utility import dtypes

    from atom.model_ops import layernorm

    calls = []
    monkeypatch.setattr(dtypes, "fp8", torch.float8_e4m3fn)
    monkeypatch.setattr(
        norm_ops, "fused_qk_rmsnorm_group_quant", lambda **kw: calls.append("bf16")
    )
    monkeypatch.setattr(
        norm_ops,
        "fused_qk_rmsnorm_per_token_quant",
        lambda *args, **kwargs: calls.append("fp8"),
    )

    def quant(x, **kwargs):
        calls.append("quant")
        return x, x[:, :1]

    monkeypatch.setattr(aiter, "get_hip_quant", lambda quant_type: quant)
    x = torch.ones(rows, 8, dtype=torch.bfloat16)
    out, res = layernorm.GemmaRMSNorm(8).forward_cuda(x, x)
    assert out.shape == res.shape == x.shape
    for fp32 in (True, False):
        out, scale, res = layernorm.gemma_rms_norm_quant(
            x, x, _Norm(), quantize_fp32_norm=fp32
        )
        assert out.shape == res.shape == x.shape and scale.shape == (rows, 1)
    assert calls == (["bf16", "fp8", "quant"] if rows else [])
