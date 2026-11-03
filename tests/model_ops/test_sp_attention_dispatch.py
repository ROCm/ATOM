# SPDX-License-Identifier: MIT
"""SP FP8 attention transport and head exchange fall back off their contract."""

from types import SimpleNamespace

import pytest
import torch
from aiter_stubs import atom_without_aiter

from atom import config


@pytest.fixture(scope="module")
def modules():
    with atom_without_aiter() as load:
        yield SimpleNamespace(
            aiter=load("aiter"),
            chip_info=load("aiter.jit.utils.chip_info"),
            norm_ops=load("aiter.ops.fused_qk_rmsnorm_group_quant"),
            dtypes=load("aiter.utility.dtypes"),
            token_transport=load("atom.distributed.token_transport"),
            sp_attention=load("atom.model_ops.sp_attention"),
            layernorm=load("atom.model_ops.layernorm"),
        )


@pytest.fixture
def supported(monkeypatch, modules):
    settings = SimpleNamespace(torch_dtype=torch.bfloat16, sequence_parallel_size=4)
    monkeypatch.setattr(modules.chip_info, "get_gfx_runtime", lambda: "gfx950")
    monkeypatch.setattr(modules.sp_attention.dtypes, "fp8", torch.float8_e4m3fn)
    monkeypatch.setenv("ATOM_USE_CUSTOM_ALL_GATHER", "1")
    with config.use_custom_atom_config(settings):
        yield settings


def test_validated_layout_selects_fp8_without_opt_in(modules, supported):
    assert modules.sp_attention.sp_fp8_attention_output_supported()


@pytest.mark.parametrize("case", ["sp_off", "dtype", "gfx"])
def test_unsupported_runtime_keeps_existing_kernels(
    monkeypatch, modules, supported, case
):
    if case == "sp_off":
        supported.sequence_parallel_size = 1
    elif case == "dtype":
        supported.torch_dtype = torch.float16
    else:
        monkeypatch.setattr(modules.chip_info, "get_gfx_runtime", lambda: "gfx942")
    assert not modules.sp_attention.sp_fp8_attention_output_supported()


def test_head_exchange_requires_usable_collective(monkeypatch, modules, supported):
    transport = modules.token_transport
    ca = SimpleNamespace(
        disabled=False,
        should_custom_ag=lambda x: True,
        should_head_exchange=lambda x: True,
        registered_input_view=lambda shape, dtype: torch.empty(shape, dtype=dtype),
    )
    group = SimpleNamespace(device_communicator=SimpleNamespace(ca_comm=ca))
    x = torch.ones(8, 16, dtype=torch.bfloat16)

    def view(*args):
        return transport.head_exchange_view(ca, x, *args)

    assert transport.head_exchange_comm(group) is ca
    assert view(8) is x
    # Padded rows, or the FP8 payload, are staged in the registered pool.
    assert view(12).shape == (12, 16)
    assert view(8, torch.float8_e4m3fn).dtype == torch.float8_e4m3fn
    assert transport.head_exchange_view(None, x, 8) is None
    assert transport.head_exchange_view(ca, x.half(), 8) is None
    for check in ("should_custom_ag", "should_head_exchange"):
        setattr(ca, check, lambda x: False)
        assert view(8) is None
        setattr(ca, check, lambda x: True)
    ca.registered_input_view = lambda shape, dtype: None
    assert view(12) is None and view(8) is x
    ca.disabled = True
    assert transport.head_exchange_comm(group) is None
    ca.disabled = False
    monkeypatch.setattr(modules.chip_info, "get_gfx_runtime", lambda: "gfx942")
    assert transport.head_exchange_comm(group) is None


class _Norm:
    weight = torch.ones(8, dtype=torch.bfloat16)
    variance_epsilon = 1e-6

    def __call__(self, hidden_states, residual):
        return hidden_states, residual


def test_fp32_norm_quant_only_when_asked(monkeypatch, modules):
    """Plain TP keeps the BF16-rounded quantization that mono matches."""
    norm_ops, dtypes, layernorm = modules.norm_ops, modules.dtypes, modules.layernorm
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

    monkeypatch.setattr(modules.aiter, "get_hip_quant", lambda quant_type: quant)
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
def test_empty_rows_launch_no_norm_kernel(monkeypatch, modules, rows):
    """An SP rank can hold no rows; an empty grid fails the next launch."""
    norm_ops, dtypes, layernorm = modules.norm_ops, modules.dtypes, modules.layernorm
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

    monkeypatch.setattr(modules.aiter, "get_hip_quant", lambda quant_type: quant)
    x = torch.ones(rows, 8, dtype=torch.bfloat16)
    out, res = layernorm.GemmaRMSNorm(8).forward_cuda(x, x)
    assert out.shape == res.shape == x.shape
    for fp32 in (True, False):
        out, scale, res = layernorm.gemma_rms_norm_quant(
            x, x, _Norm(), quantize_fp32_norm=fp32
        )
        assert out.shape == res.shape == x.shape and scale.shape == (rows, 1)
    assert calls == (["bf16", "fp8", "quant"] if rows else [])
