# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""CPU contracts for MiniMax-M3 dense MLP SwiGLU -> prequantized down_proj."""

from types import SimpleNamespace

import pytest
import torch

# swiglu_oai imports Triton, which the non-GPU CI image does not install.
pytest.importorskip("triton")

from atom.model_ops import swiglu_oai
from atom.models.minimax_m3 import model as m3_model
from atom.models.minimax_m3.model import MiniMaxM3MLP


def _per_token_fp8_state(per_token_fp8: bool) -> dict:
    # Online quantization sets these after load; the fused path depends on them.
    return {
        "quant_type": (
            m3_model.QuantType.per_Token if per_token_fp8 else m3_model.QuantType.No
        ),
        "params_dtype": m3_model.dtypes.fp8 if per_token_fp8 else torch.bfloat16,
    }


class _GateUp:
    def __init__(self, splitk_consumer_max_m: int = 0):
        self.splitk_consumer_max_m = splitk_consumer_max_m
        self.__dict__.update(_per_token_fp8_state(True))

    def __call__(self, x, x_scale=None):
        return torch.cat([x, x], dim=-1)


class _DownProj:
    def __init__(self, per_token_fp8: bool):
        self.calls = []
        self.__dict__.update(_per_token_fp8_state(per_token_fp8))

    def __call__(self, x, x_scale=None):
        self.calls.append((x, x_scale))
        return x


def _record(monkeypatch):
    calls = []

    def fake(name, quant):
        def op(gate_up, alpha=None, beta=None, limit=None):
            calls.append(name)
            n = gate_up.shape[-1] // 2
            if not quant:
                return gate_up[..., :n] + 1
            return (
                torch.zeros((*gate_up.shape[:-1], n), dtype=torch.uint8),
                torch.full((*gate_up.shape[:-1], 1), 0.5),
            )

        monkeypatch.setattr(m3_model, name, op)

    fake("swiglu_oai_split", False)
    fake("swiglu_oai_split_reducing", False)
    fake("swiglu_oai_split_quant", True)
    fake("swiglu_oai_split_reducing_quant", True)
    return calls


def _mlp(*, supported=True, down_per_token_fp8=True, splitk_consumer_max_m=0):
    return SimpleNamespace(
        gate_up_proj=_GateUp(splitk_consumer_max_m),
        down_proj=_DownProj(down_per_token_fp8),
        swiglu_alpha=1.702,
        swiglu_beta=1.0,
        swiglu_limit=7.0,
        swiglu_quant_supported=supported,
    )


@pytest.mark.parametrize(
    "splitk_consumer_max_m, op",
    [(0, "swiglu_oai_split_quant"), (64, "swiglu_oai_split_reducing_quant")],
)
def test_dense_mlp_hands_fused_fp8_and_token_scale_to_down_proj(
    monkeypatch, splitk_consumer_max_m, op
):
    calls = _record(monkeypatch)
    mlp = _mlp(splitk_consumer_max_m=splitk_consumer_max_m)
    out = MiniMaxM3MLP.forward(mlp, torch.randn(3, 8))

    assert calls == [op]
    ((values, scale),) = mlp.down_proj.calls
    assert out is values
    assert values.dtype == torch.uint8 and values.shape == (3, 8)
    assert scale.shape == (3, 1) and scale.dtype == torch.float32


@pytest.mark.parametrize(
    "splitk_consumer_max_m, op",
    [(0, "swiglu_oai_split"), (64, "swiglu_oai_split_reducing")],
)
@pytest.mark.parametrize(
    "supported, down_per_token_fp8", [(False, True), (True, False)]
)
def test_dense_mlp_keeps_the_bf16_activation_otherwise(
    monkeypatch, splitk_consumer_max_m, op, supported, down_per_token_fp8
):
    calls = _record(monkeypatch)
    mlp = _mlp(
        supported=supported,
        down_per_token_fp8=down_per_token_fp8,
        splitk_consumer_max_m=splitk_consumer_max_m,
    )
    x = torch.randn(3, 8)
    MiniMaxM3MLP.forward(mlp, x)

    assert calls == [op]
    ((values, scale),) = mlp.down_proj.calls
    # down_proj quantizes the bf16 activation itself, exactly as before.
    assert scale is None
    assert torch.equal(values, x + 1)


def test_dense_mlp_checks_down_proj_quant_at_forward_time(monkeypatch):
    # At construction down_proj is not yet per-token FP8 (online quantization
    # runs at load time); the fused path must follow its state when forward runs.
    calls = _record(monkeypatch)
    mlp = _mlp(down_per_token_fp8=False)
    MiniMaxM3MLP.forward(mlp, torch.randn(2, 8))
    mlp.down_proj.__dict__.update(_per_token_fp8_state(True))
    MiniMaxM3MLP.forward(mlp, torch.randn(2, 8))

    assert calls == ["swiglu_oai_split", "swiglu_oai_split_quant"]
    assert mlp.down_proj.calls[0][1] is None
    assert mlp.down_proj.calls[1][1] is not None


def test_swiglu_quant_support_is_ocp_fp8_up_to_one_program_wide():
    limit = swiglu_oai.SWIGLU_OAI_QUANT_MAX_INTER
    assert swiglu_oai.swiglu_oai_quant_supported(3072, torch.float8_e4m3fn)
    assert swiglu_oai.swiglu_oai_quant_supported(limit, torch.float8_e4m3fn)
    assert not swiglu_oai.swiglu_oai_quant_supported(limit + 1, torch.float8_e4m3fn)
    # gfx942's FNUZ FP8 saturates at 240 in aiter's quant; not reproduced here.
    assert not swiglu_oai.swiglu_oai_quant_supported(3072, torch.float8_e4m3fnuz)
    assert not swiglu_oai.swiglu_oai_quant_supported(3072, torch.int8)
