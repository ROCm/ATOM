# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""MiniMax-M3 dense MLP: SwiGLU-OAI quantized per token in its own kernel.

The FP8 activation and its scales must be bitwise what aiter's per-token quant
gives for the bf16 activation (GPU), and the MLP must hand them to down_proj
only when down_proj consumes per-token FP8, decided at forward time (CPU)."""

from types import SimpleNamespace

import pytest
import torch
from conftest import gpu_stack_import

try:  # both need the aiter/triton stack, absent on the CPU-only CI runner
    swiglu_oai = gpu_stack_import("atom.model_ops.swiglu_oai")
    m3_model = gpu_stack_import("atom.models.minimax_m3.model")
except pytest.skip.Exception as e:
    pytest.skip(str(e), allow_module_level=True)

gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _per_token_fp8_state(per_token_fp8: bool) -> dict:
    # Online quantization sets these after load; the fused path depends on them.
    return {
        "quant_type": (
            m3_model.QuantType.per_Token if per_token_fp8 else m3_model.QuantType.No
        ),
        "params_dtype": m3_model.dtypes.fp8 if per_token_fp8 else torch.bfloat16,
    }


class _DownProj:
    def __init__(self, per_token_fp8: bool):
        self.calls = []
        self.__dict__.update(_per_token_fp8_state(per_token_fp8))

    def __call__(self, x, x_scale=None):
        self.calls.append((x, x_scale))
        return x


def _mlp(monkeypatch, supported, down_per_token_fp8):
    calls = []

    def act(gate_up, alpha=None, beta=None, limit=None):
        calls.append("bf16")
        return gate_up[..., : gate_up.shape[-1] // 2] + 1

    def act_quant(gate_up, alpha=None, beta=None, limit=None):
        calls.append("fp8")
        rows = gate_up.shape[:-1]
        return torch.zeros((*rows, gate_up.shape[-1] // 2)), torch.ones((*rows, 1))

    monkeypatch.setattr(m3_model, "swiglu_oai_split", act)
    monkeypatch.setattr(m3_model, "swiglu_oai_split_quant", act_quant)
    mlp = SimpleNamespace(
        gate_up_proj=lambda x, x_scale=None: torch.cat([x, x], dim=-1),
        down_proj=_DownProj(down_per_token_fp8),
        swiglu_alpha=1.702,
        swiglu_beta=1.0,
        swiglu_limit=7.0,
        swiglu_quant_supported=supported,
    )
    return mlp, calls


def test_dense_mlp_hands_fp8_and_token_scale_to_down_proj(monkeypatch):
    mlp, calls = _mlp(monkeypatch, supported=True, down_per_token_fp8=True)
    out = m3_model.MiniMaxM3MLP.forward(mlp, torch.randn(3, 8))

    assert calls == ["fp8"]
    ((values, scale),) = mlp.down_proj.calls
    assert out is values and values.shape == (3, 8) and scale.shape == (3, 1)


@pytest.mark.parametrize(
    "supported, down_per_token_fp8", [(False, True), (True, False)]
)
def test_dense_mlp_keeps_the_bf16_activation_otherwise(
    monkeypatch, supported, down_per_token_fp8
):
    mlp, calls = _mlp(monkeypatch, supported, down_per_token_fp8)
    x = torch.randn(3, 8)
    m3_model.MiniMaxM3MLP.forward(mlp, x)

    # down_proj quantizes the bf16 activation itself, as before.
    assert calls == ["bf16"]
    ((values, scale),) = mlp.down_proj.calls
    assert scale is None and torch.equal(values, x + 1)
    if supported:
        # Online quantization makes down_proj per-token FP8 after construction.
        mlp.down_proj.__dict__.update(_per_token_fp8_state(True))
        m3_model.MiniMaxM3MLP.forward(mlp, x)
        assert calls == ["bf16", "fp8"]


def test_swiglu_quant_support_is_ocp_fp8_up_to_one_program_wide():
    limit = swiglu_oai.SWIGLU_OAI_QUANT_MAX_INTER
    supported = swiglu_oai.swiglu_oai_quant_supported
    assert supported(3072, torch.float8_e4m3fn) and supported(
        limit, torch.float8_e4m3fn
    )
    assert not supported(limit + 1, torch.float8_e4m3fn)
    # gfx942's FNUZ FP8 saturates at 240 in aiter's quant; not reproduced here.
    assert not supported(3072, torch.float8_e4m3fnuz)
    assert not supported(3072, torch.int8)


def _hip_per_token_fp8(x):
    # What a per-token FP8 down_proj runs on the bf16 activation it is given.
    from aiter import QuantType, dtypes
    from aiter.ops.quant import get_hip_quant

    return get_hip_quant(QuantType.per_Token)(x, quant_dtype=dtypes.fp8)


def _assert_bitwise_quant(actual, expected):
    (q, s), (q_ref, s_ref) = actual, expected
    assert q.dtype == q_ref.dtype and s.dtype == s_ref.dtype == torch.float32
    assert s.shape == s_ref.shape == (*q.shape[:-1], 1)
    assert torch.equal(q.view(torch.uint8), q_ref.view(torch.uint8))
    assert torch.equal(s.view(torch.int32), s_ref.view(torch.int32))


def _gate_up(rows, seed, edge_rows=True):
    # MiniMax-M3 TP4 dense MLP: 2 x 3072 per rank.
    torch.manual_seed(seed)
    gate_up = torch.randn(rows, 6144, device="cuda") * 3
    if edge_rows and rows >= 8:
        gate, up = gate_up[:, :3072], gate_up[:, 3072:]
        gate[0], up[0] = 0.0, 0.0  # zero row (a pad row): scale 0, 0 * inf
        gate[1] = -60.0  # sigmoid underflows: -0 everywhere
        gate[2], up[2] = 1e4, -1e4  # clamped by the limit
        gate[3, ::7] = float("nan")
        gate[4, ::5] = float("-inf")  # -inf * sigmoid(-inf) = NaN
        up[5, ::3] = float("inf")
        gate[6] = -51.5  # amax ~1e-36: the scale is subnormal, flushed to 0
        gate[7] = -45.0  # tiny activations, normal scale
    return gate_up.to(torch.bfloat16)


@gpu
@pytest.mark.parametrize("rows", [1, 4, 16, 64, 2517])
@pytest.mark.parametrize("limit", [7.0, None])
def test_swiglu_quant_is_swiglu_then_hip_per_token_quant(rows, limit):
    gate_up = _gate_up(rows, rows)
    if limit is None and rows >= 9:
        gate_up[8] = 1e30  # inf activations without the limit
    _assert_bitwise_quant(
        swiglu_oai.swiglu_oai_split_quant(gate_up, 1.702, 1.0, limit),
        _hip_per_token_fp8(swiglu_oai.swiglu_oai_split(gate_up, 1.702, 1.0, limit)),
    )


@gpu
def test_swiglu_quant_compiled():
    def activate(x, alpha, beta, cap):
        return swiglu_oai.swiglu_oai_split_quant(x, alpha, beta, cap)

    torch._dynamo.reset()
    compiled = torch.compile(activate, fullgraph=True, dynamic=True)
    for rows in (12, 257):
        x = _gate_up(rows, rows, edge_rows=False)
        expected = _hip_per_token_fp8(swiglu_oai.swiglu_oai_split(x, 1.702, 1.0, 7.0))
        _assert_bitwise_quant(compiled(x, 1.702, 1.0, 7.0), expected)
