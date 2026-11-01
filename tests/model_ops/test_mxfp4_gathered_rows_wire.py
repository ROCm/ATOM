# SPDX-License-Identifier: MIT
"""`Mxfp4MoEMethod.gathered_rows_wire` against AITER's own fused_moe.

The method moves gathered rows in MXFP4 only where it predicts fused_moe
would quantize them to MXFP4 itself and take them prequantized. That
prediction mirrors fused_moe's private dispatch, so it is checked here against
fused_moe on MiniMax-M3's SP MoE shape: rows the method sends in MXFP4 must
come out of `apply` as their BF16 originals do, up to the order of GEMM2's
atomic adds. A prediction fused_moe does not honour hands FP4 bytes to a
BF16-reading GEMM and fails this.
"""

from types import SimpleNamespace

import pytest
import torch
from import_guard import skip_if_dependency_missing

try:
    from aiter import ActivationType, QuantType, dtypes
    from aiter.jit.utils.chip_info import get_gfx
    from aiter.ops.shuffle import moe_shuffle_scale, moe_shuffle_weight

    import atom.model_ops.moe as moe_mod
except ImportError as exc:
    skip_if_dependency_missing(exc, "requires AITER")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_gfx() != "gfx950",
    reason="the MXFP4 rows wire is gfx950 only",
)

# MiniMax-M3 at SP4, as deployed: 128 routed experts, top-4, one shared expert
# fused last, and the MoE TP-sharded over the 4 SP ranks (3072 / 4 = 768).
HIDDEN, INTER, ROUTED, TOPK = 6144, 768, 128, 4
EXPERTS = ROUTED + 1


class _Layer:
    def __init__(self, **attrs):
        self.__dict__.update(attrs)


def _shuffled(experts, rows, cols, gate_up):
    weight = torch.randint(0, 256, (experts, rows, cols // 2), dtype=torch.uint8)
    # e8m0 2^-8..2^-6 keeps Swiglu's inputs well inside its 7.0 limit.
    scale = torch.randint(119, 122, (experts * rows, cols // 32), dtype=torch.uint8)
    weight = moe_shuffle_weight(
        weight.cuda().view(dtypes.fp4x2),
        experts_cnt=experts,
        is_guinterleave=False,
        gate_up=gate_up,
    )
    weight.is_shuffled = True
    scale = moe_shuffle_scale(
        scale.cuda(), experts, is_guinterleave=False, gate_up=gate_up
    )
    return weight, scale


@pytest.fixture(scope="module")
def m3_experts():
    torch.manual_seed(0)
    w13, w13_scale = _shuffled(EXPERTS, 2 * INTER, HIDDEN, gate_up=True)
    w2, w2_scale = _shuffled(EXPERTS, HIDDEN, INTER, gate_up=False)
    return _Layer(
        w13_weight=w13,
        w2_weight=w2,
        w13_weight_scale=w13_scale,
        w2_weight_scale=w2_scale,
        w13_input_scale=None,
        w2_input_scale=None,
        w13_bias=None,
        w2_bias=None,
        expert_mask=None,
        custom_routing_function=None,
        apply_router_weight_on_input=False,
        activation=ActivationType.Swiglu,
        hidden_size=HIDDEN,
        top_k=TOPK,
        num_fused_shared_experts=1,
        swiglu_limit=7.0,
    )


@pytest.fixture
def method(monkeypatch):
    monkeypatch.setattr(
        moe_mod, "get_current_atom_config", lambda: SimpleNamespace(eplb_enable=False)
    )
    for flag in (
        "ATOM_USE_TRITON_MOE",
        "ATOM_MOE_GU_ITLV",
        "ATOM_MORI_FP4_DISPATCH",
        "ATOM_USE_TRITON_MOE_A4W4",
    ):
        monkeypatch.setattr(moe_mod.envs, flag, False)
    monkeypatch.setattr(moe_mod.envs, "is_set", lambda _name: True)
    quant = SimpleNamespace(
        quant_type=QuantType.per_1x32,
        quant_dtype=dtypes.fp4x2,
        quant_method=None,
        is_dynamic=True,
    )
    moe = SimpleNamespace(
        a_quant_dtype=None,
        mxfp4_fp8_activations=False,
        use_ep=False,
        in_dtype=torch.bfloat16,
        hidden_dim=HIDDEN,
    )
    return moe_mod.Mxfp4MoEMethod(quant, moe)


def _routes(rows):
    scores = torch.rand(rows, ROUTED, device="cuda")
    weights, ids = scores.topk(TOPK, dim=-1)
    shared = torch.full((rows, 1), ROUTED, dtype=ids.dtype, device="cuda")
    return (
        torch.cat([weights, torch.ones_like(weights[:, :1])], dim=1),
        torch.cat([ids, shared], dim=1).to(torch.int32),
    )


@pytest.mark.parametrize("rows", [8192, 16384, 32768])
def test_rows_sent_in_mxfp4_match_their_bf16_originals(method, m3_experts, rows):
    wire = method.gathered_rows_wire(m3_experts, rows)
    assert wire is not None and wire.dtype == dtypes.fp4x2, (
        "M3's SP prefill no longer gathers MXFP4 rows: recheck the predictor "
        "against fused_moe before loosening this"
    )
    x = torch.randn(rows, HIDDEN, dtype=torch.bfloat16, device="cuda")
    topk_weights, topk_ids = _routes(rows)
    run = {
        "layer": m3_experts,
        "topk_weights": topk_weights,
        "topk_ids": topk_ids,
        "activation": ActivationType.Swiglu,
    }

    bf16 = method.apply(x=x, **run)
    x_fp4, x_scale = wire.quantize(x)
    fp4 = method.apply(x=x_fp4, x_scale=x_scale, **run)

    assert fp4.dtype == torch.bfloat16 and bf16.abs().amax() > 0
    # GEMM2 accumulates with atomics: two BF16 runs differ by ~2.5e-4 of the
    # norm; a scale read for the wrong rows is off by ~0.5.
    error = ((fp4.float() - bf16.float()).norm() / bf16.float().norm()).item()
    assert error < 1e-3, f"relative error {error:.3e}"
