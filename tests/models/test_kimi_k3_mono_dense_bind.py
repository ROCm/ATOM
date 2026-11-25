# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Which dense-projection layouts the K3 mono kernels bind, and which they refuse.

The kernels read a dense weight one of two ways: plain bf16, or the production
per-token FP8 plane in place (E4M3 in AITER's (16, 16) preshuffle, rows padded
to the GEMM's 128-row tile, an fp32 scale a padded row). Anything else the
weight's records describe would be indexed as one of those and read wrong, with
no error, so ``_dense`` has to refuse it at bind.
The binder compares quant types by name, so these run without AITER. So does
the check of K1's f_b addresses (``kda_plan``) against the preshuffle written
out in ``test_mono_plan``.
"""

from types import SimpleNamespace

import pytest
import torch

from atom.models.kimi_k3.mono import kda_plan
from atom.models.kimi_k3.mono import weights as W
from atom.models.kimi_k3.mono.config import EXPERTS, HEAD_DIM, HIDDEN, Dims
from atom.mono.runtime.consensus import MonoUnsupported

FP8 = torch.float8_e4m3fn
SHAPE = (200, 256)
PADDED = 256
# AITER's QuantType as the binder reads it
PER_TOKEN = SimpleNamespace(name="per_Token")
PER_1X128 = SimpleNamespace(name="per_1x128")


def fp8_linear(rows=PADDED, k=SHAPE[1], weight=None, **overrides):
    weight = torch.nn.Parameter(
        torch.zeros(rows, k, dtype=FP8) if weight is None else weight,
        requires_grad=False,
    )
    weight.is_shuffled = overrides.pop("is_shuffled", True)
    fields = {
        "weight": weight,
        "params_dtype": weight.dtype,
        "quant_type": PER_TOKEN,
        "bias": None,
        "weight_scale": torch.ones(weight.shape[0], 1, dtype=torch.float32),
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def test_bf16_binds_without_a_scale():
    weight = torch.nn.Parameter(torch.zeros(SHAPE, dtype=torch.bfloat16))
    w, scale = W._dense(SimpleNamespace(weight=weight, bias=None), "proj", SHAPE)
    assert scale is None
    assert w.data_ptr() == weight.data_ptr()


@pytest.mark.parametrize("padded", [False, True])
def test_per_token_fp8_binds_in_place(padded):
    # ATOM records the logical rows of a plane it padded to the GEMM's tile;
    # one already a multiple of 128 rows is held as is
    record = {"is_output_padded": True, "_output_size_before_padding": SHAPE[0]}
    module = fp8_linear(**(record if padded else {}))
    w, scale = W._dense(module, "proj", SHAPE if padded else (PADDED, SHAPE[1]))
    assert w.data_ptr() == module.weight.data_ptr()
    assert scale.data_ptr() == module.weight_scale.data_ptr()
    assert scale.shape == (PADDED,)


@pytest.mark.parametrize(
    "module, shape, why",
    [
        (fp8_linear(is_shuffled=False), SHAPE, "preshuffle"),
        (fp8_linear(quant_type=PER_1X128), SHAPE, "per-token"),
        (fp8_linear(rows=SHAPE[0]), SHAPE, "padded to 16"),
        (fp8_linear(rows=192), SHAPE, "weight shape"),
        (fp8_linear(k=192), (SHAPE[0], 192), "multiple of 128"),
        (
            fp8_linear(is_output_padded=True, _output_size_before_padding=SHAPE[0] + 1),
            SHAPE,
            "output padding",
        ),
        (fp8_linear(bias=torch.zeros(PADDED)), SHAPE, "bias"),
        (
            fp8_linear(weight=torch.zeros(SHAPE[1], PADDED, dtype=FP8).t()),
            SHAPE,
            "not contiguous",
        ),
        (fp8_linear(weight_scale=None), SHAPE, "scale"),
        (fp8_linear(weight_scale=torch.ones(SHAPE[0])), SHAPE, "scale"),
        (
            fp8_linear(weight_scale=torch.ones(PADDED, dtype=torch.bfloat16)),
            SHAPE,
            "scale",
        ),
        (
            fp8_linear(weight_scale=torch.ones(2 * PADDED)[::2]),
            SHAPE,
            "scale not contiguous",
        ),
    ],
)
def test_unreadable_layouts_are_refused(module, shape, why):
    with pytest.raises(MonoUnsupported, match=why):
        W._dense(module, "proj", shape)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float8_e4m3fnuz])
def test_other_dtypes_are_refused(dtype):
    # E4M3FNUZ is FP8 too (gfx942's), but not the E4M3 the kernels decode
    module = fp8_linear(weight=torch.zeros(PADDED, SHAPE[1], dtype=dtype))
    with pytest.raises(MonoUnsupported, match="bf16 or preshuffled"):
        W._dense(module, "proj", SHAPE)


def _dense_as(module, name, shape):
    """`_dense` stubbed to the layout its module stands for: a scale or None."""
    return None, module.scale


AS_FP8 = SimpleNamespace(scale=torch.ones(1))
AS_BF16 = SimpleNamespace(scale=None)


@pytest.mark.parametrize("fp8", ["in_proj", "f_b_proj"])
def test_a_kda_layer_with_one_fp8_projection_is_refused(monkeypatch, fp8):
    # one K1 build reads in_proj and f_b_proj one way
    monkeypatch.setattr(W, "_dense", _dense_as)
    dims = Dims(8)
    attn = SimpleNamespace(
        _in_proj_fused=True,
        num_local_heads=dims.heads,
        head_dim=HEAD_DIM,
        conv_kernel_size=4,
        _kda_gate_lower_bound=-5.0,
        **{p: AS_FP8 if p == fp8 else AS_BF16 for p in ("in_proj", "f_b_proj")},
    )
    layer = SimpleNamespace(is_linear_attn=True, self_attn=attn)
    with pytest.raises(MonoUnsupported, match="not both bf16 or both FP8"):
        W.bind_kda(layer, 1, dims)


@pytest.mark.parametrize("bf16", ["o_proj", "gate_up_proj", "down_proj"])
def test_a_moe_layer_with_mixed_projections_is_refused(monkeypatch, bf16):
    # one K2 build reads o_proj and both shared-expert projections one way
    monkeypatch.setattr(W, "_dense", _dense_as)
    monkeypatch.setattr(W, "_experts", lambda *a: (None,) * 4)
    monkeypatch.setattr(W, "_mxfp4_to_bf16", lambda *a, **k: None)
    proj = {
        p: AS_BF16 if p == bf16 else AS_FP8
        for p in ("o_proj", "gate_up_proj", "down_proj")
    }
    moe = SimpleNamespace(
        use_latent_moe=True,
        shared_experts=SimpleNamespace(
            gate_up_proj=proj["gate_up_proj"], down_proj=proj["down_proj"]
        ),
        routed_expert_norm=object(),
        gate=SimpleNamespace(weight=torch.zeros(EXPERTS, HIDDEN, dtype=torch.bfloat16)),
        routed_expert_up_proj=None,
    )
    layer = SimpleNamespace(
        block_sparse_moe=moe, self_attn=SimpleNamespace(o_proj=proj["o_proj"])
    )
    with pytest.raises(MonoUnsupported, match="not all bf16 or all FP8"):
        W.bind_moe(layer, 1, Dims(8), 0)


def test_the_kernels_index_through_these():
    # what the test below checks is what K1 calls
    kda = pytest.importorskip(
        "atom.models.kimi_k3.mono.kernels.attention.kda", exc_type=ImportError
    )
    assert kda.fb_w8_dword is kda_plan.fb_w8_dword
    assert kda.HD == HEAD_DIM
    for h, wave, lane in ((0, 0, 0), (1, 7, 63), (3, 2, 17)):
        c = {"lane": lane, "wave": wave, "w8": True}
        assert kda.fb_offset(c, h) == kda_plan.fb_w8_offset(h, wave, lane)


def test_fb_offset_indexes_the_production_preshuffle():
    # kda_gates' f_b operands as fb_loads reads them (fb_offset's FP8 branch,
    # fb_w8_dword's 16 B halves, then operand q's 2 dwords of half q / 2): wave
    # w's rows 16 w .. of head h, lane group g's K columns 32 g + 8 q
    from tests.test_mono_plan import preshuffled

    hd = HEAD_DIM
    plain, flat = preshuffled(2 * hd, hd)
    words = flat.view(torch.int32)
    for h in range(2):
        for wave in range(hd // 16):
            for lane in range(64):
                base = kda_plan.fb_w8_offset(h, wave, lane)
                ats = (kda_plan.fb_w8_dword(base, j) for j in range(2))
                halves = [words[at : at + 4] for at in ats]
                for q in range(4):
                    op = torch.stack([halves[q // 2][q % 2 * 2 + d] for d in range(2)])
                    row = h * hd + 16 * wave + lane % 16
                    col = 32 * (lane // 16) + 8 * q
                    assert torch.equal(op.view(torch.uint8), plain[row, col : col + 8])
