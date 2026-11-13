# SPDX-License-Identifier: MIT
"""GPU contract checks against independent PyTorch oracles and checkpoint slices."""

import json
import os
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from . import oracle_kernels as oracle

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="ROCm GPU required"
)


@pytest.mark.parametrize("shape", [(1, 32), (3, 288), (17, 5120), (2, 3, 128), (0, 32)])
def test_fp8_quantization_bytes_and_qat(shape):
    from atom.model_ops.blockscale import quantize_fp8
    from atom.model_ops.blockscale_kernels.quantization import AMAX_FLOOR

    torch.manual_seed(312)
    x = torch.randn(shape, dtype=torch.bfloat16)
    if x.numel():
        x.reshape(-1)[: min(32, x.numel())] = 0
    q, s = quantize_fp8(x.cuda())
    # the zero group's scale is the floor's: ATOM's, not the reference's 1e-4
    floor = AMAX_FLOOR.value
    ref_q, ref_s = oracle.act_quant(
        x, 32, "ue8m0", torch.float8_e8m0fnu, amax_floor=floor
    )
    assert torch.equal(q.cpu().view(torch.uint8), ref_q.view(torch.uint8))
    assert torch.equal(s.cpu().view(torch.uint8), ref_s.view(torch.uint8))
    expected = oracle.act_quant(x.clone(), 32, "ue8m0", inplace=True, amax_floor=floor)
    assert torch.equal(quantize_fp8(x.cuda(), dequantize=True).cpu(), expected)


@pytest.mark.parametrize(
    "scale_dtype,group", [(torch.float8_e8m0fnu, 32), (torch.float8_e4m3fn, 16)]
)
def test_fp4_quantization_bytes_midpoints_and_qat(scale_dtype, group):
    from atom.model_ops.blockscale import quantize_fp4

    torch.manual_seed(777)
    x = torch.randn((11, 512), dtype=torch.bfloat16)
    x[0] = 0
    # A maximum of six makes scale=1; the rows cover every rounding midpoint
    # and both signs, including signed zero.
    values = torch.tensor(
        [
            0,
            -0.0,
            0.25,
            0.75,
            1.25,
            1.75,
            2.5,
            3.5,
            5.0,
            6.0,
            -0.25,
            -0.75,
            -1.25,
            -1.75,
            -2.5,
            -6.0,
        ]
    )
    x[1] = values.repeat(32)
    x[2] *= 2**-14
    q, s = quantize_fp4(x.cuda(), group_size=group, scale_dtype=scale_dtype)
    rq, rs = oracle.fp4_act_quant(x, group, scale_dtype=scale_dtype)
    assert torch.equal(q.cpu().view(torch.uint8), rq.view(torch.uint8))
    assert torch.equal(s.cpu().view(torch.uint8), rs.view(torch.uint8))
    expected = oracle.fp4_act_quant(
        x.clone(), group, inplace=True, scale_dtype=scale_dtype
    )
    actual = quantize_fp4(
        x.cuda(), group_size=group, scale_dtype=scale_dtype, dequantize=True
    ).cpu()
    assert torch.equal(actual, expected)


@pytest.mark.parametrize(
    "fp4,n,k,m,splits",
    [
        (False, 96, 288, 1, 1),
        (False, 96, 288, 3, 4),
        (False, 288, 5120, 1, 16),
        (False, 96, 1280, 33, 1),
        (False, 33, 96, 7, 2),
        (True, 96, 288, 1, 1),
        (True, 96, 288, 3, 4),
        (True, 288, 5120, 1, 16),
        (True, 96, 2304, 33, 1),
    ],
)
def test_native_gemm_nonuniform_scales_and_split_tails(fp4, n, k, m, splits):
    from atom.model_ops.blockscale import native_quant_linear

    torch.manual_seed(312)
    x = torch.randn(m, k, dtype=torch.bfloat16)
    a, a_scale = oracle.act_quant(x, 32, "ue8m0", torch.float8_e8m0fnu)
    if fp4:
        b, b_scale = oracle.fp4_act_quant(torch.randn(n, k, dtype=torch.bfloat16))
        b_values = oracle.unpack_fp4(b)
        group_n = 1
    else:
        b = (torch.randn(n, k) * 16).to(torch.float8_e4m3fn)
        b_scale = torch.empty((n + 31) // 32, k // 32, dtype=torch.float8_e8m0fnu)
        b_values = b.float()
        group_n = 32
    b_scale = torch.exp2(torch.randint(-7, 1, b_scale.shape).float()).to(
        torch.float8_e8m0fnu
    )
    full_scale = (
        b_scale.float().repeat_interleave(group_n, 0)[:n].repeat_interleave(32, 1)
    )
    a_values = (
        a.float().reshape(m, k // 32, 32) * a_scale.float()[:, :, None]
    ).reshape(m, k)
    expected = F.linear(a_values, b_values * full_scale)
    actual = native_quant_linear(
        x.cuda(),
        b.cuda(),
        b_scale.cuda(),
        weight_group_rows=group_n,
        dtype=torch.float32,
        split_k=splits,
    ).cpu()
    # The FP8 GEMM issues the CDNA4 v_mfma_scale_f32_*_f8f6f4, whose block
    # accumulator carries about 15 bits against the block's largest term --
    # not the 24 an FP32 sum would. The resulting error scales with the
    # result, not with the element, so the bound is tied to the magnitude of
    # the output: measured 1.6e-5 to 1.8e-5 of peak across this file's cases,
    # and 5e-5 leaves room. A per-element rtol cannot express this -- rows
    # that cancel to near zero carry the same absolute error as the rest.
    torch.testing.assert_close(
        actual, expected, rtol=3e-5, atol=5e-5 * expected.abs().max().item()
    )
    # The pre-quantized entry must use exactly the supplied scale layout.
    supplied = native_quant_linear(
        a.cuda(),
        b.cuda(),
        b_scale.cuda(),
        x_scale=a_scale.cuda(),
        weight_group_rows=group_n,
        dtype=torch.float32,
        split_k=splits,
    ).cpu()
    assert torch.equal(actual, supplied)


@pytest.mark.parametrize(
    "prefix,fp4", [("layers.2.attn.wq_a", False), ("layers.2.ffn.experts.0.w1", True)]
)
def test_real_weight_slice_gemm(prefix, fp4):
    from safetensors import safe_open

    from atom.model_ops.blockscale import native_quant_linear

    directory = os.environ.get("ATOM_DSV41_REFERENCE")
    if not directory:
        pytest.skip("Set ATOM_DSV41_REFERENCE for real checkpoint tests")
    root = Path(directory)
    index = json.loads((root / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    with safe_open(root / index[prefix + ".weight"], framework="pt") as handle:
        b = handle.get_slice(prefix + ".weight")[:96, : 144 if fp4 else 288]
    with safe_open(root / index[prefix + ".scale"], framework="pt") as handle:
        s = handle.get_slice(prefix + ".scale")[: 96 if fp4 else 3, :9]
    b = b.contiguous().view(torch.float4_e2m1fn_x2) if fp4 else b.contiguous()
    torch.manual_seed(188)
    x = torch.randn(7, 288, dtype=torch.bfloat16)
    aq, a_s = oracle.act_quant(x, 32, "ue8m0", torch.float8_e8m0fnu)
    a_values = (aq.float().reshape(7, 9, 32) * a_s.float()[:, :, None]).reshape(7, 288)
    group_n = 1 if fp4 else 32
    weight_values = oracle.unpack_fp4(b) if fp4 else b.float()
    weight_values *= s.float().repeat_interleave(group_n, 0).repeat_interleave(32, 1)
    expected = F.linear(a_values, weight_values)
    actual = native_quant_linear(
        x.cuda(), b.cuda(), s.cuda(), weight_group_rows=group_n, dtype=torch.float32
    ).cpu()
    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-5)


def _preshuffled(weight):
    from atom.model_ops.utils import shuffle_weights

    weight = torch.nn.Parameter(weight.clone(), requires_grad=False)
    shuffle_weights(weight)
    return weight.data


def _oracle(x, b, b_scale):
    m, k = x.shape
    a, a_scale = oracle.act_quant(x, 32, "ue8m0", torch.float8_e8m0fnu)
    a_values = (
        a.float().reshape(m, k // 32, 32) * a_scale.float()[:, :, None]
    ).reshape(m, k)
    full_scale = b_scale.float().repeat_interleave(32, 0).repeat_interleave(32, 1)
    return F.linear(a_values, b.float() * full_scale)


@pytest.mark.parametrize("m", [1, 6, 24, 300])
def test_preshuffled_gemm_zero_padded_k(m):
    """AITER's preshuffled group32 FP8 GEMM steps K in 64s; a K it cannot step
    (the V4.1 shared expert's down projection at TP8: 288) runs on weights and
    activations zero-padded to the next multiple (``GROUP32_K_STEP``). TP4's K
    (576) steps and runs unpadded. The padding's products are exact zeros: the
    padded GEMM equals the K = 320 GEMM whose last 32 weight columns are zero
    under any activations there, bit for bit, and the fp32 oracle."""
    from atom.model_ops.blockscale import (
        GROUP32_K_STEP,
        native_quant_linear,
        pad_group32_k,
    )

    def run(x, b, b_scale):
        k = -(-b.shape[1] // GROUP32_K_STEP) * GROUP32_K_STEP
        b, b_scale = pad_group32_k(b, b_scale, k)
        return native_quant_linear(
            x.cuda(),
            _preshuffled(b.cuda()),
            b_scale.cuda(),
            weight_group_rows=32,
            weight_preshuffled=True,
            dtype=torch.bfloat16,
        ).cpu()

    def tolerance(expected):
        return {"rtol": 1e-2, "atol": 1e-2 * expected.abs().max().item()}

    torch.manual_seed(288)
    n = 4096

    def operands(k):
        x = torch.randn(m, k, dtype=torch.bfloat16)
        b = (torch.randn(n, k) * 16).to(torch.float8_e4m3fn)
        b_scale = torch.exp2(torch.randint(-7, 1, (n // 32, k // 32)).float()).to(
            torch.float8_e8m0fnu
        )
        return x, b, b_scale

    x, b, b_scale = operands(576)  # TP4: no padding
    expected = _oracle(x, b, b_scale)
    torch.testing.assert_close(
        run(x, b, b_scale).float(), expected, **tolerance(expected)
    )

    x, b, b_scale = operands(288)  # TP8
    with pytest.raises(
        (RuntimeError, ValueError)
    ):  # the symptom: no kernel steps K = 288
        native_quant_linear(
            x.cuda(),
            _preshuffled(b.cuda()),
            b_scale.cuda(),
            weight_group_rows=32,
            weight_preshuffled=True,
            dtype=torch.bfloat16,
        )
    # activations of a K the GEMM steps are never padded: 256 against a K of 320
    # or 288 is refused (only a K it cannot step takes the padded copy of itself)
    b320, s320 = pad_group32_k(b, b_scale, 320)
    for w, s in ((b320, s320), (b, b_scale)):
        with pytest.raises(ValueError, match="Invalid matrix dimensions"):
            native_quant_linear(
                torch.randn(m, 256, dtype=torch.bfloat16).cuda(),
                _preshuffled(w.cuda()),
                s.cuda(),
                weight_group_rows=32,
                weight_preshuffled=True,
                dtype=torch.bfloat16,
            )
    padded = run(x, b, b_scale)
    expected = _oracle(x, b, b_scale)
    torch.testing.assert_close(padded.float(), expected, **tolerance(expected))
    # the same GEMM at K = 320 with zero weight columns under nonzero activations
    x320 = torch.cat([x, torch.randn(m, 32, dtype=torch.bfloat16)], dim=1)
    assert torch.equal(padded, run(x320, b320, s320))
