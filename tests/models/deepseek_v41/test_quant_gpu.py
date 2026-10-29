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


@pytest.mark.parametrize("rows", [0, 1, 120, 2048])
@pytest.mark.parametrize("limit", [0.0, 10.0])
def test_silu_mul_quantize_fp8_matches_activation_then_quant(rows, limit):
    """The shared expert's fused activation + quant is `quantize_fp8` of what
    `aiter.silu_and_mul` stores: same E8M0 scales, and FP8 bytes equal but for
    the rare one-code difference `exp`'s rounding can make on a BF16 tie."""
    from aiter import silu_and_mul

    from atom.model_ops.blockscale import quantize_fp8, silu_mul_quantize_fp8

    torch.manual_seed(rows)
    inter = 2304  # V4.1's moe_intermediate_size: 72 groups, not a power of two
    x = (torch.randn(rows, 2 * inter, device="cuda") * 4).bfloat16()
    if rows:
        x[0] = 0  # an all-zero row: the amax floor's scale
    out = torch.empty(rows, inter, device="cuda", dtype=torch.bfloat16)
    if rows:
        silu_and_mul(out, x, limit)
    ref_q, ref_s = quantize_fp8(out)
    q, s = silu_mul_quantize_fp8(x, limit)
    assert q.shape == (rows, inter) and s.shape == (rows, inter // 32)
    assert torch.equal(s.view(torch.uint8), ref_s.view(torch.uint8))
    diff = (q.view(torch.uint8).int() - ref_q.view(torch.uint8).int()).abs()
    assert diff.numel() == 0 or (diff.max() <= 1 and (diff > 0).float().mean() < 1e-5)


@pytest.mark.parametrize("rows", [1, 120, 192, 3000])
@pytest.mark.parametrize("with_post", [False, True])
def test_triton_seam_quant_is_quantize_fp8_of_its_bf16(rows, with_post):
    """`pre_delayed(..., fuse_quant=True)` on the Triton seam: its MXFP8 copy is
    bit-equal to quantizing the BF16 normed input it returns."""
    from atom.model_ops.blockscale import quantize_fp8
    from atom.model_ops.deepseek_v41.mhc_pre_delayed import (
        pre_delayed,
        triton_quant_out_supported,
    )

    if not triton_quant_out_supported():
        pytest.skip("AITER's Triton mHC seam has no quant_out")
    torch.manual_seed(rows)
    hidden = 5120
    residual = torch.randn(rows, 4, hidden, device="cuda", dtype=torch.bfloat16)
    args = (
        residual,
        torch.rand(rows, 4, device="cuda"),
        torch.randn(24, 4 * hidden, device="cuda") * 0.01,
        torch.rand(3, device="cuda"),
        torch.randn(24, device="cuda") * 0.1,
        (torch.rand(hidden, device="cuda") * 2).bfloat16(),
    )
    options = {
        "rms_eps": 1e-6,
        "hc_eps": 1e-6,
        "sinkhorn_iters": 20,
        "post_mult": 2.0,
        "norm_eps": 1e-6,
    }
    if with_post:
        options.update(
            sublayer_output=torch.randn(
                rows, hidden, device="cuda", dtype=torch.bfloat16
            ),
            post_mix=torch.rand(rows, 4, device="cuda"),
            combination=torch.rand(rows, 4, 4, device="cuda"),
        )
    plain = pre_delayed(*args, **options)
    fused = pre_delayed(*args, fuse_quant=True, **options)
    bf16, q, s = fused[1]
    assert torch.equal(bf16, plain[1])
    for a, b in zip(fused[2:], plain[2:]):
        assert torch.equal(a, b)
    ref_q, ref_s = quantize_fp8(bf16)
    assert torch.equal(q.view(torch.uint8), ref_q.view(torch.uint8))
    assert torch.equal(s.view(torch.uint8), ref_s.view(torch.uint8))


def test_shared_expert_reads_the_seams_quant_and_fuses_its_own():
    """With a native FP8 gate_up and w2, the shared expert takes the seam's
    MXFP8 input as is (when it covers the call's rows) and hands w2 the fused
    activation's quant."""
    from atom.model_ops.blockscale import quantize_fp8, silu_mul_quantize_fp8
    from atom.models.deepseek_v4 import Expert

    calls = {}

    def gate_up(x, x_scale=None):
        calls["gate_up"] = (x, x_scale)
        return combined

    def w2(x, x_scale=None):
        calls["w2"] = (x, x_scale)
        return torch.zeros(x.shape[0], 64, device="cuda")

    torch.manual_seed(0)
    rows, inter = 7, 256
    combined = (torch.randn(rows, 2 * inter, device="cuda") * 3).bfloat16()
    expert = Expert.__new__(Expert)
    torch.nn.Module.__init__(expert)
    expert.gate_up_proj = gate_up
    expert.w2 = w2
    w2.native_a8_group_rows = 32
    expert.swiglu_limit = 10.0
    expert.use_fused_clamp_act_mul = False

    x = torch.randn(rows, 64, device="cuda", dtype=torch.bfloat16)
    expert.quantized_input = quantize_fp8(x)
    expert(x)
    assert calls["gate_up"][0] is expert.quantized_input[0]
    assert calls["gate_up"][1] is expert.quantized_input[1]
    want_q, want_s = silu_mul_quantize_fp8(combined, 10.0)
    assert torch.equal(calls["w2"][0].view(torch.uint8), want_q.view(torch.uint8))
    assert torch.equal(calls["w2"][1].view(torch.uint8), want_s.view(torch.uint8))

    # A quant of other rows (a padded pass) is not this call's input.
    expert.quantized_input = quantize_fp8(x[:3])
    expert(x)
    assert calls["gate_up"][0] is x and calls["gate_up"][1] is None
