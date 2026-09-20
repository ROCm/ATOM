# SPDX-License-Identifier: MIT
"""`per_1x32` fused RMSNorm+quant carrying E4M3 values, not packed E2M1.

DeepSeek-V4.1's `native_quant_config` is `per_1x32` with an FP8 value dtype, a
combination the fused dispatch used to read as MXFP4 and answer with a
half-width output. What this pins is that the FP8 branch produces the
`(values, scales)` pair `native_quant_linear` would have produced for itself,
because that is the whole point of fusing it -- the linear is supposed to skip
its own `quantize_fp8` and read these.

The widths below straddle every bucket of aiter's shape dispatch, and 5120 is
there on purpose: that bucket's default is 24 elements per thread, which no
quant group divides, so a grouped norm at DeepSeek-V4.1's own `hidden_size`
used to abort inside the kernel. It now falls back to a divisor-friendly shape,
and this is where that fallback is checked to compute the same answer as the
rest.
"""

import pytest
import torch

pytest.importorskip("aiter")

from aiter import QuantType

from atom.model_ops.blockscale import quantize_fp8
from atom.model_ops.layernorm import (
    _aiter_rms_quant,
    _is_mxfp8,
    rmsnorm2d_fwd_,
    rmsnorm2d_fwd_with_add_,
)


def test_mxfp8_is_selected_by_the_value_dtype_not_the_quant_type():
    """Both MX formats are `per_1x32`; only the value width separates them."""
    per_1x32 = QuantType.per_1x32.value
    assert _is_mxfp8(per_1x32, torch.float8_e4m3fn)
    assert _is_mxfp8(per_1x32, torch.float8_e4m3fnuz)
    assert not _is_mxfp8(per_1x32, torch.float4_e2m1fn_x2)
    # A per-block FP8 layer also has an FP8 value dtype and must not be caught.
    assert not _is_mxfp8(QuantType.per_1x128.value, torch.float8_e4m3fn)


def reference_norm(x, weight, eps, residual=None):
    """The FP32 value both paths are approximations of."""
    f32 = x.float() if residual is None else x.float() + residual.float()
    return f32 * torch.rsqrt(f32.pow(2).mean(-1, keepdim=True) + eps) * weight.float()


def unfused(x, weight, eps, dim, residual=None):
    """The two kernels the fusion replaces, not a third implementation of them.

    A hand-written torch RMSNorm answers a different question: it disagrees with
    the aiter norm by a bf16 ulp here and there, and after an E4M3 cast that
    reads as a one-code value difference in 2% of elements, which looks like a
    quantization bug and is not one. Running the same norm ATOM runs unfused
    leaves the quantization as the only variable.
    """
    if residual is None:
        return rmsnorm2d_fwd_(x, weight, eps, dim), None
    return rmsnorm2d_fwd_with_add_(x, weight, residual, eps, dim)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm GPU required")
@pytest.mark.parametrize("residual", [False, True])
@pytest.mark.parametrize(
    "shape", [(16, 64), (37, 1280), (256, 4096), (128, 5120), (8, 6144), (8, 8192)]
)
def test_fused_mxfp8_matches_rmsnorm_then_quantize_fp8(shape, residual):
    torch.manual_seed(20260918)
    rows, dim = shape
    eps = 1e-6
    x = torch.randn(rows, dim, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(dim, dtype=torch.bfloat16, device="cuda") * 0.1 + 1.0
    res = torch.randn_like(x) if residual else None

    values, scales, res_out = _aiter_rms_quant(
        x, weight, eps, QuantType.per_1x32.value, False, res, torch.float8_e4m3fn
    )
    normed, expected_res = unfused(x, weight, eps, dim, res)
    want_values, want_scales = quantize_fp8(normed)

    assert values.dtype == torch.float8_e4m3fn
    assert values.shape == (rows, dim)
    assert scales.dtype == torch.float8_e8m0fnu
    assert scales.shape == (rows, dim // 32)
    if residual:
        torch.testing.assert_close(res_out, expected_res, rtol=0, atol=0)
    else:
        assert res_out is None

    # The two are NOT bit-identical, and that is the fusion working rather than
    # failing. The unfused pair rounds twice: the norm lands in BF16 memory and
    # the quantizer reads it back, so both the group amax that picks the
    # exponent and a value sitting within a BF16 ulp of an E4M3 bin boundary
    # can fall either way. The fused kernel quantizes the FP32 value it still
    # holds in registers and rounds once.
    #
    # So the claim is not equality. It is that fusing only ever moves an answer
    # toward the FP32 norm, never away -- which double rounding explains and a
    # wrong scale rule, a wrong group or a wrong layout would not.
    exact = reference_norm(x, weight, eps, res)
    exponent_gap = scales.view(torch.uint8).int() - want_scales.view(torch.uint8).int()
    assert exponent_gap.abs().max() <= 1, (
        f"{int((exponent_gap.abs() > 1).sum())} groups differ by more than one "
        "UE8M0 step, which a BF16 amax cannot explain"
    )
    fused = values.float() * scales.float().repeat_interleave(32, dim=-1)
    unfused_q = want_values.float() * want_scales.float().repeat_interleave(32, dim=-1)
    fused_error = (fused - exact).abs()
    unfused_error = (unfused_q - exact).abs()
    worse = fused_error > unfused_error
    assert not bool(worse.any()), (
        f"{int(worse.sum())} elements are further from the FP32 norm after "
        "fusing, so the difference is not the BF16 round trip"
    )
    # And the two do agree wherever the extra rounding could not have mattered,
    # which is most of them -- a fusion that quantized something else entirely
    # would still pass the inequality above by being uniformly closer.
    agree = (fused == unfused_q).float().mean()
    assert agree > 0.9, f"only {agree:.1%} of values survived the fusion unchanged"
