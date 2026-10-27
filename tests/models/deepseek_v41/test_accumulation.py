# SPDX-License-Identifier: MIT
"""Block-scale cancellation must survive both GEMM and split-K reduction."""

import pytest
import torch

if not torch.cuda.is_available():
    # Before the import, which reaches Triton through `blockscale`. As a
    # `pytestmark` this ran after it, so a CPU runner failed collection
    # instead of skipping.
    pytest.skip("ROCm GPU required", allow_module_level=True)

from atom.model_ops.blockscale import native_quant_linear


@pytest.mark.parametrize("fp4", [False, True])
@pytest.mark.parametrize("splits", [1, 3])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_scaled_block_cancellation_preserves_small_term(fp4, splits, dtype):
    # Three groups contribute 2**23, 1, -2**23. Rounding either the block
    # accumulation or the split reduction prematurely loses the answer, 1.
    #
    # 2**23 is the widest spread an FP32 accumulator carries exactly: the
    # significand is 24 bits, so 2**23 + 1 is representable and 2**24 + 1 is
    # not. The spread was 2**24 while the sums were kept in FP64.
    activation = torch.zeros(3, 96, device="cuda")
    activation[:, ::32] = 1
    activation = activation.to(torch.float8_e4m3fn)
    a_scale = torch.ones(3, 3, device="cuda").to(torch.float8_e8m0fnu)
    if fp4:
        packed = torch.zeros(32, 48, dtype=torch.uint8, device="cuda")
        packed[:, 0], packed[:, 16], packed[:, 32] = 2, 2, 10
        weight = packed.view(torch.float4_e2m1fn_x2)
        group_rows = 1
    else:
        weight = torch.zeros(32, 96, device="cuda")
        weight[:, 0], weight[:, 32], weight[:, 64] = 1, 1, -1
        weight = weight.to(torch.float8_e4m3fn)
        group_rows = 32
    weight_scale = (
        torch.tensor([2**23, 1, 2**23], device="cuda", dtype=torch.float32)
        .expand(32 // group_rows, -1)
        .contiguous()
        .to(torch.float8_e8m0fnu)
    )
    actual = native_quant_linear(
        activation,
        weight,
        weight_scale,
        x_scale=a_scale,
        weight_group_rows=group_rows,
        dtype=dtype,
        split_k=splits,
    )
    torch.testing.assert_close(actual, torch.ones_like(actual), rtol=0, atol=0)


# Decoding an E8M0 code happens before the split reduction and before the
# output cast, so the boundary codes sweep on their own; the balanced code
# carries the split/dtype cross for the paths behind it.
@pytest.mark.parametrize("fp4", [False, True])
@pytest.mark.parametrize(
    "splits,dtype,activation_code,weight_code",
    [
        (1, torch.bfloat16, 127, 127),
        (4, torch.bfloat16, 127, 127),
        (1, torch.float32, 127, 127),
        (4, torch.float32, 127, 127),
        (1, torch.bfloat16, 0, 254),
        (1, torch.bfloat16, 254, 0),
        (1, torch.bfloat16, 128, 0),
        (1, torch.bfloat16, 0, 128),
        (1, torch.bfloat16, 255, 127),
        (1, torch.bfloat16, 127, 255),
    ],
)
def test_e8m0_scale_boundaries(fp4, splits, dtype, activation_code, weight_code):
    # E8M0 has no zero: code 0 is 2**-127 and code 255 is NaN. Balancing
    # minimum and maximum scales makes a dropped minimum scale observable.
    activation = torch.ones(3, 128, device="cuda").to(torch.float8_e4m3fn)
    if fp4:
        # Two E2M1 values of one per byte, low nibble first.
        weight = torch.full((32, 64), 0x22, device="cuda", dtype=torch.uint8).view(
            torch.float4_e2m1fn_x2
        )
        group_rows = 1
    else:
        weight = torch.ones(32, 128, device="cuda").to(torch.float8_e4m3fn)
        group_rows = 32
    a_scale = torch.full(
        (3, 4), activation_code, device="cuda", dtype=torch.uint8
    ).view(torch.float8_e8m0fnu)
    weight_scale = torch.full(
        (32 // group_rows, 4), weight_code, device="cuda", dtype=torch.uint8
    ).view(torch.float8_e8m0fnu)
    actual = native_quant_linear(
        activation,
        weight,
        weight_scale,
        x_scale=a_scale,
        weight_group_rows=group_rows,
        dtype=dtype,
        split_k=splits,
    )
    if 255 in (activation_code, weight_code):
        assert torch.isnan(actual).all()
    else:
        expected = torch.full_like(
            actual, 128 * 2.0 ** (activation_code + weight_code - 254)
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
