# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

from __future__ import annotations

import torch
from atomrl.engine.inference.fp8_weight_quantizer import (
    _per_block_cast_to_fp8,
)
from vllm.utils.deep_gemm import per_block_cast_to_fp8

torch.manual_seed(17)
weight = (torch.randn(257, 259, device="cuda", dtype=torch.bfloat16) * 3).contiguous()
actual_weight, actual_scale = _per_block_cast_to_fp8(weight)
reference_weight, reference_scale = per_block_cast_to_fp8(
    weight,
    block_size=[128, 128],
    use_ue8m0=False,
)

scale_error = (actual_scale - reference_scale).abs().max().item()
weight_mismatch = (
    (actual_weight.view(torch.uint8) != reference_weight.view(torch.uint8))
    .float()
    .mean()
    .item()
)
actual_reconstructed = (
    actual_weight.float()
    * actual_scale.repeat_interleave(
        128,
        dim=0,
    ).repeat_interleave(
        128, dim=1
    )[: weight.shape[0], : weight.shape[1]]
)
reference_reconstructed = (
    reference_weight.float()
    * reference_scale.repeat_interleave(
        128,
        dim=0,
    ).repeat_interleave(
        128, dim=1
    )[: weight.shape[0], : weight.shape[1]]
)

summary = {
    "actual_dtype": str(actual_weight.dtype),
    "reference_dtype": str(reference_weight.dtype),
    "scale_max_abs_error": scale_error,
    "weight_byte_mismatch_fraction": weight_mismatch,
    "actual_reconstruction_max_abs_error": (actual_reconstructed - weight.float())
    .abs()
    .max()
    .item(),
    "reference_reconstruction_max_abs_error": (reference_reconstructed - weight.float())
    .abs()
    .max()
    .item(),
}
print(summary)
assert actual_weight.dtype == reference_weight.dtype
torch.testing.assert_close(actual_scale, reference_scale, rtol=0, atol=0)
assert weight_mismatch <= 0.01
