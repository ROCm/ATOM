# SPDX-License-Identifier: MIT
"""Which GEMMs the startup trace covers, and at which M it launches them."""

import torch
from torch import nn

from atom.model_ops.flydsl_gemm_warmup import first_m_per_kernel, mxscale_gemm_shapes


class _Linear(nn.Module):
    def __init__(self, weight, scale):
        super().__init__()
        self.register_buffer("weight", weight)
        self.register_buffer("weight_scale", scale)


def _fp8(*shape):
    return torch.empty(shape, dtype=torch.float8_e4m3fn)


def test_shapes_take_fp8_weights_with_128x128_byte_scales_only():
    model = nn.ModuleList(
        [
            _Linear(_fp8(7168, 16384), torch.empty((56, 128), dtype=torch.uint8)),
            _Linear(_fp8(1536, 7168), torch.empty((12, 56), dtype=torch.uint8)),
            # A second layer of the same shape adds nothing.
            _Linear(_fp8(1536, 7168), torch.empty((12, 56), dtype=torch.uint8)),
            # FP32 block scales take another GEMM.
            _Linear(_fp8(2048, 7168), torch.empty((16, 56), dtype=torch.float32)),
            # A 1x32 MX scale is not a 128x128 block scale.
            _Linear(_fp8(2048, 7168), torch.empty((2048, 224), dtype=torch.uint8)),
            # Expert weights are 3-D.
            _Linear(_fp8(8, 2048, 7168), torch.empty((8, 16, 56), dtype=torch.uint8)),
            nn.Linear(16, 16),
        ]
    )
    assert mxscale_gemm_shapes(model) == {
        (7168, 16384): (torch.float8_e4m3fn, torch.uint8, (56, 128)),
        (1536, 7168): (torch.float8_e4m3fn, torch.uint8, (12, 56)),
    }


def test_each_tuned_kernel_launches_once_at_its_smallest_m():
    rows = {1: "a", 2: "a", 3: None, 4: "b", 5: "a", 6: "c", 7: "b"}

    def lookup(m):
        name = rows.get(m)
        return None if name is None else {"kernelName": name}

    assert first_m_per_kernel(lookup, 7) == [1, 4, 6]
    assert first_m_per_kernel(lookup, 3) == [1]
    assert first_m_per_kernel(lambda m: None, 7) == []
