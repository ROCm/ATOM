# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""SiTUv2-and-mul addresses rows past 2**31 elements.

Kimi-K3's layer-1 dense FFN is 33792 wide, so at TP1 (each stage of a PP8 x TP1
prefill) its gate_up output row holds 2 x 33792 elements and a batch of more
than 31,775 tokens puts `row * stride` past 2**31. The kernel formed that
offset in 32 bits and faulted on a 32K-token prefill batch.
"""

import pytest
import torch
from conftest import gpu_stack_import

D = 33792
ROWS = 2**31 // (2 * D) + 25  # the last rows start past 2**31 elements


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton kernel needs a GPU")
def test_rows_past_two_to_the_31_elements():
    act = gpu_stack_import("atom.model_ops.kimi_k3.activations")
    torch.manual_seed(0)
    x = torch.empty(ROWS, 2 * D, dtype=torch.bfloat16, device="cuda")
    x[:-4].zero_()
    tail = torch.randn(4, 2 * D, dtype=torch.bfloat16, device="cuda")
    x[-4:] = tail
    assert x[-1].data_ptr() - x.data_ptr() >= 2**31 * x.element_size()

    y = act.situ_and_mul(x, beta=25.0, linear_beta=None)

    want = act._situ_and_mul_torch(tail, 25.0, None)
    torch.testing.assert_close(y[-4:], want, rtol=1e-2, atol=1e-2)
