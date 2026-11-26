# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""`hip_copy.copy_async` declines what a plain memcpy cannot do, so the caller
keeps `Tensor.copy_` for it."""

import pytest
import torch

from atom.utils import hip_copy


@pytest.mark.parametrize(
    "dst,src",
    [
        # host -> host: not a D2H, whatever the shapes
        (torch.zeros(8, dtype=torch.int32), torch.zeros(8, dtype=torch.int32)),
        (torch.zeros(8, dtype=torch.int32), torch.zeros(8, dtype=torch.int64)),
        (torch.zeros(8, dtype=torch.int32), torch.zeros(9, dtype=torch.int32)),
        (
            torch.zeros(8, 2, dtype=torch.int32).t(),
            torch.zeros(2, 8, dtype=torch.int32),
        ),
    ],
)
def test_hip_copy_declines_what_memcpy_cannot_do(dst, src):
    assert hip_copy.copy_async(dst, src, hip_copy.D2H) is False
