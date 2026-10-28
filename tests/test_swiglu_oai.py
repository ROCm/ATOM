# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from atom.model_ops.swiglu_oai import (
    _launch_config,
    swiglu_oai_split,
)


def test_gfx1250_launch_config():
    assert _launch_config(12, 3072, "gfx1250") == (256, 4)
    assert _launch_config(256, 3072, "gfx1250") == (1024, 4)
    assert _launch_config(256, 2560, "gfx1250") == (512, 4)
    assert _launch_config(2517, 3072, "gfx1250") == (1024, 2)
    assert _launch_config(8192, 3072, "gfx1250") == (1024, 2)
    assert _launch_config(16384, 3072, "gfx1250") == (512, 1)
    assert _launch_config(2517, 3072, "gfx950") == (512, 4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm GPU required")
@pytest.mark.parametrize("rows", [12, 257])
@pytest.mark.parametrize("limit", [None, 7.0])
def test_swiglu_oai_matches_torch(rows, limit):
    gate_up = torch.randn((rows, 6144), device="cuda", dtype=torch.bfloat16)
    gate, up = gate_up.float().chunk(2, dim=-1)
    if limit is not None:
        gate = gate.clamp(max=limit)
        up = up.clamp(min=-limit, max=limit)
    expected = (gate * torch.sigmoid(1.702 * gate) * (up + 0.3)).bfloat16()

    actual = swiglu_oai_split(
        gate_up,
        alpha=1.702,
        beta=0.3,
        limit=limit,
    )

    # Triton's hardware sigmoid is approximate; compare after the BF16 round.
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
