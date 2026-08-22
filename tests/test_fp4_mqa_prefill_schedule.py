# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import runpy
from pathlib import Path

import pytest

_SCHEDULE = runpy.run_path(
    Path(__file__).parents[1]
    / "atom"
    / "model_ops"
    / "v4_kernels"
    / "fp4_mqa_schedule.py"
)
select_grid = _SCHEDULE["fp4_mqa_prefill_parallel_unit_num"]


@pytest.mark.parametrize(
    ("rows", "seq_len", "expected"),
    [
        (1, 256, 1),  # never launch dummy CTAs just to satisfy the 512 floor
        (256, 512, 512),  # useful row/chunk pairs cap the small-Q target
        (512, 16384, 2560),  # measured small-Q optimum: 5 CTA/row
        (1536, 16384, 7680),
        (1537, 16384, 3074),  # medium Q: 2 CTA/row
        (8192, 16384, 16384),
        (16384, 16384, 16384),  # full Q + <=64 chunks: 1 CTA/row
        (16384, 16640, 32768),  # longer than 64 chunks: restore 2 CTA/row
    ],
)
def test_fp4_prefill_grid_policy(rows, seq_len, expected):
    assert select_grid(rows, seq_len) == expected


@pytest.mark.parametrize(
    ("rows", "seq_len", "block_k", "minimum"),
    [
        (0, 256, 256, 512),
        (1, -1, 256, 512),
        (1, 256, 0, 512),
        (1, 256, 256, 0),
    ],
)
def test_fp4_prefill_grid_policy_rejects_invalid_inputs(
    rows, seq_len, block_k, minimum
):
    with pytest.raises(ValueError):
        select_grid(
            rows,
            seq_len,
            block_k=block_k,
            min_parallel_unit_num=minimum,
        )
