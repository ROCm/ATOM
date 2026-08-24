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
select_config = _SCHEDULE["fp4_mqa_prefill_config"]
select_grid = _SCHEDULE["fp4_mqa_prefill_parallel_unit_num"]
select_wave_tasks = _SCHEDULE["fp4_mqa_prefill_wave_tasks_per_row"]


@pytest.mark.parametrize(
    ("rows", "seq_len", "expected"),
    [
        # Small Q needs several device turnovers. The serial-depth cap stops
        # the launch from degenerating into tiny tasks.
        (256, 16512, 20),
        (512, 16896, 20),
        (768, 16896, 16),
        (1024, 16896, 12),
        (1280, 16896, 12),
        # Once row parallelism covers the machine, K-loop depth sets the floor.
        (1536, 16896, 8),
        (3584, 16896, 8),
        (16384, 16896, 8),
        # Longer contexts raise the budget to keep about 33 K tasks per wave.
        (4096, 65536, 32),
    ],
)
def test_fp4_prefill_wave_budget_uses_device_and_pipeline_model(
    rows, seq_len, expected
):
    assert select_wave_tasks(rows, seq_len) == expected


@pytest.mark.parametrize(
    ("rows", "seq_len", "block_k", "warps", "wave_tasks", "expected"),
    [
        (1, 256, 256, 4, 8, 1),
        (256, 512, 256, 4, 8, 512),
        (512, 16384, 256, 4, 20, 2560),
        (512, 16384, 64, 1, 20, 10240),
        # The useful row/chunk count caps the requested grid.
        (256, 512, 64, 1, 20, 2048),
    ],
)
def test_fp4_prefill_grid_is_derived_from_wave_budget(
    rows, seq_len, block_k, warps, wave_tasks, expected
):
    assert (
        select_grid(
            rows,
            seq_len,
            block_k=block_k,
            num_warps=warps,
            wave_tasks_per_row=wave_tasks,
        )
        == expected
    )


def test_fp4_prefill_coarse_and_fine_grids_preserve_wave_work():
    coarse_ctas = select_grid(
        4096,
        16896,
        block_k=256,
        num_warps=4,
        wave_tasks_per_row=8,
    )
    fine_ctas = select_grid(
        4096,
        16896,
        block_k=64,
        num_warps=1,
        wave_tasks_per_row=8,
    )

    assert coarse_ctas * 4 == fine_ctas


@pytest.mark.parametrize(
    (
        "rows",
        "seq_len",
        "max_query_len",
        "block_k",
        "warps",
        "wave_tasks",
        "parallel_units",
    ),
    [
        # A tiny launch stays coarse: four times as many workgroups cost more
        # than the independent-wave scheduling can recover.
        (256, 16512, 256, 256, 4, 20, 1280),
        # Small-Q latency regime: enough independent waves to cover gfx950.
        (512, 16896, 512, 64, 1, 20, 10240),
        (1536, 16896, 768, 64, 1, 8, 12288),
        # Two long sequences retain the cache-friendly coarse workgroup.
        (3584, 16896, 1792, 256, 4, 8, 7168),
        # At least three shorter sequence-equivalents select one-wave CTAs.
        (3584, 16896, 1536, 64, 1, 8, 28672),
        (8192, 16896, 2048, 64, 1, 8, 65536),
        # Cache pressure alone is insufficient: at least three independent KV
        # streams are needed to repay the fine launch's extra descriptors.
        (3584, 32768, 1792, 256, 4, 16, 14336),
        (3584, 32768, 896, 64, 1, 16, 57344),
        # Conversely, eight short working sets still fit usable L2.
        (3584, 4096, 448, 256, 4, 4, 3584),
        # Long single-sequence Q has enough row parallelism and strong KV reuse.
        (8192, 16384, 8192, 256, 4, 4, 8192),
        (16384, 16384, 16384, 256, 4, 4, 16384),
        # Q=16K is not itself a reason to stay coarse: sequence composition is.
        (16384, 16384, 2048, 64, 1, 8, 131072),
    ],
)
def test_fp4_prefill_config_uses_query_composition(
    rows,
    seq_len,
    max_query_len,
    block_k,
    warps,
    wave_tasks,
    parallel_units,
):
    config = select_config(rows, seq_len, max_query_len)

    assert config.block_k == block_k
    assert config.num_warps == warps
    assert config.wave_tasks_per_row == wave_tasks
    assert config.parallel_unit_num == parallel_units


@pytest.mark.parametrize(
    ("rows", "seq_len", "block_k", "warps", "wave_tasks", "minimum"),
    [
        (0, 256, 256, 4, 8, 512),
        (1, -1, 256, 4, 8, 512),
        (1, 256, 0, 4, 8, 512),
        (1, 256, 256, 0, 8, 512),
        (1, 256, 256, 4, 0, 512),
        (1, 256, 256, 4, 8, 0),
    ],
)
def test_fp4_prefill_grid_rejects_invalid_inputs(
    rows, seq_len, block_k, warps, wave_tasks, minimum
):
    with pytest.raises(ValueError):
        select_grid(
            rows,
            seq_len,
            block_k=block_k,
            num_warps=warps,
            wave_tasks_per_row=wave_tasks,
            min_parallel_unit_num=minimum,
        )


def test_fp4_prefill_config_rejects_invalid_query_length():
    with pytest.raises(ValueError, match="max_query_len"):
        select_config(512, 16384, 0)
