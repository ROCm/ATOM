# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Differential coverage for the two equivalent FP4 prefill wave layouts."""

import math

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("requires a real ROCm GPU", allow_module_level=True)

from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl import flydsl_pa_mqa_logits_fp4_prefill
from aiter.ops.flydsl.kernels.mqa_logits.pa_mqa_logits_fp4_prefill import (
    compute_prefill_schedule,
)

if get_gfx() != "gfx950":
    pytest.skip(
        "the FP4 indexer kernel is a gfx950 specialization",
        allow_module_level=True,
    )


HEADS = 64
HEAD_DIM = 128
KV_BLOCK_SIZE = 64


def _launch(
    *,
    block_k,
    num_warps,
    wave_tasks_per_row,
    q_fp4,
    q_scale,
    kv_cache,
    kv_scale,
    block_tables,
    weights,
    row_to_batch,
    local_starts,
    local_ends,
    max_seq_len,
):
    rows = q_fp4.size(0)
    chunks_per_row = math.ceil(max_seq_len / block_k)
    ctas_per_row = math.ceil(wave_tasks_per_row / num_warps)
    parallel_units = min(rows * chunks_per_row, rows * ctas_per_row)
    _, cta_info, n_ctas = compute_prefill_schedule(
        row_to_batch,
        local_starts,
        local_ends,
        block_k,
        parallel_units,
        max_seq_len,
    )
    out = torch.full(
        (rows, max_seq_len),
        float("nan"),
        dtype=torch.float32,
        device=q_fp4.device,
    )
    flydsl_pa_mqa_logits_fp4_prefill(
        q_fp4,
        q_scale,
        kv_cache,
        kv_scale,
        block_tables,
        weights,
        row_to_batch,
        local_starts,
        local_ends,
        max_seq_len,
        block_k=block_k,
        kv_block_size=KV_BLOCK_SIZE,
        num_warps=num_warps,
        parallel_unit_num=parallel_units,
        out=out,
        cta_info=cta_info,
        n_ctas=n_ctas,
    )
    return out


def test_one_wave_matches_four_waves_for_random_ragged_pages():
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(20260824)
    row_to_batch = torch.tensor(
        [0] * 5 + [1] * 6 + [2] * 7, dtype=torch.int32, device=device
    )
    local_starts = torch.tensor(
        [0, 0, 1, 31, 63, 0, 1, 63, 64, 65, 127, 0, 32, 63, 64, 128, 192, 255],
        dtype=torch.int32,
        device=device,
    )
    local_ends = torch.tensor(
        [1, 63, 64, 65, 129, 2, 64, 65, 127, 128, 193, 33, 64, 127, 129, 193, 256, 257],
        dtype=torch.int32,
        device=device,
    )
    rows = row_to_batch.numel()
    max_seq_len = 257
    pages_per_sequence = math.ceil(max_seq_len / KV_BLOCK_SIZE)
    num_physical_pages = 3 * pages_per_sequence + 3
    block_tables = (
        torch.randperm(num_physical_pages, generator=generator, device=device)[
            : 3 * pages_per_sequence
        ]
        .reshape(3, pages_per_sequence)
        .to(torch.int32)
    )

    q_fp4 = torch.randint(
        0,
        256,
        (rows, HEADS, HEAD_DIM // 2),
        dtype=torch.uint8,
        device=device,
        generator=generator,
    )
    q_scale = torch.randint(
        118,
        132,
        (rows, 1, 4, 16, 4),
        dtype=torch.uint8,
        device=device,
        generator=generator,
    )
    kv_cache = torch.randint(
        0,
        256,
        (num_physical_pages, 1, 4, KV_BLOCK_SIZE, 16),
        dtype=torch.uint8,
        device=device,
        generator=generator,
    )
    kv_scale = torch.randint(
        118,
        132,
        (num_physical_pages, 1, 4, KV_BLOCK_SIZE),
        dtype=torch.uint8,
        device=device,
        generator=generator,
    )
    weights = torch.randn(
        rows,
        HEADS,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    args = {
        "q_fp4": q_fp4,
        "q_scale": q_scale,
        "kv_cache": kv_cache,
        "kv_scale": kv_scale,
        "block_tables": block_tables,
        "weights": weights,
        "row_to_batch": row_to_batch,
        "local_starts": local_starts,
        "local_ends": local_ends,
        "max_seq_len": max_seq_len,
        "wave_tasks_per_row": 8,
    }

    coarse = _launch(block_k=256, num_warps=4, **args)
    fine = _launch(block_k=64, num_warps=1, **args)
    torch.cuda.synchronize(device)

    for row, (start, end) in enumerate(
        zip(local_starts.cpu().tolist(), local_ends.cpu().tolist(), strict=True)
    ):
        torch.testing.assert_close(
            fine[row, start:end], coarse[row, start:end], rtol=0, atol=0
        )
