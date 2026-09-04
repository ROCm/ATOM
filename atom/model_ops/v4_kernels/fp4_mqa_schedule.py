# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Shared persistent-grid parameters for the DeepSeek-V4 FP4 indexer."""

from typing import NamedTuple

FP4_MQA_PARALLEL_UNIT_NUM = 512
FP4_MQA_BLOCK_K = 256
FP4_MQA_NUM_WARPS = 4

# One wave in the FlyDSL kernel always owns four 16-token MFMA N tiles.  These
# two launch shapes therefore perform the same work per wave; only the hardware
# workgroup scheduling granularity changes:
#
#   coarse: 256 K rows / CTA / 4 waves = 64 K rows / wave
#   fine:    64 K rows / CTA / 1 wave  = 64 K rows / wave
FP4_MQA_FINE_BLOCK_K = 64
FP4_MQA_FINE_NUM_WARPS = 1

# gfx950 exposes 256 CUs and at most 32 resident waves per CU.  Ragged causal
# rows do not all retire together, so one resident set plus half a refill is a
# better coverage target than exactly one set.  Expressing this in waves keeps
# the model invariant across the 256x4 and 64x1 launch shapes.
FP4_MQA_GFX950_COMPUTE_UNITS = 256
FP4_MQA_GFX950_MAX_WAVES_PER_CU = 32
FP4_MQA_RESIDENT_WAVE_SLOTS = (
    FP4_MQA_GFX950_COMPUTE_UNITS * FP4_MQA_GFX950_MAX_WAVES_PER_CU
)
FP4_MQA_TARGET_DEVICE_WAVE_TASKS = 3 * FP4_MQA_RESIDENT_WAVE_SLOTS // 2

# Each logical K token contributes 64 packed FP4 bytes plus four e8m0 scale
# bytes.  rocminfo reports a 4 MiB L2 slice on gfx950.  Leave one quarter for Q,
# weights, descriptors, and unrelated traffic when deciding whether concurrent
# sequence working sets have displaced the cache-local launch's advantage.
FP4_MQA_KV_BYTES_PER_TOKEN = 64 + 4
FP4_MQA_GFX950_L2_BYTES = 4 * 1024 * 1024
FP4_MQA_L2_USABLE_NUMERATOR = 3
FP4_MQA_L2_USABLE_DENOMINATOR = 4
FP4_MQA_MIN_FINE_SEQUENCE_EQUIVALENTS = 3

# The inner loop is pipelined across 64-row wave tasks. Keep enough independent
# tasks to cover the device while retaining useful serial work in each task:
#
# * 12,288 tasks are 1.5 full resident-wave sets on gfx950;
# * about 33 serial K tasks keeps the prologue/epilogue amortized;
# * fewer than 12 serial K tasks loses to dispatch and descriptor overhead.
FP4_MQA_TARGET_SERIAL_K_TASKS = 33
FP4_MQA_MIN_SERIAL_K_TASKS = 12
FP4_MQA_WAVE_TASK_QUANTUM = FP4_MQA_NUM_WARPS

# These boundaries are consequences of the device model above rather than
# independently tuned query-size cutoffs.
FP4_MQA_FINE_MIN_ROWS = 2 * FP4_MQA_GFX950_COMPUTE_UNITS
FP4_MQA_FINE_LATENCY_MAX_ROWS = FP4_MQA_TARGET_DEVICE_WAVE_TASKS // (
    2 * FP4_MQA_NUM_WARPS
)
FP4_MQA_MAX_CACHE_LOCAL_QUERY_ROWS = (
    FP4_MQA_TARGET_DEVICE_WAVE_TASKS // FP4_MQA_NUM_WARPS
)
FP4_MQA_LONG_QUERY_MIN_ROWS = FP4_MQA_TARGET_DEVICE_WAVE_TASKS // 2
FP4_MQA_LONG_MAX_COARSE_CHUNKS = 64


class FP4MQAPrefillConfig(NamedTuple):
    """A complete, internally consistent FP4 prefill launch configuration."""

    block_k: int
    num_warps: int
    wave_tasks_per_row: int
    parallel_unit_num: int


def _round_up(value: int, quantum: int) -> int:
    return ((value + quantum - 1) // quantum) * quantum


def fp4_mqa_prefill_wave_tasks_per_row(
    num_rows: int,
    max_seq_len: int,
) -> int:
    """Derive K parallelism from device coverage and inner-loop depth.

    A wave always computes one 64-row K task per inner-loop iteration. Too few
    waves leave the device underfilled or make each wave's dependency chain too
    long; too many make the loop too short to amortize its Q/scale/weight and
    descriptor prologue. This computes the smallest four-wave-aligned budget
    satisfying both lower bounds, capped before the serial loop gets too short.
    """
    if num_rows <= 0:
        raise ValueError(f"num_rows must be positive, got {num_rows}")
    if max_seq_len < 0:
        raise ValueError(f"max_seq_len must be non-negative, got {max_seq_len}")

    k_tasks_per_row = max(
        1, (max_seq_len + FP4_MQA_FINE_BLOCK_K - 1) // FP4_MQA_FINE_BLOCK_K
    )
    waves_for_device = (FP4_MQA_TARGET_DEVICE_WAVE_TASKS + num_rows - 1) // num_rows
    waves_for_pipeline = (
        k_tasks_per_row + FP4_MQA_TARGET_SERIAL_K_TASKS - 1
    ) // FP4_MQA_TARGET_SERIAL_K_TASKS
    requested = _round_up(
        max(waves_for_device, waves_for_pipeline), FP4_MQA_WAVE_TASK_QUANTUM
    )

    max_useful_waves = max(
        FP4_MQA_WAVE_TASK_QUANTUM,
        (k_tasks_per_row // FP4_MQA_MIN_SERIAL_K_TASKS)
        // FP4_MQA_WAVE_TASK_QUANTUM
        * FP4_MQA_WAVE_TASK_QUANTUM,
    )
    return min(requested, max_useful_waves)


def fp4_mqa_prefill_parallel_unit_num(
    num_rows: int,
    max_seq_len: int,
    *,
    block_k: int = FP4_MQA_BLOCK_K,
    num_warps: int = FP4_MQA_NUM_WARPS,
    wave_tasks_per_row: int = 8,
    min_parallel_unit_num: int = FP4_MQA_PARALLEL_UNIT_NUM,
) -> int:
    """Convert a wave-task budget into the persistent-grid CTA count.

    ``compute_prefill_schedule`` launches exactly ``parallel_unit_num`` CTAs,
    not merely up to that many. Too small a grid folds multiple K chunks onto
    one CTA; too large a grid launches dummy CTAs after all useful row/chunk
    pairs have been assigned.

    The policy is expressed in *wave tasks per row*, not CTAs per row. That is
    the invariant shared by the 256x4 and 64x1 kernels: both assign exactly 64
    K rows to a wave. A 64x1 launch consequently uses four times as many CTAs
    for the same wave budget. The final cap prevents empty CTAs when a row has
    fewer K chunks than requested.
    """
    if num_rows <= 0:
        raise ValueError(f"num_rows must be positive, got {num_rows}")
    if max_seq_len < 0:
        raise ValueError(f"max_seq_len must be non-negative, got {max_seq_len}")
    if block_k <= 0:
        raise ValueError(f"block_k must be positive, got {block_k}")
    if num_warps <= 0:
        raise ValueError(f"num_warps must be positive, got {num_warps}")
    if wave_tasks_per_row <= 0:
        raise ValueError(
            f"wave_tasks_per_row must be positive, got {wave_tasks_per_row}"
        )
    if min_parallel_unit_num <= 0:
        raise ValueError(
            "min_parallel_unit_num must be positive, got " f"{min_parallel_unit_num}"
        )

    chunks_per_row = max(1, (max_seq_len + block_k - 1) // block_k)
    cta_splits_per_row = (wave_tasks_per_row + num_warps - 1) // num_warps
    target_grid = max(min_parallel_unit_num, num_rows * cta_splits_per_row)
    useful_grid_cap = num_rows * chunks_per_row
    return min(target_grid, useful_grid_cap)


def fp4_mqa_prefill_config(
    num_rows: int,
    max_seq_len: int,
    max_query_len: int,
) -> FP4MQAPrefillConfig:
    """Choose the gfx950 FP4 prefill workgroup granularity and wave budget.

    The kernel has no LDS, scratch, cross-wave reduction, or cross-wave data
    dependency. Its four waves merely process adjacent 64-row K tasks inside a
    single workgroup. Splitting that workgroup into four independent 1-wave
    workgroups preserves the work for full tiles, avoids up to three padded
    64-row wave tasks at each row's causal tail, and gives the hardware scheduler
    finer load-balancing units.

    That freedom is useful when a prefill contains several sequence-sized KV
    working sets. A long single-sequence prefill instead strongly reuses one KV
    working set and benefits from the lower dispatch/descriptor cost of 4-wave
    workgroups. ``max_query_len`` distinguishes those cases; ``num_rows`` alone
    cannot (for example, measured Q=8192 selects opposite kernels for batch 1
    and batch 4).

    The ordinary wave budget is calculated from the number of 64-row K tasks,
    the kernel's useful serial pipeline depth, and a gfx950 device-wide wave
    target. A sufficiently large, long-query launch is the one exception: it
    already fills the device, strongly reuses one sequence's KV working set,
    and measured best with one four-wave CTA per row. The grid itself is then
    derived by ``fp4_mqa_prefill_parallel_unit_num`` and represents the same
    wave budget for coarse and fine launches.
    """
    if max_query_len <= 0:
        raise ValueError(f"max_query_len must be positive, got {max_query_len}")

    coarse_chunks_per_row = max(
        1, (max_seq_len + FP4_MQA_BLOCK_K - 1) // FP4_MQA_BLOCK_K
    )
    long_reuse = (
        num_rows >= FP4_MQA_RESIDENT_WAVE_SLOTS
        and max_query_len >= FP4_MQA_LONG_QUERY_MIN_ROWS
        and coarse_chunks_per_row <= FP4_MQA_LONG_MAX_COARSE_CHUNKS
    )
    if long_reuse:
        wave_tasks_per_row = 4
    else:
        wave_tasks_per_row = fp4_mqa_prefill_wave_tasks_per_row(num_rows, max_seq_len)

    # A 1-wave CTA is useful once there are at least two rows per CU to amortize
    # dispatch. In the latency regime, two coarse CTAs per row still fit within
    # the device wave target, so independently scheduling their eight waves
    # improves tail balance. Beyond that regime, use the fine launch only when
    # the concurrent KV working sets exceed the usable gfx950 L2 capacity and
    # no individual sequence already supplies a device-filling coarse grid.
    sequence_equivalents = max(1, (num_rows + max_query_len - 1) // max_query_len)
    kv_working_set_bytes = max(1, max_seq_len * FP4_MQA_KV_BYTES_PER_TOKEN)
    usable_l2_bytes = (
        FP4_MQA_GFX950_L2_BYTES
        * FP4_MQA_L2_USABLE_NUMERATOR
        // FP4_MQA_L2_USABLE_DENOMINATOR
    )
    sequences_to_fill_l2 = (
        usable_l2_bytes + kv_working_set_bytes - 1
    ) // kv_working_set_bytes
    min_fine_sequences = max(
        FP4_MQA_MIN_FINE_SEQUENCE_EQUIVALENTS, sequences_to_fill_l2
    )
    use_fine_workgroups = num_rows >= FP4_MQA_FINE_MIN_ROWS and (
        num_rows <= FP4_MQA_FINE_LATENCY_MAX_ROWS
        or (
            sequence_equivalents >= min_fine_sequences
            and max_query_len <= FP4_MQA_MAX_CACHE_LOCAL_QUERY_ROWS
        )
    )
    if use_fine_workgroups:
        block_k = FP4_MQA_FINE_BLOCK_K
        num_warps = FP4_MQA_FINE_NUM_WARPS
    else:
        block_k = FP4_MQA_BLOCK_K
        num_warps = FP4_MQA_NUM_WARPS

    parallel_unit_num = fp4_mqa_prefill_parallel_unit_num(
        num_rows,
        max_seq_len,
        block_k=block_k,
        num_warps=num_warps,
        wave_tasks_per_row=wave_tasks_per_row,
    )
    return FP4MQAPrefillConfig(
        block_k=block_k,
        num_warps=num_warps,
        wave_tasks_per_row=wave_tasks_per_row,
        parallel_unit_num=parallel_unit_num,
    )
