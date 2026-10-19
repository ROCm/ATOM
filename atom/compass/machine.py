# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Machine description used by a Compass estimate.

Values are configuration, not a reading from a live device. The MI355X
constructor copies the public product brief (dense peaks, no sparsity credit).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MachineSpec:
    """One GPU. A TP group does not change these peaks; it changes the shard."""

    name: str
    memory_bytes: int
    hbm_bandwidth_bytes_per_s: float
    fp8_flops: float
    fp4_flops: float
    bf16_flops: float
    fp32_flops: float
    # Aggregate injection bandwidth of this GPU's scale-up links.
    interconnect_bandwidth_bytes_per_s: float
    compute_derate: float = 1.0
    memory_derate: float = 1.0
    interconnect_derate: float = 1.0

    def __post_init__(self) -> None:
        for name in (
            "memory_bytes",
            "hbm_bandwidth_bytes_per_s",
            "fp8_flops",
            "fp4_flops",
            "bf16_flops",
            "fp32_flops",
            "interconnect_bandwidth_bytes_per_s",
            "compute_derate",
            "memory_derate",
            "interconnect_derate",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")


def mi355x() -> MachineSpec:
    """AMD Instinct MI355X, dense peaks from the public CDNA 4 brief.

    Memory is 288 GB as stated by AMD (decimal). HBM bandwidth is 8 TB/s.
    Matrix peaks are the dense figures: OCP-FP8 5.0332 PFLOP/s, MXFP4 10.0663
    PFLOP/s, BF16 2.5166 PFLOP/s, FP32 157.3 TFLOP/s. The 7 scale-up links are
    153.6 GB/s each.
    """
    peta = 1e15
    return MachineSpec(
        name="MI355X",
        memory_bytes=288 * 10**9,
        hbm_bandwidth_bytes_per_s=8e12,
        fp8_flops=5.0332 * peta,
        fp4_flops=10.0663 * peta,
        bf16_flops=2.5166 * peta,
        fp32_flops=157.3e12,
        interconnect_bandwidth_bytes_per_s=7 * 153.6e9,
    )
