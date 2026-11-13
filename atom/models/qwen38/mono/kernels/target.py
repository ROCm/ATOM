# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The GPU the Qwen layer programs are built for: MI355X (gfx950), one resident CTA per CU.

A ``Target`` is a build input like ``Scenario``: a ``Program`` carries one (``MI355X``). ``cus``: the GPU's CUs;
``grid``: the CTAs of a resident launch (task ``t`` of a stage runs on CTA ``(base + t) % grid``); ``threads`` /
``wave``: a CTA's threads and the wave size; ``lds_bytes``: the LDS one CTA owns; ``mcpu``: the LLVM CPU builds and
disassembly use. A deployment on fewer CUs (a compute partition) is refused before any build
(``atom.mono.runtime.deployment.resident_grid_refusals``).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Target:
    arch: str
    cus: int
    threads: int
    wave: int
    lds_bytes: int
    grid: int = 0  # resident CTAs of a launch (0: one per CU)
    mcpu: str = ""  # LLVM CPU (default: arch)

    def __post_init__(self):
        if not self.grid:
            object.__setattr__(self, "grid", self.cus)
        if not self.mcpu:
            object.__setattr__(self, "mcpu", self.arch)
        if not 1 <= self.grid <= self.cus:
            raise ValueError(
                f"{self.arch}: grid {self.grid} outside 1..{self.cus} (one resident CTA per CU)"
            )

    @property
    def waves(self) -> int:
        return self.threads // self.wave


MI355X = Target(
    arch="gfx950", cus=256, threads=512, wave=64, lds_bytes=160 * 1024, grid=256
)


def device_arch() -> str | None:
    """The arch of the visible device 0 (torch), or None without one."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return torch.cuda.get_device_properties(0).gcnArchName.split(":", 1)[0]
    except Exception:  # noqa: BLE001 - no device runtime: no device
        return None


def device_cus() -> int | None:
    """The CUs of the visible device 0 (torch), or None without one."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return torch.cuda.get_device_properties(0).multi_processor_count
    except Exception:  # noqa: BLE001 - no device runtime: no device
        return None
