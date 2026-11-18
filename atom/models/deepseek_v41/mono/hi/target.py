# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The GPU the hi programs are built for: MI355X (gfx950), every CU (``hi_gate`` refuses any other device or a
compute partition).

A ``Target`` is a build input like ``Scenario``: a ``Program`` carries one (``MI355X``), and the planner, emitter,
compiler and resource gate read it from there.

- ``cus``: the GPU's CUs; ``grid``: the CTAs of a resident launch (one per CU). Task ``t`` of a stage runs on CTA
  ``(base + t) % grid``;
- ``threads`` / ``wave``: a CTA's threads and the wave size; ``lds_bytes``: the LDS one CTA owns (the CU's);
- ``mcpu``: the LLVM CPU name builds and disassembly use;
- ``xcds``: CTA c runs on XCD c % xcds (each XCD has its own L2), what the memory model checks (``plan.check``).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Target:
    arch: str
    cus: int  # CUs of the GPU
    threads: int  # threads per CTA (: 512 on gfx950, re-checkable)
    wave: int
    lds_bytes: int  # one CTA owns the CU's LDS
    grid: int = 0  # resident CTAs of a launch (0: one per CU)
    mcpu: str = ""  # LLVM CPU (default: arch)
    xcds: int = (
        0  # CTA c runs on XCD c % xcds (each XCD has its own L2: plain stores are not seen across them);
    )
    # 0: not modeled (``plan.check.check_memory``, the memory model, is calibrated on gfx950 only)
    # NUMA: two I/O dies; an address's home IOD flips every 8 KiB inside a 2 MiB page
    # (bit 13 XOR a bit of the page). Dual-copy regions (``Program.dual``, ``Region.home``) keep a copy on each, laid
    # out by that rule in one page (``plan.program.DUAL_*``); which XCDs read which copy is measured per scratch
    # (``dsv41hi.runtime.dual``).

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
    arch="gfx950",
    cus=256,
    threads=512,
    wave=64,
    lds_bytes=160 * 1024,
    grid=256,
    xcds=8,
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
