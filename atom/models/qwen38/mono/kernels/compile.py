# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Compile a launcher without launching, trace its hand-offs, read its resources.

A resident kernel is only safe if every CTA is resident at once: zero scratch
(the runtime can dispatch a scratch-using grid in part), zero spills, LDS within
the CU's, and the target's grid admitted (at least one CTA per CU, and the CUs
times the CTAs per CU HIP admits cover the grid). ``gate`` raises on any of those.

``compile_traced`` sets FlyDSL's arch (``ARCH`` / ``FLYDSL_GPU_ARCH``) to the
program's ``target.mcpu`` while it compiles, so it needs no GPU (no launch, no
admission query without one). The code object reading is main's
(``atom.mono.runtime.compile``).
"""

import os
from contextlib import contextmanager

from atom.models.qwen38.mono.kernels.program import Program
from atom.models.qwen38.mono.kernels.target import device_arch
from atom.mono.plan.check import RegionDecl, check
from atom.mono.plan.trace import recording
from atom.mono.runtime import compile as _main
from atom.mono.runtime.compile import _code_object, _note_uint

_RESOURCE_KEYS = {**_main._RESOURCE_KEYS, "agpr": b".agpr_count"}
_OPTIONAL = (*_main._OPTIONAL, "agpr")


class ResourceError(Exception):
    """A build that cannot run as a resident grid."""


@contextmanager
def _env(**values):
    before = {k: os.environ.get(k) for k in values}
    os.environ.update({k: str(v) for k, v in values.items()})
    try:
        yield
    finally:
        for k, v in before.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def compile_traced(launcher, program: Program):
    """Compile ``launcher`` (no launch, no disk cache so the body is traced) and hold its recorded hand-offs to the
    program's regions (``atom.mono.plan.check``). Raises ContractError."""
    arch = program.target.mcpu
    with (
        recording() as rec,
        _env(
            COMPILE_ONLY=1,
            FLYDSL_RUNTIME_ENABLE_CACHE=0,
            ARCH=arch,
            FLYDSL_GPU_ARCH=arch,
        ),
    ):
        launcher(0, 0, 0, 0, 0)
    if not rec.stages:
        raise RuntimeError("nothing traced (a cached build?)")
    check(
        rec,
        [RegionDecl(r.name, r.space, r.writers, r.exchange) for r in program.regions],
    )
    return rec


def resources(launcher, arch: str | None = None) -> list[dict]:
    """Per compiled binary held by ``launcher``: name-less resource counts, the executable bytes
    (``code_bytes``) and the CTAs per CU HIP admits (``max_blocks_per_cu``; None without a GPU of
    ``arch``, the binary's)."""
    out = []
    for artifact in launcher._mem_cache.values():
        elf = _code_object(artifact._ir_text)
        r = {
            k: _note_uint(elf, key, optional=k in _OPTIONAL)
            for k, key in _RESOURCE_KEYS.items()
        }
        r["max_blocks_per_cu"] = _admission(artifact, arch)
        r["code_bytes"] = code_bytes(elf)
        out.append(r)
    if not out:
        raise RuntimeError("no compiled binary to read")
    return out


def _admission(artifact, arch: str | None = None):
    """Resident CTAs per CU the HIP runtime admits for this binary (None without a GPU, or when
    the visible GPU is not ``arch``: it cannot load the binary)."""
    if arch is not None and device_arch() != arch:
        return None
    try:
        return int(artifact.max_blocks_per_cu())
    except Exception:  # noqa: BLE001 - compile-only hosts have no device to ask
        return None


def code_bytes(elf: bytes) -> int:
    """Bytes of executable code in a code object (its ``SHF_EXECINSTR`` sections, ``.text``,
    with the linker's tail padding)."""
    import struct

    shoff = struct.unpack_from("<Q", elf, 0x28)[0]
    shentsize, shnum = struct.unpack_from("<HH", elf, 0x3A)
    total = 0
    for i in range(shnum):
        flags, _, _, size = struct.unpack_from("<QQQQ", elf, shoff + i * shentsize + 8)
        if flags & 0x4:  # SHF_EXECINSTR
            total += size
    return total


def gate(
    res: dict,
    lds_limit: int,
    sgpr_spill_ok: bool = False,
    grid: int | None = None,
    cus: int | None = None,
):
    """Raise ResourceError unless ``res`` can run as a resident grid. ``sgpr_spill_ok``: SGPR spills
    into VGPR lanes pass (any spill to memory still shows as scratch). ``grid`` / ``cus`` (the target's
    grid, the device's CUs): HIP's admission (``max_blocks_per_cu``, when known) must cover the grid.
    HIP's per-CU bound is necessary, not proof of placement."""
    problems = []
    if res["scratch"]:
        problems.append(f"{res['scratch']} B scratch per lane")
    for k in ("vgpr_spill",) if sgpr_spill_ok else ("vgpr_spill", "sgpr_spill"):
        if res.get(k):
            problems.append(f"{res[k]} {k}")
    if res["lds"] > lds_limit:
        problems.append(f"LDS {res['lds']} > {lds_limit}")
    blocks = res.get("max_blocks_per_cu")
    if blocks is not None and blocks < 1:
        problems.append("not even one CTA per CU fits")
    elif blocks is not None and grid and cus and blocks * cus < grid:
        problems.append(f"HIP admits {blocks} x {cus} CUs < the grid of {grid} CTAs")
    if problems:
        raise ResourceError("; ".join(problems))
