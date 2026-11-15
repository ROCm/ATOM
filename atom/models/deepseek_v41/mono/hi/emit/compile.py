# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Resource reading adapted from ATOM atom/mono/runtime/compile.py (MIT,
# Copyright (C) 2026 Advanced Micro Devices, Inc.).
"""Compile a launcher without launching, trace its hand-offs, read its resources.

A resident kernel is only safe if every CTA is resident at once: zero scratch
(the runtime can dispatch a scratch-using grid in part), zero spills, LDS within
the CU's, and the target's grid admitted (at least one CTA per CU, and the CUs
times the CTAs per CU HIP admits cover the grid). ``gate`` raises on any of those.

Builds are for the program's target (``Program.target``): ``compile_traced`` sets
FlyDSL's arch (``ARCH`` / ``FLYDSL_GPU_ARCH``) to ``target.mcpu`` while it compiles, so a
program for another target than the visible GPU's compiles too (no launch, no
admission query: a GPU cannot load another arch's code object).

A build traces its kernel and holds the trace to the hand-off contract and the memory model (``check``,
``check_memory``), FlyDSL's disk cache off so that every build traces. Inside ``serving()`` (the engine's builds at
load) the disk cache is on instead and those checks are left to the CPU checks of the served programs (every one of
them traced and checked there): a build FlyDSL has cached loads without tracing or compiling, and the ranks of a node
share one compile of a program (FlyDSL's compile lock). The resource gate and the code's hazard scan read the built
code object either way.
"""

import functools
import hashlib
import os
import re
import subprocess
from contextlib import contextmanager
from contextvars import ContextVar

from atom.mono.runtime import srcloc

from ..plan.check import check, check_memory
from ..plan.program import Program
from ..plan.trace import recording

_RESOURCE_KEYS = {
    "vgpr": b".vgpr_count",
    "agpr": b".agpr_count",
    "sgpr": b".sgpr_count",
    "scratch": b".private_segment_fixed_size",
    "lds": b".group_segment_fixed_size",
    "vgpr_spill": b".vgpr_spill_count",
    "sgpr_spill": b".sgpr_spill_count",
}
_OPTIONAL = ("vgpr_spill", "sgpr_spill", "agpr")
_BIN = re.compile(r'bin\s*=\s*"')


class ResourceError(Exception):
    """A build that cannot run as a resident grid."""


# environment names a build does not depend on: where data and caches live, GPU exclusion, labels, and the cache
# switch ``compile_traced`` sets itself
_ENV_IGNORED = re.compile(r"CHECKPOINT|CKPT|FIXTURES|CAPTURES|GSM8K|GPU_EXCLUDE|COMMIT|SERVE_|CACHE_DIR|LINEAR_FITS"
                          r"|RUNTIME_ENABLE_CACHE")  # fmt: skip


@functools.cache
def _toolchain() -> tuple:
    """FlyDSL's wheel (its version and the digest of its RECORD, which lists every file's hash: the bundled MLIR /
    LLVM included; a source install without one: its native libraries' names, sizes and times) and ROCm's LLVM
    (``ld.lld --version``, ``.info/version``), which FlyDSL's serializer links with."""
    import importlib.metadata as md

    out = []
    try:
        dist = md.distribution("flydsl")
        record = dist.read_text("RECORD")
        out.append(
            (
                "flydsl",
                dist.version,
                hashlib.sha256(record.encode()).hexdigest()[:16] if record else None,
            )
        )
        if not record:
            libs = []
            for f in dist.files or ():
                path = f.locate()
                if ".so" in f.name and path.exists():
                    libs.append((str(f), path.stat().st_size, path.stat().st_mtime_ns))
            out.append(
                (
                    "flydsl-libs",
                    hashlib.sha256(repr(sorted(libs)).encode()).hexdigest()[:16],
                )
            )
    except md.PackageNotFoundError:
        out.append(("flydsl", None))
    rocm = os.environ.get("ROCM_PATH", "/opt/rocm")
    try:
        lld = subprocess.run([os.path.join(rocm, "llvm", "bin", "ld.lld"), "--version"], capture_output=True,
                             text=True, timeout=30, check=False).stdout.strip().splitlines()[:1]  # fmt: skip
    except (OSError, subprocess.SubprocessError):
        lld = []
    try:
        with open(os.path.join(rocm, ".info", "version")) as f:
            ver = f.read().strip()
    except OSError:
        ver = None
    out.append(("rocm", os.path.realpath(rocm), tuple(lld), ver))
    return tuple(out)


def toolchain_identity() -> tuple:
    """What a build's code and resources depend on besides its source and its plan: the toolchain (``_toolchain``),
    the target (arch, mcpu, grid), and every ``FLYDSL_*`` / ``ARCH`` environment variable but data and cache
    locations. Part of the Linear / MoE fit cache key (``blocks.gfx950.linear.fit_key``): a fit made by another
    toolchain or environment (another container on a shared cache) is another key."""
    from ..target import MI355X as t

    env = tuple(sorted((k, v) for k, v in os.environ.items() if _keyed_env(k)))
    return _toolchain() + ((t.arch, t.mcpu, t.grid),) + env


def _keyed_env(name: str) -> bool:
    return (name.startswith("FLYDSL_") or name == "ARCH") and not _ENV_IGNORED.search(
        name
    )


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


_SERVING = ContextVar("dsv41hi_serving", default=False)


@contextmanager
def serving():
    """Builds inside: from FlyDSL's disk cache where it holds them, the trace checks left to the CPU checks (module
    docstring)."""
    token = _SERVING.set(True)
    try:
        yield
    finally:
        _SERVING.reset(token)


def compile_traced(launcher, program: Program):
    """Compile ``launcher`` (no launch) and hold its recorded hand-offs to the program's regions and its memory
    operations to the memory model (``plan.check.check_memory``; raises ContractError): the body traced, FlyDSL's
    disk cache off. Inside ``serving()``: the disk cache on (a cached build loads untraced), no checks.
    """
    arch = program.target.mcpu
    serve = _SERVING.get()
    cache = os.environ.get("FLYDSL_RUNTIME_ENABLE_CACHE", "1") if serve else "0"
    with (
        recording() as rec,
        _env(
            COMPILE_ONLY=1,
            FLYDSL_RUNTIME_ENABLE_CACHE=cache,
            ARCH=arch,
            FLYDSL_GPU_ARCH=arch,
        ),
        srcloc.scoped(),  # (the source locations at a fraction of FlyDSL's cost)
    ):
        launcher(0, 0, 0, 0, 0)
    if serve:
        return rec
    if not rec.stages:
        raise RuntimeError("nothing traced (a cached build?)")
    return check_record(rec, program)


def check_record(rec, program: Program):
    """``compile_traced``'s checks of a traced record against ``program``: its hand-offs to the regions,
    its memory operations to the memory model (sets ``rec.waived``). Raises ContractError.
    """
    prev = {st.name: st.prev_reads for st in program.stages if st.prev_reads}
    check(rec, program.regions, prev)
    rec.waived = check_memory(rec, program, program.plain_waive)
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
        if r["scratch"]:  # (``gate``: a private segment the code never reaches)
            r["scratch_ops"] = scratch_ops(elf, arch)
        out.append(r)
    if not out:
        raise RuntimeError("no compiled binary to read")
    return out


_PRIVATE_OPS = re.compile(r"^\s*(scratch_|flat_)", re.MULTILINE)
_PRIVATE_REGS = re.compile(
    r"\b(flat_scratch(_lo|_hi)?|src_private_(base|limit)|src_flat_scratch_base_(lo|hi))\b"
)
_PRIVATE_BUFFER = re.compile(r"\.amdhsa_user_sgpr_private_segment_buffer\s+1\b")


def private_access_count(text: str, descriptor: str = "") -> int:
    """Instructions in a disassembly (``text``) that can reach the kernel's private segment: ``scratch_*`` and
    ``flat_*`` (a flat address may be private) and any use of the flat-scratch / private-aperture registers. With
    architected flat scratch (gfx940+, gfx12) these are the only ways in; a kernel descriptor (``descriptor``, its
    ``.amdhsa_*`` directives) that hands the kernel the private segment buffer (pre-gfx940 scratch through buffer
    instructions) counts every instruction as one."""
    if _PRIVATE_BUFFER.search(descriptor):
        return max(1, sum(1 for ln in text.splitlines() if ln.strip()))
    return len(_PRIVATE_OPS.findall(text)) + len(_PRIVATE_REGS.findall(text))


def scratch_ops(elf: bytes, arch: str | None) -> int | None:
    """``private_access_count`` of a code object (its code and kernel descriptor, disassembled with ROCm's
    llvm-objdump for ``arch``); None when it cannot be disassembled (the gate then treats the scratch as reached).
    """
    import tempfile

    from .isa import _objdump

    if not arch:
        return None
    try:
        with tempfile.NamedTemporaryFile(suffix=".co") as f:
            f.write(elf)
            f.flush()
            run = [_objdump(), f"--mcpu={arch}", "--no-show-raw-insn"]
            text = subprocess.run(
                run + ["-d", f.name], capture_output=True, text=True, check=True
            ).stdout
            kd = subprocess.run(
                run + ["-D", "--section=.rodata", f.name],
                capture_output=True,
                text=True,
                check=False,
            ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    return private_access_count(text, kd)


def _admission(artifact, arch: str | None = None):
    """Resident CTAs per CU the HIP runtime admits for this binary (None without a GPU, or when
    the visible GPU is not ``arch``: it cannot load the binary)."""
    if arch is not None:
        from ..target import device_arch

        if device_arch() != arch:
            return None
    try:
        return int(artifact.max_blocks_per_cu())
    except Exception:  # noqa: BLE001 - compile-only hosts have no device to ask
        return None


def device_cus() -> int | None:
    """CUs of the visible device 0 (None without one)."""
    try:
        import torch

        return (
            torch.cuda.get_device_properties(0).multi_processor_count
            if torch.cuda.is_available()
            else None
        )
    except Exception:  # noqa: BLE001
        return None


def code_bytes(elf: bytes) -> int:
    """Bytes of executable code in a code object (its ``SHF_EXECINSTR`` sections, ``.text``,
    with the linker's tail padding): what the CUs fetch, a per-layer
    cost measured on the reference."""
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
    """Raise ResourceError unless ``res`` can run as a resident grid. Scratch is refused unless the code provably never
    reaches it (``res["scratch_ops"] == 0``: no scratch / flat instruction, no private-aperture register;
    ``resources``). ``sgpr_spill_ok`` (``Program.sgpr_spill_ok``): SGPR spills into VGPR lanes pass (any spill
    to memory still shows as scratch); otherwise the build stays spill-free. ``grid`` / ``cus`` (the target's grid,
    the device's CUs): HIP's admission (``max_blocks_per_cu``, when known) must cover the grid.
    HIP's per-CU bound is necessary, not proof of placement (the reference's
    ``occupancy.py``); the toy and layer runs on the device are the rest of the evidence.
    """
    problems = []
    # a private segment the code never reaches passes: FlyDSL 0.4.0 reserves 36 B in V4.1's 1M-context KV-source
    # layers (an SGPR-spill save slot) with no instruction that can touch it (``scratch_ops`` 0; None or missing:
    # unknown, refused as before)
    if res["scratch"] and res.get("scratch_ops") != 0:
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


def _code_object(module_text: str) -> bytes:
    """The gpu.binary's ELF from the module text (``bin = "..."``, MLIR-escaped)."""
    m = _BIN.search(module_text)
    if m is None:
        raise RuntimeError("no gpu.binary in the compiled module")
    out, i = bytearray(), m.end()
    while module_text[i] != '"':
        ch = module_text[i]
        if ch != "\\":
            out.append(ord(ch))
            i += 1
        elif module_text[i + 1] in '\\"':
            out.append(ord(module_text[i + 1]))
            i += 2
        else:
            out.append(int(module_text[i + 1 : i + 3], 16))
            i += 3
    return bytes(out)


def _note_uint(elf: bytes, key: bytes, optional: bool = False):
    at = elf.find(key)
    if at < 0:
        if optional:
            return None
        raise RuntimeError(f"no {key.decode()} in the code object")
    v = elf[at + len(key) :]
    if v[0] < 0x80:
        return v[0]
    width = {0xCC: 1, 0xCD: 2, 0xCE: 4, 0xCF: 8}.get(v[0])
    if width is None:
        raise RuntimeError(f"unexpected msgpack tag {v[0]:#x} for {key.decode()}")
    return int.from_bytes(v[1 : 1 + width], "big")
