# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The code objects a decode step launches, by TileRT kernel symbol (the plan's ``kernels``), and their launch.

A table (``CodeTable``) maps a TileRT kernel symbol to a code object file, the function to load from it and that
function's explicit kernarg layout (``(offset, size)`` per argument):

- ``CodeTable.from_source(cache)``: our FlyDSL ports (``kernels``) of the step's 28 kernels and ``admit_kernel``,
  built from source into a node-local cache (``flybuild``), each under the TileRT symbol it replaces, the layout read
  from its code object's AMDGPU metadata note (``kernels``: a small msgpack reader, no dependency).

A launch (``Launcher.launch``) is ``hipModuleLaunchKernel`` in params mode: ``kernelParams[i]`` points at byte
``offset_i`` of the launch's kernarg blob (``decode_launches.kernargs``), so HIP packs the explicit arguments by the
loaded function's own metadata and fills its hidden arguments.
"""

from __future__ import annotations

import ctypes
import struct
from dataclasses import dataclass
from pathlib import Path

NT_AMDGPU_METADATA = 32


# ---------------------------------------------------------------------------------------------------------------
# code object metadata (CPU)


def _mp(b: bytes, i: int):
    """One msgpack value of ``b`` at ``i`` -> (value, next index)."""
    t = b[i]
    i += 1
    if t <= 0x7F:
        return t, i
    if t >= 0xE0:
        return t - 256, i
    if 0x80 <= t <= 0x8F:
        return _seq(b, i, t & 15, True)
    if 0x90 <= t <= 0x9F:
        return _seq(b, i, t & 15, False)
    if 0xA0 <= t <= 0xBF:
        n = t & 31
        return b[i : i + n].decode(), i + n
    if t in (0xC0, 0xC2, 0xC3):
        return {0xC0: None, 0xC2: False, 0xC3: True}[t], i
    fixed = {0xCA: ">f", 0xCB: ">d", 0xCC: ">B", 0xCD: ">H", 0xCE: ">I", 0xCF: ">Q"}
    fixed.update({0xD0: ">b", 0xD1: ">h", 0xD2: ">i", 0xD3: ">q"})
    if t in fixed:
        return struct.unpack_from(fixed[t], b, i)[0], i + struct.calcsize(fixed[t])
    sized = {0xC4: 1, 0xC5: 2, 0xC6: 4, 0xD9: 1, 0xDA: 2, 0xDB: 4}
    if t in sized:
        n = int.from_bytes(b[i : i + sized[t]], "big")
        i += sized[t]
        v = b[i : i + n]
        return (v.decode() if t >= 0xD9 else bytes(v)), i + n
    if t in (0xDC, 0xDD, 0xDE, 0xDF):
        w = 2 if t in (0xDC, 0xDE) else 4
        n = int.from_bytes(b[i : i + w], "big")
        return _seq(b, i + w, n, t >= 0xDE)
    raise ValueError(f"msgpack tag {t:#x} not read here")


def _seq(b: bytes, i: int, n: int, is_map: bool):
    out: dict | list = {} if is_map else []
    for _ in range(n):
        k, i = _mp(b, i)
        if is_map:
            out[k], i = _mp(b, i)
        else:
            out.append(k)
    return out, i


def metadata(elf: bytes) -> dict:
    """The AMDGPU metadata (``amdhsa.kernels`` ...) of a code object."""
    if elf[:4] != b"\x7fELF" or elf[4] != 2:
        raise ValueError("not an ELF64 code object")
    (shoff,) = struct.unpack_from("<Q", elf, 0x28)
    shentsize, shnum = struct.unpack_from("<HH", elf, 0x3A)
    for k in range(shnum):
        o = shoff + k * shentsize
        if struct.unpack_from("<I", elf, o + 4)[0] != 7:  # SHT_NOTE
            continue
        off, size = struct.unpack_from("<QQ", elf, o + 0x18)
        p = off
        while p < off + size:
            nsz, dsz, typ = struct.unpack_from("<III", elf, p)
            p += 12
            name = elf[p : p + nsz].rstrip(b"\0")
            p += (nsz + 3) & ~3
            if name == b"AMDGPU" and typ == NT_AMDGPU_METADATA:
                return _mp(elf[p : p + dsz], 0)[0]
            p += (dsz + 3) & ~3
    raise ValueError("no AMDGPU metadata note")


def kernels(elf: bytes) -> dict[str, list[tuple[int, int]]]:
    """``{function: [(offset, size), ...]}``: each kernel's explicit (non-hidden) kernarg layout."""
    out = {}
    for k in metadata(elf)["amdhsa.kernels"]:
        args = [
            a for a in k.get(".args", []) if not a[".value_kind"].startswith("hidden_")
        ]
        out[k[".name"]] = [(a[".offset"], a[".size"]) for a in args]
    return out


def parse_layout(s: str) -> list[tuple[int, int]]:
    return [tuple(int(x) for x in t.split(":")) for t in s.split(",") if t]


@dataclass(frozen=True)
class Entry:
    path: str
    function: str
    layout: tuple[tuple[int, int], ...]


class CodeTable:
    """TileRT kernel symbol -> ``Entry`` (module docstring)."""

    def __init__(self, entries: dict[str, Entry] | None = None):
        self.entries = dict(entries or {})

    @classmethod
    def from_source(
        cls, cache_dir: str | Path | None = None, jobs: int | None = None, log=None
    ) -> CodeTable:
        """Our FlyDSL builds of every kernel the host launches (``flybuild.code_table``)."""
        from . import flybuild

        return flybuild.code_table(cache_dir, jobs=jobs, log=log)

    def missing(self, symbols) -> list[str]:
        return [s for s in symbols if s not in self.entries]


# ---------------------------------------------------------------------------------------------------------------
# HIP (device)


def _hip():
    """The process's HIP library (``atom.mono.runtime.hip_ipc``) with the module API's prototypes."""
    from atom.mono.runtime import hip_ipc

    lib = hip_ipc.load_runtime()
    if not getattr(lib, "_tk_module_api", False):
        vp, u = ctypes.c_void_p, ctypes.c_uint
        lib.hipModuleLoadData.argtypes = [ctypes.POINTER(vp), vp]
        lib.hipModuleGetFunction.argtypes = [ctypes.POINTER(vp), vp, ctypes.c_char_p]
        lib.hipModuleLaunchKernel.argtypes = [
            vp,
            u,
            u,
            u,
            u,
            u,
            u,
            u,
            vp,
            ctypes.POINTER(vp),
            ctypes.POINTER(vp),
        ]
        for f in ("hipModuleLoadData", "hipModuleGetFunction", "hipModuleLaunchKernel"):
            getattr(lib, f).restype = ctypes.c_int
        lib._tk_module_api = True
    return lib, hip_ipc.check_hip_error


class Launcher:
    """The table's functions loaded on the current device (``hipModuleLoadData``, each file once) and launched."""

    def __init__(self, table: CodeTable, symbols):
        lib, check = _hip()
        self.lib, self.check = lib, check
        self._images: dict[str, ctypes.Array] = {}  # (kept alive with their modules)
        modules: dict[str, ctypes.c_void_p] = {}
        self.functions: dict[str, tuple[ctypes.c_void_p, tuple]] = {}
        missing = table.missing(symbols)
        if missing:
            raise KeyError(
                f"no code object for {len(missing)} kernel(s), e.g. {missing[0]}"
            )
        for sym in dict.fromkeys(symbols):
            e = table.entries[sym]
            if e.path not in modules:
                image = ctypes.create_string_buffer(Path(e.path).read_bytes())
                m = ctypes.c_void_p()
                check(
                    lib.hipModuleLoadData(ctypes.byref(m), image),
                    operation=f"hipModuleLoadData({e.path})",
                )
                self._images[e.path], modules[e.path] = image, m
            fn = ctypes.c_void_p()
            check(
                lib.hipModuleGetFunction(
                    ctypes.byref(fn), modules[e.path], e.function.encode()
                ),
                operation=f"hipModuleGetFunction({e.function})",
            )
            self.functions[sym] = (fn, e.layout)

    def arg_bytes(self, symbol: str) -> int:
        """The end of the loaded function's last explicit argument: a blob passed to it holds at least this many
        bytes (params mode reads each argument's full size)."""
        return max((o + n for o, n in self.functions[symbol][1]), default=0)

    def params(self, symbol: str, blob) -> ctypes.Array:
        """``kernelParams`` for ``blob`` (a ctypes buffer kept alive by the caller)."""
        base = ctypes.addressof(blob)
        _, layout = self.functions[symbol]
        return (ctypes.c_void_p * max(len(layout), 1))(
            *[base + off for off, _ in layout]
        )

    def launch(
        self, symbol: str, grid, block, shm: int, stream: int, params: ctypes.Array
    ) -> None:
        fn, _ = self.functions[symbol]
        self.check(
            self.lib.hipModuleLaunchKernel(
                fn, *grid, *block, shm, ctypes.c_void_p(stream), params, None
            ),
            operation=f"hipModuleLaunchKernel({symbol[:60]})",
        )
