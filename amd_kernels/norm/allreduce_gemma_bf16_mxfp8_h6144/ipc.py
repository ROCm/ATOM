# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Graph-input registration, adapted from the existing norm family."""
import contextlib
import ctypes
import functools
import struct

import torch
import torch.distributed as dist

from amd_kernels.common import hip_runtime as ops
from amd_kernels.common.hip_runtime import load_hip_runtime

WORLD = 4
H = 6144
MAX_ENTRIES = 1 << 15  # captured calls (rows of the address table)


def _run_compiled(exe, *args):
    """First call compiles and runs (``flyc.compile``); later calls dispatch the
    cached CompiledFunction."""
    import flydsl.compiler as flyc

    cf = getattr(exe, "_cf", None)
    if cf is None:
        exe._cf = flyc.compile(exe, *args)
    else:
        cf(*args)


def _f32_bits(v: float) -> int:
    return struct.unpack("<i", struct.pack("<f", v))[0]


class _DevicePointer:
    """A raw device allocation as a tensor (``__cuda_array_interface__``)."""

    def __init__(self, ptr: int, nbytes: int):
        self.__cuda_array_interface__ = {
            "shape": (nbytes,),
            "typestr": "|u1",
            "data": (ptr, False),
            "version": 3,
        }


class _IpcHandle(ctypes.Structure):
    _fields_ = [("reserved", ctypes.c_char * 64)]


@functools.cache
def _hip():
    lib = load_hip_runtime()
    lib.hipMemGetAddressRange.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_void_p,
    ]
    lib.hipIpcGetMemHandle.argtypes = [ctypes.POINTER(_IpcHandle), ctypes.c_void_p]
    lib.hipIpcOpenMemHandle.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        _IpcHandle,
        ctypes.c_uint,
    ]
    return lib


def _alloc_handle(ptr: int) -> tuple[int, bytes]:
    """(base of the allocation holding ptr, its IPC handle's 64 raw bytes)."""
    hip = _hip()
    base, size = ctypes.c_void_p(), ctypes.c_size_t()
    rc = hip.hipMemGetAddressRange(ctypes.byref(base), ctypes.byref(size), ptr)
    assert rc == 0, f"hipMemGetAddressRange failed: {rc}"
    h = _IpcHandle()
    rc = hip.hipIpcGetMemHandle(ctypes.byref(h), base)
    assert rc == 0, f"hipIpcGetMemHandle failed: {rc}"
    # the raw bytes: a c_char field reads back only up to its first NUL
    return base.value, ctypes.string_at(ctypes.addressof(h), ctypes.sizeof(h))


def _open_handle(raw: bytes) -> int:
    hip = _hip()
    h = _IpcHandle()
    ctypes.memmove(ctypes.addressof(h), raw, len(raw))
    ptr = ctypes.c_void_p()
    rc = hip.hipIpcOpenMemHandle(ctypes.byref(ptr), h, 1)  # lazy peer access
    assert rc == 0, f"hipIpcOpenMemHandle failed: {rc}"
    return ptr.value


def exchange(obj, rank: int, group) -> list:
    """Every rank's ``obj`` (broadcasts on the CPU group: all_gather_object fails
    on gloo under inference mode)."""
    out = [[None] for _ in range(WORLD)]
    out[rank][0] = obj
    ranks = sorted(dist.get_process_group_ranks(group=group))
    for i, r in enumerate(ranks):
        dist.broadcast_object_list(out[i], src=r, group=group, device="cpu")
    return [o[0] for o in out]


class GraphInputs:
    """IPC mapping of graph inputs, exchanged only after capture ends."""
    @contextlib.contextmanager
    def capture(self):
        """Around graph capture (every rank, same captures): the fused calls
        recorded inside take table rows, filled when it exits."""
        self.capturing = True
        try:
            yield
        finally:
            self.capturing = False
            self.register_graph_buffers()

    def _row(self, x: torch.Tensor) -> int:
        n = len(self._inputs)
        assert n < MAX_ENTRIES, "allreduce_add_rmsnorm: address table full"
        self._inputs.append(x.data_ptr())
        return self.table.data_ptr() + n * WORLD * 8

    def register_graph_buffers(self) -> None:
        """Fill the rows taken since the last call with the 4 ranks' input
        addresses (a collective: the ranks captured the same calls in order)."""
        new = self._inputs[self._filled :]
        counts = exchange(len(new), self.rank, self.group)
        assert len(set(counts)) == 1, f"allreduce_add_rmsnorm: captured calls differ {counts}"
        if not new:
            return
        handles: list[bytes] = []
        index: dict[int, int] = {}
        metas = []
        for p in new:
            base, raw = _alloc_handle(p)
            if base not in index:
                index[base] = len(handles)
                handles.append(raw)
            metas.append((index[base], p - base))
        every = exchange((handles, metas), self.rank, self.group)
        rows = torch.empty((len(new), WORLD), dtype=torch.int64)
        for r, (hs, ms) in enumerate(every):
            if r == self.rank:
                rows[:, r] = torch.tensor(new, dtype=torch.int64)
                continue
            bases = []
            for raw in hs:
                if raw not in self._opened:
                    self._opened[raw] = _open_handle(raw)
                bases.append(self._opened[raw])
            rows[:, r] = torch.tensor([bases[i] + off for i, off in ms])
        self.table[self._filled : self._filled + len(new)].copy_(rows)
        torch.cuda.synchronize(self.device)
        self._filled += len(new)
