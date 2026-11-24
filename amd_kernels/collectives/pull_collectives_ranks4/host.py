# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The pools and the phase slots the pull collectives read (``kernel.py``).

``Pools`` is created once per rank (a collective: every rank of the group calls
it): a data pool of ``sets x 2`` slots in coarse-grained device memory, mapped on
the peers by IPC, and a flag word per set in uncached memory. A set belongs to a
DBO microbatch (its collectives are issued on its own comm stream hand-offs, so
each set keeps its own sequence); without DBO only set 0 is used.

A phase takes one slot (alternating, ``Collectives``): its collectives copy their
inputs into the slot, publish, wait for the peers and pull. Two slots per set
suffice: a rank writes slot ``k`` for phase ``c + 2`` only after its phase
``c + 1`` completed, i.e. after every peer published phase ``c + 1``, which a peer
does after its own pull of phase ``c`` (the previous use of slot ``k``) finished.
"""

import ctypes
import functools
import logging
import math

import torch
import torch.distributed as dist

from amd_kernels.common import hip_runtime as ops
from . import kernel as _kernel
from amd_kernels.common.hip_runtime import load_hip_runtime

logger = logging.getLogger(__name__)

WORLD = _kernel.WORLD
SLOTS = 2
_ALIGN = 256


def _run_compiled(exe, *args):
    """First call compiles and runs (``flyc.compile``); later calls dispatch the
    cached CompiledFunction."""
    import flydsl.compiler as flyc

    cf = getattr(exe, "_cf", None)
    if cf is None:
        exe._cf = flyc.compile(exe, *args)
    else:
        cf(*args)


@functools.cache
def _kernels():
    k = _kernel
    return k.compile_publish(), k.compile_fence_publish(), k.compile_sum4()


class _IpcHandle(ctypes.Structure):
    _fields_ = [("reserved", ctypes.c_char * 64)]


@functools.cache
def _hip():
    lib = load_hip_runtime()
    lib.hipMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
    lib.hipIpcGetMemHandle.argtypes = [ctypes.POINTER(_IpcHandle), ctypes.c_void_p]
    lib.hipIpcOpenMemHandle.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        _IpcHandle,
        ctypes.c_uint,
    ]
    return lib


class _DevicePointer:
    """A device allocation as torch sees it (``torch.as_tensor`` wraps it)."""

    def __init__(self, ptr: int, nbytes: int):
        self.__cuda_array_interface__ = {
            "data": (ptr, False),
            "shape": (nbytes,),
            "typestr": "|u1",
            "strides": None,
            "version": 3,
        }


def _alloc_pool(nbytes: int, device: torch.device) -> tuple[int, bytes, torch.Tensor]:
    """A hipMalloc pool (coarse-grained, never freed: it lives as long as the
    process) -> (pointer, IPC handle bytes, the pool as a uint8 tensor)."""
    hip = _hip()
    ptr = ctypes.c_void_p()
    with torch.cuda.device(device):
        rc = hip.hipMalloc(ctypes.byref(ptr), nbytes)
    assert rc == 0, f"hipMalloc({nbytes}) failed: {rc}"
    handle = _IpcHandle()
    rc = hip.hipIpcGetMemHandle(ctypes.byref(handle), ptr)
    assert rc == 0, f"hipIpcGetMemHandle failed: {rc}"
    pool = torch.as_tensor(_DevicePointer(ptr.value, nbytes), device=device)
    assert pool.data_ptr() == ptr.value
    # the raw 64 bytes (a c_char field reads back only up to its first NUL)
    raw = ctypes.string_at(ctypes.addressof(handle), ctypes.sizeof(handle))
    return ptr.value, raw, pool


def _open_pool(handle: bytes, device: torch.device) -> int:
    hip = _hip()
    ptr = ctypes.c_void_p()
    h = _IpcHandle()
    assert len(handle) == ctypes.sizeof(h)
    ctypes.memmove(ctypes.addressof(h), handle, len(handle))
    with torch.cuda.device(device):
        rc = hip.hipIpcOpenMemHandle(ctypes.byref(ptr), h, 1)  # lazy peer access
    assert rc == 0, f"hipIpcOpenMemHandle failed: {rc}"
    return ptr.value


def _exchange(obj, rank: int, world: int, group) -> list:
    """Every rank's ``obj`` (broadcasts: ``all_gather_object`` is unavailable on
    gloo under inference mode, which the profile run enters)."""
    out = [[None] for _ in range(world)]
    out[rank][0] = obj
    ranks = sorted(dist.get_process_group_ranks(group=group))
    for i, r in enumerate(ranks):
        dist.broadcast_object_list(out[i], src=r, group=group, device="cpu")
    return [o[0] for o in out]


class Pools:
    """One rank's pools and flags, with the peers' mapped. ``slot_bytes`` is the
    largest phase (the inputs of its collectives, 256 B aligned each)."""

    def __init__(
        self,
        rank: int,
        world: int,
        group,
        device: torch.device,
        slot_bytes: int,
        sets: int = 2,
    ):
        assert world == WORLD, world
        self.rank, self.world, self.device = rank, world, device
        self.sets, self.slot_bytes = sets, slot_bytes
        nbytes = sets * SLOTS * slot_bytes
        ptr, handle, self.pool = _alloc_pool(nbytes, device)
        flag_ptr, flag_handle = ops.allocate_shared_buffer_and_handle(_ALIGN)
        peers = _exchange((handle, flag_handle), rank, world, group)
        self.pool_ptrs = [
            ptr if r == rank else _open_pool(h, device)
            for r, (h, _) in enumerate(peers)
        ]
        self.flag_ptrs = [
            flag_ptr if r == rank else ops.open_mem_handle(h)
            for r, (_, h) in enumerate(peers)
        ]
        # per set: the sequence (the flag's next value) and the phase count
        self.seq = [0] * sets
        self.phase = [0] * sets
        publish, self.fence_publish, _ = _kernels()
        stream = torch.cuda.current_stream(device)
        # the flag word and the ticket counter of every set start at 0
        for s in range(sets):
            base = self.flag_ptrs[rank] + _kernel.FLAG_STRIDE * s
            _run_compiled(publish, base, 0, stream)
            _run_compiled(publish, base + 4, 0, stream)
        torch.cuda.synchronize(device)
        dist.barrier(group=group)
        logger.info(
            "pull_collectives_ranks4: %d sets x %d slots x %.0f MB pools",
            sets,
            SLOTS,
            slot_bytes / 2**20,
        )

    def slot(self, set_id: int) -> tuple[int, int]:
        """Begin a phase of ``set_id``: (byte offset of its slot in every pool,
        the set's flag offset)."""
        k = self.phase[set_id] % SLOTS
        self.phase[set_id] += 1
        return (set_id * SLOTS + k) * self.slot_bytes, _kernel.FLAG_STRIDE * set_id

    def next_seq(self, set_id: int) -> int:
        self.seq[set_id] += 1
        return self.seq[set_id]


class Collectives:
    """One phase in the next slot of ``set_id`` (the unpacking all-to-all of
    ``pull_unpack_q2048_shuffled`` reads it). Created before the compute that
    produces the phase's inputs: ``alloc`` hands the producers views of the slot,
    which the collective then reads in place; other inputs are copied in
    (``_stage``). Bound to the comm stream when the phase runs (``bind``)."""

    def __init__(self, pools: Pools, set_id: int):
        self.p = pools
        self.set_id = set_id
        self.base, self.flag_off = pools.slot(set_id)
        self.off = 0
        self.stream: torch.cuda.Stream | None = None

    def bind(self, stream: torch.cuda.Stream) -> "Collectives":
        self.stream = stream
        return self

    def alloc(self, shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor | None:
        """A ``shape`` tensor of ``dtype`` in the slot for a producer to write
        (the collective it feeds reads it in place); None when it does not fit."""
        n = math.prod(shape) * dtype.itemsize
        off = self._take(n)
        if off is None:
            return None
        return self.p.pool[off : off + n].view(dtype).view(shape)

    def _take(self, nbytes: int) -> int | None:
        """Reserve ``nbytes`` of the slot (256 B aligned); None when it does not fit."""
        off = (self.off + _ALIGN - 1) // _ALIGN * _ALIGN
        if off + nbytes > self.p.slot_bytes:
            return None
        self.off = off + nbytes
        return self.base + off

    def _stage(self, x: torch.Tensor) -> int | None:
        """``x``'s byte offset in every pool: where it is when ``alloc`` placed it
        (the same offset on every rank: the producers allocate in one order),
        else after a copy into the slot; None when it does not fit."""
        n = x.numel() * x.element_size()
        start = self.p.pool.data_ptr() + self.base
        if start <= x.data_ptr() < start + self.p.slot_bytes:
            assert x.data_ptr() + n <= start + self.p.slot_bytes
            return x.data_ptr() - self.p.pool.data_ptr()
        off = self._take(n)
        if off is None:
            return None
        dst = self.p.pool[off : off + n].view(x.dtype).view(x.shape)
        dst.copy_(x)
        return off

    def _publish_wait(self) -> None:
        """Fence + publish this rank's slot; the pull that follows waits for the
        peers' flags itself."""
        p = self.p
        self.seq = p.next_seq(self.set_id)
        me_flag = p.flag_ptrs[p.rank] + self.flag_off
        _run_compiled(p.fence_publish, me_flag, self.seq, self.stream)
