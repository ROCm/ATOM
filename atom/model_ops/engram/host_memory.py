# SPDX-License-Identifier: MIT
"""Page-locked host memory for an Engram UVA shard: pinning, copying, release.

`pin_rows` takes the row ranges a device kernel will read across the bus (a
table's values and, when quantized, its block scales) and returns them as
page-locked tensors -- copied into `hipHostMalloc` memory of exactly their
size, or the checkpoint mapping registered in place (`ATOM_ENGRAM_UVA_COPY=0`,
or no HIP runtime). Whole tables every DP-attention rank of the node holds
are copied once into memory those ranks share (`_shared_copy`).

Registrations go through a process-wide, reference-counted set of page runs
(`_PageRuns`): ranges are widened to page boundaries, so ranges stored back
to back (a layer's scales right before its values) share a page, and HIP
refuses to register a range overlapping one already registered. Gaps are
registered in 1 GiB pieces.

Every refused or failed HIP call clears HIP's last error (`_clear_hip_error`):
HIP also leaves it as the runtime's sticky last error, which torch raises at
its next kernel-launch check -- in the host-path fallback the refusal was
meant to allow.

Ownership: each returned tensor's storage holds the lease on its memory, so
the memory is released when the last tensor viewing it is collected -- never
while one is alive, and not leaked when its owner is dropped without an
explicit release. In-place runs are also `mlock`ed while registered and
unlocked with their last reference (`_PageRuns`).

Release and the device: `hipHostFree` synchronizes the device before freeing,
and an in-place registration synchronizes before unregistering, so kernels
already launched on the memory finish first. A CUDA graph captured over the
memory keeps its address without a reference, so the memory must outlive any
graph that reads it; Engram releases a shard only when the TP group falls back
at load, before capture, or at exit. A release that would have to synchronize
during a capture is skipped, with a warning, rather than breaking it.
"""

import ctypes
import dataclasses
import errno
import functools
import logging
import mmap
import os
import resource
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait

import numpy as np
import torch

from atom.distributed.group_agreement import group_agree
from atom.utils import envs

logger = logging.getLogger("atom")

_PAGE = 4096
_MADV_DONTNEED = 4
_MADV_POPULATE_READ = 22  # Linux 5.14+; elsewhere EINVAL, and nothing is lost
_MLOCK_ONFAULT = 1
# `hipHostRegisterReadOnly`: the GPU only reads the range. Needed for a range
# the process cannot write (the shared read-only checkpoint mapping), which
# the default flags would ask to map writable.
_HOST_REGISTER_READ_ONLY = 0x08
# Not exported by Python's `mmap`: map at exactly the address given, over
# whatever was reserved there (`_StitchedMapping`).
_MAP_FIXED = 0x10
_PROT_NONE = 0
_MAP_FAILED = ctypes.c_void_p(-1).value
_PR_SET_THP_DISABLE = 41
_PR_GET_THP_DISABLE = 42
# Copy policy, from `tools/engram/bench_host_copy.py` on MI355X (warm source,
# OMP_NUM_THREADS=4; one copy_ against N memmove threads):
#   256 MiB  copy_ 3.8 ms   2 thr 8.0   4 thr 4.4   8 thr 4.1
#   512 MiB  copy_ 7.5 ms   4 thr 7.0   8 thr 5.9
#     1 GiB  copy_ 15.3 ms  8 thr 8.8   16 thr 10.4
#    16 GiB  copy_ 215 ms   8 thr 120   16 thr 107
# Two or three threads always lose to one copy_. So: one `copy_` below
# `_THREADS_FROM` x `_BYTES_PER_THREAD` (512 MiB), otherwise one thread per
# `_BYTES_PER_THREAD` (128 MiB) but at least `_MIN_COPY_THREADS` (8), at most
# `_MAX_COPY_THREADS` and the CPUs this process may run on.
_BYTES_PER_THREAD = 128 << 20
_THREADS_FROM = 4
_MIN_COPY_THREADS = 8
# Largest chunk one copy thread takes at a time; bounds how much copied
# source is still mapped before it is dropped (`_source_dropper`).
_CHUNK_BYTES = 256 << 20
_MAX_COPY_THREADS = 16
# Largest single cudaHostRegister. One ioctl covering a whole Engram table
# does not return when several ranks issue it together.
_REG_CHUNK_BYTES = 1 << 30


@functools.cache
def _libc():
    """This process's libc, with the calls used here declared once."""
    libc = ctypes.CDLL(None, use_errno=True)
    libc.madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    libc.mlock2.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    libc.munlock.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    libc.prctl.argtypes = [ctypes.c_int] + [ctypes.c_ulong] * 4
    libc.mmap.restype = ctypes.c_void_p
    libc.mmap.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_long,
    ]
    libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    return libc


def _page_span(t: torch.Tensor) -> tuple[int, int]:
    """`[lo, hi)`: the whole pages `t`'s bytes lie on."""
    base = t.data_ptr()
    lo = base - base % _PAGE
    hi = -(-(base + t.numel() * t.element_size()) // _PAGE) * _PAGE
    return lo, hi


def _mapped_libraries(prefix: str) -> list[str]:
    """Paths of the shared libraries named `prefix*` this process has loaded
    (`/proc/self/maps`), in load order."""
    paths: list[str] = []
    try:
        with open("/proc/self/maps") as maps:
            for line in maps:
                path = line.split(maxsplit=5)[-1].strip()
                if os.path.basename(path).startswith(prefix) and path not in paths:
                    paths.append(path)
    except OSError:
        pass
    return paths


@functools.cache
def _hip_runtime():
    """The HIP runtime this process already runs -- the `libamdhip64` torch
    loaded (a ROCm wheel bundles its own, e.g. `torch/lib/libamdhip64.so.7`)
    -- or None.

    Not `CDLL("libamdhip64.so")`: that is the system's unversioned dev link,
    which a wheel-only install lacks (the copy would quietly turn into an
    in-place registration) and which can name a second, different runtime
    than torch's, with its own allocations and last error.
    """
    if not torch.version.hip:
        return None
    candidates = _mapped_libraries("libamdhip64.so")
    torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
    try:
        candidates += sorted(
            os.path.join(torch_lib, n)
            for n in os.listdir(torch_lib)
            if n.startswith("libamdhip64.so")
        )
    except OSError:
        pass
    candidates.append("libamdhip64.so")
    for path in candidates:
        try:
            hip = ctypes.CDLL(path)
        except OSError:
            continue
        logger.info("engram: HIP runtime for page-locking: %s", path)
        return hip
    logger.warning(
        "engram: no HIP runtime found (tried %s); a UVA shard is registered in "
        "place instead of copied",
        ", ".join(candidates),
    )
    return None


@functools.cache
def _host_get_device_pointer():
    """`hipHostGetDevicePointer` of the HIP runtime that page-locks
    (`_hip_runtime`), resolved once; None without one."""
    hip = _hip_runtime()
    fn = None if hip is None else getattr(hip, "hipHostGetDevicePointer", None)
    if fn is not None:
        fn.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_uint]
    return fn


def device_address(
    host_ptr: int, base: int | None = None, *, identity_without_hip: bool = False
) -> int:
    """The device's address of `host_ptr`, a byte of the page-locked host
    range starting at `base` (default: `host_ptr` itself). Not always the host
    address: memory registered in place is mapped at another device address,
    and a kernel handed the host one faults.

    Without a HIP runtime to ask it raises, or with `identity_without_hip`
    returns `host_ptr` itself (where the caller's host memory is read at its
    host address: copies the HIP allocator page-locked, or no device at all).
    A run ROCr locked for this GPU alone is read at its host address too."""
    if _PAGE_RUNS.locked_alone(host_ptr if base is None else base):
        return host_ptr
    fn = _host_get_device_pointer()
    if fn is None:
        if identity_without_hip:
            return host_ptr
        raise RuntimeError("no HIP runtime loaded: no device address of host memory")
    base = host_ptr if base is None else base
    dev = ctypes.c_void_p()
    rc = fn(ctypes.byref(dev), base, 0)
    if rc != 0 or not dev.value:
        raise RuntimeError(f"hipHostGetDevicePointer({base:#x}) failed ({rc})")
    return dev.value + (host_ptr - base)


def shard_device_address(shard: torch.Tensor) -> int:
    """The device's address of a page-locked shard tensor (`pin_rows`), for
    a kernel that dereferences it from a table of addresses: its storage
    starts at its page-locked region's base."""
    return device_address(shard.data_ptr(), shard.untyped_storage().data_ptr())


def _clear_hip_error() -> None:
    """Reset HIP's last error after a refused or failed call.

    HIP reports the failure through the return code and also leaves it as the
    runtime's last error, and torch raises that at its next kernel-launch
    check: the host-path fallback would die there ("part or all of the
    requested memory range is already mapped").
    """
    hip = _hip_runtime()
    if hip is not None:
        hip.hipGetLastError()


# ROCr's agent queries (hsa.h, hsa_ext_amd.h)
_HSA_AGENT_INFO_DEVICE = 17
_HSA_DEVICE_TYPE_GPU = 1
_HSA_AMD_AGENT_INFO_BDFID = 0xA006
_HSA_AMD_AGENT_INFO_DOMAIN = 0xA00F


class _HsaAgent(ctypes.Structure):
    _fields_ = [("handle", ctypes.c_uint64)]


_HSA_VISIT = ctypes.CFUNCTYPE(ctypes.c_int, _HsaAgent, ctypes.c_void_p)


def _hsa_gpu_agent(hsa, domain: int, bus: int, device: int):
    """ROCr's GPU agent at PCI `domain:bus:device` (its BDF id is bus, device,
    function packed), or None. By address, not by order: HIP_VISIBLE_DEVICES
    renumbers HIP's devices, not ROCr's agents."""
    hsa.hsa_agent_get_info.argtypes = [_HsaAgent, ctypes.c_int, ctypes.c_void_p]
    found = []

    def info(agent, attribute, value) -> bool:
        return hsa.hsa_agent_get_info(agent, attribute, ctypes.byref(value)) == 0

    def visit(agent, _data):
        kind, bdf, dom = ctypes.c_int(), ctypes.c_uint32(), ctypes.c_uint32()
        if not info(agent, _HSA_AGENT_INFO_DEVICE, kind):
            return 0
        if kind.value != _HSA_DEVICE_TYPE_GPU:
            return 0
        if info(agent, _HSA_AMD_AGENT_INFO_BDFID, bdf) and info(
            agent, _HSA_AMD_AGENT_INFO_DOMAIN, dom
        ):
            here = (dom.value, bdf.value >> 8, (bdf.value >> 3) & 0x1F)
            if here == (domain, bus, device):
                found.append(_HsaAgent(agent.handle))
        return 0

    callback = _HSA_VISIT(visit)
    hsa.hsa_iterate_agents.argtypes = [_HSA_VISIT, ctypes.c_void_p]
    status = hsa.hsa_iterate_agents(callback, None)
    if status != 0:
        logger.info("engram: listing ROCr agents failed (status %#x)", status)
        return None
    return found[0] if len(found) == 1 else None


class _DeviceLock:
    """Page-locking for this process's GPU alone, through ROCr
    (`hsa_amd_memory_lock` with its agent). `hipHostRegister` maps a range
    into every visible GPU, whatever its flags: a node's ranks each see all
    of the node's GPUs, so each registration costs one mapping a GPU. The
    agent's pointer must be the host pointer, which the UVA kernels address."""

    def __init__(self, hsa, agent):
        self._lock_fn, self._unlock_fn = (
            hsa.hsa_amd_memory_lock,
            hsa.hsa_amd_memory_unlock,
        )
        self._lock_fn.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.POINTER(_HsaAgent),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self._unlock_fn.argtypes = [ctypes.c_void_p]
        self._agents = (_HsaAgent * 1)(agent)

    def lock(self, ptr: int, nbytes: int) -> bool:
        out = ctypes.c_void_p()
        if self._lock_fn(ptr, nbytes, self._agents, 1, ctypes.byref(out)) != 0:
            return False
        if out.value != ptr:
            self._unlock_fn(ptr)
            return False
        return True

    def unlock(self, ptr: int) -> bool:
        return self._unlock_fn(ptr) == 0


def _device_lock():
    """This process's GPU's `_DeviceLock`, or None (no GPU, no ROCr to call,
    no agent at the GPU's PCI address): registration then goes through HIP."""
    if not torch.cuda.is_available():
        return None
    return _device_lock_for(torch.cuda.current_device())


@functools.cache
def _device_lock_for(device: int):
    # The ROCr HIP already runs on, never a second copy by name: one not
    # initialized lists no agents (a wheel bundling its own ROCr)
    mapped = _mapped_libraries("libhsa-runtime64.so")
    try:
        hsa = ctypes.CDLL(mapped[0]) if mapped else None
    except OSError:
        hsa = None
    if hsa is None:
        logger.info("engram: no ROCr loaded; registering through HIP")
        return None
    props = torch.cuda.get_device_properties(device)
    agent = _hsa_gpu_agent(
        hsa, props.pci_domain_id, props.pci_bus_id, props.pci_device_id
    )
    if agent is None:
        logger.info(
            "engram: no ROCr agent for device %d; registering through HIP", device
        )
        return None
    return _DeviceLock(hsa, agent)


def _register_range(rt, ptr: int, nbytes: int, flags: int):
    """Page-lock `[ptr, ptr + nbytes)` for this process's GPU: through ROCr to
    it alone where it can, else `cudaHostRegister`. `(registered, lock)`: the
    `_DeviceLock` that holds it, None for HIP's; refused either way, HIP's
    error cleared."""
    lock = _device_lock()
    if lock is not None and lock.lock(ptr, nbytes):
        return True, lock
    if int(rt.cudaHostRegister(ptr, nbytes, flags)):
        _clear_hip_error()
        return False, None
    return True, None


def _unregister_range(rt, ptr: int, lock) -> bool:
    """Undo `_register_range` (`lock`: the `_DeviceLock` that holds it, None
    for HIP's); False: refused."""
    if lock is not None:
        return lock.unlock(ptr)
    if int(rt.cudaHostUnregister(ptr) or 0):
        _clear_hip_error()
        return False
    return True


class _HipHostAllocator:
    """`hipHostMalloc`/`hipHostFree` through ctypes, on base pages.

    Not torch's pinned allocator: that one rounds every block up to a power of
    two, which for a large table shard wastes a large share of page-locked
    memory. ctypes also drops the GIL for the call.

    Transparent huge pages are disabled for this process around the call
    (`PR_SET_THP_DISABLE`, restored after). `hipHostMalloc` otherwise asks for
    huge pages, and once the page cache has fragmented memory -- as loading
    a checkpoint does -- the kernel compacts memory inside the call: a TP2
    shard took ~197 s per rank that way (~143k compaction stalls), ~12 s on
    base pages. The setting is process-wide, so another thread faulting in
    memory at that moment also gets base pages; that costs it nothing but
    huge-page TLB reach.
    """

    def __init__(self, hip, libc):
        self._malloc, self._free = hip.hipHostMalloc, hip.hipHostFree
        self._malloc.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_size_t,
            ctypes.c_uint,
        ]
        self._free.argtypes = [ctypes.c_void_p]
        self._prctl = libc.prctl

    def alloc(self, nbytes: int) -> int | None:
        was = self._prctl(_PR_GET_THP_DISABLE, 0, 0, 0, 0)
        self._prctl(_PR_SET_THP_DISABLE, 1, 0, 0, 0)
        try:
            ptr = ctypes.c_void_p()
            rc = self._malloc(ctypes.byref(ptr), max(nbytes, 1), 0)
        finally:
            if was == 0:
                self._prctl(_PR_SET_THP_DISABLE, 0, 0, 0, 0)
        if rc != 0 or not ptr.value:
            _clear_hip_error()
            return None
        return ptr.value

    def free(self, ptr: int) -> None:
        # Synchronizes the device first (an implicit hipDeviceSynchronize).
        rc = self._free(ptr)
        if rc != 0:
            _clear_hip_error()
            logger.warning("engram: hipHostFree failed (%d)", rc)


def _host_allocator():
    """The page-locked allocator a shard is copied into, or None when there is
    no HIP runtime to call (the shard is then registered in place)."""
    hip = _hip_runtime()
    if hip is None:
        return None
    try:
        return _HipHostAllocator(hip, _libc())
    except (OSError, AttributeError):
        logger.warning("engram: hipHostMalloc unavailable", exc_info=True)
        return None


def copies() -> bool:
    """Whether a rank's shard is copied into page-locked memory (else
    registered in place): `ATOM_ENGRAM_UVA_COPY`."""
    return envs.ATOM_ENGRAM_UVA_COPY


def _capturing() -> bool:
    return (
        torch.cuda.is_available()
        and torch.cuda.is_initialized()
        and torch.cuda.is_current_stream_capturing()
    )


# Leases whose release came during a graph capture (`_Lease.release`).
_RETAINED: list = []


class _Lease:
    """One page-locked region, released exactly once: explicitly, or when the
    last tensor over it is collected."""

    def __init__(self, ptr: int, nbytes: int, release_fn, keep=None):
        self.ptr, self.nbytes = ptr, nbytes
        self._release_fn = release_fn
        # What the region lies in, kept alive with it (the checkpoint mapping
        # of an in-place registration).
        self._keep = keep

    def release(self) -> None:
        ptr, self.ptr = self.ptr, None
        if ptr is None or sys.is_finalizing():
            return  # at exit the runtime may be gone; the OS reclaims it
        if _capturing():
            # A graph being captured may hold this address and nothing tells
            # when it can no longer run: keep the region, and what it lies in,
            # for the life of the process.
            logger.warning(
                "engram: page-locked shard released during graph capture; "
                "keeping it for the life of the process"
            )
            self.ptr = ptr
            _RETAINED.append(self)
            return
        self._release_fn(ptr)
        self._keep = None

    def __del__(self):
        try:
            self.release()
        except Exception:  # noqa: BLE001 - never raise from a finalizer
            logger.warning("engram: releasing a page-locked shard failed")


def _owning_bytes(lease: _Lease, offset: int, nbytes: int) -> torch.Tensor:
    """`nbytes` uint8 at `lease.ptr + offset`, as a tensor whose storage keeps
    `lease` alive (`torch.frombuffer` holds a reference to the buffer object,
    and the buffer holds the lease)."""
    buf_type = type("_LeasedBytes", (ctypes.c_char * max(lease.nbytes, 1),), {})
    buf = buf_type.from_address(lease.ptr)
    buf.lease = lease
    return torch.frombuffer(buf, dtype=torch.uint8)[offset : offset + nbytes]


def copy_threads(nbytes: int) -> int:
    """Threads to copy `nbytes` on (1: a single `copy_`)."""
    per = nbytes // _BYTES_PER_THREAD
    if per < _THREADS_FROM:
        return 1
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else 1
    return max(1, min(_MAX_COPY_THREADS, cpus, max(_MIN_COPY_THREADS, per)))


def _copy_into(dst: torch.Tensor, src: torch.Tensor, pool, drop=None) -> None:
    """`dst.copy_(src)` for flat byte tensors; large ones in chunks on `pool`,
    each chunk's source then handed to `drop(offset, nbytes)`.

    The chunks use `ctypes.memmove`, which drops the GIL and, unlike `copy_`,
    does not fan out again over torch's intra-op pool, so the thread count is
    what `copy_threads` chose and torch's global setting is left alone.
    """
    nbytes = src.numel()
    threads = copy_threads(nbytes)
    s = src.data_ptr()
    if threads == 1 or pool is None:
        if drop is None:
            dst.copy_(src)
            return
        # One worker, still in chunks, so the copied part of the source goes
        # before the rest comes in (peak ~one shard).
        for o, n in _chunks(s, nbytes, _CHUNK_BYTES):
            dst[o : o + n].copy_(src[o : o + n])
            drop(o, n)
        return
    d = dst.data_ptr()

    def chunk(piece) -> None:
        o, n = piece
        ctypes.memmove(d + o, s + o, n)
        if drop is not None:
            drop(o, n)

    step = min(-(-nbytes // (threads * 4)), _CHUNK_BYTES)
    # Every submitted chunk finishes before this returns or raises: the
    # caller then closes `drop`, whose descriptor the chunks still read.
    futures = []
    try:
        for piece in _chunks(s, nbytes, step):
            futures.append(pool.submit(chunk, piece))
    except BaseException:
        # A failed submit (no thread to start) can leave its chunk queued
        # with no future to wait on: cancel what is queued and wait for the
        # workers themselves, so nothing writes `dst` or reads `drop` after.
        pool.shutdown(wait=True, cancel_futures=True)
        # Cancelled by shutdown, never run: `wait` would not count these.
        futures = [f for f in futures if not f.cancelled()]
        raise
    finally:
        wait(futures)
    for f in futures:
        f.result()


def _file_mapped(addr: int, nbytes: int) -> bool:
    """Whether `[addr, addr + nbytes)` lies wholly, without a gap, in mappings
    of one regular file (`/proc/self/maps`; cheap, no page-table walk)."""
    end, covered, path = addr + nbytes, addr, None
    try:
        with open("/proc/self/maps") as maps:
            for line in maps:
                fields = line.split(maxsplit=5)
                lo, hi = (int(x, 16) for x in fields[0].split("-"))
                if hi <= addr or lo >= end:
                    continue
                name = fields[5].strip() if len(fields) > 5 else ""
                if lo > covered or not name.startswith("/"):
                    return False
                if name.startswith("/dev/") or path not in (None, name):
                    return False
                path, covered = name, max(covered, hi)
    except (OSError, ValueError, IndexError):
        return False
    return path is not None and covered >= end


# /proc/self/pagemap, one little-endian u64 per page: bit 63 present, 62
# swapped, 61 file page (or shared anonymous).
_PM_PRESENT, _PM_SWAPPED, _PM_FILE = 1 << 63, 1 << 62, 1 << 61


def _source_dropper(src: torch.Tensor):
    """`drop(offset, nbytes)` unmapping that byte range of `src`'s checkpoint
    pages from this process, or None when `src` is not a file mapping.

    Clean file pages: dropping them leaves the data in the page cache
    (reclaimable, not this process's RSS), and a later read faults it back.
    Without this the copy and its source are both resident, twice the shard;
    dropping each chunk as it is copied also keeps the peak near one shard.

    Only pages that still match the file are dropped. A page written through
    a private mapping is an anonymous copy (whatever the mapping's current
    protection, and wherever `mprotect` split it); dropping it would put the
    file's bytes back under the table. Each chunk's pages are checked in
    `/proc/self/pagemap` just before, and a chunk holding any such page is
    kept whole. This reads 8 bytes per page of the chunk, on the copy thread
    (~120 ms per 46 GiB, single-threaded), rather than walking every mapping
    of the process as `/proc/self/smaps` does (~0.36 s per call there).
    """
    base = src.data_ptr()
    if not _file_mapped(base, src.numel() * src.element_size()):
        return None
    try:
        pagemap = os.open("/proc/self/pagemap", os.O_RDONLY)
    except OSError:
        return None
    libc = _libc()

    def unmodified(lo: int, hi: int) -> bool:
        first, count = lo // _PAGE, (hi - lo) // _PAGE
        step = 1 << 16  # pages per read
        for p in range(0, count, step):
            n = min(step, count - p)
            raw = os.pread(pagemap, 8 * n, 8 * (first + p))
            if len(raw) != 8 * n:
                return False
            entries = np.frombuffer(raw, dtype="<u8")
            used = (entries & np.uint64(_PM_PRESENT | _PM_SWAPPED)) != 0
            if np.any(used & ((entries & np.uint64(_PM_FILE)) == 0)):
                return False
        return True

    # `close` waits for drops in progress; a drop after it does nothing (the
    # descriptor number may belong to another file by then).
    state = threading.Condition()
    active, closed = [0], []

    def drop(offset: int, nbytes: int) -> None:
        with state:
            if closed:
                return
            active[0] += 1
        try:
            _drop(offset, nbytes)
        finally:
            with state:
                active[0] -= 1
                state.notify_all()

    def _drop(offset: int, nbytes: int) -> None:
        # Whole pages inside the range only (`_chunks` aligns chunk edges).
        lo = -(-(base + offset) // _PAGE) * _PAGE
        hi = (base + offset + nbytes) // _PAGE * _PAGE
        if hi <= lo:
            return
        if not unmodified(lo, hi):
            logger.debug("engram: source chunk holds written pages; kept")
            return
        if libc.madvise(lo, hi - lo, _MADV_DONTNEED) != 0:
            logger.debug("engram: madvise(DONTNEED) failed: %d", ctypes.get_errno())

    def close() -> None:
        with state:
            if closed:
                return
            closed.append(True)
            state.wait_for(lambda: active[0] == 0)
        os.close(pagemap)

    drop.close = close
    return drop


def _chunks(addr: int, nbytes: int, step: int):
    """`(offset, size)` covering `[0, nbytes)` in about `step` bytes, cut at
    page boundaries of `addr + offset`, so a chunk's whole pages are its own
    and only the range's two edge pages belong to no chunk."""
    step = max(_PAGE, step // _PAGE * _PAGE)
    first = min(nbytes, (-addr) % _PAGE or step)
    out, o = [], 0
    for end in [first, *range(first + step, nbytes, step), nbytes]:
        if end > o:
            out.append((o, end - o))
            o = end
    return out


def _as_bytes(t: torch.Tensor) -> torch.Tensor:
    if not t.is_contiguous():
        raise ValueError("engram UVA rows must be one contiguous range")
    return t.reshape(-1).view(torch.uint8)


def pin_rows(
    sources: list[torch.Tensor],
    copy: bool = False,
    share: "NodeShare | None" = None,
) -> tuple[torch.Tensor, ...] | None:
    """`sources` (contiguous row ranges) as page-locked tensors of the same
    dtype and shape, owning their memory; all-or-nothing.

    `share`: these are whole tables every rank of `share` holds alike (DP
    attention), one copy of them on the node (`_shared_copy`); any failure
    there raises `SharedTablesError` on every rank of `share`.

    `copy` copies whatever `ATOM_ENGRAM_UVA_COPY` says, and never registers
    in place, not even without an allocator.

    Otherwise None, with everything taken released, on any failure -- no
    memory, a refused registration, no thread to copy on -- so the group can
    fall back as one.
    """
    if share is not None:
        return _shared_copy(share, sources)
    plan = ["copy"] if copy or copies() else ["register"]
    for how in plan:
        allocator = _host_allocator() if how == "copy" else None
        if how == "copy" and allocator is None:
            if copy or "register" in plan:
                break  # registering is not wanted, or was refused already
            how = "register"
        leases: list[_Lease] = []
        try:
            if how == "copy":
                out = _copy(allocator, sources, leases)
            else:
                out = _register(sources, leases)
        except Exception:  # any failure is a fallback
            logger.warning("engram: page-locking the UVA shard failed", exc_info=True)
            out = None
        if out is not None:
            return out
        for lease in leases:
            lease.release()
    return None


def _copy(allocator, sources, leases):
    largest = max((s.numel() * s.element_size() for s in sources), default=0)
    threads = copy_threads(largest)
    pool = ThreadPoolExecutor(threads) if threads > 1 else None
    try:
        out = []
        for src in sources:
            nbytes = src.numel() * src.element_size()
            ptr = allocator.alloc(nbytes)
            if ptr is None:
                logger.info("engram: no page-locked memory for %d bytes", nbytes)
                return None
            lease = _Lease(ptr, nbytes, allocator.free)
            leases.append(lease)
            dst = _owning_bytes(lease, 0, nbytes)
            drop = _source_dropper(src)
            try:
                _copy_into(dst, _as_bytes(src), pool, drop)
            finally:
                if drop is not None:
                    drop.close()
            out.append(dst.view(src.dtype).view(src.shape))
        return tuple(out)
    finally:
        if pool is not None:
            pool.shutdown()


class SharedTablesError(RuntimeError):
    """The ranks of a `NodeShare` could not hold one copy of the tables a
    node; raised on every rank of it alike."""


@dataclasses.dataclass(frozen=True)
class NodeShare:
    """The ranks that hold the same whole tables (DP attention: one-GPU TP
    groups), on one node or several: `group` their gloo group, `rank` this
    rank's place in it. Each node's ranks page-lock one copy of a table, each
    rank filling its own segment of it (`_shared_copy`); a failure on any rank
    is raised on every one (`SharedTablesError`)."""

    group: object
    rank: int

    @property
    def size(self) -> int:
        return torch.distributed.get_world_size(self.group)

    def all_gather(self, value) -> list:
        out = [None] * self.size
        torch.distributed.all_gather_object(out, value, group=self.group)
        return out

    def barrier(self) -> None:
        torch.distributed.barrier(group=self.group)

    def agree(self, ok: bool) -> bool:
        return group_agree(ok, self.group)


def _segment_bounds(total: int, ranks: int) -> list[int]:
    """`ranks + 1` page-aligned cut points of `[0, total)`: rank r's segment
    is `[bounds[r], bounds[r + 1])`, possibly empty."""
    return [total * r // ranks // _PAGE * _PAGE for r in range(ranks)] + [total]


class _StitchedMapping:
    """One reserved address range with the ranks' segments, each its own
    memfd, mapped back to back over it: one contiguous view of the tables."""

    def __init__(self, nbytes: int):
        flags = mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS
        base = _libc().mmap(None, nbytes, _PROT_NONE, flags, -1, 0)
        if base in (None, _MAP_FAILED):
            raise OSError(ctypes.get_errno(), "reserving the shared tables' range")
        self.base, self.nbytes = base, nbytes

    def place(self, fd: int, offset: int, nbytes: int) -> None:
        want = self.base + offset
        got = _libc().mmap(
            want,
            nbytes,
            mmap.PROT_READ | mmap.PROT_WRITE,
            mmap.MAP_SHARED | _MAP_FIXED,
            fd,
            0,
        )
        if got != want:
            raise OSError(ctypes.get_errno(), "mapping a shared tables' segment")

    def close(self) -> None:
        base, self.base = self.base, None
        if base is not None and _libc().munmap(base, self.nbytes) != 0:
            logger.warning("engram: munmap failed: %d", ctypes.get_errno())


def _proc_domain() -> tuple[str, int] | None:
    """Which ranks can open each other's descriptors through `/proc/<pid>/fd`:
    those of one kernel boot and one PID namespace (a node, a container).
    None, sharing nothing, where it cannot be read."""
    try:
        with open("/proc/sys/kernel/random/boot_id") as f:
            boot = f.read().strip()
        return boot, os.stat("/proc/self/ns/pid").st_ino
    except OSError:
        logger.warning("engram: no /proc domain to share in", exc_info=True)
        return None


def _segment_memfd(nbytes: int) -> int | None:
    try:
        fd = os.memfd_create("atom_engram", os.MFD_CLOEXEC)
    except OSError:
        logger.warning("engram: no memfd for the shared tables", exc_info=True)
        return None
    try:
        os.ftruncate(fd, nbytes)
    except OSError:
        logger.warning("engram: cannot size a shared tables' segment", exc_info=True)
        os.close(fd)
        return None
    return fd


def _attempt(what: str, step):
    """`step()`, or None with a warning if it raises: a local failure becomes
    this rank's vote at the group's next agreement, never an exception that
    leaves the other ranks waiting in it."""
    try:
        return step()
    except Exception:  # any local failure is a vote to fall back
        logger.warning("engram: %s failed", what, exc_info=True)
        return None


def _stitch(owners: list, bounds: list[int]) -> _StitchedMapping:
    """Every rank's segment (`owners[r]`: its `(pid, fd)`) mapped in place, a
    peer's through its descriptor (`/proc/<pid>/fd`)."""
    stitched = _StitchedMapping(bounds[-1])
    try:
        for r, (pid, fd) in enumerate(owners):
            lo, hi = bounds[r], bounds[r + 1]
            if hi == lo:
                continue
            if pid == os.getpid():
                stitched.place(fd, lo, hi - lo)
                continue
            peer = os.open(f"/proc/{pid}/fd/{fd}", os.O_RDWR)
            try:
                stitched.place(peer, lo, hi - lo)
            finally:
                os.close(peer)
    except BaseException:
        stitched.close()
        raise
    return stitched


def _raw_bytes(addr: int, nbytes: int) -> torch.Tensor:
    return torch.frombuffer(
        (ctypes.c_char * nbytes).from_address(addr), dtype=torch.uint8
    )


def _fill_segment(sources, offsets, base: int, lo: int, hi: int) -> bool:
    """The part of `sources` (laid out at `offsets` from `base`) that falls in
    `[lo, hi)`, copied there; True."""
    start = time.monotonic()
    threads = copy_threads(hi - lo)
    pool = ThreadPoolExecutor(threads) if threads > 1 else None
    try:
        for src, off in zip(sources, offsets, strict=True):
            a = max(off, lo)
            b = min(off + src.numel() * src.element_size(), hi)
            if a < b:
                _copy_into(
                    _raw_bytes(base + a, b - a), _as_bytes(src)[a - off : b - off], pool
                )
    finally:
        if pool is not None:
            pool.shutdown()
    logger.info(
        "engram: shared tables' segment filled, %.2f GiB in %.1f s",
        (hi - lo) / 2**30,
        time.monotonic() - start,
    )
    return True


def _unshare(rt, runs, stitched: _StitchedMapping, _ptr: int) -> None:
    """A shared copy's release: kernels launched on it finish, its runs are
    unlocked, every segment unmapped."""
    if torch.cuda.is_initialized():
        torch.cuda.synchronize()
    _PAGE_RUNS.release(rt, runs)
    stitched.close()


def _one_domain_a_node(domains: list) -> None:
    """Raises `SharedTablesError` unless every rank has a `_proc_domain` and
    each node's ranks are in one: ranks of one node in several PID namespaces
    cannot open each other's descriptors, and each namespace would hold a
    whole copy of the tables."""
    missing = [r for r, d in enumerate(domains) if d is None]
    if missing:
        raise SharedTablesError(
            f"engram: DP ranks {missing} have no /proc domain to share the " "tables in"
        )
    namespaces: dict[str, set[int]] = {}
    for boot, pid_ns in domains:
        namespaces.setdefault(boot, set()).add(pid_ns)
    split = sorted(len(ns) for ns in namespaces.values() if len(ns) > 1)
    if split:
        raise SharedTablesError(
            f"engram: a node's DP ranks are in {split[-1]} PID namespaces, "
            "which cannot share one copy of the tables"
        )


def _shared_copy(share: NodeShare, sources):
    """`sources` copied once a node (a `_proc_domain`'s ranks of `share`),
    each at a page boundary of one layout cut into a segment a rank of the
    node: each rank fills its own memfd, and maps
    every rank's back to back into one contiguous range, which it page-locks
    -- a memfd's registration is as fast as anonymous memory's, a file-backed
    mapping registered in place is not. A memfd a rank: the ranks' copies
    into one memfd fault its pages in against each other. Every step is
    collective, and a failure on any rank is raised on every rank
    (`SharedTablesError`): each local step votes at the next agreement
    (`_attempt`), and what this rank holds is released on any way out until
    a lease owns it."""
    offsets, total = [], 0
    for src in sources:
        offsets.append(total)
        total += max(-(-src.numel() * src.element_size() // _PAGE), 1) * _PAGE
    # one copy a node: its ranks of `share` (a DP group may span nodes), which
    # open each other's descriptors through /proc
    domains = share.all_gather(_proc_domain())
    _one_domain_a_node(domains)
    members = [r for r, d in enumerate(domains) if d == domains[share.rank]]
    bounds = _segment_bounds(total, len(members))
    local = members.index(share.rank)
    lo, hi = bounds[local], bounds[local + 1]
    fd = _segment_memfd(hi - lo)
    stitched = runs = rt = None
    try:
        try:
            owners = share.all_gather(None if fd is None else (os.getpid(), fd))
            if None not in owners:
                mine = [owners[r] for r in members]
                stitched = _attempt(
                    "mapping the shared tables", lambda: _stitch(mine, bounds)
                )
            # every rank holds every segment before any lets go of its descriptor
            share.barrier()
        finally:
            if fd is not None:
                os.close(fd)
        if not share.agree(stitched is not None):
            raise SharedTablesError("engram: a DP rank could not map the shared tables")
        ptr = stitched.base
        # filled before any rank reads it (the agreement is collective)
        filled = _attempt(
            "filling the shared tables",
            lambda: _fill_segment(sources, offsets, ptr, lo, hi),
        )
        if not share.agree(bool(filled)):
            raise SharedTablesError(
                "engram: a DP rank could not fill the shared tables"
            )
        rt = _attempt("loading the CUDA runtime", torch.cuda.cudart)
        # in pieces, as an in-place registration: one call over a whole
        # table, issued by every rank at once, does not return
        if rt is not None:
            runs = _attempt(
                "page-locking the shared tables",
                lambda: _PAGE_RUNS.acquire(rt, ptr, ptr + total, 0),
            )
        if not share.agree(runs is not None):
            raise SharedTablesError(
                "engram: a DP rank could not page-lock the shared tables"
            )
        lease = _Lease(ptr, total, functools.partial(_unshare, rt, runs, stitched))
        stitched = runs = None  # the lease owns them
    finally:
        if runs is not None:
            _PAGE_RUNS.release(rt, runs)
        if stitched is not None:
            stitched.close()
    return tuple(
        _owning_bytes(lease, off, src.numel() * src.element_size())
        .view(src.dtype)
        .view(src.shape)
        for src, off in zip(sources, offsets, strict=True)
    )


def _read_only(lo: int, hi: int) -> bool:
    """Whether some mapping of `[lo, hi)` is not writable (`/proc/self/maps`)."""
    try:
        with open("/proc/self/maps") as maps:
            for line in maps:
                fields = line.split(maxsplit=2)
                a, b = (int(x, 16) for x in fields[0].split("-"))
                if b > lo and a < hi and "w" not in fields[1]:
                    return True
    except (OSError, ValueError, IndexError):
        pass
    return False


def _reg_chunk_bytes() -> int:
    """Largest single `cudaHostRegister`, in bytes."""
    return _REG_CHUNK_BYTES


@dataclasses.dataclass
class _Run:
    """One page-locked run `[base, end)`, its references, and the
    `_DeviceLock` that holds it (None: registered through HIP)."""

    end: int
    refs: int
    lock: object = None


class _PageRuns:
    """This process's in-place page-locked runs, reference-counted, each
    `mlock`ed for as long as it is registered.

    `cudaHostRegister` needs page boundaries and refuses a range overlapping
    one already registered. Ranges stored back to back share their boundary
    page -- a layer's scales sit right before its values in the checkpoint
    shard, and a rank holding every hash head (one DP-attention rank per GPU)
    registers both whole -- so each range takes a reference on the runs that
    cover it and registers only the gaps between them. A run is unregistered
    with its last reference.

    Why the `mlock`: a registration pins page-cache pages through a KFD
    userptr BO, which the kernel still treats as reclaimable. Under host
    memory pressure kswapd reclaim touches them, and every such MMU notifier
    evicts all of the process's GPU queues until the BO is revalidated --
    decode stalls of seconds, or a hang. `mlock` keeps reclaim off the pages;
    kcompactd still migrates locked pages, firing the same notifier, unless
    `vm.compact_unevictable_allowed=0`. Runs never overlap, so a run is
    locked when it is created (`_fault_and_lock`, which also faults it in)
    and unlocked when its last reference goes, before it is unregistered. A
    refused lock warns; the registration stands.
    """

    def __init__(self):
        self._runs: dict[int, _Run] = {}
        self._lock = threading.RLock()

    def acquire(self, rt, lo: int, hi: int, flags: int = 0) -> list[int] | None:
        """References on the runs covering `[lo, hi)` (page-aligned), the
        uncovered gaps locked and registered first, in `_REG_CHUNK_BYTES`
        pieces (`_register_range`); None, holding nothing, if one is refused.
        A read-only range (`flags`) is faulted in even where its lock is
        refused. Rollback goes through `_release_locked`, on a refusal or an
        exception alike."""
        with self._lock:
            held: list[int] = []
            created = created_bytes = 0
            alone = False
            cursor = lo
            overlapping = sorted(
                b for b, run in self._runs.items() if b < hi and run.end > lo
            )
            for base in [*overlapping, None]:
                gap_end = hi if base is None else base
                if cursor < gap_end:
                    read_only = bool(flags & _HOST_REGISTER_READ_ONLY)
                    _fault_and_lock(cursor, gap_end, read_only)
                    c = cursor
                    while c < gap_end:
                        e = min(gap_end, c + _reg_chunk_bytes())
                        logger.debug("engram: page-locking %.2f GiB", (e - c) / 2**30)
                        registered, lock = False, None
                        try:
                            registered, lock = _register_range(rt, c, e - c, flags)
                        finally:
                            if not registered:
                                _libc().munlock(c, gap_end - c)
                                self._release_locked(rt, held)
                        if not registered:
                            return None
                        self._runs[c] = _Run(e, 1, lock)
                        alone = lock is not None
                        held.append(c)
                        created += 1
                        created_bytes += e - c
                        c = e
                if base is None:
                    break
                self._runs[base].refs += 1
                held.append(base)
                cursor = max(cursor, self._runs[base].end)
            if created_bytes >= _REG_CHUNK_BYTES:
                logger.info(
                    "engram: page-locked %.2f GiB in %d piece(s)%s",
                    created_bytes / 2**30,
                    created,
                    " for this GPU alone" if alone else "",
                )
            return held

    def release(self, rt, bases) -> None:
        """Drop references taken by `acquire`."""
        with self._lock:
            self._release_locked(rt, bases)

    def _release_locked(self, rt, bases) -> None:
        """`release` with `self._lock` already held by the caller."""
        for base in bases:
            run = self._runs[base]
            run.refs -= 1
            if run.refs == 0:
                del self._runs[base]
                _libc().munlock(base, run.end - base)
                if not _unregister_range(rt, base, run.lock):
                    logger.warning("engram: unregistering a page-locked run failed")

    def locked_alone(self, addr: int) -> bool:
        """Whether `addr` lies in a run ROCr locked for this GPU alone, which
        the GPU reads at its host address (`_DeviceLock.lock`) and HIP has no
        record of."""
        with self._lock:
            return any(
                base <= addr < run.end and run.lock is not None
                for base, run in self._runs.items()
            )

    def __len__(self) -> int:
        return len(self._runs)


_PAGE_RUNS = _PageRuns()


def _fault_and_lock(lo: int, hi: int, populate: bool) -> None:
    """Lock `[lo, hi)` and fault it in on several threads, ahead of
    registering it.

    `mlock2(MLOCK_ONFAULT)` marks the whole range locked in one call without
    faulting it; `MADV_POPULATE_READ` per chunk then faults the pages in, in
    parallel, and each is locked as it arrives. (A plain `mlock` faults on
    the calling thread, and per-chunk `mlock`s scale poorly: on a warm 8 GiB
    shared read-only mapping, 16 threads, this takes
    0.27 s against 0.40 s for per-chunk `mlock` and 0.49 s for populate then
    one `mlock`; populate alone 0.15 s.) Registration would otherwise fault
    every page itself, on one thread and under the HIP runtime lock (which
    stalls GPU allocations elsewhere in the process). Only a read-only range
    (`populate`) is faulted in here, as before; the rest is locked as
    registration faults it. A kernel without `MADV_POPULATE_READ` declines
    and registration faults the pages as before. A refused lock warns once
    per process and the range is still faulted in.
    """
    if _libc().mlock2(lo, hi - lo, _MLOCK_ONFAULT) != 0:
        _warn_unlocked(ctypes.get_errno(), hi - lo)
    if populate:
        _populate(lo, hi)


def _populate(lo: int, hi: int) -> None:
    """`MADV_POPULATE_READ` over `[lo, hi)`, one chunk per thread."""
    libc = _libc()
    pieces = _chunks(lo, hi - lo, _CHUNK_BYTES)

    def fault(piece) -> bool:
        o, n = piece
        return libc.madvise(lo + o, n, _MADV_POPULATE_READ) == 0

    threads = min(copy_threads(hi - lo), len(pieces))
    if threads <= 1:
        for piece in pieces:
            if not fault(piece):
                return
        return
    with ThreadPoolExecutor(threads) as pool:
        list(pool.map(fault, pieces))


def _warn_unlocked(err: int, nbytes: int) -> None:
    if _UNLOCKED_WARNED:
        return
    _UNLOCKED_WARNED.append(True)
    hint = ""
    soft, _ = resource.getrlimit(resource.RLIMIT_MEMLOCK)
    if err in (errno.EPERM, errno.ENOMEM) and soft != resource.RLIM_INFINITY:
        hint = f"; RLIMIT_MEMLOCK is {soft / 1024**3:.2f} GiB, raise it"
    logger.warning(
        "engram: mlock of %.2f GiB refused (%s)%s; reclaim can then stall "
        "the GPU queues under host memory pressure",
        nbytes / 1024**3,
        os.strerror(err),
        hint,
    )


# Set once a refused `mlock` has been logged (`_fault_and_lock`).
_UNLOCKED_WARNED: list = []


def _register(sources, leases):
    """Register the mapping in place; each range is widened to page
    boundaries, which registration requires (the extra bytes are neighbouring
    rows the kernel never addresses), and shares any page another range of
    this process already holds (`_PageRuns`). A read-only mapping (the
    checkpoint mapping) is registered read-only."""
    rt = torch.cuda.cudart()
    spans = []
    for src in sources:
        lo, hi = _page_span(src)
        flags = _HOST_REGISTER_READ_ONLY if _read_only(lo, hi) else 0
        spans.append((lo, hi, flags))
    return _register_spans(rt, sources, spans, leases)


def _register_spans(rt, sources, spans, leases):
    out = []
    for src, (lo, hi, flags) in zip(sources, spans, strict=True):
        base = src.data_ptr()
        nbytes = src.numel() * src.element_size()
        runs = _PAGE_RUNS.acquire(rt, lo, hi, flags)
        if runs is None:
            logger.info("engram: page-lock registration of %d bytes refused", hi - lo)
            return None

        def unregister(_ptr, runs=runs):
            if torch.cuda.is_initialized():
                torch.cuda.synchronize()
            _PAGE_RUNS.release(rt, runs)

        lease = _Lease(lo, hi - lo, unregister, keep=src)
        leases.append(lease)
        view = _owning_bytes(lease, base - lo, nbytes)
        out.append(view.view(src.dtype).view(src.shape))
    return tuple(out)
