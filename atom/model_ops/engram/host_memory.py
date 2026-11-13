# SPDX-License-Identifier: MIT
"""Page-locked host memory for an Engram UVA shard: pinning, copying, release.

`pin_rows` takes the row ranges a device kernel will read across the bus (a
table's values and, when quantized, its block scales) and returns them as
page-locked tensors -- copied into `hipHostMalloc` memory of exactly their
size, or the checkpoint mapping registered in place (`ATOM_ENGRAM_UVA_COPY=0`,
no HIP runtime, or -- unless `ATOM_ENGRAM_UVA_COPY` is set -- a shard that
every DP-attention rank of the node holds whole, see `pin_rows`).

In-place registrations go through a process-wide, reference-counted set of
page runs (`_PageRuns`): ranges are widened to page boundaries, so ranges
stored back to back (a layer's scales right before its values) share a page,
and HIP refuses to register a range overlapping one already registered.
Gaps are registered in 1 GiB pieces, and a table every DP-attention rank
shares registers and unregisters under a per-checkpoint file lock
(`_register_lock_file`), one process at a time.

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

import contextlib
import ctypes
import dataclasses
import errno
import fcntl
import functools
import hashlib
import logging
import os
import resource
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait

import numpy as np
import torch

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
# How long a waiter tolerates a registration lock whose holder shows no
# progress (the holder touches the file after every faulted-in piece and every
# registered chunk) before giving up. Once a path has timed out it is never
# waited on again in this process.
_REG_LOCK_TIMEOUT_S = 600
_logged_lock_paths: set[str] = set()
_timed_out_lock_paths: set[str] = set()
# Descriptors of the registration lock files this process holds. Fault-in
# worker threads touch them, so the list and each touch are under the lock: a
# descriptor leaves the list before it is closed and is never touched after.
_held_lock_fds: list[int] = []
_held_lock_fds_lock = threading.Lock()


def _lock_heartbeat() -> None:
    """Touch the held registration lock files, so waiters see progress."""
    with _held_lock_fds_lock:
        for fd in _held_lock_fds:
            with contextlib.suppress(OSError):
                os.utime(fd)


@functools.cache
def _libc():
    """This process's libc, with the calls used here declared once."""
    libc = ctypes.CDLL(None, use_errno=True)
    libc.madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    libc.mlock2.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    libc.munlock.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    libc.prctl.argtypes = [ctypes.c_int] + [ctypes.c_ulong] * 4
    return libc


def _page_span(t: torch.Tensor) -> tuple[int, int]:
    """`[lo, hi)`: the whole pages `t`'s bytes lie on."""
    base = t.data_ptr()
    lo = base - base % _PAGE
    hi = -(-(base + t.numel() * t.element_size()) // _PAGE) * _PAGE
    return lo, hi


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
    candidates = []
    try:
        with open("/proc/self/maps") as maps:
            for line in maps:
                path = line.split(maxsplit=5)[-1].strip()
                name = os.path.basename(path)
                if name.startswith("libamdhip64.so") and path not in candidates:
                    candidates.append(path)
    except OSError:
        pass
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
    host address: copies the HIP allocator page-locked, or no device at all)."""
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


_COPY_ENV = "ATOM_ENGRAM_UVA_COPY"


def copies(shared: bool) -> bool:
    """Whether a shard is copied into page-locked memory (else registered in
    place).

    `ATOM_ENGRAM_UVA_COPY`, when set, decides. Unset, a shard copies unless it
    is `shared`: the same whole table every DP-attention rank of the node
    holds. A copy there is one per rank -- ~183 GiB each for V4.1-Flash --
    where the shared read-only checkpoint mapping registered in place pins the
    page cache once for all of them.
    """
    if _COPY_ENV in os.environ:
        return envs.ATOM_ENGRAM_UVA_COPY
    return not shared


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
    sources: list[torch.Tensor], shared: bool = False, copy: bool = False
) -> tuple[torch.Tensor, ...] | None:
    """`sources` (contiguous row ranges) as page-locked tensors of the same
    dtype and shape, owning their memory; all-or-nothing.

    `shared`: these are whole tables that every DP-attention rank of the node
    page-locks too. Unless `ATOM_ENGRAM_UVA_COPY` says otherwise they are then
    registered in place (`copies`), and copied only if that is refused.

    `copy` copies whatever `ATOM_ENGRAM_UVA_COPY` and `shared` say, and never
    registers in place, not even without an allocator.

    None, with everything taken released, on any failure -- no memory, a
    refused registration, no thread to copy on -- so the TP group can fall
    back as one.
    """
    plan = ["copy"] if copy or copies(shared) else ["register"]
    if shared and _COPY_ENV not in os.environ:
        plan.append("copy")
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
                out = _register(sources, leases, shared=shared)
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


def _mapping_path(addr: int) -> str | None:
    """Regular file mapped at `addr` (`/proc/self/maps`), or None."""
    try:
        with open("/proc/self/maps") as maps:
            for line in maps:
                fields = line.split(maxsplit=5)
                lo, hi = (int(x, 16) for x in fields[0].split("-"))
                if not lo <= addr < hi:
                    continue
                name = fields[5].strip() if len(fields) > 5 else ""
                if name.startswith("/") and not name.startswith("/dev/"):
                    return name
                return None
    except (OSError, ValueError, IndexError):
        return None
    return None


def _registration_lock_path(sources) -> str:
    """Lock file for `sources`.

    `ATOM_ENGRAM_REG_LOCK`, when set. Otherwise one file per checkpoint under
    `/dev/shm`, so two servers loading different checkpoints do not share it.
    """
    explicit = envs.ATOM_ENGRAM_REG_LOCK
    if explicit:
        return explicit
    mapped = _mapping_path(sources[0].data_ptr()) if sources else None
    key = os.path.realpath(mapped) if mapped else "anonymous"
    digest = hashlib.sha256(key.encode()).hexdigest()[:16]
    return f"/dev/shm/engram-uva-{digest}.lock"


def _flock_logging_wait(fd: int, path: str) -> None:
    """Exclusive `flock` on `fd`.

    A wait is logged. It is bounded by the holder's progress, not by its
    total hold time: the holder touches the file after every faulted-in piece
    and every registered chunk, and only `_REG_LOCK_TIMEOUT_S` without a
    touch raises `TimeoutError` (an `OSError` subclass; callers must not
    swallow it with `except OSError`).
    """
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return
    except BlockingIOError:
        pass
    logger.info("engram: waiting for %s; another process is page-locking", path)
    start = time.monotonic()
    wait_from = time.time()
    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            last = max(wait_from, os.fstat(fd).st_mtime)
            if time.time() - last >= _REG_LOCK_TIMEOUT_S:
                logger.error(
                    "engram: gave up on %s: its holder made no progress for "
                    "%d s; not waiting on it again in this process",
                    path,
                    _REG_LOCK_TIMEOUT_S,
                )
                raise TimeoutError(
                    f"engram registration lock {path}: holder made no progress "
                    f"for {_REG_LOCK_TIMEOUT_S}s"
                )
            time.sleep(0.05)
        else:
            logger.info("engram: took %s after %.1f s", path, time.monotonic() - start)
            return


@contextlib.contextmanager
def _register_lock_file(path: str):
    """Hold the registration lock file `path`.

    A file that cannot be opened or locked is logged and skipped. The `yield`
    is outside the `except`, so an error during registration is not chained
    onto that failure. A holder that makes no progress for
    `_REG_LOCK_TIMEOUT_S` raises `TimeoutError`, and after that the path is
    never waited on again in this process: every later call raises at once.
    """
    if path in _timed_out_lock_paths:
        raise TimeoutError(f"engram registration lock {path} timed out earlier")
    fd = None
    held = False
    open_error = None
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o666)
    except OSError as exc:
        open_error = exc
    if open_error is not None:
        logger.warning(
            "engram: cannot open registration lock %s (%s); page-locking without it",
            path,
            open_error,
        )
    else:
        if path not in _logged_lock_paths:
            _logged_lock_paths.add(path)
            logger.info("engram: page-locking serialized on %s", path)
        try:
            _flock_logging_wait(fd, path)
        except TimeoutError:
            _timed_out_lock_paths.add(path)
            os.close(fd)
            fd = None
            raise
        except OSError as exc:
            logger.warning(
                "engram: cannot lock %s (%s); page-locking without it", path, exc
            )
        else:
            held = True
            with _held_lock_fds_lock:
                _held_lock_fds.append(fd)
            _lock_heartbeat()
    try:
        yield
    finally:
        if held:
            with _held_lock_fds_lock:
                _held_lock_fds.remove(fd)
            fcntl.flock(fd, fcntl.LOCK_UN)
        if fd is not None:
            os.close(fd)


@dataclasses.dataclass
class _Run:
    """One in-place registration `[base, end)` and its references."""

    end: int
    refs: int


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
        pieces; None, holding nothing, if HIP refuses one. A read-only range
        (`flags`) is faulted in even where its lock is refused. Rollback goes
        through `_release_locked` and does not take the registration lock."""
        with self._lock:
            held: list[int] = []
            created = created_bytes = 0
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
                        logger.debug(
                            "engram: cudaHostRegister %.2f GiB", (e - c) / 2**30
                        )
                        if int(rt.cudaHostRegister(c, e - c, flags)):
                            _clear_hip_error()
                            _libc().munlock(c, gap_end - c)
                            self._release_locked(rt, held)
                            return None
                        self._runs[c] = _Run(e, 1)
                        held.append(c)
                        created += 1
                        created_bytes += e - c
                        _lock_heartbeat()
                        c = e
                if base is None:
                    break
                self._runs[base].refs += 1
                held.append(base)
                cursor = max(cursor, self._runs[base].end)
            if created_bytes >= _REG_CHUNK_BYTES:
                logger.info(
                    "engram: registered %.2f GiB in-place as %d userptr BO(s)",
                    created_bytes / 2**30,
                    created,
                )
            return held

    def release(self, rt, bases, lock_path: str | None = None) -> None:
        """Drop references taken by `acquire`. Lock order: the registration
        lock file `lock_path` (a shared registration), then `self._lock`.

        Best effort: if the lock file times out the runs are released without
        it, so they never stay registered and locked for the process's life.
        """
        if lock_path:
            try:
                with _register_lock_file(lock_path), self._lock:
                    self._release_locked(rt, bases)
                return
            except TimeoutError:
                logger.warning(
                    "engram: unregistering without %s, which timed out", lock_path
                )
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
                if int(rt.cudaHostUnregister(base) or 0):
                    _clear_hip_error()
                    logger.warning("engram: cudaHostUnregister failed")
                _lock_heartbeat()

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
    """`MADV_POPULATE_READ` over `[lo, hi)`, one chunk per thread.

    Each finished chunk touches any registration lock this process holds:
    `acquire` faults in under the lock what reclaim took back, and on a cold
    cache that can outlast `_REG_LOCK_TIMEOUT_S` with no chunk registered.
    """
    libc = _libc()
    pieces = _chunks(lo, hi - lo, _CHUNK_BYTES)

    def fault(piece) -> bool:
        o, n = piece
        ok = libc.madvise(lo + o, n, _MADV_POPULATE_READ) == 0
        _lock_heartbeat()
        return ok

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


def _register(sources, leases, *, shared: bool = False):
    """Register the mapping in place; each range is widened to page
    boundaries, which registration requires (the extra bytes are neighbouring
    rows the kernel never addresses), and shares any page another range of
    this process already holds (`_PageRuns`). A read-only mapping (the shared
    checkpoint mapping) is faulted in on several threads first and registered
    read-only.

    A `shared` table takes the registration lock file around `acquire` and
    the matching `release`. Its fault-in runs before the lock, so ranks read
    the page cache concurrently; `acquire` then faults in only what reclaim
    took back before its `mlock`."""
    rt = torch.cuda.cudart()
    spans = []
    for src in sources:
        lo, hi = _page_span(src)
        flags = _HOST_REGISTER_READ_ONLY if _read_only(lo, hi) else 0
        if shared and flags:
            _populate(lo, hi)
        spans.append((lo, hi, flags))
    if not shared:
        return _register_spans(rt, sources, spans, leases, None)
    lock_path = _registration_lock_path(sources)
    with _register_lock_file(lock_path):
        return _register_spans(rt, sources, spans, leases, lock_path)


def _register_spans(rt, sources, spans, leases, lock_path):
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
            _PAGE_RUNS.release(rt, runs, lock_path)

        lease = _Lease(lo, hi - lo, unregister, keep=src)
        leases.append(lease)
        view = _owning_bytes(lease, base - lo, nbytes)
        out.append(view.view(src.dtype).view(src.shape))
    return tuple(out)
