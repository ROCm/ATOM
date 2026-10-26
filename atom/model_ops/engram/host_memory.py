# SPDX-License-Identifier: MIT
"""Page-locked host memory for an Engram UVA shard: pinning, copying, release.

`pin_rows` takes the row ranges a device kernel will read across the bus (a
table's values and, when quantized, its block scales) and returns them as
page-locked tensors -- copied into `hipHostMalloc` memory of exactly their
size, or (`ATOM_ENGRAM_UVA_COPY=0`, or no HIP runtime) the checkpoint mapping
registered in place.

Ownership: each returned tensor's storage holds the lease on its memory, so
the memory is released when the last tensor viewing it is collected -- never
while one is alive, and not leaked when its owner is dropped without an
explicit release.

Release and the device: `hipHostFree` synchronizes the device before freeing,
and an in-place registration synchronizes before unregistering, so kernels
already launched on the memory finish first. A CUDA graph captured over the
memory keeps its address without a reference, so the memory must outlive any
graph that reads it; Engram releases a shard only when the TP group falls back
at load, before capture, or at exit. A release that would have to synchronize
during a capture is skipped, with a warning, rather than breaking it.
"""

import ctypes
import logging
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, wait

import numpy as np
import torch

from atom.utils import envs

logger = logging.getLogger("atom")

_PAGE = 4096
_MADV_DONTNEED = 4
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
        self._prctl.argtypes = [ctypes.c_int] + [ctypes.c_ulong] * 4

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
            return None
        return ptr.value

    def free(self, ptr: int) -> None:
        # Synchronizes the device first (an implicit hipDeviceSynchronize).
        self._free(ptr)


def _host_allocator():
    """The page-locked allocator a shard is copied into, or None to register
    the checkpoint mapping in place (`ATOM_ENGRAM_UVA_COPY=0`, or no HIP
    runtime to call)."""
    if not envs.ATOM_ENGRAM_UVA_COPY or not torch.version.hip:
        return None
    try:
        return _HipHostAllocator(
            ctypes.CDLL("libamdhip64.so"), ctypes.CDLL(None, use_errno=True)
        )
    except OSError:
        return None


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
    libc = ctypes.CDLL(None, use_errno=True)
    libc.madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]

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


def pin_rows(sources: list[torch.Tensor]) -> tuple[torch.Tensor, ...] | None:
    """`sources` (contiguous row ranges) as page-locked tensors of the same
    dtype and shape, owning their memory; all-or-nothing.

    None, with everything taken released, on any failure -- no memory, a
    refused registration, no thread to copy on -- so the TP group can fall
    back as one.
    """
    allocator = _host_allocator()
    leases: list[_Lease] = []
    try:
        if allocator is not None:
            out = _copy(allocator, sources, leases)
        else:
            out = _register(sources, leases)
    except Exception:  # any failure is a fallback
        logger.warning("engram: page-locking the UVA shard failed", exc_info=True)
        out = None
    if out is None:
        for lease in leases:
            lease.release()
    return out


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


def _register(sources, leases):
    """Register the mapping in place; the range is widened to page boundaries,
    which registration requires (the extra bytes are neighbouring rows the
    kernel never addresses)."""
    rt = torch.cuda.cudart()

    def unregister(base):
        if torch.cuda.is_initialized():
            torch.cuda.synchronize()
        rt.cudaHostUnregister(base)

    out = []
    for src in sources:
        base = src.data_ptr()
        nbytes = src.numel() * src.element_size()
        lo = base - (base % _PAGE)
        size = -(-(base + nbytes - lo) // _PAGE) * _PAGE
        if int(rt.cudaHostRegister(lo, size, 0)) != 0:
            logger.info("engram: page-lock registration of %d bytes refused", size)
            return None
        lease = _Lease(lo, size, unregister, keep=src)
        leases.append(lease)
        view = _owning_bytes(lease, base - lo, nbytes)
        out.append(view.view(src.dtype).view(src.shape))
    return tuple(out)
