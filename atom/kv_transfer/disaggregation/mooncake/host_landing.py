# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Decode host landing for Mooncake P/D (opt-in: ``ATOM_PD_HOST_LANDING_BLOCKS``).

Without it a P/D request takes its full-context HBM blocks at admission and
holds them, idle, for the whole P->D pull. Under load that parks tens of
requests' KV in HBM while running decodes starve for blocks.

With it the pull lands in a pinned host pool instead:

    LANDING  RDMA writes the pulled suffix into host blocks. The request holds
             only its local HBM prefix-cache hit (claimed, so it cannot be
             evicted under the pull) and its host blocks.
    LANDED   The transfer is complete. The request waits until the HBM pool
             can take its suffix without starving running decodes.
    COPYING  HBM blocks are allocated and the host blocks are copied into them
             on a side stream. The copy's completion wakes the request exactly
             like a finished direct pull, and it decodes.

The host pool has the GPU KV cache's per-region layout (same regions, same
bytes per block), and the consumer advertises its host base addresses instead
of its HBM ones, so the producer's RDMA path -- DCP relayout, staged index
pages, incremental skip -- places every byte exactly where it would have in
HBM, just at a host address. The H2D copy is then a plain block-to-block copy
per region.

Host block ids are scheduler-owned, like HBM block ids, and every TP rank's
pool has the same block count, so one id names the same slot on every rank.
"""

from __future__ import annotations

import bisect
import contextlib
import enum
import logging
import mmap
import time
from dataclasses import dataclass, field
from typing import Any

from atom.kv_transfer.disaggregation.types import LoadOperationId

logger = logging.getLogger("atom")

#: `ConnectorCompletion` channel the workers report finished H2D copies on.
HOST_LANDING_COPY_CHANNEL = "mooncake.host_landing_h2d"
#: `kv_transfer_params` key marking a pull whose destination is host blocks.
HOST_LANDING_PARAM = "host_landing"

_POOL_LOG_INTERVAL_S = 10.0
_PAGE = 4096
_HUGE_PAGE = 2 << 20


# ---------------------------------------------------------------------------
# Scheduler side
# ---------------------------------------------------------------------------


class HostBlockAllocator:
    """Free-interval allocator over host landing block ids ``[0, num_blocks)``.

    Prefers one contiguous run (best fit) so a request's H2D copy is one
    host-contiguous source per region; falls back to the largest free runs
    when the pool is fragmented.
    """

    def __init__(self, num_blocks: int) -> None:
        if num_blocks <= 0:
            raise ValueError(f"host landing pool needs blocks, got {num_blocks}")
        self.num_blocks = num_blocks
        # Sorted, disjoint, non-adjacent [start, end) runs.
        self._starts: list[int] = [0]
        self._ends: list[int] = [num_blocks]
        self.num_free = num_blocks

    @property
    def num_used(self) -> int:
        return self.num_blocks - self.num_free

    def allocate(self, count: int) -> list[int] | None:
        """``count`` free ids, or None (nothing taken) when fewer are free."""
        if count <= 0:
            return []
        if count > self.num_free:
            return None
        best = -1
        best_len = 0
        for i, (start, end) in enumerate(zip(self._starts, self._ends)):
            length = end - start
            if length >= count and (best < 0 or length < best_len):
                best, best_len = i, length
        if best >= 0:
            start = self._starts[best]
            self._take(best, count)
            return list(range(start, start + count))
        # Fragmented: take whole runs, largest first.
        order = sorted(
            range(len(self._starts)),
            key=lambda i: self._ends[i] - self._starts[i],
            reverse=True,
        )
        taken: list[tuple[int, int]] = []
        need = count
        for i in order:
            start, end = self._starts[i], self._ends[i]
            take = min(need, end - start)
            taken.append((start, start + take))
            need -= take
            if need == 0:
                break
        for start, end in taken:
            i = bisect.bisect_right(self._starts, start) - 1
            assert self._starts[i] == start
            self._take(i, end - start)
        ids: list[int] = []
        for start, end in sorted(taken):
            ids.extend(range(start, end))
        return ids

    def _take(self, i: int, count: int) -> None:
        self._starts[i] += count
        self.num_free -= count
        if self._starts[i] == self._ends[i]:
            del self._starts[i]
            del self._ends[i]

    def free(self, ids: list[int]) -> None:
        for start, end in _runs(sorted(ids)):
            self._insert(start, end)

    def _insert(self, start: int, end: int) -> None:
        if not 0 <= start < end <= self.num_blocks:
            raise ValueError(f"host landing ids [{start}, {end}) out of range")
        i = bisect.bisect_left(self._starts, start)
        if (i < len(self._starts) and self._starts[i] < end) or (
            i > 0 and self._ends[i - 1] > start
        ):
            raise ValueError(f"host landing ids [{start}, {end}) freed twice")
        merge_left = i > 0 and self._ends[i - 1] == start
        merge_right = i < len(self._starts) and self._starts[i] == end
        if merge_left and merge_right:
            self._ends[i - 1] = self._ends[i]
            del self._starts[i]
            del self._ends[i]
        elif merge_left:
            self._ends[i - 1] = end
        elif merge_right:
            self._starts[i] = start
        else:
            self._starts.insert(i, start)
            self._ends.insert(i, end)
        self.num_free += end - start


def _runs(sorted_ids: list[int]) -> list[tuple[int, int]]:
    """Maximal ``[start, end)`` runs of consecutive ids."""
    out: list[tuple[int, int]] = []
    for x in sorted_ids:
        if out and out[-1][1] == x:
            out[-1] = (out[-1][0], x + 1)
        else:
            out.append((x, x + 1))
    return out


class HostLandingPhase(enum.Enum):
    LANDING = "landing"
    LANDED = "landed"
    COPYING = "copying"


@dataclass
class HostLandingRecord:
    """One request's host landing, from admission until its first decode."""

    host_block_ids: list[int]
    # Leading block-table entries that are local HBM prefix-cache hits; the
    # host blocks land block-table entries [hbm_prefix_blocks, end).
    hbm_prefix_blocks: int
    pulled_tokens: int
    phase: HostLandingPhase = HostLandingPhase.LANDING
    generation: int = -1
    admitted_at: float = field(default_factory=time.monotonic)
    landed_at: float = 0.0
    copy_started_at: float = 0.0


@dataclass(frozen=True)
class HostLandingCopy:
    """Worker instruction: copy host blocks into HBM blocks, pairwise."""

    operation: LoadOperationId
    host_block_ids: tuple[int, ...]
    hbm_block_ids: tuple[int, ...]


class HostLandingController:
    """Scheduler-side host landing state: pool, per-request records, copies.

    The scheduler drives the phases; this object owns the bookkeeping so every
    exit (copy done, transfer failure, abort) returns host blocks exactly once.
    """

    def __init__(
        self,
        num_blocks: int,
        hbm_reserve_blocks: int = 0,
        *,
        incremental_geometry=None,
    ) -> None:
        self.allocator = HostBlockAllocator(num_blocks)
        self.hbm_reserve_blocks = hbm_reserve_blocks
        # ``params -> geometry | None``: None when the producer's page geometry
        # forces a full transfer (see `MooncakeConnectorScheduler`).
        self._incremental_geometry = incremental_geometry
        self._records: dict[str, HostLandingRecord] = {}
        self._pending_copies: list[HostLandingCopy] = []
        self._copied: set[Any] = set()
        self._copy_failed: set[Any] = set()
        self._generation = 0
        self._next_pool_log = 0.0
        self._next_stall_log = 0.0
        self.num_fallbacks = 0
        self.num_landed_total = 0

    # -- records ----------------------------------------------------------
    def eligible(self, seq) -> bool:
        """Whether ``seq``'s pull may land in host memory.

        Needs a producer geometry that keeps the pull incremental (the host
        blocks cover exactly the suffix past the local HBM hit, so a forced
        full transfer would not fit them) and no per-request state, which the
        host pool does not stage.
        """
        if seq.has_per_req_cache:
            return False
        params = seq.kv_transfer_params or {}
        if not params.get("do_remote_prefill"):
            return False
        return (
            self._incremental_geometry is None
            or self._incremental_geometry(params) is not None
        )

    def record(self, seq) -> HostLandingRecord | None:
        return self._records.get(str(seq.id))

    def reserve(self, seq, hbm_prefix_blocks: int, total_blocks: int) -> bool:
        """Take host blocks for block-table entries ``[hbm_prefix_blocks,
        total_blocks)``. False (nothing taken) when the pool cannot hold them."""
        count = total_blocks - hbm_prefix_blocks
        if count <= 0:
            return False
        ids = self.allocator.allocate(count)
        if ids is None:
            self.num_fallbacks += 1
            return False
        self._records[str(seq.id)] = HostLandingRecord(
            host_block_ids=ids,
            hbm_prefix_blocks=hbm_prefix_blocks,
            pulled_tokens=max(0, seq.num_prompt_tokens - seq.num_cached_tokens),
        )
        return True

    def release(self, seq) -> bool:
        """Return this request's host blocks and forget it. Idempotent."""
        rec = self._records.pop(str(seq.id), None)
        if rec is None:
            return False
        if rec.host_block_ids:
            self.allocator.free(rec.host_block_ids)
            rec.host_block_ids = []
        return True

    def is_landed(self, seq) -> bool:
        rec = self._records.get(str(seq.id))
        return rec is not None and rec.phase is HostLandingPhase.LANDED

    def has_landed(self) -> bool:
        return any(r.phase is HostLandingPhase.LANDED for r in self._records.values())

    def has_copying(self) -> bool:
        return any(r.phase is HostLandingPhase.COPYING for r in self._records.values())

    def warn_hbm_stall(self, seq, need_blocks: int, now: float | None = None) -> None:
        """Rate-limited: a landed request cannot get HBM and nothing will free
        any (no running request, no copy in flight) -- the prefix claims of
        the requests still on host hold what it needs."""
        now = time.monotonic() if now is None else now
        if now < self._next_stall_log:
            return
        self._next_stall_log = now + _POOL_LOG_INTERVAL_S
        counts = self.phase_counts()
        logger.warning(
            "[PD-HOST-LAND] HBM stall: req=%s needs %d blocks, nothing running "
            "and no copy in flight (landing=%d landed=%d); the requests on host "
            "hold HBM prefix claims",
            seq.id,
            need_blocks,
            counts["landing"],
            counts["landed"],
        )

    # -- phase transitions ------------------------------------------------
    def mark_landed(self, seq) -> None:
        rec = self._records[str(seq.id)]
        rec.phase = HostLandingPhase.LANDED
        rec.landed_at = time.monotonic()
        self.num_landed_total += 1

    def queue_copy(self, seq, hbm_block_ids: list[int]) -> None:
        rec = self._records[str(seq.id)]
        if len(hbm_block_ids) != len(rec.host_block_ids):
            raise RuntimeError(
                f"host landing copy for req {seq.id}: {len(rec.host_block_ids)} "
                f"host blocks but {len(hbm_block_ids)} HBM blocks"
            )
        self._generation += 1
        rec.generation = self._generation
        rec.phase = HostLandingPhase.COPYING
        rec.copy_started_at = time.monotonic()
        self._pending_copies.append(
            HostLandingCopy(
                operation=LoadOperationId(seq.id, rec.generation),
                host_block_ids=tuple(rec.host_block_ids),
                hbm_block_ids=tuple(hbm_block_ids),
            )
        )

    def take_pending_copies(self) -> list[HostLandingCopy]:
        copies, self._pending_copies = self._pending_copies, []
        return copies

    # -- completions ------------------------------------------------------
    def consume_completions(self, completions) -> set:
        """Take this channel's completions out of ``completions``.

        Returns the completions it consumed; a copy is recognised only for the
        generation the request is currently copying, so a stale or replayed
        report cannot wake a later admission.
        """
        mine = {c for c in completions if c.channel == HOST_LANDING_COPY_CHANNEL}
        for c in mine:
            op = c.operation_id
            rec = self._records.get(str(op.req_id))
            if (
                rec is None
                or rec.phase is not HostLandingPhase.COPYING
                or rec.generation != op.generation
            ):
                logger.warning(
                    "[PD-HOST-LAND] ignoring stale copy report %s (record=%s)",
                    op,
                    None if rec is None else (rec.phase.value, rec.generation),
                )
                continue
            (self._copied if c.succeeded else self._copy_failed).add(op.req_id)
        return mine

    def take_copy_results(self) -> tuple[set, set]:
        copied, failed = self._copied, self._copy_failed
        self._copied, self._copy_failed = set(), set()
        return copied, failed

    # -- observability ----------------------------------------------------
    def log_request_done(self, seq, hbm_prefix_tokens: int) -> None:
        rec = self._records.get(str(seq.id))
        if rec is None:
            return
        now = time.monotonic()
        logger.info(
            "[PD-HOST-LAND] req=%s h=%d pulled_tokens=%d host_blocks=%d "
            "land_ms=%.1f wait_hbm_ms=%.1f h2d_ms=%.1f",
            seq.id,
            hbm_prefix_tokens,
            rec.pulled_tokens,
            len(rec.host_block_ids),
            (rec.landed_at - rec.admitted_at) * 1e3,
            (rec.copy_started_at - rec.landed_at) * 1e3,
            (now - rec.copy_started_at) * 1e3,
        )

    def phase_counts(self) -> dict[str, int]:
        counts = {phase.value: 0 for phase in HostLandingPhase}
        for rec in self._records.values():
            counts[rec.phase.value] += 1
        return counts

    def maybe_log_pool(self, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        if now < self._next_pool_log or not (self._records or self.num_fallbacks):
            return
        self._next_pool_log = now + _POOL_LOG_INTERVAL_S
        counts = self.phase_counts()
        logger.info(
            "[PD-HOST-LAND] pool used=%d/%d blocks (%.1f%%) landing=%d landed=%d "
            "copying=%d landed_total=%d hbm_fallbacks=%d",
            self.allocator.num_used,
            self.allocator.num_blocks,
            100.0 * self.allocator.num_used / self.allocator.num_blocks,
            counts["landing"],
            counts["landed"],
            counts["copying"],
            self.num_landed_total,
            self.num_fallbacks,
        )


# ---------------------------------------------------------------------------
# Worker side
# ---------------------------------------------------------------------------


def copy_runs(host_block_ids, hbm_block_ids) -> list[tuple[int, int, int, bool]]:
    """Split a pairwise copy into host-contiguous runs.

    Returns ``(pair_start, host_start, length, hbm_contiguous)`` per run:
    pairs ``[pair_start, pair_start + length)`` read host blocks
    ``[host_start, host_start + length)``; ``hbm_contiguous`` says their HBM
    destinations are consecutive too, so the run is one direct copy.
    """
    out: list[tuple[int, int, int, bool]] = []
    n = len(host_block_ids)
    i = 0
    while i < n:
        j = i + 1
        while j < n and host_block_ids[j] == host_block_ids[j - 1] + 1:
            j += 1
        contiguous = all(
            hbm_block_ids[k] == hbm_block_ids[k - 1] + 1 for k in range(i + 1, j)
        )
        out.append((i, int(host_block_ids[i]), j - i, contiguous))
        i = j
    return out


_MPOL_PREFERRED = 1


def gpu_numa_node(device_index: int) -> int | None:
    """NUMA node of CUDA/HIP device ``device_index`` (logical), or None."""
    import torch

    try:
        props = torch.cuda.get_device_properties(device_index)
        bdf = (
            f"{int(props.pci_domain_id):04x}:{int(props.pci_bus_id):02x}:"
            f"{int(props.pci_device_id):02x}.0"
        )
        with open(f"/sys/bus/pci/devices/{bdf}/numa_node") as f:
            node = int(f.read())
        if node >= 0:
            return node
    except Exception as e:  # noqa: BLE001 - locality only, never correctness
        logger.debug("host landing: PCI NUMA lookup failed: %s", e)
    try:
        from atom.utils.numa_utils import _query_node_sysfs

        return _query_node_sysfs(device_index)
    except Exception as e:  # noqa: BLE001
        logger.debug("host landing: sysfs NUMA lookup failed: %s", e)
        return None


def _prefer_numa_node(ptr: int, size: int, node: int) -> bool:
    """``mbind(MPOL_PREFERRED)`` a not-yet-faulted range to ``node``.

    Preferred, not bound: a full node spills instead of OOM-killing a peer
    process constrained to it. Must run before the pages are first touched
    (page-locking faults them in).
    """
    import ctypes

    try:
        libnuma = ctypes.CDLL("libnuma.so.1", use_errno=True)
        if libnuma.numa_available() < 0 or not 0 <= node < 63:
            return False
        mask = ctypes.c_ulong(1 << node)
        libnuma.mbind.argtypes = [
            ctypes.c_void_p,
            ctypes.c_ulong,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_ulong),
            ctypes.c_ulong,
            ctypes.c_uint,
        ]
        rc = libnuma.mbind(
            ctypes.c_void_p(ptr), size, _MPOL_PREFERRED, ctypes.byref(mask), 65, 0
        )
        if rc != 0:
            logger.warning(
                "[PD-HOST-LAND] mbind to NUMA node %d failed: errno %d",
                node,
                ctypes.get_errno(),
            )
            return False
        return True
    except Exception as e:  # noqa: BLE001
        logger.warning("[PD-HOST-LAND] NUMA placement skipped: %s", e)
        return False


def _sample_numa_nodes(ptr: int, size: int, samples: int = 64) -> dict[int, int]:
    """Node -> count over ``samples`` pages spread across a faulted range."""
    import ctypes

    try:
        libnuma = ctypes.CDLL("libnuma.so.1")
        n = max(1, min(samples, size // _PAGE))
        step = (size // _PAGE) // n * _PAGE
        pages = (ctypes.c_void_p * n)(*[ptr + i * step for i in range(n)])
        status = (ctypes.c_int * n)()
        libnuma.numa_move_pages.argtypes = [
            ctypes.c_int,
            ctypes.c_ulong,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
        ]
        if libnuma.numa_move_pages(0, n, pages, None, status, 0) != 0:
            return {}
        out: dict[int, int] = {}
        for st in status:
            out[int(st)] = out.get(int(st), 0) + 1
        return out
    except Exception:  # noqa: BLE001
        return {}


class HostLandingBuffer:
    """One rank's pinned host landing pool, laid out like its GPU KV cache.

    Region ``r`` is ``num_blocks`` units of ``unit_bytes[r]`` bytes -- the
    exact per-block geometry of GPU PAGE region ``r`` -- so a producer that
    writes unit ``b`` of region ``r`` at ``base + b * unit_bytes`` lands the
    same bytes whether ``base`` is the GPU region or this one.

    ``gpu_views`` are the published ``uint8 [num_units, 1, unit_bytes]`` PAGE
    views (``KVTransferTensors.block_tensor_views``) in region order.
    """

    def __init__(
        self,
        gpu_views,
        num_blocks: int,
        *,
        device,
        stream=None,
        pin: bool = True,
        event_factory=None,
        numa_node: int | None = None,
    ) -> None:
        import torch

        if num_blocks <= 0:
            raise ValueError(f"host landing pool needs blocks, got {num_blocks}")
        self.num_blocks = num_blocks
        self.device = device
        self.stream = stream
        self._event_factory = event_factory
        self.gpu_views = [v.view(v.shape[0], -1) for v in gpu_views]
        self.unit_bytes = [int(v.shape[1]) for v in self.gpu_views]
        self.bytes_per_block = sum(self.unit_bytes)
        total = num_blocks * self.bytes_per_block
        started = time.monotonic()
        # One pageable allocation, page-locked in place: an exact size, where
        # the caching host allocator would round ~100 GiB up to a power of 2.
        # Registration wants whole pages, so over-allocate and align the pool
        # to a page; the pinned range then never touches another allocation.
        # Back the pool with 2 MiB transparent huge pages. The NIC caps how
        # many host pages one RDMA context may register (~4 GiB of 4 KiB
        # pages on ionic), which ~100 GiB of small pages overflows; huge
        # pages cut the page count 512x.
        span = -(-total // _HUGE_PAGE) * _HUGE_PAGE
        self._mmap = mmap.mmap(
            -1, span + _HUGE_PAGE, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS
        )
        self._raw = torch.frombuffer(self._mmap, dtype=torch.uint8)
        start = (-self._raw.data_ptr()) % _HUGE_PAGE
        self._mmap.madvise(mmap.MADV_HUGEPAGE, start, span)
        self.host = self._raw[start : start + total]
        self._registered_ptr = 0
        # Place the pool on the GPU's NUMA node before anything faults it in:
        # unbound, first touch lands wherever the thread happens to run, and
        # the prefill LMCache on the other node needs that node's memory.
        numa_placed = numa_node is not None and _prefer_numa_node(
            self.host.data_ptr(), span, numa_node
        )
        # Fault the pool in with parallel writes so every huge page exists
        # before page-locking and RDMA registration pin it.
        self._raw[start : start + span].fill_(0)
        if pin:
            rt = torch.cuda.cudart()
            ptr = self.host.data_ptr()
            if int(rt.cudaHostRegister(ptr, span, 0)) != 0:
                raise RuntimeError(
                    f"host landing: page-locking {total / 2**30:.1f} GiB failed"
                )
            self._registered_ptr = ptr
        self.host_views = []
        offset = 0
        for unit in self.unit_bytes:
            size = num_blocks * unit
            self.host_views.append(
                self.host[offset : offset + size].view(num_blocks, unit)
            )
            offset += size
        self.base_addrs = [v.data_ptr() for v in self.host_views]
        placement = (
            _sample_numa_nodes(self.host.data_ptr(), span)
            if pin and numa_node is not None
            else {}
        )
        logger.info(
            "[PD-HOST-LAND] host pool: %d blocks x %d regions, %d B/block, "
            "%.1f GiB, pinned=%s numa_node=%s (preferred=%s, sampled pages by "
            "node=%s) (%.1fs)",
            num_blocks,
            len(self.unit_bytes),
            self.bytes_per_block,
            total / 2**30,
            pin,
            numa_node,
            numa_placed,
            placement,
            time.monotonic() - started,
        )

    def regions(self):
        """RDMA-registerable regions, in the GPU region order."""
        from atom.kv_transfer.disaggregation.types import KVTransferRegion

        return [
            KVTransferRegion(
                base_addr=base,
                total_bytes=self.num_blocks * unit,
                unit_bytes=unit,
                semantic_role="host_landing",
            )
            for base, unit in zip(self.base_addrs, self.unit_bytes)
        ]

    def copy_to_hbm(self, host_block_ids, hbm_block_ids):
        """Issue the host -> HBM copies of ``host_block_ids[i]`` into
        ``hbm_block_ids[i]`` for every region.

        Returns ``(start_event, done_event, bytes)``; both events are None when
        the copy ran synchronously (no side stream, as in tests).
        """
        import torch

        if len(host_block_ids) != len(hbm_block_ids):
            raise ValueError("host landing copy pairs mismatch")
        for b in host_block_ids:
            if not 0 <= b < self.num_blocks:
                raise ValueError(f"host landing block {b} out of range")
        runs = copy_runs(host_block_ids, hbm_block_ids)
        on_stream = self.stream is not None
        factory = self._event_factory or torch.cuda.Event
        start_event = done_event = None
        ctx = torch.cuda.stream(self.stream) if on_stream else contextlib.nullcontext()
        if on_stream:
            # Fresh HBM blocks may have been freed by a request whose last
            # forward is still queued on the compute stream; copy after it.
            self.stream.wait_stream(torch.cuda.current_stream(self.device))
        with ctx:
            if on_stream:
                start_event = factory(enable_timing=True)
                start_event.record(self.stream)
            index_tensors = {}
            for pair_start, _host_start, length, contiguous in runs:
                if not contiguous:
                    index_tensors[pair_start] = torch.tensor(
                        hbm_block_ids[pair_start : pair_start + length],
                        dtype=torch.long,
                        device="cpu",
                    ).to(self.device, non_blocking=True)
            for host_view, gpu_view in zip(self.host_views, self.gpu_views):
                for pair_start, host_start, length, contiguous in runs:
                    src = host_view[host_start : host_start + length]
                    if contiguous:
                        dst0 = int(hbm_block_ids[pair_start])
                        gpu_view[dst0 : dst0 + length].copy_(src, non_blocking=True)
                    else:
                        staged = src.to(self.device, non_blocking=True)
                        gpu_view.index_copy_(0, index_tensors[pair_start], staged)
            if on_stream:
                done_event = factory(enable_timing=True)
                done_event.record(self.stream)
        return start_event, done_event, len(host_block_ids) * self.bytes_per_block

    def close(self) -> None:
        if self._registered_ptr:
            import torch

            torch.cuda.cudart().cudaHostUnregister(self._registered_ptr)
            self._registered_ptr = 0
