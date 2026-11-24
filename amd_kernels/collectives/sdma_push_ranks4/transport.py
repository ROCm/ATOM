# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Pushes between four ranks on the SDMA engines, ordered by the GPU (no host waits, no CUs).

Every rank owns receive regions (one IPC pool, coarse-grained) that its peers map. ``push(region, items)`` copies
this rank's bytes into the peers' copies of a region:

1. (host, enqueued at once) one ``hsa_amd_memory_async_copy_on_engine`` per item on the engine of its peer; every
   copy depends on the push's trigger signal (HSA, GPU_ONLY attribute: its value lives where a stream can write it);
2. per peer, an 8-byte copy of the push's sequence number into that peer's flag word of (region, this rank), after
   the data copies to it completed;
3. (stream) the sequence number into the push's device word, then 0 into the trigger: the copies start once the
   stream's work ahead of the push is done.

``wait(region, seq)`` puts a ``hipStreamWaitValue64`` (flag >= seq) per peer on the stream: the work after it sees
the peers' bytes. The command processor waits; no workgroup spins. Every rank issues the same pushes and waits on a
region in the same order. Flag words are uncached: a peer's SDMA write lands in HBM, and the waiting command processor
must not read a stale L2 line.

Lifetimes, the caller's part: an SDMA engine reads a source after the push's stream position, outside stream order,
so a source tensor must stay alive until those reads are done. ``push(..., keep=...)`` holds the tensors;
``wait(..., release=True)`` drops the ones of every push but the latest: correct when, as in a chain of all-to-all
phases, a peer triggers its next push only after its wait for this rank's previous one (that wait saw this rank's
flag, which follows the data copies). A region's bytes are overwritten by the next push into it: the consumer of a
region must be ahead (in stream order on its rank) of that rank's next push that every peer waits for before
pushing into the region again.

Engines: one per peer, by default HSA_AMD_SDMA_ENGINE_2 / 4 / 6 (143-155 GB/s out per GPU at 4096-24576 rows of
6 KB), else the engines ``hsa_amd_memory_copy_engine_status`` reports free.
"""

import ctypes
import logging

import torch

from amd_kernels.common import hip_runtime as ops
from amd_kernels.collectives.pull_collectives_ranks4.host import _alloc_pool, _exchange, _open_pool

from . import hsa_runtime as rt

logger = logging.getLogger(__name__)

WORLD = 4
RING = 32  # pushes in flight per transport; a slot is reused only after its copies are done
FLAG_STRIDE = 64  # bytes per (region, writer) flag word
REGION_ALIGN = 4096
PREFERRED_ENGINES = (2, 4, 6)


def _round_up(x: int, m: int) -> int:
    return (x + m - 1) // m * m


class _Slot:
    __slots__ = ("trigger", "done", "fdone", "used")

    def __init__(self):
        self.trigger = rt.signal()
        self.done = [rt.signal() for _ in range(WORLD)]
        self.fdone = [rt.signal() for _ in range(WORLD)]
        self.used = []  # peers whose signals this slot's last push armed


class SdmaPush:
    """Receive regions of ``region_bytes[k]`` bytes on every rank, the flags, the SDMA engines and the signal ring.
    Collective: the four ranks of ``cpu_group`` (gloo) construct it at the same point."""

    def __init__(self, rank: int, cpu_group, device: torch.device, region_bytes):
        device = torch.device(device)
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        self.rank, self.device = rank, device
        self.offsets, off = [], 0
        for n in region_bytes:
            self.offsets.append(off)
            off += _round_up(n, REGION_ALIGN)
        self.region_bytes = list(region_bytes)
        ptr, handle, self.pool = _alloc_pool(max(off, REGION_ALIGN), device)
        nflags = _round_up(len(self.offsets) * WORLD * FLAG_STRIDE, 4096)
        flag_ptr, flag_handle = ops.allocate_shared_buffer_and_handle(nflags)
        # the sequence words the flag copies read (uncached: a stream write must reach HBM before an engine reads it)
        self.seq_words, _ = ops.allocate_shared_buffer_and_handle(RING * 8)
        peers = _exchange((handle, flag_handle, rt.pci_key(device)), rank, WORLD, cpu_group)
        self.pool_ptrs = [ptr if r == rank else _open_pool(h, device) for r, (h, _, _) in enumerate(peers)]
        self.flag_ptrs = [flag_ptr if r == rank else ops.open_mem_handle(h) for r, (_, h, _) in enumerate(peers)]
        agents = rt.gpu_agents()
        self.agents = [agents[key] for _, _, key in peers]
        self.engines = self._pick_engines()
        self.ring = [_Slot() for _ in range(RING)]
        self.pushes = 0
        self.seq = [0] * len(self.offsets)
        self._held: list[tuple] = []
        self._hip, self._hsa = rt.hip(), rt.hsa()
        logger.info("sdma_push_ranks4: rank %d, %d regions, %.0f MB, engines %s", rank, len(self.offsets),
                    off / 2**20, self.engines)  # fmt: skip

    def _pick_engines(self) -> list[int]:
        me, taken, out = self.agents[self.rank], set(), [0] * WORLD
        for p in range(WORLD):
            if p == self.rank:
                continue
            mask = ctypes.c_uint32()
            rc = rt.hsa().hsa_amd_memory_copy_engine_status(self.agents[p], me, ctypes.byref(mask))
            free = [e for e in range(32) if mask.value >> e & 1] if rc == 0 else []
            pick = next((e for e in PREFERRED_ENGINES if e not in taken and (not free or e in free)), None)
            if pick is None:
                pick = next(e for e in free if e not in taken)
            taken.add(pick)
            out[p] = 1 << pick
        return out

    def region(self, k: int) -> torch.Tensor:
        """This rank's receive region ``k`` (uint8)."""
        return self.pool[self.offsets[k] : self.offsets[k] + self.region_bytes[k]]

    def _flag_off(self, k: int, writer: int) -> int:
        return (k * WORLD + writer) * FLAG_STRIDE

    def push(self, k: int, items, stream=None, keep=()) -> int:
        """Copy ``items`` = [(peer, src address, byte offset in the peer's region k, bytes)] once the stream's
        preceding work is done; returns region k's sequence number for ``wait``."""
        stream = torch.cuda.current_stream(self.device) if stream is None else stream
        self.seq[k] += 1
        seq = self.seq[k]
        slot = self.ring[self.pushes % RING]
        word = self.seq_words + (self.pushes % RING) * 8
        self.pushes += 1
        lib = self._hsa
        for p in slot.used:  # the slot's previous copies must be done before its signals are reset
            for s in (slot.done[p], slot.fdone[p]):
                while lib.hsa_signal_wait_scacquire(s, rt.HSA_SIGNAL_CONDITION_LT, 1, 1 << 40,
                                                    rt.HSA_WAIT_STATE_ACTIVE) > 0:  # fmt: skip
                    pass
        by_peer: dict[int, list] = {}
        for p, src, off, n in items:
            assert p != self.rank and 0 <= off and off + n <= self.region_bytes[k] and n > 0, (p, off, n)
            by_peer.setdefault(p, []).append((src, off, n))
        lib.hsa_signal_store_relaxed(slot.trigger, 1)
        trigger = (rt.Signal * 1)(slot.trigger)
        me = self.agents[self.rank]
        for p, copies in by_peer.items():
            done, fdone = slot.done[p], slot.fdone[p]
            lib.hsa_signal_store_relaxed(done, len(copies))
            lib.hsa_signal_store_relaxed(fdone, 1)
            dst = self.pool_ptrs[p] + self.offsets[k]
            for src, off, n in copies:
                rc = lib.hsa_amd_memory_async_copy_on_engine(dst + off, self.agents[p], src, me, n, 1, trigger, done,
                                                             self.engines[p], True)  # fmt: skip
                assert rc == 0, f"SDMA copy to rank {p} failed: {rc:#x}"
            dep = (rt.Signal * 1)(done)
            rc = lib.hsa_amd_memory_async_copy_on_engine(self.flag_ptrs[p] + self._flag_off(k, self.rank),
                                                         self.agents[p], word, me, 8, 1, dep, fdone,
                                                         self.engines[p], True)  # fmt: skip
            assert rc == 0, f"SDMA flag copy to rank {p} failed: {rc:#x}"
        slot.used = list(by_peer)
        s = ctypes.c_void_p(stream.cuda_stream)
        assert self._hip.hipStreamWriteValue64(s, word, seq, 0) == 0
        assert self._hip.hipStreamWriteValue64(s, rt.value_pointer(slot.trigger), 0, 0) == 0
        self._held.append(tuple(keep))
        return seq

    def wait(self, k: int, seq: int, peers=None, stream=None, release: bool = True) -> None:
        """The stream waits for every peer's push ``seq`` into this rank's region k (or those of ``peers``)."""
        stream = torch.cuda.current_stream(self.device) if stream is None else stream
        s = ctypes.c_void_p(stream.cuda_stream)
        for p in range(WORLD) if peers is None else peers:
            if p == self.rank:
                continue
            rc = self._hip.hipStreamWaitValue64(s, self.flag_ptrs[self.rank] + self._flag_off(k, p), seq,
                                                rt.HIP_STREAM_WAIT_VALUE_GTE, 0xFFFFFFFFFFFFFFFF)  # fmt: skip
            assert rc == 0, f"hipStreamWaitValue64 failed: {rc}"
        if release:
            del self._held[:-1]

    def drain(self) -> None:
        """Host: wait until every push's copies are done, and drop the held sources."""
        lib = self._hsa
        for slot in self.ring:
            for p in slot.used:
                for s in (slot.done[p], slot.fdone[p]):
                    while lib.hsa_signal_wait_scacquire(s, rt.HSA_SIGNAL_CONDITION_LT, 1, 1 << 40,
                                                        rt.HSA_WAIT_STATE_ACTIVE) > 0:  # fmt: skip
                        pass
        self._held.clear()
