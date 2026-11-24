# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Reduce-scatter and all-gather over SDMA for Qwen3.8-Flash-Next sequence
parallelism (see `_SequenceParallel` in `atom/models/qwen4_exp.py`).

Rank r owns rows [r * L, (r + 1) * L) of the padded [N * L, hidden] tensors.
Every copy is an SDMA put posted from the compute stream by a one-wave FlyDSL
kernel through mori CCO (`Sdma.put_ptr`) into a peer's IPC-shared receive
buffer; the copy engine bumps the receiver's per-sender mori signal once the
bytes have landed, and the consumer spins on that counter in another one-wave
kernel on its compute stream right before it reads the rows. There are no side
streams or events: the copies run on the copy engines under the compute
kernels that follow the push.

Reduce-scatter: the partial sums are pushed in row chunks, the blocks this rank
does not own first, and each chunk is reduced as soon as it has landed.
All-gather: this rank's rows are pushed range by range as they are computed.

Only mori's SDMA queues and signal counters are used. The destinations are HIP
IPC mappings, not mori's symmetric windows, so hosts whose kernel lacks
CONFIG_DMABUF_MOVE_NOTIFY work too.

Receive buffers are double-buffered: a rank pushes into a slot only after it
waited for a later transfer every peer issued after consuming that slot, which
the reduce-scatter / all-gather alternation of the decoder layers guarantees.
"""

import logging
import os
import socket
from collections.abc import Iterator

import torch
import torch.distributed as dist
import triton
import triton.language as tl
from aiter.jit.utils.chip_info import get_gfx_runtime

from atom.distributed.group_agreement import group_agree
from atom.mono.runtime.hip_ipc import HipRuntime

logger = logging.getLogger("atom")

_WAVE = 64
_REDUCE_BLOCK = 512


def _local_unavailable_reason() -> str | None:
    """Why this rank cannot use SDMA, or None. Runs before any mori queue
    exists: a failing KFD setup exits the process."""
    # gfx942 only: chunking and engine use were measured on MI308X.
    if get_gfx_runtime() != "gfx942":
        return f"enabled on gfx942 only, not {get_gfx_runtime()}"
    # mori creates no SDMA queue without it; requesting this feature turns it
    # on unless the user set it explicitly.
    os.environ.setdefault("MORI_ENABLE_SDMA", "1")
    if os.environ["MORI_ENABLE_SDMA"].lower() not in ("1", "true", "on", "yes"):
        return "MORI_ENABLE_SDMA is disabled"
    try:
        import flydsl.compiler  # noqa: F401
        import mori.cco  # noqa: F401
    except ImportError as e:
        return f"cannot import mori / flydsl: {e}"
    # torch's HIP may load ROCm's private libdrm copy; mori links the system one
    # unless the private copy is preloaded, and two copies cannot share the GPU VM.
    with open("/proc/self/maps") as f:
        hip_uses_sysdeps_drm = "librocm_sysdeps_drm_amdgpu" in f.read()
    if hip_uses_sysdeps_drm and "rocm_sysdeps_drm_amdgpu" not in os.getenv(
        "LD_PRELOAD", ""
    ):
        return "LD_PRELOAD lacks ROCm's librocm_sysdeps_drm(_amdgpu)"
    return None


def _build_kernels(rank: int, world: int):
    # mori and FlyDSL are imported only once SDMA is enabled: loading mori's
    # library links the system libdrm next to HIP's.
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    import mori.cco.device.flydsl as cco
    from flydsl._mlir import ir
    from flydsl._mlir.dialects import llvm, scf
    from flydsl.expr import gpu as fgpu
    from flydsl.expr.typing import Int32, Int64
    from mori.cco.device.flydsl.signals import i32_type, signal_ptr

    default_stream = fx.Stream(None)

    def load_i64(addr):
        return llvm.load(ir.IntegerType.get_signless(64), signal_ptr(addr), alignment=8)

    def load_u32_system(ptr):
        return llvm.load(
            i32_type(),
            ptr,
            alignment=4,
            ordering=llvm.AtomicOrdering.monotonic,
            syncscope="",
        )

    def spin_until(ptr, flag):
        first = load_u32_system(ptr)
        i32 = i32_type()
        loop = scf.WhileOp([i32], [first])
        cond = ir.Block.create_at_start(loop.before, [i32])
        body = ir.Block.create_at_start(loop.after, [i32])
        with ir.InsertionPoint(cond):
            cur = fx.Int32(cond.arguments[0])
            # Signed difference: correct across the 32-bit counter wrap.
            scf.ConditionOp(
                ((cur - fx.Int32(flag)) < fx.Int32(0)).ir_value(), [cond.arguments[0]]
            )
        with ir.InsertionPoint(body):
            scf.YieldOp([load_u32_system(ptr)])

    # The same source to every peer; dst_table holds one base address per peer.
    @flyc.kernel(known_block_size=[_WAVE, 1, 1])
    def push_all(
        dev_comm: Int64, dst_table: Int64, src: Int64, dst_off: Int64, nbytes: Int64
    ):
        tid = fx.thread_idx.x
        sdma = cco.DevComm(dev_comm).sdma()
        if tid < world:  # noqa: SIM102 (one dynamic condition per FlyDSL if)
            if tid != rank:
                base = fx.Int64(
                    load_i64(fx.Int64(dst_table) + fx.Int64(tid) * fx.Int64(8))
                )
                sdma.put_ptr(
                    tid,
                    base + fx.Int64(dst_off),
                    fx.Int64(src),
                    fx.Int64(nbytes),
                    0,
                    remote_signal=True,
                )
        fgpu.barrier()

    @flyc.jit
    def push_all_run(
        dev_comm: Int64,
        dst_table: Int64,
        src: Int64,
        dst_off: Int64,
        nbytes: Int64,
        stream: fx.Stream = default_stream,
    ):
        push_all(dev_comm, dst_table, src, dst_off, nbytes).launch(
            grid=(1, 1, 1), block=[_WAVE, 1, 1], stream=stream
        )

    @flyc.kernel(known_block_size=[_WAVE, 1, 1])
    def push_one(dev_comm: Int64, dst: Int64, src: Int64, nbytes: Int64, peer: Int32):
        tid = fx.thread_idx.x
        sdma = cco.DevComm(dev_comm).sdma()
        if tid == peer:
            sdma.put_ptr(
                tid,
                fx.Int64(dst),
                fx.Int64(src),
                fx.Int64(nbytes),
                0,
                remote_signal=True,
            )
        fgpu.barrier()

    @flyc.jit
    def push_one_run(
        dev_comm: Int64,
        dst: Int64,
        src: Int64,
        nbytes: Int64,
        peer: Int32,
        stream: fx.Stream = default_stream,
    ):
        push_one(dev_comm, dst, src, nbytes, peer).launch(
            grid=(1, 1, 1), block=[_WAVE, 1, 1], stream=stream
        )

    @flyc.kernel(known_block_size=[_WAVE, 1, 1])
    def wait_all(dev_comm: Int64, expected: Int32):
        tid = fx.thread_idx.x
        sdma = cco.DevComm(dev_comm).sdma()
        if tid < world:  # noqa: SIM102 (one dynamic condition per FlyDSL if)
            if tid != rank:
                spin_until(
                    signal_ptr(fx.Int64(sdma.signal_slot(tid, 0))), fx.Int32(expected)
                )
        fgpu.barrier()

    @flyc.jit
    def wait_all_run(
        dev_comm: Int64, expected: Int32, stream: fx.Stream = default_stream
    ):
        wait_all(dev_comm, expected).launch(
            grid=(1, 1, 1), block=[_WAVE, 1, 1], stream=stream
        )

    return flyc, fx, push_all_run, push_one_run, wait_all_run


class _MoriSdma:
    """mori SDMA queue 0 and its signal counters, shared by both collectives.

    mori's counter for a sender counts every signalled put it sent here on
    queue 0, in issue order. Every rank issues the same sequence of transfers,
    so `received` (puts each peer has sent this rank so far) is the same for
    every peer and is what a wait compares against."""

    def __init__(self, cpu_group: dist.ProcessGroup, device: torch.device):
        from mori.cco import (
            GDA_CONNECTION_NONE,
            CCODevCommRequirements,
            Communicator,
            UniqueId,
        )

        self.rank = dist.get_rank(cpu_group)
        self.world = dist.get_world_size(cpu_group)
        # Rank 0 sends its error with the uid, so every rank leaves the
        # broadcast and raises alike instead of waiting on a uid never sent.
        uid, err = None, None
        if self.rank == 0:
            try:
                uid = bytes(Communicator.get_unique_id())
            except Exception as e:  # noqa: BLE001 - every rank must reach the broadcast
                err = f"mori get_unique_id failed on rank 0: {e}"
        msg = [(uid, err)]
        dist.broadcast_object_list(
            msg, src=dist.get_global_rank(cpu_group, 0), group=cpu_group
        )
        uid, err = msg[0]
        if err is not None:
            raise RuntimeError(err)
        self._comm = Communicator.init(
            self.world, self.rank, UniqueId.from_bytes(uid), per_rank_vmm=64 << 20
        )
        reqs = CCODevCommRequirements()
        reqs.gda_connection_type = GDA_CONNECTION_NONE
        reqs.gda_signal_count = 0
        reqs.gda_counter_count = 0
        reqs.sdma_queue_count = 1
        self._dev_comm = self._comm.create_dev_comm(reqs)
        # 0 when mori was built without BUILD_CCO_SDMA or no peer has SDMA; puts
        # then move no bytes and waits would never return.
        self.queues = self._dev_comm.sdma_num_queue
        (
            self._flyc,
            self._fx,
            self._push_all_jit,
            self._push_one_jit,
            self._wait_all_jit,
        ) = _build_kernels(self.rank, self.world)
        self._compiled = {}
        self.received = 0

    def _launch(self, jit_fn, *args) -> None:
        stream = self._fx.Stream(torch.cuda.current_stream())
        fn = self._compiled.get(jit_fn)
        if fn is None:
            # flyc.compile also runs the launcher once; that run is this call.
            self._compiled[jit_fn] = self._flyc.compile(jit_fn, *args, stream)
        else:
            fn(*args, stream)

    def push_all(self, dst_table: torch.Tensor, src: int, dst_off: int, nbytes: int):
        self._launch(
            self._push_all_jit,
            self._dev_comm.ptr,
            dst_table.data_ptr(),
            src,
            dst_off,
            nbytes,
        )

    def push_one(self, peer: int, dst: int, src: int, nbytes: int) -> None:
        self._launch(self._push_one_jit, self._dev_comm.ptr, dst, src, nbytes, peer)

    def wait(self, expected: int) -> None:
        """Wait until every peer's `expected`-th put to this rank has landed."""
        expected = (expected + (1 << 31)) % (1 << 32) - (1 << 31)
        self._launch(self._wait_all_jit, self._dev_comm.ptr, expected)


@triton.jit
def _reduce_rows_kernel(
    local_ptr,
    local_stride,
    recv_ptr,
    peer_stride,
    out_ptr,
    PEERS: tl.constexpr,
    HIDDEN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # out = local + the PEERS received rows, accumulated in fp32.
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < HIDDEN
    acc = tl.load(local_ptr + row * local_stride + cols, mask=mask).to(tl.float32)
    for p in tl.static_range(PEERS):
        rec = recv_ptr + p * peer_stride + row * HIDDEN + cols
        acc += tl.load(rec, mask=mask).to(tl.float32)
    tl.store(out_ptr + row * HIDDEN + cols, acc.to(out_ptr.dtype.element_ty), mask=mask)


class PendingReduceScatter:
    """A reduce-scatter in flight; consume it (`chunks` / `result`) before the
    next one on this rank."""

    def __init__(self, local_chunks, recv, bounds, wait_chunk):
        self._local_chunks = local_chunks
        self._recv = recv  # [peers, local capacity, hidden]
        self.bounds = bounds  # chunk boundaries in this rank's local rows
        self._wait_chunk = wait_chunk

    def chunks(self) -> Iterator[tuple[int, int, torch.Tensor]]:
        """`(r0, r1, summed rows)` per chunk, in local row coordinates, each
        reduced once it has landed from every peer."""
        for j, local in enumerate(self._local_chunks):
            self._wait_chunk(j)
            r0, r1 = self.bounds[j], self.bounds[j + 1]
            out = torch.empty_like(local)
            if r1 > r0:
                hidden = local.shape[1]
                _reduce_rows_kernel[(r1 - r0, triton.cdiv(hidden, _REDUCE_BLOCK))](
                    local,
                    local.stride(0),
                    self._recv[:, r0:],
                    self._recv.stride(0),
                    out,
                    PEERS=self._recv.shape[0],
                    HIDDEN=hidden,
                    BLOCK=_REDUCE_BLOCK,
                )
            yield r0, r1, out

    def result(self) -> torch.Tensor:
        return torch.cat([rows for _, _, rows in self.chunks()])


class AllGatherWriter:
    """An all-gather filled range by range: write this rank's rows [r0, r1)
    into `rows(r0, r1)`, `push(r0, r1)` them, then `finish()` waits for every
    peer and returns the gathered [num_tokens, hidden] rows."""

    def __init__(self, buf, dst_table, sdma, rank, local_rows, num_tokens, pushes):
        self._buf = buf  # this slot, [world * capacity, hidden]
        self._dst_table = dst_table
        self._sdma = sdma
        self._start = rank * local_rows
        self._gathered = buf[: sdma.world * local_rows][:num_tokens]
        self._pushes = pushes
        self._pushed = 0
        sdma.received += pushes
        self._expected = sdma.received

    def rows(self, r0: int, r1: int) -> torch.Tensor:
        return self._buf[self._start + r0 : self._start + r1]

    def push(self, r0: int, r1: int) -> None:
        assert self._pushed < self._pushes, "more pushes than announced"
        own = self.rows(r0, r1)
        self._sdma.push_all(
            self._dst_table,
            own.data_ptr(),
            (self._start + r0) * own.stride(0) * own.element_size(),
            own.numel() * own.element_size(),
        )
        self._pushed += 1

    def finish(self) -> torch.Tensor:
        assert self._pushed == self._pushes, "fewer pushes than announced"
        self._sdma.wait(self._expected)
        return self._gathered


class SdmaCollectives:
    """Reduce-scatter and all-gather of [tokens, hidden] rows over the TP group.

    Construction is collective. A rank-local failure leaves `error` set
    instead of raising, so every rank still runs the same collectives and can
    then agree on the outcome (`get_sdma_collectives`)."""

    def __init__(self, capacity_tokens, hidden, dtype, device, cpu_group):
        self.rank = rank = dist.get_rank(cpu_group)
        self.world = world = dist.get_world_size(cpu_group)
        self.capacity = -(-capacity_tokens // world)
        self.hidden = hidden
        self.dtype = dtype
        self.error = None
        rs_rows = 2 * (world - 1) * self.capacity
        ag_rows = 2 * world * self.capacity
        exported = None
        try:
            hip = HipRuntime()
            # One allocation, so a peer maps exactly one IPC handle per rank.
            buf = torch.empty(rs_rows + ag_rows, hidden, dtype=dtype, device=device)
            torch.cuda.synchronize(device)
            # The caching allocator hands out sub-ranges; handles cover whole blocks.
            base = hip.get_allocation_base(buf.data_ptr())
            exported = (hip.get_ipc_handle(base), buf.data_ptr() - base)
        except (RuntimeError, OSError) as e:  # HIP errors and torch OOM
            self.error = f"cannot export the SDMA buffer: {e}"
        handles = [None] * world
        dist.all_gather_object(handles, exported, group=cpu_group)
        peers = None
        if all(h is not None for h in handles):
            try:
                peers = {
                    p: hip.open_ipc_handle(handles[p][0]) + handles[p][1]
                    for p in range(world)
                    if p != rank
                }
            except RuntimeError as e:
                self.error = f"cannot map a peer's SDMA buffer: {e}"
        else:
            self.error = self.error or "a peer cannot export its SDMA buffer"
            return
        try:
            self._sdma = _MoriSdma(cpu_group, device)
            if self._sdma.queues < 1:
                self.error = "mori set up no SDMA queue (built without BUILD_CCO_SDMA?)"
        except Exception as e:  # noqa: BLE001 - mori / FlyDSL setup: agree, fall back
            self.error = self.error or f"mori SDMA setup failed: {e}"
        if self.error is not None:
            return
        row_bytes = hidden * buf.element_size()
        # [slot, source (rank - owner - 1) mod N, row, hidden]
        self._rs = buf[:rs_rows].view(2, world - 1, self.capacity, hidden)
        self._rs_peer = peers
        # [slot, row, hidden]
        self._ag = buf[rs_rows:].view(2, world * self.capacity, hidden)
        ag_off, slot_bytes = rs_rows * row_bytes, world * self.capacity * row_bytes
        # [slot, peer]: base address of the peer's all-gather slot (0 for this rank).
        self._ag_tables = torch.tensor(
            [
                [
                    peers[p] + ag_off + s * slot_bytes if p != rank else 0
                    for p in range(world)
                ]
                for s in range(2)
            ],
            dtype=torch.int64,
            device=device,
        )
        self._rs_slot = 0
        self._ag_slot = 0
        # Sources the copy engine may still read, kept for two sends.
        self._sources = [[], []]

    def fits(self, local_rows: int) -> bool:
        return local_rows <= self.capacity

    def reduce_scatter(
        self, x: torch.Tensor, local_rows: int, chunks: int
    ) -> PendingReduceScatter:
        """Reduce-scatter this rank's partial sums `x` ([num_tokens, hidden])
        over the padded [world * local_rows] rows, in `chunks` row chunks."""
        world, rank, slot = self.world, self.rank, self._rs_slot
        x = x.contiguous()
        num_tokens, hidden = x.shape
        assert x.dtype == self.dtype and hidden == self.hidden, (x.dtype, hidden)
        sources = self._sources[slot] = []
        bounds = [local_rows * j // chunks for j in range(chunks + 1)]
        local_chunks = [None] * chunks
        row_bytes = hidden * x.element_size()
        recv_slot = self._rs[slot]
        for i in range(1, world + 1):
            owner = (rank + i) % world
            for j in range(chunks):
                r0 = owner * local_rows + bounds[j]
                r1 = owner * local_rows + bounds[j + 1]
                rows = x[r0:r1]
                if r1 > num_tokens:
                    rows = x.new_zeros(r1 - r0, hidden)
                    if r0 < num_tokens:
                        rows[: num_tokens - r0] = x[r0:]
                if owner == rank:
                    local_chunks[j] = rows
                    continue
                sources.append(rows)
                source = (rank - owner - 1) % world
                dst = (
                    self._rs_peer[owner]
                    + recv_slot.data_ptr()
                    - self._rs.data_ptr()
                    + (source * self.capacity + bounds[j]) * row_bytes
                )
                self._sdma.push_one(
                    owner, dst, rows.data_ptr(), rows.shape[0] * row_bytes
                )
        self._rs_slot ^= 1
        received = self._sdma.received
        self._sdma.received += chunks
        return PendingReduceScatter(
            local_chunks,
            recv_slot,
            bounds,
            # Each peer pushes this rank's chunks in order on one queue.
            lambda j: self._sdma.wait(received + j + 1),
        )

    def all_gather(
        self, local_rows: int, num_tokens: int, pushes: int
    ) -> AllGatherWriter:
        slot = self._ag_slot
        self._ag_slot ^= 1
        return AllGatherWriter(
            self._ag[slot],
            self._ag_tables[slot],
            self._sdma,
            self.rank,
            local_rows,
            num_tokens,
            pushes,
        )


_COLLECTIVES: list = []


def get_sdma_collectives(
    capacity_tokens: int, hidden: int, dtype: torch.dtype, cpu_group
) -> SdmaCollectives | None:
    """The process' SDMA collectives over `cpu_group`, or None (with one
    warning) unless every rank can use them. The first call is collective."""
    if not _COLLECTIVES:
        world = dist.get_world_size(cpu_group)
        infos = [None] * world
        local = (socket.gethostname(), _local_unavailable_reason())
        dist.all_gather_object(infos, local, group=cpu_group)
        reason = next((r for _, r in infos if r is not None), None)
        # HIP IPC mappings exist only within one node.
        if reason is None and len({host for host, _ in infos}) > 1:
            reason = "the TP group spans nodes"
        collectives = None
        if reason is None:
            device = torch.device("cuda", torch.cuda.current_device())
            collectives = SdmaCollectives(
                capacity_tokens, hidden, dtype, device, cpu_group
            )
            reason = collectives.error
        if not group_agree(reason is None, cpu_group):
            reason = reason or "another TP rank cannot use SDMA"
            collectives = None
        if collectives is None:
            logger.warning(
                "Sequence-parallel collectives stay on RCCL: SDMA unavailable (%s)",
                reason,
            )
        else:
            logger.info(
                "Sequence-parallel reduce-scatter / all-gather over SDMA (mori), "
                "up to %d tokens",
                capacity_tokens,
            )
        _COLLECTIVES.append(collectives)
    return _COLLECTIVES[0]


def chunk_count(local_rows: int) -> int:
    """Reduce-scatter chunks, and all-gather pushes, for `local_rows` rows a
    rank: one per ~1500 rows, at most 4. Each chunk costs a push and a split
    hyper-connection (~0.4 ms of host time); below ~1500 rows its compute no
    longer hides that, so short prefills use one chunk (MI308X, TP2 / TP4,
    2k-12k tokens). Depends on the row count only, so every rank agrees."""
    return max(1, min(4, local_rows // 1500))
