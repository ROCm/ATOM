# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

###############################################################################
#
# See LICENSE for license information.
###############################################################################

"""Tests for atomrl.lumen.modules.sdma_comm — SDMA TP communication primitives.

Covers:
  - SdmaTpContext singleton lifecycle
  - SdmaTpComm construction and singleton pattern
  - allgather_dim0: correctness, shape, multi-dtype
  - allgather_last_dim: correctness, shape
  - allreduce_sum: correctness, multi-dtype
  - allreduce_sum_inplace: correctness
  - reduce_scatter_dim0: correctness, shape
  - Autograd functions: forward/backward gradient flow
  - Performance: throughput for TP-sized collectives

Multi-GPU tests use torch.multiprocessing.spawn with mori shmem init,
following the same pattern as ``mori/tests/python/ccl/test_allreduce.py``.

Hardware requirements:
  - Unit tests (SdmaTpContext, SdmaTpComm): single GPU, no RDMA needed.
  - Distributed tests (TestSdmaTpCommDistributed): >= 2 GPUs with SDMA
    support (``is_sdma_available()`` must return True).
  - Performance tests (TestSdmaTpCommPerformance): same as distributed.

How to run::

    # All tests (unit + multi-GPU); SDMA-capable hardware required:
    pytest tests/modules/test_sdma_comm.py -v

    # Unit tests only (single GPU):
    pytest tests/modules/test_sdma_comm.py -v -k "Unit"

    # Multi-GPU TP communication tests (mp.spawn, >= 2 GPUs):
    pytest tests/modules/test_sdma_comm.py -v -k "Distributed"

    # Performance benchmarks only (>= 2 GPUs):
    pytest tests/modules/test_sdma_comm.py -v -k "Performance"

    # Override SDMA spawn cooldown for fast local iteration:
    LUMEN_SDMA_SPAWN_COOLDOWN_SECS=0 pytest tests/modules/test_sdma_comm.py -v
"""

import os
import time

import pytest
import torch
from atomrl.lumen.ops.sdma import is_sdma_available

_requires_sdma_hw = pytest.mark.skipif(
    not is_sdma_available(),
    reason="SDMA not available (is_sdma_available() returned False)",
)

_SDMA_SPAWN_COOLDOWN_SECS = float(os.environ.get("LUMEN_SDMA_SPAWN_COOLDOWN_SECS", "2"))


_SDMA_SPAWN_TIMEOUT_SECS = int(os.environ.get("LUMEN_SDMA_SPAWN_TIMEOUT_SECS", "120"))


def _sdma_spawn(fn, args, nprocs, join=True):
    """``mp.spawn`` with a KFD SDMA-queue reclamation cooldown and timeout.

    See ``tests/ops/test_sdma.py::_sdma_spawn`` for the full rationale.
    Override with ``LUMEN_SDMA_SPAWN_COOLDOWN_SECS=0`` for fast local runs.

    ``mp.spawn(join=True)`` blocks forever if a worker hangs (e.g. a
    ``dist.barrier`` waiting on a crashed peer).  We use ``join=False``
    and poll with a deadline so the test fails instead of hanging.
    """
    import torch.multiprocessing as mp

    os.environ.setdefault("MORI_ENABLE_SDMA", "1")

    if _SDMA_SPAWN_COOLDOWN_SECS > 0:
        time.sleep(_SDMA_SPAWN_COOLDOWN_SECS)

    if not join:
        return mp.spawn(fn, args=args, nprocs=nprocs, join=False)

    ctx = mp.spawn(fn, args=args, nprocs=nprocs, join=False)
    deadline = time.monotonic() + _SDMA_SPAWN_TIMEOUT_SECS
    while not ctx.join(timeout=5):
        if time.monotonic() > deadline:
            for proc in ctx.processes:
                if proc.is_alive():
                    proc.kill()
            pytest.fail(
                f"_sdma_spawn timed out after {_SDMA_SPAWN_TIMEOUT_SECS}s — "
                f"workers likely hung on dist.barrier or SDMA op"
            )


def _get_free_port():
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


class _TorchDistContext:
    """Context manager for torch.distributed init / teardown.

    Mirrors ``TorchDistContext`` from ``mori/tests/python/utils.py``
    so that Lumen SDMA tests follow the identical process lifecycle
    that the mori ccl test suite relies on.
    """

    def __init__(self, rank, world_size, master_port, backend="cpu:gloo,cuda:nccl"):
        self.rank = rank
        self.world_size = world_size
        self.master_port = master_port
        self.backend = backend

    def __enter__(self):
        import torch.distributed as dist

        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(self.master_port)
        torch.cuda.set_device(self.rank)
        device = torch.device("cuda", self.rank)
        dist.init_process_group(
            backend=self.backend,
            rank=self.rank,
            world_size=self.world_size,
            device_id=device,
        )
        world_group = torch.distributed.group.WORLD
        assert world_group is not None
        torch._C._distributed_c10d._register_process_group("default", world_group)

    def __exit__(self, exc_type, exc_val, exc_tb):
        import torch.distributed as dist

        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()


def _sdma_tp_cleanup(shmem_mod, comm):
    """Release SdmaTpComm handles and finalize shmem (mori cleanup order).

    Must be called *inside* the ``_TorchDistContext`` block.  Pass the
    ``SdmaTpComm`` instance so its C++ SDMA handles are released at the
    correct point — after ``cuda.synchronize()`` + ``barrier``, before
    ``shmem_finalize()``.
    """
    import torch.distributed as dist
    from atomrl.lumen.modules.sdma_comm import SdmaTpComm, SdmaTpContext

    torch.cuda.synchronize()
    dist.barrier()
    for attr in ("_ag", "_ag_chunk", "_rs_chunk_ar"):
        wrapper = getattr(comm, attr, None)
        if wrapper is not None and hasattr(wrapper, "_handle"):
            wrapper._handle = None
    for ar in getattr(comm, "_ar_handles", {}).values():
        if hasattr(ar, "_handle"):
            ar._handle = None
    SdmaTpComm.reset()
    SdmaTpContext.reset()
    dist.barrier()
    shmem_mod.shmem_finalize()


# ===================================================================
# SdmaTpContext singleton
# ===================================================================


class TestSdmaTpContextUnit:
    def test_reset_clears_singleton(self):
        from atomrl.lumen.modules.sdma_comm import SdmaTpContext

        SdmaTpContext.reset()
        assert SdmaTpContext._instance is None


# ===================================================================
# SdmaTpComm singleton
# ===================================================================


class TestSdmaTpCommUnit:
    def test_reset_clears_singleton(self):
        from atomrl.lumen.modules.sdma_comm import SdmaTpComm

        SdmaTpComm.reset()
        assert SdmaTpComm._instance is None


# ===================================================================
# Multi-GPU distributed tests
# ===================================================================


def _worker_tp_allgather_dim0(rank, world_size, port):
    """Worker: test SdmaTpComm.allgather_dim0 correctness."""
    os.environ["MORI_ENABLE_SDMA"] = "1"

    import torch.distributed as dist
    from atomrl.lumen.modules.sdma_comm import SdmaTpComm, SdmaTpContext
    from mori import shmem

    with _TorchDistContext(rank, world_size, port):
        tp_group = dist.new_group(list(range(world_size)))
        SdmaTpContext.reset()
        SdmaTpComm.reset()
        comm = SdmaTpComm.get(tp_group)

        seq_local = 16
        hidden = 64
        local_tensor = torch.full(
            (seq_local, hidden),
            float(rank + 1),
            dtype=torch.bfloat16,
            device=f"cuda:{rank}",
        )

        gathered = comm.allgather_dim0(local_tensor)

        errors = []
        if gathered.shape != (seq_local * world_size, hidden):
            errors.append(
                f"PE {rank}: shape mismatch {gathered.shape} != {(seq_local * world_size, hidden)}"
            )

        for pe in range(world_size):
            chunk = gathered[pe * seq_local : (pe + 1) * seq_local]
            expected = float(pe + 1)
            if not torch.allclose(
                chunk.float(), torch.full_like(chunk.float(), expected), atol=1e-3
            ):
                errors.append(f"PE {rank}: allgather_dim0 chunk from PE {pe} mismatch")

        _sdma_tp_cleanup(shmem, comm)

        if errors:
            raise AssertionError("\n".join(errors))


def _worker_tp_allreduce(rank, world_size, port):
    """Worker: test SdmaTpComm.allreduce_sum correctness."""
    os.environ["MORI_ENABLE_SDMA"] = "1"

    import torch.distributed as dist
    from atomrl.lumen.modules.sdma_comm import SdmaTpComm, SdmaTpContext
    from mori import shmem

    with _TorchDistContext(rank, world_size, port):
        tp_group = dist.new_group(list(range(world_size)))
        SdmaTpContext.reset()
        SdmaTpComm.reset()
        comm = SdmaTpComm.get(tp_group)

        n_elems = 256
        local = torch.full(
            (n_elems,),
            float(rank + 1),
            dtype=torch.float32,
            device=f"cuda:{rank}",
        )

        result = comm.allreduce_sum(local)

        expected_sum = sum(range(1, world_size + 1))
        ok = torch.allclose(
            result,
            torch.full_like(result, float(expected_sum)),
            atol=0.5,
        )

        _sdma_tp_cleanup(shmem, comm)

        if not ok:
            raise AssertionError(f"PE {rank}: allreduce_sum mismatch")


def _worker_tp_reduce_scatter(rank, world_size, port):
    """Worker: test SdmaTpComm.reduce_scatter_dim0 correctness."""
    os.environ["MORI_ENABLE_SDMA"] = "1"

    import torch.distributed as dist
    from atomrl.lumen.modules.sdma_comm import SdmaTpComm, SdmaTpContext
    from mori import shmem

    with _TorchDistContext(rank, world_size, port):
        tp_group = dist.new_group(list(range(world_size)))
        SdmaTpContext.reset()
        SdmaTpComm.reset()
        comm = SdmaTpComm.get(tp_group)

        chunk_size = 16
        full_size = chunk_size * world_size
        local = torch.full(
            (full_size,),
            float(rank + 1),
            dtype=torch.float32,
            device=f"cuda:{rank}",
        )

        result = comm.reduce_scatter_dim0(local)

        errors = []
        if result.shape != (chunk_size,):
            errors.append(
                f"PE {rank}: shape mismatch {result.shape} != {(chunk_size,)}"
            )

        expected_sum = sum(range(1, world_size + 1))
        if not torch.allclose(
            result,
            torch.full_like(result, float(expected_sum)),
            atol=0.5,
        ):
            errors.append(f"PE {rank}: reduce_scatter_dim0 mismatch")

        _sdma_tp_cleanup(shmem, comm)

        if errors:
            raise AssertionError("\n".join(errors))


def _worker_tp_allgather_last_dim(rank, world_size, port):
    """Worker: test SdmaTpComm.allgather_last_dim correctness."""
    os.environ["MORI_ENABLE_SDMA"] = "1"

    import torch.distributed as dist
    from atomrl.lumen.modules.sdma_comm import SdmaTpComm, SdmaTpContext
    from mori import shmem

    with _TorchDistContext(rank, world_size, port):
        tp_group = dist.new_group(list(range(world_size)))
        SdmaTpContext.reset()
        SdmaTpComm.reset()
        comm = SdmaTpComm.get(tp_group)

        batch, seq, d_local = 2, 8, 16
        local = torch.full(
            (batch, seq, d_local),
            float(rank + 1),
            dtype=torch.bfloat16,
            device=f"cuda:{rank}",
        )

        gathered = comm.allgather_last_dim(local)

        errors = []
        if gathered.shape != (batch, seq, d_local * world_size):
            errors.append(
                f"PE {rank}: shape mismatch {gathered.shape} != {(batch, seq, d_local * world_size)}"
            )

        for pe in range(world_size):
            chunk = gathered[..., pe * d_local : (pe + 1) * d_local]
            expected = float(pe + 1)
            if not torch.allclose(
                chunk.float(), torch.full_like(chunk.float(), expected), atol=1e-3
            ):
                errors.append(
                    f"PE {rank}: allgather_last_dim chunk from PE {pe} mismatch"
                )

        _sdma_tp_cleanup(shmem, comm)

        if errors:
            raise AssertionError("\n".join(errors))


def _worker_tp_allgather_chunk(rank, world_size, port):
    """Worker: test SdmaTpComm.allgather_dim0_chunk correctness."""
    os.environ["MORI_ENABLE_SDMA"] = "1"

    import torch.distributed as dist
    from atomrl.lumen.modules.sdma_comm import SdmaTpComm, SdmaTpContext
    from mori import shmem

    with _TorchDistContext(rank, world_size, port):
        tp_group = dist.new_group(list(range(world_size)))
        SdmaTpContext.reset()
        SdmaTpComm.reset()
        comm = SdmaTpComm.get(tp_group)

        chunk_rows = 8
        hidden = 64
        input_chunk = torch.full(
            (chunk_rows, hidden),
            float(rank + 1),
            dtype=torch.bfloat16,
            device=f"cuda:{rank}",
        )
        output_buf = torch.zeros(
            chunk_rows * world_size,
            hidden,
            dtype=torch.bfloat16,
            device=f"cuda:{rank}",
        )
        stream = torch.cuda.Stream(device=f"cuda:{rank}")
        ev = comm.allgather_dim0_chunk(input_chunk, output_buf, stream)
        torch.cuda.current_stream(f"cuda:{rank}").wait_event(ev)
        torch.cuda.synchronize()

        errors = []
        if output_buf.shape != (chunk_rows * world_size, hidden):
            errors.append(f"PE {rank}: shape {output_buf.shape}")

        for pe in range(world_size):
            chunk = output_buf[pe * chunk_rows : (pe + 1) * chunk_rows]
            expected = float(pe + 1)
            if not torch.allclose(
                chunk.float(), torch.full_like(chunk.float(), expected), atol=1e-3
            ):
                errors.append(f"PE {rank}: chunk from PE {pe} mismatch")

        _sdma_tp_cleanup(shmem, comm)

        if errors:
            raise AssertionError("\n".join(errors))


def _worker_tp_reduce_scatter_chunk_allreduce(rank, world_size, port):
    """Worker: test SdmaTpComm.reduce_scatter_dim0_chunk with allreduce backend."""
    os.environ["MORI_ENABLE_SDMA"] = "1"

    import torch.distributed as dist
    from atomrl.lumen.modules.sdma_comm import SdmaTpComm, SdmaTpContext
    from mori import shmem

    with _TorchDistContext(rank, world_size, port):
        tp_group = dist.new_group(list(range(world_size)))
        SdmaTpContext.reset()
        SdmaTpComm.reset()
        comm = SdmaTpComm(tp_group, rs_chunk_method="allreduce")

        chunk_full = 16 * world_size
        out_dim = 32
        input_chunk = torch.full(
            (chunk_full, out_dim),
            float(rank + 1),
            dtype=torch.bfloat16,
            device=f"cuda:{rank}",
        )
        output_buf = torch.zeros(
            chunk_full // world_size,
            out_dim,
            dtype=torch.bfloat16,
            device=f"cuda:{rank}",
        )
        stream = torch.cuda.Stream(device=f"cuda:{rank}")
        ev = comm.reduce_scatter_dim0_chunk(input_chunk, output_buf, stream)
        torch.cuda.current_stream(f"cuda:{rank}").wait_event(ev)
        torch.cuda.synchronize()

        expected_sum = float(sum(range(1, world_size + 1)))
        ok = torch.allclose(
            output_buf.float(),
            torch.full_like(output_buf.float(), expected_sum),
            atol=1.0,
        )

        _sdma_tp_cleanup(shmem, comm)

        if not ok:
            raise AssertionError(
                f"PE {rank}: reduce_scatter_chunk (allreduce) mismatch, "
                f"got {output_buf[0, 0].item()}, expected {expected_sum}"
            )


def _worker_tp_reduce_scatter_chunk_nccl(rank, world_size, port):
    """Worker: test SdmaTpComm.reduce_scatter_dim0_chunk with NCCL fallback."""
    os.environ["MORI_ENABLE_SDMA"] = "1"

    import torch.distributed as dist
    from atomrl.lumen.modules.sdma_comm import SdmaTpComm, SdmaTpContext
    from mori import shmem

    with _TorchDistContext(rank, world_size, port):
        tp_group = dist.new_group(list(range(world_size)))
        SdmaTpContext.reset()
        SdmaTpComm.reset()
        comm = SdmaTpComm(tp_group, rs_chunk_method="nccl")

        chunk_full = 16 * world_size
        out_dim = 32
        input_chunk = torch.full(
            (chunk_full, out_dim),
            float(rank + 1),
            dtype=torch.bfloat16,
            device=f"cuda:{rank}",
        )
        output_buf = torch.zeros(
            chunk_full // world_size,
            out_dim,
            dtype=torch.bfloat16,
            device=f"cuda:{rank}",
        )
        stream = torch.cuda.Stream(device=f"cuda:{rank}")
        ev = comm.reduce_scatter_dim0_chunk(input_chunk, output_buf, stream)
        torch.cuda.current_stream(f"cuda:{rank}").wait_event(ev)
        torch.cuda.synchronize()

        expected_sum = float(sum(range(1, world_size + 1)))
        ok = torch.allclose(
            output_buf.float(),
            torch.full_like(output_buf.float(), expected_sum),
            atol=0.5,
        )

        _sdma_tp_cleanup(shmem, comm)

        if not ok:
            raise AssertionError(
                f"PE {rank}: reduce_scatter_chunk (nccl) mismatch, "
                f"got {output_buf[0, 0].item()}, expected {expected_sum}"
            )


@_requires_sdma_hw
class TestSdmaTpCommDistributed:
    """Multi-GPU SdmaTpComm correctness tests."""

    @property
    def world_size(self):
        return min(torch.cuda.device_count(), 8)

    def test_allgather_dim0(self):
        port = _get_free_port()
        _sdma_spawn(
            _worker_tp_allgather_dim0,
            args=(self.world_size, port),
            nprocs=self.world_size,
        )

    def test_allreduce_sum(self):
        port = _get_free_port()
        _sdma_spawn(
            _worker_tp_allreduce,
            args=(self.world_size, port),
            nprocs=self.world_size,
        )

    def test_reduce_scatter_dim0(self):
        port = _get_free_port()
        _sdma_spawn(
            _worker_tp_reduce_scatter,
            args=(self.world_size, port),
            nprocs=self.world_size,
        )

    def test_allgather_last_dim(self):
        port = _get_free_port()
        _sdma_spawn(
            _worker_tp_allgather_last_dim,
            args=(self.world_size, port),
            nprocs=self.world_size,
        )

    def test_allgather_dim0_chunk(self):
        port = _get_free_port()
        _sdma_spawn(
            _worker_tp_allgather_chunk,
            args=(self.world_size, port),
            nprocs=self.world_size,
        )

    def test_reduce_scatter_dim0_chunk_allreduce(self):
        port = _get_free_port()
        _sdma_spawn(
            _worker_tp_reduce_scatter_chunk_allreduce,
            args=(self.world_size, port),
            nprocs=self.world_size,
        )

    def test_reduce_scatter_dim0_chunk_nccl(self):
        port = _get_free_port()
        _sdma_spawn(
            _worker_tp_reduce_scatter_chunk_nccl,
            args=(self.world_size, port),
            nprocs=self.world_size,
        )


# ===================================================================
# Performance benchmarks
# ===================================================================


def _worker_tp_perf(rank, world_size, port, op_name, n_elems, iterations, warmup):
    """Worker: measure TP comm throughput."""
    os.environ["MORI_ENABLE_SDMA"] = "1"

    import numpy as np
    import torch.distributed as dist
    from atomrl.lumen.modules.sdma_comm import SdmaTpComm, SdmaTpContext
    from mori import shmem

    with _TorchDistContext(rank, world_size, port):
        tp_group = dist.new_group(list(range(world_size)))
        SdmaTpContext.reset()
        SdmaTpComm.reset()
        comm = SdmaTpComm.get(tp_group)
        device = f"cuda:{rank}"

        local = torch.randn(n_elems, dtype=torch.bfloat16, device=device)

        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)
        stream = torch.cuda.current_stream(device)

        op_fn = {
            "allgather_dim0": lambda c=comm, t=local: c.allgather_dim0(t),
            "allreduce_sum": lambda c=comm, t=local: c.allreduce_sum(t),
        }[op_name]

        times = []
        for i in range(warmup + iterations):
            start_ev.record(stream)
            _ = op_fn()
            end_ev.record(stream)
            stream.synchronize()
            if i >= warmup:
                times.append(start_ev.elapsed_time(end_ev))

        avg_ms = np.mean(times) if times else 0
        total_bytes = n_elems * 2 * world_size
        bw = (total_bytes / (avg_ms / 1000.0)) / (1024**3) if avg_ms > 0 else 0

        if rank == 0:
            print(
                f"\n  TP {op_name} SDMA PE0: {total_bytes / 1024:.1f} KB, "
                f"avg={avg_ms:.3f}ms, BW={bw:.2f} GB/s"
            )

        _sdma_tp_cleanup(shmem, comm)


@_requires_sdma_hw
class TestSdmaTpCommPerformance:
    """TP communication performance benchmarks."""

    @property
    def world_size(self):
        return min(torch.cuda.device_count(), 8)

    @pytest.mark.parametrize("n_elems", [4096, 65536, 1048576])
    def test_allgather_dim0_perf(self, n_elems):
        port = _get_free_port()
        _sdma_spawn(
            _worker_tp_perf,
            args=(self.world_size, port, "allgather_dim0", n_elems, 10, 5),
            nprocs=self.world_size,
        )

    @pytest.mark.parametrize("n_elems", [4096, 65536, 1048576])
    def test_allreduce_sum_perf(self, n_elems):
        port = _get_free_port()
        _sdma_spawn(
            _worker_tp_perf,
            args=(self.world_size, port, "allreduce_sum", n_elems, 10, 5),
            nprocs=self.world_size,
        )
