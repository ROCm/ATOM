# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

###############################################################################
#
# See LICENSE for license information.
###############################################################################

"""Tests for atomrl.lumen.ops.sdma — SDMA collective wrappers and communication.

Covers:
  - is_sdma_available detection
  - SdmaContext singleton lifecycle (get/reset)
  - SdmaAllgather: basic allgather, buffer reuse, shape consistency
  - SdmaAllreduce: in-place and out-of-place sum, dtype variants
  - SdmaAll2all: basic all-to-all, correctness, buffer reuse
  - sdma_allgather_max: element-wise MAX across PEs
  - Performance: SDMA bandwidth measurement (latency, throughput)

Multi-GPU tests use torch.multiprocessing.spawn with mori shmem init,
following the same pattern as ``mori/tests/python/ccl/test_allreduce.py``.

Hardware requirements:
  - Unit tests (is_sdma_available, SdmaContext): single GPU, no RDMA needed.
  - Distributed tests (TestSdmaDistributed, TestSdmaVsNcclGolden): >= 2 GPUs
    with SDMA support (``is_sdma_available()`` must return True).
  - Performance tests (TestSdmaPerformance): same as distributed.

How to run::

    # All tests (unit + multi-GPU); requires SDMA-capable hardware:
    pytest tests/ops/test_sdma.py -v

    # Unit tests only (single GPU):
    pytest tests/ops/test_sdma.py -v -k "not Distributed and not VsNccl and not Performance"

    # Multi-GPU correctness only (>= 2 GPUs):
    pytest tests/ops/test_sdma.py -v -k "Distributed or VsNccl"

    # Performance benchmarks only (>= 2 GPUs):
    pytest tests/ops/test_sdma.py -v -k "Performance"

    # Override SDMA spawn cooldown for fast local iteration:
    LUMEN_SDMA_SPAWN_COOLDOWN_SECS=0 pytest tests/ops/test_sdma.py -v
"""

import os
import time

import pytest
import torch
from atomrl.lumen.ops.sdma import is_sdma_available


def _get_free_port():
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


_requires_sdma_hw = pytest.mark.skipif(
    not is_sdma_available(),
    reason="SDMA not available (is_sdma_available() returned False)",
)


# ===================================================================
# TorchDistContext — matches mori/tests/python/utils.py exactly
# ===================================================================


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


# ===================================================================
# is_sdma_available
# ===================================================================


def test_is_sdma_available_returns_true():
    """is_sdma_available returns True in this environment."""
    assert is_sdma_available() is True


# ===================================================================
# SdmaContext singleton
# ===================================================================


class TestSdmaContext:
    def setup_method(self):
        from atomrl.lumen.ops.sdma import SdmaContext

        SdmaContext.reset()

    def test_reset_clears_singleton(self):
        from atomrl.lumen.ops.sdma import SdmaContext

        SdmaContext.reset()
        assert SdmaContext._instance is None


# ===================================================================
# SdmaAllgather unit tests
# ===================================================================


class TestSdmaAllgatherUnit:
    """Unit tests that validate the SdmaAllgather API contracts without full distributed init."""

    def test_npes_property(self):
        """npes should delegate to the context."""
        from unittest.mock import MagicMock

        from atomrl.lumen.ops.sdma import SdmaAllgather

        fake_ctx = MagicMock()
        fake_ctx.npes = 4
        ag = SdmaAllgather.__new__(SdmaAllgather)
        ag._ctx = fake_ctx
        ag._handle = None
        ag._capacity_elems = 0
        ag._gathered = None
        assert ag.npes == 4

    def test_ensure_handle_skips_when_capacity_sufficient(self):
        """_ensure_handle should not recreate when capacity is sufficient."""
        from unittest.mock import MagicMock

        from atomrl.lumen.ops.sdma import SdmaAllgather

        fake_ctx = MagicMock()
        fake_ctx.my_pe = 0
        fake_ctx.npes = 2
        ag = SdmaAllgather.__new__(SdmaAllgather)
        ag._ctx = fake_ctx
        sentinel = MagicMock()
        ag._handle = sentinel
        ag._capacity_elems = 512
        ag._gathered = None
        ag._ensure_handle(256)
        assert ag._handle is sentinel


# ===================================================================
# SdmaAll2all unit tests
# ===================================================================


class TestSdmaAll2allUnit:
    """Validate SdmaAll2all API shape contracts."""

    def test_npes_property(self):
        from unittest.mock import MagicMock

        from atomrl.lumen.ops.sdma import SdmaAll2all

        fake_ctx = MagicMock()
        fake_ctx.npes = 8
        a2a = SdmaAll2all.__new__(SdmaAll2all)
        a2a._ctx = fake_ctx
        a2a._handle = None
        a2a._capacity_bytes = 0
        assert a2a.npes == 8

    def test_ensure_handle_reuses_when_fits(self):
        from unittest.mock import MagicMock

        from atomrl.lumen.ops.sdma import SdmaAll2all

        fake_ctx = MagicMock()
        fake_ctx.my_pe = 0
        fake_ctx.npes = 2
        a2a = SdmaAll2all.__new__(SdmaAll2all)
        a2a._ctx = fake_ctx
        sentinel = MagicMock()
        a2a._handle = sentinel
        a2a._capacity_bytes = 1024
        a2a._ensure_handle(512)
        assert a2a._handle is sentinel


# ===================================================================
# Spawn helpers
# ===================================================================


_SDMA_SPAWN_COOLDOWN_SECS = float(os.environ.get("LUMEN_SDMA_SPAWN_COOLDOWN_SECS", "2"))

_SDMA_SPAWN_TIMEOUT_SECS = int(os.environ.get("LUMEN_SDMA_SPAWN_TIMEOUT_SECS", "120"))


def _sdma_spawn(fn, args, nprocs, join=True):
    """``mp.spawn`` wrapper with a cooldown for KFD SDMA queue reclamation.

    A short pre-spawn sleep prevents back-to-back ``mp.spawn`` calls from
    exhausting the SDMA queue pool on KFD.  Override with
    ``LUMEN_SDMA_SPAWN_COOLDOWN_SECS=0`` for fast local runs.

    Uses ``join=False`` + polling with a deadline to avoid indefinite hangs.
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


def _sdma_cleanup(shmem_mod, *ccl_wrappers):
    """Cleanup helper matching mori ccl test teardown order **exactly**.

    Must be called before leaving the ``_TorchDistContext`` block.  Pass all
    CCL wrapper objects (``SdmaAllgather``, ``SdmaAllreduce``, ``SdmaAll2all``)
    so their C++ handles are released at the right point in the sequence.

    Matches ``mori/tests/python/ccl/test_allgather.py`` lines 276-281::

        torch.cuda.synchronize()          # 1  all local GPU work done
        dist.barrier()                    # 2  all ranks synchronised
        del allgather                     # 3  release SHMEM buffers
        dist.barrier()                    # 4  all ranks released
        shmem.shmem_finalize()            # 5  close SDMA/HSA resources
    """
    import torch.distributed as dist
    from atomrl.lumen.ops.sdma import SdmaContext

    torch.cuda.synchronize()
    dist.barrier()
    for w in ccl_wrappers:
        if hasattr(w, "_handle"):
            w._handle = None
    SdmaContext.reset()
    dist.barrier()
    shmem_mod.shmem_finalize()


# ===================================================================
# Multi-GPU distributed SDMA tests
# ===================================================================


def _worker_allgather(rank, world_size, port):
    """Worker: test SdmaAllgather correctness (mori pattern)."""
    os.environ["MORI_ENABLE_SDMA"] = "1"

    from atomrl.lumen.ops.sdma import SdmaAllgather, SdmaContext
    from mori import shmem

    with _TorchDistContext(rank, world_size, port):
        ctx = SdmaContext.get()
        ag = SdmaAllgather(ctx)

        n_elems = 1024
        local = torch.full(
            (n_elems,), float(rank + 1), dtype=torch.float32, device=f"cuda:{rank}"
        )
        gathered = ag(local)

        errors = []
        for pe in range(world_size):
            expected_val = float(pe + 1)
            chunk = gathered[pe]
            if not torch.allclose(chunk, torch.full_like(chunk, expected_val)):
                errors.append(f"PE {rank}: chunk from PE {pe} mismatch")

        _sdma_cleanup(shmem, ag)

        if errors:
            raise AssertionError("\n".join(errors))


def _worker_all2all(rank, world_size, port):
    """Worker: test SdmaAll2all correctness (mori pattern)."""
    os.environ["MORI_ENABLE_SDMA"] = "1"

    from atomrl.lumen.ops.sdma import SdmaAll2all, SdmaContext
    from mori import shmem

    with _TorchDistContext(rank, world_size, port):
        ctx = SdmaContext.get()
        a2a = SdmaAll2all(ctx)

        elems_per_pe = 256
        device = f"cuda:{rank}"
        input_tensor = torch.zeros(
            elems_per_pe * world_size, dtype=torch.uint32, device=device
        )
        for dest_pe in range(world_size):
            value = (rank + 1) * 1000 + dest_pe
            input_tensor[dest_pe * elems_per_pe : (dest_pe + 1) * elems_per_pe] = value

        output_tensor = torch.zeros_like(input_tensor)
        a2a(input_tensor, output_tensor)

        errors = []
        for src_pe in range(world_size):
            expected_value = (src_pe + 1) * 1000 + rank
            chunk = output_tensor[src_pe * elems_per_pe : (src_pe + 1) * elems_per_pe]
            if not torch.all(chunk == expected_value):
                errors.append(f"PE {rank}: chunk from PE {src_pe} mismatch")

        _sdma_cleanup(shmem, a2a)

        if errors:
            raise AssertionError("\n".join(errors))


def _worker_allreduce(rank, world_size, port):
    """Worker: test SdmaAllreduce correctness (mori pattern)."""
    os.environ["MORI_ENABLE_SDMA"] = "1"

    from atomrl.lumen.ops.sdma import SdmaAllreduce, SdmaContext
    from mori import shmem

    with _TorchDistContext(rank, world_size, port):
        ctx = SdmaContext.get()
        ar = SdmaAllreduce(ctx=ctx)

        n_elems = 1024
        local = torch.full(
            (n_elems,), float(rank + 1), dtype=torch.float32, device=f"cuda:{rank}"
        )
        expected_sum = float(sum(range(1, world_size + 1)))

        result = local.clone()
        ar.inplace(result)

        ok = torch.allclose(result, torch.full_like(result, expected_sum), atol=0.5)

        _sdma_cleanup(shmem, ar)

        if not ok:
            raise AssertionError(f"PE {rank}: allreduce sum mismatch")


@_requires_sdma_hw
class TestSdmaDistributed:
    """Multi-GPU SDMA correctness tests."""

    @property
    def world_size(self):
        return min(torch.cuda.device_count(), 8)

    def test_allgather_correctness(self):
        """SdmaAllgather produces correct gathered data across PEs."""
        port = _get_free_port()
        _sdma_spawn(
            _worker_allgather,
            args=(self.world_size, port),
            nprocs=self.world_size,
        )

    def test_all2all_correctness(self):
        """SdmaAll2all produces correct shuffled data across PEs."""
        port = _get_free_port()
        _sdma_spawn(
            _worker_all2all,
            args=(self.world_size, port),
            nprocs=self.world_size,
        )

    def test_allreduce_correctness(self):
        """SdmaAllreduce SUM produces correct reduced values."""
        port = _get_free_port()
        _sdma_spawn(
            _worker_allreduce,
            args=(self.world_size, port),
            nprocs=self.world_size,
        )


# ===================================================================
# SDMA vs NCCL golden correctness tests
# ===================================================================


def _worker_allgather_vs_nccl(rank, world_size, port, n_elems, seed):
    """Worker: compare SdmaAllgather output against NCCL all_gather golden."""
    os.environ["MORI_ENABLE_SDMA"] = "1"

    import torch.distributed as dist
    from atomrl.lumen.ops.sdma import SdmaAllgather, SdmaContext
    from mori import shmem

    with _TorchDistContext(rank, world_size, port):
        device = f"cuda:{rank}"
        torch.manual_seed(seed + rank)
        local = torch.randn(n_elems, dtype=torch.float32, device=device)

        ctx = SdmaContext.get()
        ag = SdmaAllgather(ctx)
        sdma_gathered = ag(local)

        nccl_gathered = [torch.empty_like(local) for _ in range(world_size)]
        dist.all_gather(nccl_gathered, local)
        nccl_stacked = torch.stack(nccl_gathered)

        ok = torch.allclose(sdma_gathered, nccl_stacked, atol=1e-6)

        _sdma_cleanup(shmem, ag)

        if not ok:
            raise AssertionError(f"PE {rank}: SdmaAllgather != NCCL (n={n_elems})")


def _worker_all2all_vs_nccl(rank, world_size, port, elems_per_pe, seed):
    """Worker: compare SdmaAll2all output against NCCL all_to_all golden."""
    os.environ["MORI_ENABLE_SDMA"] = "1"

    import torch.distributed as dist
    from atomrl.lumen.ops.sdma import SdmaAll2all, SdmaContext
    from mori import shmem

    with _TorchDistContext(rank, world_size, port):
        device = f"cuda:{rank}"
        total_elems = elems_per_pe * world_size
        torch.manual_seed(seed + rank)
        input_tensor = torch.randint(
            0, 2**31, (total_elems,), dtype=torch.uint32, device=device
        )

        sdma_output = torch.zeros_like(input_tensor)
        ctx = SdmaContext.get()
        a2a = SdmaAll2all(ctx)
        a2a(input_tensor, sdma_output)

        nccl_input = input_tensor.view(torch.int32)
        nccl_output = torch.zeros_like(nccl_input)
        dist.all_to_all_single(nccl_output, nccl_input)
        nccl_output_u32 = nccl_output.view(torch.uint32)

        ok = torch.equal(sdma_output, nccl_output_u32)

        _sdma_cleanup(shmem, a2a)

        if not ok:
            raise AssertionError(
                f"PE {rank}: SdmaAll2all != NCCL (per_pe={elems_per_pe})"
            )


def _worker_allreduce_vs_nccl(rank, world_size, port, n_elems, seed):
    """Worker: compare SdmaAllreduce SUM against NCCL all_reduce golden."""
    os.environ["MORI_ENABLE_SDMA"] = "1"

    import torch.distributed as dist
    from atomrl.lumen.ops.sdma import SdmaAllreduce, SdmaContext
    from mori import shmem

    with _TorchDistContext(rank, world_size, port):
        device = f"cuda:{rank}"
        torch.manual_seed(seed + rank)
        local = torch.randn(n_elems, dtype=torch.float32, device=device)

        ctx = SdmaContext.get()
        ar = SdmaAllreduce(ctx=ctx)
        sdma_result = local.clone()
        ar.inplace(sdma_result)

        nccl_result = local.clone()
        dist.all_reduce(nccl_result, op=dist.ReduceOp.SUM)

        ok = torch.allclose(sdma_result, nccl_result, atol=0.5, rtol=0.02)

        _sdma_cleanup(shmem, ar)

        if not ok:
            raise AssertionError(f"PE {rank}: SdmaAllreduce != NCCL (n={n_elems})")


def _worker_allreduce_outofplace(rank, world_size, port, n_elems, seed):
    """Worker: test SdmaAllreduce out-of-place __call__ path."""
    os.environ["MORI_ENABLE_SDMA"] = "1"

    import torch.distributed as dist
    from atomrl.lumen.ops.sdma import SdmaAllreduce, SdmaContext
    from mori import shmem

    with _TorchDistContext(rank, world_size, port):
        device = f"cuda:{rank}"
        torch.manual_seed(seed + rank)
        local = torch.randn(n_elems, dtype=torch.float32, device=device)

        ctx = SdmaContext.get()
        ar = SdmaAllreduce(ctx=ctx)
        sdma_output = torch.zeros_like(local)
        ar(local, sdma_output)

        nccl_result = local.clone()
        dist.all_reduce(nccl_result, op=dist.ReduceOp.SUM)

        ok = torch.allclose(sdma_output, nccl_result, atol=0.5, rtol=0.02)

        _sdma_cleanup(shmem, ar)

        if not ok:
            raise AssertionError(f"PE {rank}: SdmaAllreduce outofplace != NCCL")


def _worker_allgather_max(rank, world_size, port, n_elems, seed):
    """Worker: test sdma_allgather_max against torch.distributed MAX golden."""
    os.environ["MORI_ENABLE_SDMA"] = "1"

    import torch.distributed as dist
    from atomrl.lumen.ops.sdma import SdmaAllgather, SdmaContext, sdma_allgather_max
    from mori import shmem

    with _TorchDistContext(rank, world_size, port):
        device = f"cuda:{rank}"
        torch.manual_seed(seed + rank)
        local = torch.randn(n_elems, dtype=torch.float32, device=device)

        ctx = SdmaContext.get()
        ag = SdmaAllgather(ctx)
        sdma_max_result = sdma_allgather_max(local, ag)

        nccl_max = local.clone()
        dist.all_reduce(nccl_max, op=dist.ReduceOp.MAX)

        ok = torch.allclose(sdma_max_result, nccl_max, atol=1e-6)

        _sdma_cleanup(shmem, ag)

        if not ok:
            raise AssertionError(
                f"PE {rank}: sdma_allgather_max != NCCL MAX (n={n_elems})"
            )


def _worker_allgather_buffer_reuse(rank, world_size, port, seed):
    """Worker: call SdmaAllgather multiple times with different data/sizes."""
    os.environ["MORI_ENABLE_SDMA"] = "1"

    import torch.distributed as dist
    from atomrl.lumen.ops.sdma import SdmaAllgather, SdmaContext
    from mori import shmem

    with _TorchDistContext(rank, world_size, port):
        device = f"cuda:{rank}"
        ctx = SdmaContext.get()
        ag = SdmaAllgather(ctx)

        errors = []
        for call_idx, n_elems in enumerate([256, 1024, 512, 2048, 256]):
            torch.manual_seed(seed + rank * 100 + call_idx)
            local = torch.randn(n_elems, dtype=torch.float32, device=device)

            sdma_gathered = ag(local)

            nccl_gathered = [torch.empty_like(local) for _ in range(world_size)]
            dist.all_gather(nccl_gathered, local)
            nccl_stacked = torch.stack(nccl_gathered)

            if not torch.allclose(sdma_gathered, nccl_stacked, atol=1e-6):
                errors.append(
                    f"PE {rank}: allgather buffer reuse failed at call {call_idx}"
                )

        _sdma_cleanup(shmem, ag)

        if errors:
            raise AssertionError("\n".join(errors))


def _worker_all2all_buffer_reuse(rank, world_size, port, seed):
    """Worker: call SdmaAll2all multiple times with different sizes."""
    os.environ["MORI_ENABLE_SDMA"] = "1"

    import torch.distributed as dist
    from atomrl.lumen.ops.sdma import SdmaAll2all, SdmaContext
    from mori import shmem

    with _TorchDistContext(rank, world_size, port):
        device = f"cuda:{rank}"
        ctx = SdmaContext.get()
        a2a = SdmaAll2all(ctx)

        errors = []
        for call_idx, elems_per_pe in enumerate([64, 512, 256, 1024, 64]):
            total = elems_per_pe * world_size
            torch.manual_seed(seed + rank * 100 + call_idx)
            inp = torch.randint(0, 2**31, (total,), dtype=torch.uint32, device=device)
            sdma_out = torch.zeros_like(inp)
            a2a(inp, sdma_out)

            nccl_in = inp.view(torch.int32)
            nccl_out = torch.zeros_like(nccl_in)
            dist.all_to_all_single(nccl_out, nccl_in)

            if not torch.equal(sdma_out, nccl_out.view(torch.uint32)):
                errors.append(
                    f"PE {rank}: all2all buffer reuse failed at call {call_idx}"
                )

        _sdma_cleanup(shmem, a2a)

        if errors:
            raise AssertionError("\n".join(errors))


@_requires_sdma_hw
class TestSdmaVsNcclGolden:
    """SDMA vs NCCL golden comparison with random data."""

    @property
    def world_size(self):
        return min(torch.cuda.device_count(), 8)

    @pytest.mark.parametrize("n_elems", [128, 1024, 65536, 1048576])
    def test_allgather_vs_nccl(self, n_elems):
        """SdmaAllgather matches NCCL all_gather on random data."""
        port = _get_free_port()
        _sdma_spawn(
            _worker_allgather_vs_nccl,
            args=(self.world_size, port, n_elems, 42),
            nprocs=self.world_size,
        )

    @pytest.mark.parametrize("elems_per_pe", [64, 256, 4096, 65536])
    def test_all2all_vs_nccl(self, elems_per_pe):
        """SdmaAll2all matches NCCL all_to_all_single on random data."""
        port = _get_free_port()
        _sdma_spawn(
            _worker_all2all_vs_nccl,
            args=(self.world_size, port, elems_per_pe, 42),
            nprocs=self.world_size,
        )

    @pytest.mark.parametrize("n_elems", [128, 1024, 65536, 1048576])
    def test_allreduce_vs_nccl(self, n_elems):
        """SdmaAllreduce SUM matches NCCL all_reduce SUM on random data."""
        port = _get_free_port()
        _sdma_spawn(
            _worker_allreduce_vs_nccl,
            args=(self.world_size, port, n_elems, 42),
            nprocs=self.world_size,
        )

    def test_allreduce_outofplace_vs_nccl(self):
        """SdmaAllreduce out-of-place __call__ matches NCCL all_reduce."""
        port = _get_free_port()
        _sdma_spawn(
            _worker_allreduce_outofplace,
            args=(self.world_size, port, 4096, 42),
            nprocs=self.world_size,
        )

    @pytest.mark.parametrize("n_elems", [256, 4096, 65536])
    def test_allgather_max_vs_nccl(self, n_elems):
        """sdma_allgather_max matches NCCL all_reduce MAX on random data."""
        port = _get_free_port()
        _sdma_spawn(
            _worker_allgather_max,
            args=(self.world_size, port, n_elems, 42),
            nprocs=self.world_size,
        )

    def test_allgather_buffer_reuse(self):
        """SdmaAllgather handle reuse across varying sizes stays correct."""
        port = _get_free_port()
        _sdma_spawn(
            _worker_allgather_buffer_reuse,
            args=(self.world_size, port, 42),
            nprocs=self.world_size,
        )

    def test_all2all_buffer_reuse(self):
        """SdmaAll2all handle reuse across varying sizes stays correct."""
        port = _get_free_port()
        _sdma_spawn(
            _worker_all2all_buffer_reuse,
            args=(self.world_size, port, 42),
            nprocs=self.world_size,
        )


# ===================================================================
# SDMA performance benchmarks
# ===================================================================


def _worker_all2all_perf(rank, world_size, port, elems_per_pe, iterations, warmup):
    """Worker: measure SdmaAll2all throughput (mori pattern)."""
    os.environ["MORI_ENABLE_SDMA"] = "1"

    import numpy as np
    from atomrl.lumen.ops.sdma import SdmaAll2all, SdmaContext
    from mori import shmem

    with _TorchDistContext(rank, world_size, port):
        ctx = SdmaContext.get()
        a2a = SdmaAll2all(ctx)
        device = f"cuda:{rank}"

        input_tensor = torch.randint(
            0, 2**31, (elems_per_pe * world_size,), dtype=torch.uint32, device=device
        )
        output_tensor = torch.zeros_like(input_tensor)

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        stream = torch.cuda.current_stream(device)

        times = []
        for i in range(warmup + iterations):
            start_event.record(stream)
            a2a(input_tensor, output_tensor, stream)
            end_event.record(stream)
            stream.synchronize()
            elapsed_ms = start_event.elapsed_time(end_event)
            if i >= warmup:
                times.append(elapsed_ms)

        avg_ms = np.mean(times) if times else 0.0
        total_bytes = elems_per_pe * world_size * 4
        bandwidth_gb_s = (
            (total_bytes / (avg_ms / 1000.0)) / (1024**3) if avg_ms > 0 else 0
        )

        if rank == 0:
            print(
                f"\n  All2all SDMA PE0: {total_bytes / 1024**2:.1f} MB, "
                f"avg={avg_ms:.3f}ms, BW={bandwidth_gb_s:.2f} GB/s"
            )

        _sdma_cleanup(shmem, a2a)


def _worker_allgather_perf(rank, world_size, port, n_elems, iterations, warmup):
    """Worker: measure SdmaAllgather throughput (mori pattern)."""
    os.environ["MORI_ENABLE_SDMA"] = "1"

    import numpy as np
    from atomrl.lumen.ops.sdma import SdmaAllgather, SdmaContext
    from mori import shmem

    with _TorchDistContext(rank, world_size, port):
        ctx = SdmaContext.get()
        ag = SdmaAllgather(ctx)
        device = f"cuda:{rank}"

        local = torch.randn(n_elems, dtype=torch.float32, device=device)

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        stream = torch.cuda.current_stream(device)

        times = []
        for i in range(warmup + iterations):
            start_event.record(stream)
            _ = ag(local, stream)
            end_event.record(stream)
            stream.synchronize()
            elapsed_ms = start_event.elapsed_time(end_event)
            if i >= warmup:
                times.append(elapsed_ms)

        avg_ms = np.mean(times) if times else 0.0
        total_bytes = n_elems * 4 * world_size
        bandwidth_gb_s = (
            (total_bytes / (avg_ms / 1000.0)) / (1024**3) if avg_ms > 0 else 0
        )

        if rank == 0:
            print(
                f"\n  Allgather SDMA PE0: {total_bytes / 1024**2:.1f} MB, "
                f"avg={avg_ms:.3f}ms, BW={bandwidth_gb_s:.2f} GB/s"
            )

        _sdma_cleanup(shmem, ag)


@_requires_sdma_hw
class TestSdmaPerformance:
    """SDMA performance benchmarks.

    These tests measure latency and bandwidth. They pass unconditionally
    (no hard threshold) but print results for manual inspection.
    """

    @property
    def world_size(self):
        return min(torch.cuda.device_count(), 8)

    @pytest.mark.parametrize("elems_per_pe", [1024, 65536, 1048576, 16777216])
    def test_all2all_bandwidth(self, elems_per_pe):
        """Measure SdmaAll2all bandwidth for various data sizes."""
        port = _get_free_port()
        _sdma_spawn(
            _worker_all2all_perf,
            args=(self.world_size, port, elems_per_pe, 10, 5),
            nprocs=self.world_size,
        )

    @pytest.mark.parametrize("n_elems", [1024, 65536, 1048576, 16777216])
    def test_allgather_bandwidth(self, n_elems):
        """Measure SdmaAllgather bandwidth for various data sizes."""
        port = _get_free_port()
        _sdma_spawn(
            _worker_allgather_perf,
            args=(self.world_size, port, n_elems, 10, 5),
            nprocs=self.world_size,
        )
