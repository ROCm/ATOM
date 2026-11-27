# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Tests for atomrl.lumen.modules.comm_overlap — chunked comm-GEMM overlap engines.

Covers:
  - UserBufferPool: allocation, reuse, reallocation, multi-key isolation
  - NcclCommBackend: allgather_chunk, reduce_scatter_chunk correctness
  - PipelinedAllgatherGemm: numerical correctness vs naive for various configs
  - PipelinedGemmReduceScatter: numerical correctness vs naive
  - End-to-end pipeline: bulk vs pipeline mode equivalence

Multi-GPU tests use torch.multiprocessing.spawn with NCCL.

How to run::

    # All tests (unit + multi-GPU); multi-GPU tests require >= 2 GPUs (some up to 4):
    pytest tests/modules/test_comm_overlap.py -v

    # Unit tests only (single GPU):
    pytest tests/modules/test_comm_overlap.py -v -k "Pool"

    # Multi-GPU comm-overlap tests (mp.spawn, up to 4 GPUs):
    pytest tests/modules/test_comm_overlap.py -v -k "not Pool"
"""

import os
import time

import pytest
import torch
import torch.nn.functional as F

# ===================================================================
# Skip markers
# ===================================================================


def _has_multi_gpu() -> bool:
    try:
        return torch.cuda.device_count() >= 2
    except (AssertionError, RuntimeError):
        return False


_requires_multi_gpu = pytest.mark.skipif(
    not _has_multi_gpu(),
    reason="Requires at least 2 GPUs",
)

_SPAWN_COOLDOWN = float(os.environ.get("LUMEN_SPAWN_COOLDOWN_SECS", "1"))


def _nccl_spawn(fn, args, nprocs, join=True):
    """mp.spawn wrapper with cooldown for NCCL process group cleanup."""
    import torch.multiprocessing as mp

    if _SPAWN_COOLDOWN > 0:
        time.sleep(_SPAWN_COOLDOWN)
    return mp.spawn(fn, args=args, nprocs=nprocs, join=join)


def _find_free_port():
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


# ===================================================================
# UserBufferPool — single GPU tests
# ===================================================================


class TestUserBufferPool:
    def test_alloc_returns_correct_shape_dtype(self):
        from atomrl.lumen.modules.comm_overlap import UserBufferPool

        device = (
            torch.device("cuda", 0)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        pool = UserBufferPool(device)
        buf = pool.get("test", (4, 8), torch.float32)
        assert buf.shape == (4, 8)
        assert buf.dtype == torch.float32
        assert buf.device.type == device.type

    def test_reuse_same_shape(self):
        from atomrl.lumen.modules.comm_overlap import UserBufferPool

        device = (
            torch.device("cuda", 0)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        pool = UserBufferPool(device)
        buf1 = pool.get("test", (4, 8), torch.float32)
        buf1.fill_(42.0)
        buf2 = pool.get("test", (4, 8), torch.float32)
        assert buf2.data_ptr() == buf1.data_ptr()

    def test_realloc_on_larger_shape(self):
        from atomrl.lumen.modules.comm_overlap import UserBufferPool

        device = (
            torch.device("cuda", 0)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        pool = UserBufferPool(device)
        pool.get("test", (4, 8), torch.float32)
        buf2 = pool.get("test", (8, 16), torch.float32)
        assert buf2.shape == (8, 16)
        assert buf2.numel() >= 128

    def test_reuse_on_smaller_shape(self):
        from atomrl.lumen.modules.comm_overlap import UserBufferPool

        device = (
            torch.device("cuda", 0)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        pool = UserBufferPool(device)
        pool.get("test", (8, 16), torch.float32)
        buf2 = pool.get("test", (4, 8), torch.float32)
        assert buf2.shape == (4, 8)

    def test_multi_key_isolation(self):
        from atomrl.lumen.modules.comm_overlap import UserBufferPool

        device = (
            torch.device("cuda", 0)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        pool = UserBufferPool(device)
        buf_a = pool.get("a", (4, 8), torch.float32)
        buf_b = pool.get("b", (4, 8), torch.float32)
        assert buf_a.data_ptr() != buf_b.data_ptr()

    def test_dtype_key_isolation(self):
        from atomrl.lumen.modules.comm_overlap import UserBufferPool

        device = (
            torch.device("cuda", 0)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        pool = UserBufferPool(device)
        buf_f32 = pool.get("test", (4, 8), torch.float32)
        buf_bf16 = pool.get("test", (4, 8), torch.bfloat16)
        assert buf_f32.dtype == torch.float32
        assert buf_bf16.dtype == torch.bfloat16

    def test_reset_clears_buffers(self):
        from atomrl.lumen.modules.comm_overlap import UserBufferPool

        device = (
            torch.device("cuda", 0)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        pool = UserBufferPool(device)
        pool.get("test", (4, 8), torch.float32)
        pool.reset()
        assert len(pool._buffers) == 0


# ===================================================================
# Multi-GPU worker functions
# ===================================================================


def _worker_nccl_ag_correctness(rank, world_size, port, results_dict):
    """Worker: test NcclCommBackend.allgather_chunk correctness."""
    import torch.distributed as dist
    from atomrl.lumen.modules.comm_overlap import NcclCommBackend

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    try:
        backend = NcclCommBackend(dist.group.WORLD)
        stream = torch.cuda.Stream(device=device)

        chunk = torch.full((4, 8), float(rank + 1), device=device, dtype=torch.bfloat16)
        output = torch.zeros(4 * world_size, 8, device=device, dtype=torch.bfloat16)

        ev = backend.allgather_chunk(chunk, output, stream)
        torch.cuda.current_stream(device).wait_event(ev)
        torch.cuda.synchronize()

        for r in range(world_size):
            expected_val = float(r + 1)
            actual = output[r * 4 : (r + 1) * 4]
            if not torch.allclose(actual, torch.full_like(actual, expected_val)):
                results_dict[rank] = f"FAIL: rank {r} chunk mismatch"
                return
        results_dict[rank] = "PASS"
    finally:
        dist.destroy_process_group()


def _worker_nccl_rs_correctness(rank, world_size, port, results_dict):
    """Worker: test NcclCommBackend.reduce_scatter_chunk correctness."""
    import torch.distributed as dist
    from atomrl.lumen.modules.comm_overlap import NcclCommBackend

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    try:
        backend = NcclCommBackend(dist.group.WORLD)
        stream = torch.cuda.Stream(device=device)

        S_full = 8 * world_size
        chunk = torch.ones(S_full, 4, device=device, dtype=torch.bfloat16) * (rank + 1)
        output = torch.zeros(8, 4, device=device, dtype=torch.bfloat16)

        ev = backend.reduce_scatter_chunk(chunk, output, stream)
        torch.cuda.current_stream(device).wait_event(ev)
        torch.cuda.synchronize()

        expected_sum = sum(r + 1 for r in range(world_size))
        if not torch.allclose(
            output, torch.full_like(output, float(expected_sum)), atol=0.1
        ):
            results_dict[rank] = (
                f"FAIL: expected {expected_sum}, got {output[0, 0].item()}"
            )
            return
        results_dict[rank] = "PASS"
    finally:
        dist.destroy_process_group()


def _worker_pipeline_ag_gemm(rank, world_size, port, results_dict, num_chunks):
    """Worker: test PipelinedAllgatherGemm correctness vs naive."""
    import torch.distributed as dist
    from atomrl.lumen.modules.comm_overlap import (
        NcclCommBackend,
        PipelinedAllgatherGemm,
    )

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    try:
        torch.manual_seed(42)
        S_local = 16
        H = 32
        O_local = 16

        input_local = torch.randn(S_local, H, device=device, dtype=torch.bfloat16)
        weight = torch.randn(O_local, H, device=device, dtype=torch.bfloat16)

        gathered_chunks = [torch.zeros_like(input_local) for _ in range(world_size)]
        dist.all_gather(gathered_chunks, input_local)
        gathered_full = torch.cat(gathered_chunks, dim=0)
        expected = F.linear(gathered_full, weight)

        backend = NcclCommBackend(dist.group.WORLD)
        pipeline = PipelinedAllgatherGemm(num_chunks, backend)

        def gemm_fn(inp, w, b):
            return F.linear(inp, w, b)

        actual = pipeline.forward(input_local, weight, None, gemm_fn)

        max_diff = (expected - actual).abs().max().item()
        if max_diff > 1e-2:
            results_dict[rank] = f"FAIL: max_diff={max_diff}"
        else:
            results_dict[rank] = f"PASS: max_diff={max_diff}"
    finally:
        dist.destroy_process_group()


def _worker_pipeline_gemm_rs(rank, world_size, port, results_dict, num_chunks):
    """Worker: test PipelinedGemmReduceScatter correctness vs naive."""
    import torch.distributed as dist
    from atomrl.lumen.modules.comm_overlap import (
        NcclCommBackend,
        PipelinedGemmReduceScatter,
    )

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    try:
        torch.manual_seed(42)
        S_full = 16 * world_size
        H_local = 32
        out_dim = 16

        input_full = torch.randn(S_full, H_local, device=device, dtype=torch.bfloat16)
        weight = torch.randn(out_dim, H_local, device=device, dtype=torch.bfloat16)

        gemm_out = F.linear(input_full, weight)
        expected = torch.zeros(
            S_full // world_size, out_dim, device=device, dtype=torch.bfloat16
        )
        dist.reduce_scatter_tensor(expected, gemm_out)

        backend = NcclCommBackend(dist.group.WORLD)
        pipeline = PipelinedGemmReduceScatter(num_chunks, backend)

        def gemm_fn(inp, w, b):
            return F.linear(inp, w, b)

        actual = pipeline.forward(input_full, weight, gemm_fn)

        max_diff = (expected - actual).abs().max().item()
        if max_diff > 1e-2:
            results_dict[rank] = f"FAIL: max_diff={max_diff}"
        else:
            results_dict[rank] = f"PASS: max_diff={max_diff}"
    finally:
        dist.destroy_process_group()


def _worker_pipeline_standalone_rs(rank, world_size, port, results_dict, num_chunks):
    """Worker: test PipelinedGemmReduceScatter.forward_standalone correctness."""
    import torch.distributed as dist
    from atomrl.lumen.modules.comm_overlap import (
        NcclCommBackend,
        PipelinedGemmReduceScatter,
    )

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    try:
        torch.manual_seed(42)
        S_full = 16 * world_size
        out_dim = 32

        tensor = torch.randn(S_full, out_dim, device=device, dtype=torch.bfloat16) * (
            rank + 1
        )

        expected = torch.zeros(
            S_full // world_size, out_dim, device=device, dtype=torch.bfloat16
        )
        dist.reduce_scatter_tensor(expected, tensor)

        backend = NcclCommBackend(dist.group.WORLD)
        pipeline = PipelinedGemmReduceScatter(num_chunks, backend)
        actual = pipeline.forward_standalone(tensor)

        max_diff = (expected - actual).abs().max().item()
        if max_diff > 1e-2:
            results_dict[rank] = f"FAIL: max_diff={max_diff}"
        else:
            results_dict[rank] = f"PASS: max_diff={max_diff}"
    finally:
        dist.destroy_process_group()


# ===================================================================
# Multi-GPU test classes
# ===================================================================


@_requires_multi_gpu
class TestNcclCommBackend:
    def test_allgather_chunk_correctness(self):
        import torch.multiprocessing as mp

        port = _find_free_port()
        results = mp.Manager().dict()
        world_size = min(torch.cuda.device_count(), 2)
        _nccl_spawn(
            _worker_nccl_ag_correctness,
            (world_size, port, results),
            nprocs=world_size,
        )
        for rank, result in results.items():
            assert result == "PASS", f"Rank {rank}: {result}"

    def test_reduce_scatter_chunk_correctness(self):
        import torch.multiprocessing as mp

        port = _find_free_port()
        results = mp.Manager().dict()
        world_size = min(torch.cuda.device_count(), 2)
        _nccl_spawn(
            _worker_nccl_rs_correctness,
            (world_size, port, results),
            nprocs=world_size,
        )
        for rank, result in results.items():
            assert result == "PASS", f"Rank {rank}: {result}"


@_requires_multi_gpu
class TestPipelinedAllgatherGemm:
    @pytest.mark.parametrize("num_chunks", [1, 2, 4])
    def test_correctness_vs_naive(self, num_chunks):
        import torch.multiprocessing as mp

        port = _find_free_port()
        results = mp.Manager().dict()
        world_size = min(torch.cuda.device_count(), 2)
        _nccl_spawn(
            _worker_pipeline_ag_gemm,
            (world_size, port, results, num_chunks),
            nprocs=world_size,
        )
        for rank, result in results.items():
            assert result.startswith("PASS"), f"Rank {rank}: {result}"


@_requires_multi_gpu
class TestPipelinedGemmReduceScatter:
    @pytest.mark.parametrize("num_chunks", [1, 2, 4])
    def test_correctness_vs_naive(self, num_chunks):
        import torch.multiprocessing as mp

        port = _find_free_port()
        results = mp.Manager().dict()
        world_size = min(torch.cuda.device_count(), 2)
        _nccl_spawn(
            _worker_pipeline_gemm_rs,
            (world_size, port, results, num_chunks),
            nprocs=world_size,
        )
        for rank, result in results.items():
            assert result.startswith("PASS"), f"Rank {rank}: {result}"

    @pytest.mark.parametrize("num_chunks", [1, 2, 4])
    def test_standalone_rs_correctness(self, num_chunks):
        import torch.multiprocessing as mp

        port = _find_free_port()
        results = mp.Manager().dict()
        world_size = min(torch.cuda.device_count(), 2)
        _nccl_spawn(
            _worker_pipeline_standalone_rs,
            (world_size, port, results, num_chunks),
            nprocs=world_size,
        )
        for rank, result in results.items():
            assert result.startswith("PASS"), f"Rank {rank}: {result}"


# ===================================================================
# Autograd correctness workers
# ===================================================================


def _worker_autograd_gather_grad(rank, world_size, port, results_dict, num_chunks):
    """Worker: verify pipelined_gather_for_sp produces correct gradients."""
    import torch.distributed as dist
    from atomrl.lumen.modules.comm_overlap import (
        NcclCommBackend,
        PipelinedAllgatherGemm,
        PipelinedGemmReduceScatter,
        pipelined_gather_for_sp,
    )

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    try:
        torch.manual_seed(42 + rank)
        S_local, H = 16, 32

        input_local = torch.randn(
            S_local, H, device=device, dtype=torch.float32, requires_grad=True
        )

        backend = NcclCommBackend(dist.group.WORLD)
        pipe_ag = PipelinedAllgatherGemm(num_chunks, backend)
        pipe_rs = PipelinedGemmReduceScatter(num_chunks, backend)

        gathered = pipelined_gather_for_sp(input_local, pipe_ag, pipe_rs)

        loss = gathered.sum()
        loss.backward()

        grad = input_local.grad
        if grad is None:
            results_dict[rank] = "FAIL: grad is None"
            return
        if grad.shape != (S_local, H):
            results_dict[rank] = f"FAIL: wrong grad shape {grad.shape}"
            return

        expected_grad = torch.full_like(grad, float(world_size))
        max_diff = (grad - expected_grad).abs().max().item()
        if max_diff > 1e-3:
            results_dict[rank] = f"FAIL: grad max_diff={max_diff}"
        else:
            results_dict[rank] = f"PASS: grad max_diff={max_diff}"
    finally:
        dist.destroy_process_group()


def _worker_autograd_scatter_grad(rank, world_size, port, results_dict, num_chunks):
    """Worker: verify pipelined_scatter_for_sp produces correct gradients.

    For loss = reduce_scatter(x).sum(), the gradient w.r.t. x is:
    allgather(ones) = ones of the same shape as x, because reduce_scatter
    sums contributions from all ranks, and allgather broadcasts the upstream
    gradient (all-ones) back.  So grad should be all-ones.
    """
    import torch.distributed as dist
    from atomrl.lumen.modules.comm_overlap import (
        NcclCommBackend,
        PipelinedAllgatherGemm,
        PipelinedGemmReduceScatter,
        pipelined_scatter_for_sp,
    )

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    try:
        torch.manual_seed(42 + rank)
        S_full = 16 * world_size
        out_dim = 32

        input_full = torch.randn(
            S_full, out_dim, device=device, dtype=torch.float32, requires_grad=True
        )

        backend = NcclCommBackend(dist.group.WORLD)
        pipe_rs = PipelinedGemmReduceScatter(num_chunks, backend)
        pipe_ag = PipelinedAllgatherGemm(num_chunks, backend)

        scattered = pipelined_scatter_for_sp(input_full, pipe_rs, pipe_ag)

        loss = scattered.sum()
        loss.backward()

        grad = input_full.grad
        if grad is None:
            results_dict[rank] = "FAIL: grad is None"
            return
        if grad.shape != (S_full, out_dim):
            results_dict[rank] = f"FAIL: wrong grad shape {grad.shape}"
            return

        expected_grad = torch.ones_like(grad)
        max_diff = (grad - expected_grad).abs().max().item()
        if max_diff > 1e-3:
            results_dict[rank] = f"FAIL: grad max_diff={max_diff}, expected all-ones"
        else:
            results_dict[rank] = f"PASS: grad max_diff={max_diff}"
    finally:
        dist.destroy_process_group()


# ===================================================================
# Autograd test classes
# ===================================================================


@_requires_multi_gpu
class TestPipelinedAutograd:
    @pytest.mark.parametrize("num_chunks", [1, 2, 4])
    def test_gather_backward_produces_gradient(self, num_chunks):
        import torch.multiprocessing as mp

        port = _find_free_port()
        results = mp.Manager().dict()
        world_size = min(torch.cuda.device_count(), 2)
        _nccl_spawn(
            _worker_autograd_gather_grad,
            (world_size, port, results, num_chunks),
            nprocs=world_size,
        )
        for rank, result in results.items():
            assert result.startswith("PASS"), f"Rank {rank}: {result}"

    @pytest.mark.parametrize("num_chunks", [1, 2, 4])
    def test_scatter_backward_produces_gradient(self, num_chunks):
        import torch.multiprocessing as mp

        port = _find_free_port()
        results = mp.Manager().dict()
        world_size = min(torch.cuda.device_count(), 2)
        _nccl_spawn(
            _worker_autograd_scatter_grad,
            (world_size, port, results, num_chunks),
            nprocs=world_size,
        )
        for rank, result in results.items():
            assert result.startswith("PASS"), f"Rank {rank}: {result}"


# ===================================================================
# Fused parallel linear autograd workers
# ===================================================================


def _worker_fused_column_fwd(rank, world_size, port, results_dict, num_chunks=2):
    """Verify fused column forward == naive AG + GEMM."""
    import torch.distributed as dist
    from atomrl.lumen.modules.comm_overlap import (
        NcclCommBackend,
        PipelinedAllgatherGemm,
        PipelinedGemmReduceScatter,
        fused_column_parallel_forward,
    )

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    try:
        torch.manual_seed(42)
        S_local, H, O_tp = 64, 128, 256
        input_local = torch.randn(S_local, H, device=device, dtype=torch.bfloat16)
        weight = torch.randn(O_tp, H, device=device, dtype=torch.bfloat16)

        chunks = [torch.empty_like(input_local) for _ in range(world_size)]
        dist.all_gather(chunks, input_local)
        input_full = torch.cat(chunks, dim=0)
        expected = F.linear(input_full, weight)

        backend = NcclCommBackend(dist.group.WORLD)
        pipeline_ag = PipelinedAllgatherGemm(num_chunks, backend)
        pipeline_rs = PipelinedGemmReduceScatter(num_chunks, backend)
        weight_ref = torch.nn.Parameter(weight.clone())

        actual = fused_column_parallel_forward(
            input_local.clone(),
            weight.clone(),
            None,
            pipeline_ag,
            pipeline_rs,
            weight_ref,
            False,
            False,
            None,
            False,
        )

        max_diff = (actual - expected).abs().max().item()
        rel_err = max_diff / (expected.abs().max().item() + 1e-7)
        results_dict[rank] = (
            f"{'PASS' if rel_err < 0.02 else 'FAIL'}: max_diff={max_diff:.6f}, rel_err={rel_err:.6f}"
        )
    finally:
        dist.destroy_process_group()


def _worker_fused_column_grad(rank, world_size, port, results_dict):
    """Verify fused column backward produces numerically correct dgrad."""
    import torch.distributed as dist
    from atomrl.lumen.modules.comm_overlap import (
        NcclCommBackend,
        PipelinedAllgatherGemm,
        PipelinedGemmReduceScatter,
        fused_column_parallel_forward,
    )

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    try:
        torch.manual_seed(42)
        S_local, H, O_tp = 64, 128, 256
        input_local = torch.randn(
            S_local, H, device=device, dtype=torch.bfloat16, requires_grad=True
        )
        weight = torch.randn(O_tp, H, device=device, dtype=torch.bfloat16)

        backend = NcclCommBackend(dist.group.WORLD)
        pipeline_ag = PipelinedAllgatherGemm(2, backend)
        pipeline_rs = PipelinedGemmReduceScatter(2, backend)
        weight_param = torch.nn.Parameter(weight.clone())

        output = fused_column_parallel_forward(
            input_local,
            weight_param,
            None,
            pipeline_ag,
            pipeline_rs,
            weight_param,
            False,
            False,
            None,
            False,
        )
        loss = output.sum()
        loss.backward()

        # Reference dgrad: allgather backward is reduce_scatter.
        # Forward: output = F.linear(allgather(input_local), weight)
        # dL/d(input_full) = ones(S_local*world_size, O_tp) @ weight  (since loss = sum)
        # dL/d(input_local) = reduce_scatter(dL/d(input_full))
        input_ref = input_local.detach().clone()
        chunks = [torch.empty_like(input_ref) for _ in range(world_size)]
        dist.all_gather(chunks, input_ref)
        input_full = torch.cat(chunks, dim=0)
        grad_output = torch.ones(
            input_full.shape[0], O_tp, device=device, dtype=torch.bfloat16
        )
        grad_full = grad_output @ weight
        ref_dgrad = torch.zeros(S_local, H, device=device, dtype=torch.bfloat16)
        dist.reduce_scatter_tensor(ref_dgrad, grad_full)

        checks = []
        if input_local.grad is None:
            checks.append("FAIL: no grad on input_local")
        elif input_local.grad.shape != (S_local, H):
            checks.append(f"FAIL: grad shape {input_local.grad.shape}")
        else:
            dgrad_diff = (input_local.grad - ref_dgrad).abs().max().item()
            dgrad_rel = dgrad_diff / (ref_dgrad.abs().max().item() + 1e-7)
            if dgrad_rel < 0.02:
                checks.append(f"PASS: dgrad rel_err={dgrad_rel:.6f}")
            else:
                checks.append(
                    f"FAIL: dgrad rel_err={dgrad_rel:.6f} (max_diff={dgrad_diff:.6f})"
                )

        results_dict[rank] = ", ".join(checks)
    finally:
        dist.destroy_process_group()


def _worker_fused_row_fwd_bwd(rank, world_size, port, results_dict):
    """Verify fused row forward and backward correctness."""
    import torch.distributed as dist
    from atomrl.lumen.modules.comm_overlap import (
        NcclCommBackend,
        PipelinedAllgatherGemm,
        PipelinedGemmReduceScatter,
        fused_row_parallel_forward,
    )

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    try:
        torch.manual_seed(42)
        S, H_tp, out_dim = 128, 64, 256
        input_parallel = torch.randn(
            S, H_tp, device=device, dtype=torch.bfloat16, requires_grad=True
        )
        weight = torch.randn(out_dim, H_tp, device=device, dtype=torch.bfloat16)

        backend = NcclCommBackend(dist.group.WORLD)
        pipeline_rs = PipelinedGemmReduceScatter(2, backend)
        pipeline_ag = PipelinedAllgatherGemm(2, backend)
        weight_param = torch.nn.Parameter(weight.clone())

        output = fused_row_parallel_forward(
            input_parallel,
            weight_param,
            pipeline_rs,
            pipeline_ag,
            weight_param,
            False,
            False,
            None,
        )
        loss = output.sum()
        loss.backward()

        # Reference forward: naive GEMM + reduce_scatter (non-differentiable)
        input_ref = input_parallel.detach().clone()
        ref_local = F.linear(input_ref, weight)
        ref_rs = torch.zeros(
            S // world_size, out_dim, device=device, dtype=torch.bfloat16
        )
        dist.reduce_scatter_tensor(ref_rs, ref_local)

        # Reference dgrad: reduce_scatter backward is allgather.
        # dL/d(ref_local) = allgather(ones(S/world_size, out_dim))
        # dL/d(input_ref) = dL/d(ref_local) @ weight
        grad_rs = torch.ones(
            S // world_size, out_dim, device=device, dtype=torch.bfloat16
        )
        grad_local = torch.zeros(S, out_dim, device=device, dtype=torch.bfloat16)
        dist.all_gather_into_tensor(grad_local, grad_rs)
        ref_dgrad = grad_local @ weight

        checks = []
        expected_out_rows = S // world_size
        if output.shape[0] != expected_out_rows:
            checks.append(
                f"FAIL: fwd shape {output.shape}, expected ({expected_out_rows}, ...)"
            )
        else:
            fwd_diff = (output - ref_rs).abs().max().item()
            fwd_rel = fwd_diff / (ref_rs.abs().max().item() + 1e-7)
            if fwd_rel < 0.02:
                checks.append(f"PASS: fwd rel_err={fwd_rel:.6f}")
            else:
                checks.append(f"FAIL: fwd rel_err={fwd_rel:.6f}")
        if input_parallel.grad is None:
            checks.append("FAIL: grad is None")
        elif input_parallel.grad.shape != (S, H_tp):
            checks.append(f"FAIL: grad shape {input_parallel.grad.shape}")
        else:
            dgrad_diff = (input_parallel.grad - ref_dgrad).abs().max().item()
            dgrad_rel = dgrad_diff / (ref_dgrad.abs().max().item() + 1e-7)
            if dgrad_rel < 0.02:
                checks.append(f"PASS: dgrad rel_err={dgrad_rel:.6f}")
            else:
                checks.append(f"FAIL: dgrad rel_err={dgrad_rel:.6f}")

        all_pass = all(c.startswith("PASS") for c in checks)
        results_dict[rank] = ("PASS" if all_pass else "FAIL") + ": " + ", ".join(checks)
    finally:
        dist.destroy_process_group()


def _worker_fused_column_delay_wgrad(rank, world_size, port, results_dict):
    """Verify fused column with delay_wgrad accumulates into main_grad."""
    import torch.distributed as dist
    from atomrl.lumen.modules.comm_overlap import (
        NcclCommBackend,
        PipelinedAllgatherGemm,
        PipelinedGemmReduceScatter,
        fused_column_parallel_forward,
    )
    from atomrl.lumen.modules.parallel_linear import _DeferredWgrad

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    try:
        torch.manual_seed(42)
        S_local, H, O_tp = 64, 128, 256
        input_local = torch.randn(
            S_local, H, device=device, dtype=torch.bfloat16, requires_grad=True
        )
        weight = torch.randn(O_tp, H, device=device, dtype=torch.bfloat16)

        backend = NcclCommBackend(dist.group.WORLD)
        pipeline_ag = PipelinedAllgatherGemm(2, backend)
        pipeline_rs = PipelinedGemmReduceScatter(2, backend)
        weight_param = torch.nn.Parameter(weight.clone())
        weight_param.main_grad = torch.zeros_like(weight_param)
        deferred = _DeferredWgrad()

        output = fused_column_parallel_forward(
            input_local,
            weight_param,
            None,
            pipeline_ag,
            pipeline_rs,
            weight_param,
            True,
            True,
            deferred,
            False,
        )
        output.sum().backward()

        checks = []
        if input_local.grad is not None:
            checks.append("dgrad OK")
        else:
            checks.append("dgrad FAIL")

        if deferred.has_pending:
            deferred.execute()
            if weight_param.main_grad.abs().sum() > 0:
                checks.append("wgrad OK (accumulated into main_grad)")
            else:
                checks.append("wgrad FAIL (main_grad still zero)")
        else:
            checks.append("wgrad FAIL (nothing deferred)")

        joined = " ".join(checks)
        results_dict[rank] = (
            "PASS: " + ", ".join(checks)
            if "FAIL" not in joined
            else "FAIL: " + ", ".join(checks)
        )
    finally:
        dist.destroy_process_group()


@_requires_multi_gpu
class TestFusedAutograd:
    @pytest.mark.parametrize("num_chunks", [1, 2, 4])
    def test_fused_column_fwd(self, num_chunks):
        import torch.multiprocessing as mp

        world_size = min(torch.cuda.device_count(), 4)
        results = mp.Manager().dict()
        port = _find_free_port()
        _nccl_spawn(
            _worker_fused_column_fwd,
            (world_size, port, results, num_chunks),
            nprocs=world_size,
        )
        for rank, msg in sorted(results.items()):
            assert msg.startswith("PASS"), f"Rank {rank}: {msg}"

    def test_fused_column_grad(self):
        import torch.multiprocessing as mp

        world_size = min(torch.cuda.device_count(), 4)
        results = mp.Manager().dict()
        port = _find_free_port()
        _nccl_spawn(
            _worker_fused_column_grad,
            (world_size, port, results),
            nprocs=world_size,
        )
        for rank, msg in sorted(results.items()):
            assert msg.startswith("PASS"), f"Rank {rank}: {msg}"

    def test_fused_column_delay_wgrad(self):
        import torch.multiprocessing as mp

        world_size = min(torch.cuda.device_count(), 4)
        results = mp.Manager().dict()
        port = _find_free_port()
        _nccl_spawn(
            _worker_fused_column_delay_wgrad,
            (world_size, port, results),
            nprocs=world_size,
        )
        for rank, msg in sorted(results.items()):
            assert msg.startswith("PASS"), f"Rank {rank}: {msg}"

    def test_fused_row_fwd_bwd(self):
        import torch.multiprocessing as mp

        world_size = min(torch.cuda.device_count(), 4)
        results = mp.Manager().dict()
        port = _find_free_port()
        _nccl_spawn(
            _worker_fused_row_fwd_bwd,
            (world_size, port, results),
            nprocs=world_size,
        )
        for rank, msg in sorted(results.items()):
            assert msg.startswith("PASS"), f"Rank {rank}: {msg}"


# ===================================================================
# FP8 pipeline overlap workers and tests
# ===================================================================


def _worker_fused_column_fp8_fwd(rank, world_size, port, results_dict, scaling_type):
    """Verify fused column FP8 forward produces valid output."""
    import torch.distributed as dist
    from atomrl.lumen.modules.comm_overlap import (
        NcclCommBackend,
        PipelinedAllgatherGemm,
        PipelinedGemmReduceScatter,
        fused_column_parallel_forward,
    )
    from atomrl.lumen.quantize.config import _get_float8_e4m3

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    fp8_dtype = _get_float8_e4m3()

    try:
        from atomrl.lumen.ops.quantize.linear import quantize_input

        quantize_input(
            torch.randn(2, 4, device=device, dtype=torch.bfloat16),
            scaling_type,
            fp8_dtype,
            128,
        )
    except RuntimeError as e:
        results_dict[rank] = f"SKIP: FP8 backend not available ({e})"
        return

    try:
        torch.manual_seed(42)
        S_local, H, O_tp = 64, 128, 256
        input_local = torch.randn(S_local, H, device=device, dtype=torch.bfloat16)
        weight = torch.randn(O_tp, H, device=device, dtype=torch.bfloat16)

        backend = NcclCommBackend(dist.group.WORLD)
        pipeline_ag = PipelinedAllgatherGemm(2, backend)
        pipeline_rs = PipelinedGemmReduceScatter(2, backend)
        weight_ref = torch.nn.Parameter(weight.clone())

        actual = fused_column_parallel_forward(
            input_local.clone(),
            weight.clone(),
            None,
            pipeline_ag,
            pipeline_rs,
            weight_ref,
            False,
            False,
            None,
            False,
            scaling_manager=None,
            scaling_type=scaling_type,
            fp8_dtype=fp8_dtype,
            block_size=128,
        )

        checks = []
        if actual.shape != (S_local * world_size, O_tp):
            checks.append(f"FAIL: shape {actual.shape}")
        elif torch.isnan(actual).any():
            checks.append("FAIL: NaN in output")
        elif torch.isinf(actual).any():
            checks.append("FAIL: Inf in output")
        else:
            checks.append("PASS: output valid")

        results_dict[rank] = checks[0]
    finally:
        dist.destroy_process_group()


def _worker_fused_column_fp8_grad(rank, world_size, port, results_dict, scaling_type):
    """Verify fused column FP8 backward produces non-None gradients."""
    import torch.distributed as dist
    from atomrl.lumen.modules.comm_overlap import (
        NcclCommBackend,
        PipelinedAllgatherGemm,
        PipelinedGemmReduceScatter,
        fused_column_parallel_forward,
    )
    from atomrl.lumen.quantize.config import _get_float8_e4m3

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    fp8_dtype = _get_float8_e4m3()

    try:
        from atomrl.lumen.ops.quantize.linear import quantize_input

        quantize_input(
            torch.randn(2, 4, device=device, dtype=torch.bfloat16),
            scaling_type,
            fp8_dtype,
            128,
        )
    except RuntimeError as e:
        results_dict[rank] = f"SKIP: FP8 backend not available ({e})"
        return

    try:
        torch.manual_seed(42)
        S_local, H, O_tp = 64, 128, 256
        input_local = torch.randn(
            S_local,
            H,
            device=device,
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        weight = torch.randn(O_tp, H, device=device, dtype=torch.bfloat16)

        backend = NcclCommBackend(dist.group.WORLD)
        pipeline_ag = PipelinedAllgatherGemm(2, backend)
        pipeline_rs = PipelinedGemmReduceScatter(2, backend)
        weight_param = torch.nn.Parameter(weight.clone())

        output = fused_column_parallel_forward(
            input_local,
            weight_param,
            None,
            pipeline_ag,
            pipeline_rs,
            weight_param,
            False,
            False,
            None,
            False,
            scaling_manager=None,
            scaling_type=scaling_type,
            fp8_dtype=fp8_dtype,
            block_size=128,
        )
        output.sum().backward()

        checks = []
        if input_local.grad is None:
            checks.append("FAIL: no input grad")
        elif torch.isnan(input_local.grad).any():
            checks.append("FAIL: NaN in input grad")
        else:
            checks.append("PASS: input grad OK")

        results_dict[rank] = checks[0]
    finally:
        dist.destroy_process_group()


def _worker_fused_row_fp8_fwd_bwd(rank, world_size, port, results_dict, scaling_type):
    """Verify fused row FP8 forward + backward."""
    import torch.distributed as dist
    from atomrl.lumen.modules.comm_overlap import (
        NcclCommBackend,
        PipelinedAllgatherGemm,
        PipelinedGemmReduceScatter,
        fused_row_parallel_forward,
    )
    from atomrl.lumen.quantize.config import _get_float8_e4m3

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    fp8_dtype = _get_float8_e4m3()

    try:
        from atomrl.lumen.ops.quantize.linear import quantize_input

        quantize_input(
            torch.randn(2, 4, device=device, dtype=torch.bfloat16),
            scaling_type,
            fp8_dtype,
            128,
        )
    except RuntimeError as e:
        results_dict[rank] = f"SKIP: FP8 backend not available ({e})"
        return

    try:
        torch.manual_seed(42)
        S, H_tp, out_dim = 128, 64, 256
        input_parallel = torch.randn(
            S,
            H_tp,
            device=device,
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        weight = torch.randn(out_dim, H_tp, device=device, dtype=torch.bfloat16)

        backend = NcclCommBackend(dist.group.WORLD)
        pipeline_rs = PipelinedGemmReduceScatter(2, backend)
        pipeline_ag = PipelinedAllgatherGemm(2, backend)
        weight_param = torch.nn.Parameter(weight.clone())

        output = fused_row_parallel_forward(
            input_parallel,
            weight_param,
            pipeline_rs,
            pipeline_ag,
            weight_param,
            False,
            False,
            None,
            scaling_manager=None,
            scaling_type=scaling_type,
            fp8_dtype=fp8_dtype,
            block_size=128,
        )
        output.sum().backward()

        checks = []
        expected_rows = S // world_size
        if output.shape[0] != expected_rows:
            checks.append(f"FAIL: fwd shape {output.shape}")
        elif torch.isnan(output).any():
            checks.append("FAIL: NaN in output")
        elif input_parallel.grad is None:
            checks.append("FAIL: no input grad")
        elif torch.isnan(input_parallel.grad).any():
            checks.append("FAIL: NaN in input grad")
        else:
            checks.append("PASS: fwd + bwd OK")

        results_dict[rank] = checks[0]
    finally:
        dist.destroy_process_group()


def _worker_fused_column_bf16_compat(rank, world_size, port, results_dict):
    """Verify new FP8-extended API is backward-compatible with BF16 (scaling_type='none')."""
    import torch.distributed as dist
    from atomrl.lumen.modules.comm_overlap import (
        NcclCommBackend,
        PipelinedAllgatherGemm,
        PipelinedGemmReduceScatter,
        fused_column_parallel_forward,
    )
    from atomrl.lumen.quantize.config import _get_float8_e4m3

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    try:
        torch.manual_seed(42)
        S_local, H, O_tp = 64, 128, 256
        input_local = torch.randn(S_local, H, device=device, dtype=torch.bfloat16)
        weight = torch.randn(O_tp, H, device=device, dtype=torch.bfloat16)

        chunks = [torch.empty_like(input_local) for _ in range(world_size)]
        dist.all_gather(chunks, input_local)
        input_full = torch.cat(chunks, dim=0)
        expected = F.linear(input_full, weight)

        backend = NcclCommBackend(dist.group.WORLD)
        pipeline_ag = PipelinedAllgatherGemm(2, backend)
        pipeline_rs = PipelinedGemmReduceScatter(2, backend)
        weight_ref = torch.nn.Parameter(weight.clone())

        actual = fused_column_parallel_forward(
            input_local.clone(),
            weight.clone(),
            None,
            pipeline_ag,
            pipeline_rs,
            weight_ref,
            False,
            False,
            None,
            False,
            scaling_manager=None,
            scaling_type="none",
            fp8_dtype=_get_float8_e4m3(),
            block_size=128,
        )

        max_diff = (actual - expected).abs().max().item()
        rel_err = max_diff / (expected.abs().max().item() + 1e-7)
        results_dict[rank] = (
            f"{'PASS' if rel_err < 0.02 else 'FAIL'}: rel_err={rel_err:.6f}"
        )
    finally:
        dist.destroy_process_group()


@_requires_multi_gpu
class TestFusedFP8Pipeline:
    """FP8 pipeline overlap correctness tests."""

    @pytest.mark.parametrize(
        "scaling_type", ["dynamic", "delayed", "per_token", "blockwise"]
    )
    def test_fused_column_fp8_fwd(self, scaling_type):
        import torch.multiprocessing as mp

        world_size = min(torch.cuda.device_count(), 2)
        results = mp.Manager().dict()
        port = _find_free_port()
        _nccl_spawn(
            _worker_fused_column_fp8_fwd,
            (world_size, port, results, scaling_type),
            nprocs=world_size,
        )
        for rank, msg in sorted(results.items()):
            if msg.startswith("SKIP"):
                pytest.skip(msg)
            assert msg.startswith("PASS"), f"Rank {rank}: {msg}"

    @pytest.mark.parametrize(
        "scaling_type", ["dynamic", "delayed", "per_token", "blockwise"]
    )
    def test_fused_column_fp8_grad(self, scaling_type):
        import torch.multiprocessing as mp

        world_size = min(torch.cuda.device_count(), 2)
        results = mp.Manager().dict()
        port = _find_free_port()
        _nccl_spawn(
            _worker_fused_column_fp8_grad,
            (world_size, port, results, scaling_type),
            nprocs=world_size,
        )
        for rank, msg in sorted(results.items()):
            if msg.startswith("SKIP"):
                pytest.skip(msg)
            assert msg.startswith("PASS"), f"Rank {rank}: {msg}"

    @pytest.mark.parametrize(
        "scaling_type", ["dynamic", "delayed", "per_token", "blockwise"]
    )
    def test_fused_row_fp8_fwd_bwd(self, scaling_type):
        import torch.multiprocessing as mp

        world_size = min(torch.cuda.device_count(), 2)
        results = mp.Manager().dict()
        port = _find_free_port()
        _nccl_spawn(
            _worker_fused_row_fp8_fwd_bwd,
            (world_size, port, results, scaling_type),
            nprocs=world_size,
        )
        for rank, msg in sorted(results.items()):
            if msg.startswith("SKIP"):
                pytest.skip(msg)
            assert msg.startswith("PASS"), f"Rank {rank}: {msg}"

    def test_bf16_backward_compat(self):
        """Ensure the extended API with FP8 params works unchanged for BF16."""
        import torch.multiprocessing as mp

        world_size = min(torch.cuda.device_count(), 2)
        results = mp.Manager().dict()
        port = _find_free_port()
        _nccl_spawn(
            _worker_fused_column_bf16_compat,
            (world_size, port, results),
            nprocs=world_size,
        )
        for rank, msg in sorted(results.items()):
            assert msg.startswith("PASS"), f"Rank {rank}: {msg}"
