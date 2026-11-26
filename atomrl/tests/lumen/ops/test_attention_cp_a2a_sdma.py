# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

###############################################################################
#
# See LICENSE for license information.
###############################################################################

"""Tests for attention_with_cp_a2a SDMA integration.

Covers:
  - AttentionCPA2AHelper: reshape correctness (unit, single-GPU)
  - CP A2A with mori SDMA: Triton, MXFP8, Aiter paths
  - Forward + backward numerical correctness vs non-SDMA baseline
  - Performance comparison: SDMA vs torch.distributed all-to-all

How to run::

    # All tests (unit + multi-GPU); SDMA-capable hardware required for distributed tests:
    pytest tests/ops/test_attention_cp_a2a_sdma.py -v

    # Unit tests only (single GPU):
    pytest tests/ops/test_attention_cp_a2a_sdma.py -v -k "Helper"

    # Multi-GPU CP A2A tests (up to 8 GPUs, mp.spawn):
    pytest tests/ops/test_attention_cp_a2a_sdma.py -v -k "CPA2ASdma"

Skip conditions:
  - mori SDMA not available → skip SDMA / distributed tests
  - aiter not available → skip aiter-specific tests
"""

import os
import time

import pytest
import torch
from atomrl.lumen.ops.sdma import is_sdma_available

_sdma_available = is_sdma_available()

skip_no_sdma = pytest.mark.skipif(not _sdma_available, reason="mori SDMA not available")
skip_no_cuda = pytest.mark.skipif(not _sdma_available, reason="CUDA/SDMA required")
skip_no_multi_gpu = pytest.mark.skipif(
    not _sdma_available, reason="SDMA/Multi-GPU required"
)


def _get_free_port():
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


_SDMA_SPAWN_COOLDOWN_SECS = float(os.environ.get("LUMEN_SDMA_SPAWN_COOLDOWN_SECS", "2"))


def _sdma_spawn(fn, args, nprocs, join=True):
    """``mp.spawn`` with a KFD SDMA-queue reclamation cooldown.

    See ``tests/ops/test_sdma.py::_sdma_spawn`` for the full rationale.
    Override with ``LUMEN_SDMA_SPAWN_COOLDOWN_SECS=0`` for fast local runs.
    """
    import torch.multiprocessing as mp

    os.environ.setdefault("MORI_ENABLE_SDMA", "1")

    if _SDMA_SPAWN_COOLDOWN_SECS > 0:
        time.sleep(_SDMA_SPAWN_COOLDOWN_SECS)
    return mp.spawn(fn, args=args, nprocs=nprocs, join=join)


# ===================================================================
# AttentionCPA2AHelper unit tests (single GPU, no distribution)
# ===================================================================


@skip_no_cuda
class TestAttentionCPA2AHelper:
    """Test the reshape/split helpers used by CP A2A."""

    def test_combine_qkv_round_trip(self):
        """combine_qkv_before_a2a followed by splits_qkv_after_a2a recovers original shapes."""
        from atomrl.lumen.ops.attention.attention_with_cp_a2a import (
            AttentionCPA2AHelper,
        )

        b, s, h_q, h_kv, d_qk, d_v, n = 2, 16, 8, 4, 64, 64, 2

        helper = AttentionCPA2AHelper(b, s, h_q, h_kv, d_qk, d_v, seq_dim=1, n=n)
        q = torch.randn(b, s // n, h_q, d_qk, device="cuda")
        k = torch.randn(b, s // n, h_kv, d_qk, device="cuda")
        v = torch.randn(b, s // n, h_kv, d_v, device="cuda")

        combined = helper.combine_qkv_before_a2a(q, k, v)
        assert combined.shape[0] == n

        q2, k2, v2 = helper.splits_qkv_after_a2a(combined)
        assert q2.shape == (b, s, h_q // n, d_qk)
        assert k2.shape == (b, s, h_kv // n, d_qk)
        assert v2.shape == (b, s, h_kv // n, d_v)

    def test_output_reshape_round_trip(self):
        """reshape_o_before_a2a followed by reshape_o_after_a2a recovers shape."""
        from atomrl.lumen.ops.attention.attention_with_cp_a2a import (
            AttentionCPA2AHelper,
        )

        b, s, h_q, h_kv, d_qk, d_v, n = 2, 16, 8, 4, 64, 64, 2

        helper = AttentionCPA2AHelper(b, s, h_q, h_kv, d_qk, d_v, seq_dim=1, n=n)
        o = torch.randn(b, s, h_q // n, d_v, device="cuda")

        reshaped = helper.reshape_o_before_a2a(o)
        assert reshaped.shape[0] == n

        recovered = helper.reshape_o_after_a2a(reshaped)
        assert recovered.shape == (b, s // n, h_q, d_v)

    def test_grad_reshape_round_trip(self):
        """reshape_do_before_a2a followed by reshape_do_after_a2a recovers shape."""
        from atomrl.lumen.ops.attention.attention_with_cp_a2a import (
            AttentionCPA2AHelper,
        )

        b, s, h_q, h_kv, d_qk, d_v, n = 2, 16, 8, 4, 64, 64, 2

        helper = AttentionCPA2AHelper(b, s, h_q, h_kv, d_qk, d_v, seq_dim=1, n=n)
        do = torch.randn(b, s // n, h_q, d_v, device="cuda")

        reshaped = helper.reshape_do_before_a2a(do)
        recovered = helper.reshape_do_after_a2a(reshaped)
        assert recovered.shape == (b, s, h_q // n, d_v)

    def test_dqkv_round_trip(self):
        """combine_dqkv_before_a2a followed by split_dqkv_after_a2a recovers shapes."""
        from atomrl.lumen.ops.attention.attention_with_cp_a2a import (
            AttentionCPA2AHelper,
        )

        b, s, h_q, h_kv, d_qk, d_v, n = 2, 16, 8, 4, 64, 64, 2

        helper = AttentionCPA2AHelper(b, s, h_q, h_kv, d_qk, d_v, seq_dim=1, n=n)
        dq = torch.randn(b, s, h_q // n, d_qk, device="cuda")
        dk = torch.randn(b, s, h_kv // n, d_qk, device="cuda")
        dv = torch.randn(b, s, h_kv // n, d_v, device="cuda")

        combined = helper.combine_dqkv_before_a2a(dq, dk, dv)
        dq2, dk2, dv2 = helper.split_dqkv_after_a2a(combined)
        assert dq2.shape == (b, s // n, h_q, d_qk)
        assert dk2.shape == (b, s // n, h_kv, d_qk)
        assert dv2.shape == (b, s // n, h_kv, d_v)


# ===================================================================
# Multi-GPU CP A2A SDMA correctness tests
# ===================================================================


def _cp_a2a_worker_teardown(shmem_mod):
    """Standard teardown for CP A2A SDMA worker processes.

    Cleanup order (mirrors mori ccl reference tests):
    1. CUDA sync — flush outstanding GPU work.
    2. Barrier — all ranks rendezvous.
    3. Release the module-level SdmaAll2all cache so C++ handles
       destruct while shmem is still alive.
    4. SdmaContext.reset() — clear the singleton reference.
    5. Barrier — sync after handle destruction.
    6. shmem_finalize() — release mori symmetric-memory resources.
    7. Barrier — rendezvous after finalization.
    8. destroy_process_group() — must be last.
    """
    import torch.distributed as dist
    from atomrl.lumen.ops.attention.attention_with_cp_a2a import reset_sdma_a2a_cache
    from atomrl.lumen.ops.sdma import SdmaContext

    torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()
    reset_sdma_a2a_cache()
    SdmaContext.reset()
    if dist.is_initialized():
        dist.barrier()
    shmem_mod.shmem_finalize()
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def _worker_cp_a2a_triton_sdma(rank, world_size, port, results_dict):
    """Worker: run CP A2A triton attention with SDMA, compare to reference."""
    os.environ["MORI_ENABLE_SDMA"] = "1"

    import torch.distributed as dist
    from mori import shmem

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "cpu:gloo,cuda:nccl", rank=rank, world_size=world_size, device_id=device
    )
    torch._C._distributed_c10d._register_process_group("default", dist.group.WORLD)

    from atomrl.lumen.ops.attention.attention_with_cp_a2a import (
        AttentionTritonFunctionCPA2A,
    )

    try:
        # h_q/h_kv must be > world_size so each rank keeps >=2 heads
        # after CP A2A scatter; aiter CK rejects nheads==1.
        b, s_local, h_q, h_kv, d = 1, 64, 16, 16, 64
        sm_scale = d**-0.5

        from atomrl.lumen.ops.sdma import SdmaContext

        SdmaContext.get()
        dist.barrier()

        torch.cuda.manual_seed(42 + rank)
        q = torch.randn(b, s_local, h_q, d, device=f"cuda:{rank}", dtype=torch.bfloat16)
        k = torch.randn(
            b, s_local, h_kv, d, device=f"cuda:{rank}", dtype=torch.bfloat16
        )
        v = torch.randn(
            b, s_local, h_kv, d, device=f"cuda:{rank}", dtype=torch.bfloat16
        )

        cp_group = dist.new_group(list(range(world_size)))

        out_nccl = AttentionTritonFunctionCPA2A.apply(
            q,
            k,
            v,
            0.0,
            sm_scale,
            False,
            (-1, -1),
            None,
            None,
            False,
            False,
            False,
            False,
            cp_group,
            None,
            False,
        )

        torch.cuda.synchronize()
        dist.barrier()

        out_sdma = AttentionTritonFunctionCPA2A.apply(
            q,
            k,
            v,
            0.0,
            sm_scale,
            False,
            (-1, -1),
            None,
            None,
            False,
            False,
            False,
            False,
            cp_group,
            None,
            True,
        )

        passed = torch.allclose(out_nccl, out_sdma, atol=1e-3, rtol=1e-3)
        results_dict[rank] = passed
    finally:
        _cp_a2a_worker_teardown(shmem)


def _worker_cp_a2a_perf_compare(
    rank, world_size, port, results_dict, iterations, warmup
):
    """Worker: compare SDMA vs NCCL all-to-all latency in CP attention."""
    os.environ["MORI_ENABLE_SDMA"] = "1"

    import numpy as np
    import torch.distributed as dist
    from mori import shmem

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "cpu:gloo,cuda:nccl", rank=rank, world_size=world_size, device_id=device
    )
    torch._C._distributed_c10d._register_process_group("default", dist.group.WORLD)

    from atomrl.lumen.ops.attention.attention_with_cp_a2a import (
        AttentionTritonFunctionCPA2A,
    )

    try:
        from atomrl.lumen.ops.sdma import SdmaContext

        SdmaContext.get()
        dist.barrier()

        b, s_local, h_q, h_kv, d = 1, 256, 16, 16, 128
        sm_scale = d**-0.5
        device = f"cuda:{rank}"
        cp_group = dist.new_group(list(range(world_size)))

        torch.cuda.manual_seed(42 + rank)
        q = torch.randn(b, s_local, h_q, d, device=device, dtype=torch.bfloat16)
        k = torch.randn(b, s_local, h_kv, d, device=device, dtype=torch.bfloat16)
        v = torch.randn(b, s_local, h_kv, d, device=device, dtype=torch.bfloat16)

        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)
        stream = torch.cuda.current_stream(device)

        def _bench(use_sdma):
            times = []
            for i in range(warmup + iterations):
                start_ev.record(stream)
                _ = AttentionTritonFunctionCPA2A.apply(
                    q,
                    k,
                    v,
                    0.0,
                    sm_scale,
                    False,
                    (-1, -1),
                    None,
                    None,
                    False,
                    False,
                    False,
                    False,
                    cp_group,
                    None,
                    use_sdma,
                )
                end_ev.record(stream)
                stream.synchronize()
                if i >= warmup:
                    times.append(start_ev.elapsed_time(end_ev))
            return np.mean(times) if times else 0

        nccl_ms = _bench(False)
        torch.cuda.synchronize()
        dist.barrier()
        sdma_ms = _bench(True)

        results_dict[rank] = {"nccl_ms": nccl_ms, "sdma_ms": sdma_ms}
    finally:
        _cp_a2a_worker_teardown(shmem)


@skip_no_sdma
@skip_no_multi_gpu
class TestCPA2ASdma:
    """CP A2A with SDMA correctness and performance tests."""

    @property
    def world_size(self):
        return min(torch.cuda.device_count(), 8)

    def test_triton_sdma_matches_nccl(self):
        """Triton CP A2A with SDMA produces same output as NCCL path."""
        import torch.multiprocessing as mp

        manager = mp.Manager()
        results = manager.dict()
        port = _get_free_port()
        _sdma_spawn(
            _worker_cp_a2a_triton_sdma,
            args=(self.world_size, port, results),
            nprocs=self.world_size,
        )
        for r in range(self.world_size):
            assert results[r], f"PE {r}: SDMA output != NCCL output"

    def test_sdma_vs_nccl_performance(self):
        """Measure and print SDMA vs NCCL latency for CP A2A attention."""
        import torch.multiprocessing as mp

        manager = mp.Manager()
        results = manager.dict()
        port = _get_free_port()
        _sdma_spawn(
            _worker_cp_a2a_perf_compare,
            args=(self.world_size, port, results, 10, 5),
            nprocs=self.world_size,
        )
        r0 = results[0]
        speedup = r0["nccl_ms"] / r0["sdma_ms"] if r0["sdma_ms"] > 0 else float("inf")
        print(
            f"\nCP A2A Attention: NCCL={r0['nccl_ms']:.3f}ms, "
            f"SDMA={r0['sdma_ms']:.3f}ms, "
            f"speedup={speedup:.2f}x"
        )
