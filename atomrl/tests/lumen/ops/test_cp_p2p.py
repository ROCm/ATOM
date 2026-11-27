# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""
Tests for ring-based context parallelism (CP P2P).

Covers:
  - Online softmax accumulation: shape, identity, commutativity, associativity,
    numerical correctness vs full-attention reference
  - Ring send/recv KV (existence check)
  - AttentionCPP2PFunction autograd class (existence check)
  - attention_cp_p2p API (existence check)
  - Distributed 2-GPU: output shape, forward correctness vs single-GPU reference (SNR),
    backward gradient correctness vs single-GPU reference (SNR)

How to run::

    # All tests (unit + distributed); distributed tests require >= 2 GPUs:
    pytest tests/ops/test_cp_p2p.py -v

    # Unit tests only (single GPU):
    pytest tests/ops/test_cp_p2p.py -v -k "not Distributed"

    # Distributed correctness tests (spawns 2 GPU workers via mp.spawn):
    pytest tests/ops/test_cp_p2p.py -v -k "Distributed"
"""

import os
import tempfile

import pytest
import torch

# =========================================================================
# Unit tests for online softmax merge
# =========================================================================


class TestOnlineSoftmaxUpdate:
    """Test the online softmax accumulation utility."""

    def test_two_equal_chunks(self):
        from atomrl.lumen.ops.attention.attention_with_cp_p2p import (
            _online_softmax_update,
        )

        B, S, H, D = 2, 8, 4, 64
        out1 = torch.randn(B, S, H, D)
        out2 = torch.randn(B, S, H, D)
        lse1 = torch.randn(B, S, H)
        lse2 = torch.randn(B, S, H)

        merged_out, merged_lse = _online_softmax_update(out1, lse1, out2, lse2)
        assert merged_out.shape == (B, S, H, D)
        assert merged_lse.shape == (B, S, H)
        assert torch.isfinite(merged_out).all()
        assert torch.isfinite(merged_lse).all()

    def test_identity_with_neg_inf_lse(self):
        """If one chunk has -inf LSE, the other dominates."""
        from atomrl.lumen.ops.attention.attention_with_cp_p2p import (
            _online_softmax_update,
        )

        B, S, H, D = 1, 4, 2, 32
        out1 = torch.randn(B, S, H, D)
        lse1 = torch.randn(B, S, H)
        out2 = torch.randn(B, S, H, D)
        lse2 = torch.full((B, S, H), float("-inf"))

        merged_out, _merged_lse = _online_softmax_update(out1, lse1, out2, lse2)
        torch.testing.assert_close(merged_out, out1, atol=1e-5, rtol=1e-5)

    def test_commutativity(self):
        from atomrl.lumen.ops.attention.attention_with_cp_p2p import (
            _online_softmax_update,
        )

        B, S, H, D = 1, 4, 2, 32
        out1 = torch.randn(B, S, H, D)
        lse1 = torch.randn(B, S, H)
        out2 = torch.randn(B, S, H, D)
        lse2 = torch.randn(B, S, H)

        r1_out, r1_lse = _online_softmax_update(out1, lse1, out2, lse2)
        r2_out, r2_lse = _online_softmax_update(out2, lse2, out1, lse1)
        torch.testing.assert_close(r1_out, r2_out, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(r1_lse, r2_lse, atol=1e-5, rtol=1e-5)

    def test_three_way_associativity(self):
        """Merging (A,B) then C should equal merging A then (B,C)."""
        from atomrl.lumen.ops.attention.attention_with_cp_p2p import (
            _online_softmax_update,
        )

        B, S, H, D = 1, 4, 2, 32
        o1, l1 = torch.randn(B, S, H, D), torch.randn(B, S, H)
        o2, l2 = torch.randn(B, S, H, D), torch.randn(B, S, H)
        o3, l3 = torch.randn(B, S, H, D), torch.randn(B, S, H)

        ab_out, ab_lse = _online_softmax_update(o1, l1, o2, l2)
        abc_out, abc_lse = _online_softmax_update(ab_out, ab_lse, o3, l3)

        bc_out, bc_lse = _online_softmax_update(o2, l2, o3, l3)
        abc_out2, abc_lse2 = _online_softmax_update(o1, l1, bc_out, bc_lse)

        torch.testing.assert_close(abc_out, abc_out2, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(abc_lse, abc_lse2, atol=1e-4, rtol=1e-4)

    def test_numerical_correctness_vs_full_attention(self):
        """Merge partial attention outputs and compare against full attention reference.

        Splits K,V into two chunks, computes partial attention on each (with LSE),
        merges via _online_softmax_update, and checks against a full-sequence
        reference computed in one pass.
        """
        from atomrl.lumen.ops.attention.attention_with_cp_p2p import (
            _online_softmax_update,
        )

        B, S, H, D = 1, 16, 2, 32
        torch.manual_seed(123)
        q = torch.randn(B, S, H, D)
        k = torch.randn(B, S, H, D)
        v = torch.randn(B, S, H, D)
        sm_scale = D**-0.5

        q_t = q.float().transpose(1, 2)
        k_t = k.float().transpose(1, 2)
        v_t = v.float().transpose(1, 2)
        scores_full = torch.matmul(q_t, k_t.transpose(-2, -1)) * sm_scale
        ref_out = torch.matmul(torch.softmax(scores_full, dim=-1), v_t).transpose(1, 2)

        S_half = S // 2
        k1, k2 = k[:, :S_half], k[:, S_half:]
        v1, v2 = v[:, :S_half], v[:, S_half:]

        k1_t = k1.float().transpose(1, 2)
        v1_t = v1.float().transpose(1, 2)
        scores1 = torch.matmul(q_t, k1_t.transpose(-2, -1)) * sm_scale
        lse1 = torch.logsumexp(scores1, dim=-1).transpose(1, 2)
        out1 = torch.matmul(torch.softmax(scores1, dim=-1), v1_t).transpose(1, 2)

        k2_t = k2.float().transpose(1, 2)
        v2_t = v2.float().transpose(1, 2)
        scores2 = torch.matmul(q_t, k2_t.transpose(-2, -1)) * sm_scale
        lse2 = torch.logsumexp(scores2, dim=-1).transpose(1, 2)
        out2 = torch.matmul(torch.softmax(scores2, dim=-1), v2_t).transpose(1, 2)

        merged_out, _ = _online_softmax_update(out1, lse1, out2, lse2)
        torch.testing.assert_close(merged_out, ref_out, atol=1e-5, rtol=1e-5)


# =========================================================================
# API existence checks
# =========================================================================


class TestRingSendRecvKV:
    """Test P2P send/recv (mocked)."""

    def test_function_exists(self):
        from atomrl.lumen.ops.attention.attention_with_cp_p2p import _ring_send_recv_kv

        assert callable(_ring_send_recv_kv)


class TestAttentionCPP2P:
    """Test ring attention autograd function."""

    def test_class_exists(self):
        from atomrl.lumen.ops.attention.attention_with_cp_p2p import (
            AttentionCPP2PFunction,
        )

        assert hasattr(AttentionCPP2PFunction, "forward")
        assert hasattr(AttentionCPP2PFunction, "backward")

    def test_api_function_exists(self):
        from atomrl.lumen.ops.attention.attention_with_cp_p2p import attention_cp_p2p

        assert callable(attention_cp_p2p)


# =========================================================================
# Distributed ring attention test (2 GPUs)
# =========================================================================


def _has_multi_gpu() -> bool:
    try:
        return torch.cuda.is_available() and torch.cuda.device_count() >= 2
    except (AssertionError, RuntimeError):
        return False


_MULTI_GPU = pytest.mark.skipif(
    not _has_multi_gpu(),
    reason="Need >= 2 GPUs for CP P2P distributed test",
)


def _ck_probe_worker(rank, port):
    """Module-level worker so ``mp.spawn`` can pickle it."""
    import torch.distributed as dist

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "cpu:gloo,cuda:nccl",
        rank=rank,
        world_size=1,
        device_id=torch.device("cuda", rank),
    )
    try:
        from atomrl.lumen.ops.attention.attention import attention

        q = torch.randn(1, 32, 1, 64, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(1, 32, 1, 64, dtype=torch.bfloat16, device="cuda")
        v = torch.randn(1, 32, 1, 64, dtype=torch.bfloat16, device="cuda")
        attention(q, k, v, causal=True, return_lse=True)
        torch.cuda.synchronize()
    finally:
        dist.destroy_process_group()


def _ck_spawn_probe() -> bool:
    """Return True if CK attention survives inside a single-process mp.spawn.

    Runs a minimal forward pass in a spawned subprocess.  If the CK kernel
    SIGSEGV's (a known issue on some builds), the ProcessExitedException is
    caught and the probe returns False -- callers should skip the test.
    The result is cached so the probe runs at most once per session.
    """
    if hasattr(_ck_spawn_probe, "_cached"):
        return _ck_spawn_probe._cached

    import socket

    import torch.multiprocessing as mp
    from torch.multiprocessing.spawn import (
        ProcessExitedException,
        ProcessRaisedException,
    )

    _warmup_ck_kernels(1, 32, 1, 64, dtype=torch.bfloat16)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        port = sock.getsockname()[1]

    try:
        mp.spawn(_ck_probe_worker, args=(port,), nprocs=1, join=True)
        _ck_spawn_probe._cached = True
    except (ProcessExitedException, ProcessRaisedException):
        _ck_spawn_probe._cached = False
    return _ck_spawn_probe._cached


def _warmup_ck_kernels(B, S_local, H, D, dtype, device="cuda:0", include_bwd=False):
    """Pre-JIT CK attention kernels so mp.spawn children hit the cache.

    When *include_bwd* is True the backward kernel is also warmed up so that
    ``AttentionCPP2PFunction.backward`` can safely run inside spawned workers
    without triggering JIT compilation (which would SIGSEGV).
    """
    from atomrl.lumen.ops.attention.attention import attention

    q = torch.randn(
        B, S_local, H, D, dtype=dtype, device=device, requires_grad=include_bwd
    )
    k = torch.randn(
        B, S_local, H, D, dtype=dtype, device=device, requires_grad=include_bwd
    )
    v = torch.randn(
        B, S_local, H, D, dtype=dtype, device=device, requires_grad=include_bwd
    )
    out, _lse = attention(q, k, v, causal=True, return_lse=True)
    if include_bwd:
        out.sum().backward()
    torch.cuda.synchronize()
    del q, k, v, out, _lse


def _cp_p2p_worker(
    rank,
    world_size,
    result_dir,
    global_q,
    global_k,
    global_v,
    sm_scale,
    port,
    requires_grad=False,
):
    """Worker function for 2-GPU CP P2P test.

    CK kernels are pre-compiled by ``_warmup_ck_kernels`` before spawn,
    so ``backend_type="auto"`` safely picks the CK csrc path without
    JIT races across subprocesses.
    """
    import torch.distributed as dist
    from atomrl.lumen.ops.attention.attention_with_cp_p2p import attention_cp_p2p

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    device = torch.device("cuda", rank)
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "cpu:gloo,cuda:nccl", rank=rank, world_size=world_size, device_id=device
    )
    try:
        cp_group = dist.new_group(ranks=list(range(world_size)))

        _B, S_total, _H, _D = global_q.shape
        S_local = S_total // world_size

        q_local = (
            global_q[:, rank * S_local : (rank + 1) * S_local].to(device).contiguous()
        )
        k_local = (
            global_k[:, rank * S_local : (rank + 1) * S_local].to(device).contiguous()
        )
        v_local = (
            global_v[:, rank * S_local : (rank + 1) * S_local].to(device).contiguous()
        )

        if requires_grad:
            q_local = q_local.requires_grad_(True)
            k_local = k_local.requires_grad_(True)
            v_local = v_local.requires_grad_(True)

        dist.barrier()

        def attn_fn(q_chunk, k_chunk, v_chunk, causal, softmax_scale):
            from atomrl.lumen.ops.attention.attention import attention

            out, lse = attention(
                q_chunk,
                k_chunk,
                v_chunk,
                softmax_scale=softmax_scale,
                causal=causal,
                return_lse=True,
                backend_type="auto",
            )
            return out, lse.transpose(1, 2).contiguous()

        out_local = attention_cp_p2p(
            q_local,
            k_local,
            v_local,
            cp_group=cp_group,
            cp_size=world_size,
            cp_rank=rank,
            attn_fn=attn_fn,
            causal=True,
            softmax_scale=sm_scale,
        )

        result = {"out": out_local.detach().cpu()}

        if requires_grad:
            grad_out = torch.randn_like(out_local)
            out_local.backward(grad_out)
            result["grad_q"] = q_local.grad.cpu()
            result["grad_k"] = k_local.grad.cpu()
            result["grad_v"] = v_local.grad.cpu()
            result["grad_out"] = grad_out.cpu()

        torch.save(result, os.path.join(result_dir, f"rank{rank}.pt"))
    finally:
        torch.cuda.synchronize()
        dist.barrier()
        dist.destroy_process_group()


def _attention_ref_causal(q, k, v, sm_scale):
    """Pure PyTorch causal attention reference (BSHD layout, float32 internally)."""
    _B, SQ, _H, _D = q.shape
    _, SK, _, _DV = v.shape
    q_t = q.float().transpose(1, 2)
    k_t = k.float().transpose(1, 2)
    v_t = v.float().transpose(1, 2)
    attn = torch.matmul(q_t, k_t.transpose(-2, -1)) * sm_scale
    row_idx = torch.arange(SQ, device=q.device).unsqueeze(1)
    col_idx = torch.arange(SK, device=q.device).unsqueeze(0)
    col_offset = SQ - SK
    mask = row_idx >= (col_offset + col_idx)
    attn = attn.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float("-inf"))
    attn = torch.softmax(attn, dim=-1)
    out = torch.matmul(attn, v_t).transpose(1, 2)
    return out.to(q.dtype)


def _compute_snr(ref, test):
    """Signal-to-Noise Ratio in dB."""
    signal = torch.norm(ref.float()).pow(2)
    if signal < 1e-12:
        return (
            float("inf")
            if torch.allclose(ref.float(), test.float(), atol=1e-7)
            else 0.0
        )
    noise = torch.norm(ref.float() - test.float()).pow(2)
    return 10.0 * torch.log10(signal / (noise + 1e-12)).item()


def _spawn_cp_p2p(B, S, H, D, sm_scale, requires_grad=False):
    """Helper: spawn 2-GPU CP P2P workers and collect results."""
    import socket

    import torch.multiprocessing as mp

    torch.manual_seed(42)
    global_q = torch.randn(B, S, H, D, dtype=torch.bfloat16) * 0.02
    global_k = torch.randn(B, S, H, D, dtype=torch.bfloat16) * 0.02
    global_v = torch.randn(B, S, H, D, dtype=torch.bfloat16) * 0.02

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        port = sock.getsockname()[1]

    S_local = S // 2
    _warmup_ck_kernels(
        B, S_local, H, D, dtype=torch.bfloat16, include_bwd=requires_grad
    )

    result_dir = tempfile.mkdtemp(prefix="cp-p2p-")
    mp.spawn(
        _cp_p2p_worker,
        args=(
            2,
            result_dir,
            global_q,
            global_k,
            global_v,
            sm_scale,
            port,
            requires_grad,
        ),
        nprocs=2,
        join=True,
    )

    results = {}
    for rank in range(2):
        path = os.path.join(result_dir, f"rank{rank}.pt")
        if os.path.exists(path):
            results[rank] = torch.load(path, weights_only=False)

    return global_q, global_k, global_v, results


@_MULTI_GPU
class TestCPP2PDistributed:
    """Distributed 2-GPU test for CP P2P ring attention."""

    def setup_method(self):
        if not _ck_spawn_probe():
            pytest.skip(
                "CK attention kernel SIGSEGV in mp.spawn — known CK kernel issue"
            )

    def test_output_shape(self):
        B, S, H, D = 2, 256, 8, 64
        sm_scale = D**-0.5
        _, _, _, results = _spawn_cp_p2p(B, S, H, D, sm_scale)

        assert len(results) == 2, f"Expected 2 results, got {len(results)}"
        S_local = S // 2
        for rank in [0, 1]:
            out = results[rank]["out"]
            assert out.shape == (
                B,
                S_local,
                H,
                D,
            ), f"Rank {rank}: expected shape ({B}, {S_local}, {H}, {D}), got {out.shape}"
            assert torch.isfinite(
                out
            ).all(), f"Rank {rank} output contains non-finite values"

    def test_forward_correctness_vs_reference(self):
        """Compare CP P2P ring attention output against single-GPU full-attention reference.

        SNR threshold: >= 18 dB is typical for bf16 attention on MI300X.
        """
        B, S, H, D = 2, 256, 8, 64
        sm_scale = D**-0.5
        global_q, global_k, global_v, results = _spawn_cp_p2p(B, S, H, D, sm_scale)

        ref_out = _attention_ref_causal(global_q, global_k, global_v, sm_scale)

        cp_out = torch.cat([results[0]["out"], results[1]["out"]], dim=1)

        snr = _compute_snr(ref_out, cp_out)
        assert snr >= 18.0, (
            f"Forward SNR {snr:.1f} dB < 18 dB threshold; "
            f"max abs err = {(ref_out - cp_out).abs().max().item():.6f}"
        )

    def test_backward_correctness_vs_reference(self):
        """Compare CP P2P backward gradients against single-GPU reference gradients.

        Uses the same global Q,K,V and grad_output, computes reference gradients
        on a single device, then compares.
        """
        B, S, H, D = 2, 256, 8, 64
        sm_scale = D**-0.5
        global_q, global_k, global_v, results = _spawn_cp_p2p(
            B,
            S,
            H,
            D,
            sm_scale,
            requires_grad=True,
        )

        grad_out_full = torch.cat(
            [results[0]["grad_out"], results[1]["grad_out"]], dim=1
        )

        ref_q = global_q.float().requires_grad_(True)
        ref_k = global_k.float().requires_grad_(True)
        ref_v = global_v.float().requires_grad_(True)
        q_t = ref_q.transpose(1, 2)
        k_t = ref_k.transpose(1, 2)
        v_t = ref_v.transpose(1, 2)
        attn = torch.matmul(q_t, k_t.transpose(-2, -1)) * sm_scale
        row_idx = torch.arange(S, device=ref_q.device).unsqueeze(1)
        col_idx = torch.arange(S, device=ref_q.device).unsqueeze(0)
        attn = attn.masked_fill(
            ~(row_idx >= col_idx).unsqueeze(0).unsqueeze(0), float("-inf")
        )
        attn = torch.softmax(attn, dim=-1)
        ref_out = torch.matmul(attn, v_t).transpose(1, 2)
        ref_out.backward(grad_out_full.float())

        cp_grad_q = torch.cat([results[0]["grad_q"], results[1]["grad_q"]], dim=1)
        cp_grad_k = torch.cat([results[0]["grad_k"], results[1]["grad_k"]], dim=1)
        cp_grad_v = torch.cat([results[0]["grad_v"], results[1]["grad_v"]], dim=1)

        for name, cp_grad, ref_grad in [
            ("grad_q", cp_grad_q, ref_q.grad.to(torch.bfloat16)),
            ("grad_k", cp_grad_k, ref_k.grad.to(torch.bfloat16)),
            ("grad_v", cp_grad_v, ref_v.grad.to(torch.bfloat16)),
        ]:
            snr = _compute_snr(ref_grad, cp_grad)
            assert snr >= 12.0, (
                f"{name} SNR {snr:.1f} dB < 12 dB threshold; "
                f"max abs err = {(ref_grad - cp_grad).abs().max().item():.6f}"
            )
