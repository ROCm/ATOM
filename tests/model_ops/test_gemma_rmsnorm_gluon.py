# SPDX-License-Identifier: MIT
"""`GemmaRMSNorm.forward_cuda` on gfx1250 runs aiter's Gluon `fused_rmsnorm_add`.

The Gluon op takes the place of the HIP `fused_qk_rmsnorm_group_quant` kernel
(`ATOM_ENABLE_GLUON_GEMMA_RMSNORM`). The first two tests check the dispatch with
stand-in ops, so they need no GPU; the last one compares the real op with
`forward_native` on a gfx1250 GPU.
"""

import pytest
import torch

pytest.importorskip("aiter")

from atom.model_ops import layernorm
from atom.model_ops.layernorm import GemmaRMSNorm


def make_norm(n, dtype, device="cpu"):
    norm = GemmaRMSNorm(n, eps=1e-6)
    norm.weight.data = 0.1 * torch.randn(n, dtype=dtype, device=device)
    return norm


@pytest.mark.parametrize("with_residual", [False, True])
def test_forward_cuda_runs_the_gluon_op(monkeypatch, with_residual):
    calls = []

    def gluon_op(x, weight, epsilon, res1=None, gemma_norm=False):
        calls.append((tuple(x.shape), res1 is not None, gemma_norm))
        out = torch.zeros_like(x)
        return out if res1 is None else (out, x + res1)

    monkeypatch.setattr(layernorm, "gluon_gemma_rmsnorm", lambda: gluon_op)
    norm = make_norm(64, torch.bfloat16)
    x = torch.randn(2, 3, 64, dtype=torch.bfloat16)
    if with_residual:
        residual = torch.randn_like(x)
        out, res_out = norm.forward_cuda(x, residual)
        torch.testing.assert_close(res_out, x + residual)
    else:
        out = norm.forward_cuda(x)
    assert out.shape == x.shape
    assert calls == [((6, 64), with_residual, True)]


def test_forward_cuda_keeps_the_hip_kernel(monkeypatch):
    import aiter.ops.fused_qk_rmsnorm_group_quant as hip

    hip_calls = []
    monkeypatch.setattr(
        hip,
        "fused_qk_rmsnorm_group_quant",
        lambda **kwargs: hip_calls.append(kwargs["gemma_norm"]),
    )
    x = torch.randn(4, 64, dtype=torch.bfloat16)

    # No Gluon op: other GPUs, the flag off, or an aiter without it.
    monkeypatch.setattr(layernorm, "gluon_gemma_rmsnorm", lambda: None)
    make_norm(64, torch.bfloat16).forward_cuda(x)

    # The Gluon op takes a weight in the dtype of x only.
    def gluon_op(*args, **kwargs):
        raise AssertionError("an fp32 weight must keep the HIP kernel")

    monkeypatch.setattr(layernorm, "gluon_gemma_rmsnorm", lambda: gluon_op)
    make_norm(64, torch.float32).forward_cuda(x)
    assert hip_calls == [True, True]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("m", [1, 7, 256, 4096])
@pytest.mark.parametrize("with_residual", [False, True])
def test_gluon_op_matches_forward_native(m, with_residual):
    layernorm.gluon_gemma_rmsnorm.cache_clear()
    if layernorm.gluon_gemma_rmsnorm() is None:
        pytest.skip("needs gfx1250 and an aiter with the Gluon fused_rmsnorm_add")
    torch.manual_seed(0)
    norm = make_norm(2048, torch.bfloat16, "cuda")
    x = torch.randn(m, 2048, dtype=torch.bfloat16, device="cuda")
    residual = torch.randn_like(x) if with_residual else None
    out = norm.forward_cuda(x, residual)
    ref = norm.forward_native(x, residual)
    if with_residual:
        torch.testing.assert_close(out[1], ref[1])
        out, ref = out[0], ref[0]
    torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)
