# SPDX-License-Identifier: MIT
"""On gfx1250 the fused Q/K norm + MRoPE gives the result of the norm, then the MRoPE."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("triton")
pytest.importorskip("aiter")
pytestmark = pytest.mark.skipif(
    not torch.version.hip or not torch.cuda.is_available(), reason="requires a ROCm GPU"
)


@pytest.mark.parametrize("n", [1, 16, 64, 300])
def test_fused_qk_norm_mrope_matches_two_kernels(n):
    from aiter.jit.utils.chip_info import get_gfx

    from atom.model_ops import triton_mrope as m
    from atom.model_ops.layernorm import DualRMSNorm, GemmaRMSNorm

    if get_gfx() != "gfx1250":
        pytest.skip("gfx1250 kernel")
    torch.manual_seed(43)
    angles = torch.randn(64, 32, device="cuda")
    rotary = SimpleNamespace(
        mrope_section=[11, 11, 10],
        mrope_interleaved=True,
        rotary_dim=64,
        is_neox_style=True,
        cos_cache=angles.cos(),
        sin_cache=angles.sin(),
    )
    q_norm = GemmaRMSNorm(256, eps=1e-6).cuda()
    k_norm = GemmaRMSNorm(256, eps=1e-6).cuda()
    with torch.no_grad():
        q_norm.weight.normal_(0, 0.2)
        k_norm.weight.normal_(0, 0.2)
    qk_norm = DualRMSNorm(q_norm, k_norm, 4, 1, 256, prefix="test")
    # q and k are column slices of one projection output, as in Qwen3.5 attention.
    qkv = torch.randn(n, 2 * 1024 + 2 * 256, device="cuda", dtype=torch.bfloat16)
    q, k = qkv[:, 1024:2048], qkv[:, 2048:2304]
    positions = torch.randint(0, 64, (3, n), device="cuda")

    fused = m.try_fused_qk_norm_mrope(qk_norm, rotary, positions, q, k, 4, 1, 256)
    q_normed, k_normed = qk_norm(q, k)
    expected = m.try_mrope_qk_fused(rotary, positions, q_normed, k_normed, 4, 1, 256)
    for actual, ref in zip(fused, expected):
        torch.testing.assert_close(actual, ref, atol=0.016, rtol=0.016)


@pytest.mark.parametrize("n", [1, 16, 300])
def test_fused_qk_norm_mrope_cache_matches_three_kernels(n):
    import aiter
    from aiter.jit.utils.chip_info import get_gfx

    from atom.model_ops import triton_mrope as m
    from atom.model_ops.layernorm import DualRMSNorm, GemmaRMSNorm

    if get_gfx() != "gfx1250":
        pytest.skip("gfx1250 kernel")
    torch.manual_seed(44)
    angles = torch.randn(64, 32, device="cuda")
    rotary = SimpleNamespace(
        mrope_section=[11, 11, 10],
        mrope_interleaved=True,
        rotary_dim=64,
        is_neox_style=True,
        cos_cache=angles.cos(),
        sin_cache=angles.sin(),
    )
    q_norm = GemmaRMSNorm(256, eps=1e-6).cuda()
    k_norm = GemmaRMSNorm(256, eps=1e-6).cuda()
    with torch.no_grad():
        q_norm.weight.normal_(0, 0.2)
        k_norm.weight.normal_(0, 0.2)
    qk_norm = DualRMSNorm(q_norm, k_norm, 4, 1, 256, prefix="test")
    qkv = torch.randn(n, 2 * 1024 + 2 * 256, device="cuda", dtype=torch.bfloat16)
    q, k, v = qkv[:, 1024:2048], qkv[:, 2048:2304], qkv[:, 2304:2560]
    positions = torch.randint(0, 64, (3, n), device="cuda")
    # SHUFFLE caches with blocks of 16 slots and 8 bf16 values in 16 bytes.
    blocks = n // 16 + 4
    slots = torch.randperm(blocks * 16, device="cuda")[:n]
    caches = [
        (
            torch.zeros(blocks, 1, 32, 16, 8, device="cuda", dtype=torch.bfloat16),
            torch.zeros(blocks, 1, 2, 256, 8, device="cuda", dtype=torch.bfloat16),
        )
        for _ in range(2)
    ]

    fused = m.try_fused_qk_norm_mrope_cache(
        q_norm, k_norm, rotary, positions, q, k, v, 4, 1, 256, *caches[0], slots
    )
    q_normed, k_normed = qk_norm(q, k)
    q_ref, k_ref = m.try_mrope_qk_fused(
        rotary, positions, q_normed, k_normed, 4, 1, 256
    )
    aiter.reshape_and_cache(
        k_ref.view(-1, 1, 256),
        v.view(-1, 1, 256),
        *caches[1],
        slots,
        kv_cache_dtype="auto",
        k_scale=None,
        v_scale=None,
        asm_layout=True,
    )
    torch.testing.assert_close(fused[0], q_ref, atol=0.016, rtol=0.016)
    torch.testing.assert_close(caches[0][0], caches[1][0], atol=0.016, rtol=0.016)
    torch.testing.assert_close(caches[0][1], caches[1][1], atol=0, rtol=0)


def test_fused_qk_norm_mrope_cache_applies():
    from aiter.jit.utils.chip_info import get_gfx

    from atom.model_ops import triton_mrope as m

    if get_gfx() != "gfx1250":
        pytest.skip("gfx1250 kernel")
    qwen35 = {
        "mrope_section": [11, 11, 10],
        "mrope_interleaved": True,
        "rotary_dim": 64,
        "is_neox_style": True,
    }
    rotary = SimpleNamespace(**qwen35)
    assert m.gfx1250_fused_qk_norm_mrope_cache_applies(rotary, 256, "bf16")
    assert not m.gfx1250_fused_qk_norm_mrope_cache_applies(rotary, 256, "fp8")
    assert not m.gfx1250_fused_qk_norm_mrope_cache_applies(rotary, 128, "bf16")
    plain = SimpleNamespace(**{**qwen35, "mrope_interleaved": False})
    assert not m.gfx1250_fused_qk_norm_mrope_cache_applies(plain, 256, "bf16")
