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


@pytest.mark.parametrize("n", [1, 16, 300])
def test_fused_qk_norm_mrope_cache_fp8_matches_unfused_path(n):
    """An fp8 cache: the same bytes as the norm and MRoPE, the K/V clamps,
    reshape_and_cache with the per-tensor scale, and Q clamped and cast to fp8."""
    import aiter
    from aiter.jit.utils.chip_info import get_gfx

    from atom.model_ops import triton_mrope as m
    from atom.model_ops.layernorm import DualRMSNorm, GemmaRMSNorm

    if get_gfx() != "gfx1250":
        pytest.skip("gfx1250 kernel")
    if not m._gfx1250_fused_qk_norm_mrope_cache_fp8():
        pytest.skip("aiter without the fp8 cache write")
    torch.manual_seed(45)
    fp8 = aiter.dtypes.fp8
    fp8_max = torch.finfo(fp8).max
    kv_scale = torch.finfo(torch.float8_e4m3fn).max / fp8_max
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
    qk_norm = DualRMSNorm(q_norm, k_norm, 16, 2, 256, prefix="test")
    qkv = torch.randn(n, 2 * 4096 + 2 * 512, device="cuda", dtype=torch.bfloat16)
    qkv[:, -512:-480] = 3000.0  # V values past the fp8 range: stored clamped, not NaN
    q, k, v = qkv[:, 4096:8192], qkv[:, 8192:8704], qkv[:, 8704:9216]
    positions = torch.randint(0, 64, (3, n), device="cuda")
    # SHUFFLE fp8 caches: blocks of 64 slots, 16 fp8 values in 16 bytes.
    blocks = n // 64 + 4
    slots = torch.randperm(blocks * 64, device="cuda")[:n]
    caches = [
        (
            torch.zeros(blocks, 2, 16, 64, 16, device="cuda", dtype=torch.uint8).view(
                fp8
            ),
            torch.zeros(blocks, 2, 256, 64, device="cuda", dtype=torch.uint8).view(fp8),
        )
        for _ in range(2)
    ]

    fused = m.try_fused_qk_norm_mrope_cache(
        q_norm,
        k_norm,
        rotary,
        positions,
        q,
        k,
        v,
        16,
        2,
        256,
        *caches[0],
        slots,
        kv_scale=kv_scale,
    )
    q_ref, k_ref = m.try_fused_qk_norm_mrope(
        qk_norm, rotary, positions, q, k, 16, 2, 256
    )
    kv_max = fp8_max * kv_scale
    scale = torch.tensor(kv_scale, dtype=torch.float32, device="cuda")
    aiter.reshape_and_cache(
        k_ref.view(-1, 2, 256).clamp(-kv_max, kv_max),
        v.reshape(-1, 2, 256).clamp(-kv_max, kv_max),
        *caches[1],
        slots,
        kv_cache_dtype="fp8",
        k_scale=scale,
        v_scale=scale,
        asm_layout=True,
    )
    same = lambda a, b: torch.equal(a.view(torch.uint8), b.view(torch.uint8))
    assert fused[0].dtype == fp8
    assert same(fused[0], q_ref.clamp(-fp8_max, fp8_max).to(fp8))
    assert torch.equal(fused[1], k_ref)
    assert same(caches[0][0], caches[1][0]) and same(caches[0][1], caches[1][1])
    assert not torch.isnan(caches[0][1].float()).any()


def test_fused_qk_norm_mrope_cache_applies(monkeypatch):
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
    # fp8: only for unified attention (one per-tensor scale, Q in fp8), and only when
    # aiter's kernel writes fp8
    monkeypatch.setenv("ATOM_USE_UNIFIED_ATTN", "1")
    assert (
        m.gfx1250_fused_qk_norm_mrope_cache_applies(rotary, 256, "fp8")
        == m._gfx1250_fused_qk_norm_mrope_cache_fp8()
    )
    monkeypatch.setenv("ATOM_USE_UNIFIED_ATTN", "0")
    assert not m.gfx1250_fused_qk_norm_mrope_cache_applies(rotary, 256, "fp8")
    assert not m.gfx1250_fused_qk_norm_mrope_cache_applies(rotary, 128, "bf16")
    plain = SimpleNamespace(**{**qwen35, "mrope_interleaved": False})
    assert not m.gfx1250_fused_qk_norm_mrope_cache_applies(plain, 256, "bf16")
