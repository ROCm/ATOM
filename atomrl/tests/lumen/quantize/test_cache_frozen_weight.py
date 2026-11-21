# SPDX-License-Identifier: MIT
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

"""cache_frozen_weight: a frozen weight's FP8 quant is cached on first forward
and reused (skips per-forward re-quant), without changing numerics."""

import pytest
import torch
from torch import nn

_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _frozen_linear(out_f=256, in_f=256):
    m = nn.Linear(in_f, out_f, bias=False).cuda().to(torch.bfloat16)
    m.weight.requires_grad_(False)  # base weight is frozen (LoRA recipe)
    return m


@_CUDA
def test_cache_populated_and_reused():
    from atomrl.lumen.quantize import disable, enable
    from atomrl.lumen.quantize.config import QuantConfig

    m = _frozen_linear()
    enable(
        m,
        config=QuantConfig.from_str(
            scaling="blockwise2d", block_size=128, cache_frozen_weight=True
        ),
    )
    x = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)

    assert getattr(m, "_fp8_weight_data", None) is None
    m(x)
    assert m._fp8_weight_data is not None and m._fp8_weight_scale is not None
    # blockwise2d cache scale is the 2D tile grid (N/128, K/128).
    assert tuple(m._fp8_weight_scale.shape) == (256 // 128, 256 // 128)

    ptr = m._fp8_weight_data.data_ptr()
    m(x)
    assert m._fp8_weight_data.data_ptr() == ptr, "second forward must reuse the cache"
    disable(m)


@_CUDA
def test_cached_matches_uncached_output():
    from atomrl.lumen.quantize import disable, enable
    from atomrl.lumen.quantize.config import QuantConfig

    torch.manual_seed(0)
    w = (torch.randn(256, 256) * 0.05).cuda().to(torch.bfloat16)
    x = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)

    outs = {}
    for cache in (False, True):
        m = _frozen_linear()
        m.weight.data.copy_(w)
        enable(
            m,
            config=QuantConfig.from_str(
                scaling="blockwise2d", block_size=128, cache_frozen_weight=cache
            ),
        )
        outs[cache] = m(x).float()
        disable(m)

    # The cache stores exactly the first-forward quant of the same frozen weight,
    # so cached and uncached forwards must be numerically identical.
    torch.testing.assert_close(outs[True], outs[False], rtol=0, atol=0)


@_CUDA
def test_skip_wgrad_preserves_dgrad():
    """Skipping the frozen weight's WGrad must not change the input grad (DGrad)."""
    from atomrl.lumen.quantize import disable, enable
    from atomrl.lumen.quantize.config import QuantConfig

    torch.manual_seed(0)
    w = (torch.randn(256, 256) * 0.05).cuda().to(torch.bfloat16)
    x0 = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)

    grads = {}
    for cache in (False, True):
        m = _frozen_linear()
        m.weight.data.copy_(w)
        enable(
            m,
            config=QuantConfig.from_str(
                scaling="blockwise2d", block_size=128, cache_frozen_weight=cache
            ),
        )
        x = x0.clone().requires_grad_(True)
        m(x).sum().backward()
        grads[cache] = x.grad.float()
        # frozen weight never accumulates grad regardless
        assert m.weight.grad is None
        disable(m)

    # WGrad skip (cache=True) must leave the DGrad identical to the full path.
    torch.testing.assert_close(grads[True], grads[False], rtol=0, atol=0)


@_CUDA
def test_bpreshuffle_matches_blockscale():
    """bpreshuffle GEMM (frozen weight) must match the CK blockscale path (fwd+bwd)."""
    from atomrl.lumen.quantize import disable, enable
    from atomrl.lumen.quantize.config import QuantConfig

    try:
        import aiter
        from aiter.ops.shuffle import shuffle_weight  # noqa: F401

        aiter.gemm_a8w8_blockscale_bpreshuffle  # noqa: B018
    except (ImportError, AttributeError):
        pytest.skip("aiter bpreshuffle blockscale not available")

    from conftest import compute_snr  # type: ignore

    torch.manual_seed(0)
    w = (torch.randn(256, 256) * 0.05).cuda().to(torch.bfloat16)
    x0 = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)

    outs, gins = {}, {}
    for bp in (False, True):
        m = _frozen_linear()
        m.weight.data.copy_(w)
        enable(
            m,
            config=QuantConfig.from_str(
                scaling="blockwise2d",
                block_size=128,
                cache_frozen_weight=True,
                bpreshuffle_gemm=bp,
            ),
        )
        x = x0.clone().requires_grad_(True)
        y = m(x)
        y.sum().backward()
        outs[bp] = y.detach().float()
        gins[bp] = x.grad.float()
        disable(m)
    # bit-identical math (shuffle only changes layout) -> very high SNR
    assert compute_snr(outs[False], outs[True]) > 40
    assert compute_snr(gins[False], gins[True]) > 40


@_CUDA
def test_bpreshuffle_onfly_matches_blockscale(monkeypatch):
    """On-the-fly bpreshuffle (LUMEN_BPRESHUFFLE_ONFLY, uncached) must match CK blockscale."""
    import atomrl.lumen.ops.quantize.linear as lin

    if not hasattr(lin, "_bpreshuffle_onfly_enabled"):
        pytest.skip("gemm_blockscale no longer switches on LUMEN_BPRESHUFFLE_ONFLY")
    try:
        import aiter
        from aiter.ops.shuffle import shuffle_weight  # noqa: F401

        aiter.gemm_a8w8_blockscale_bpreshuffle  # noqa: B018
    except (ImportError, AttributeError):
        pytest.skip("aiter bpreshuffle blockscale not available")
    from conftest import compute_snr  # type: ignore

    torch.manual_seed(0)
    M, N, K = 256, 256, 256
    a = (torch.rand(M, K, device="cuda") / 10).to(
        torch.float8_e4m3fnuz
        if "gfx94" in torch.cuda.get_device_properties(0).gcnArchName
        else torch.float8_e4m3fn
    )
    w = (torch.rand(N, K, device="cuda") / 10).to(a.dtype)
    sa = torch.rand(M, K // 128, device="cuda", dtype=torch.float32)
    sw = torch.rand(N // 128, K // 128, device="cuda", dtype=torch.float32)

    lin._bpreshuffle_onfly_enabled.cache_clear()
    monkeypatch.delenv("LUMEN_BPRESHUFFLE_ONFLY", raising=False)
    y_ck = lin.gemm_blockscale(a, w, sa, sw)
    monkeypatch.setenv("LUMEN_BPRESHUFFLE_ONFLY", "1")
    lin._bpreshuffle_onfly_enabled.cache_clear()
    y_bp = lin.gemm_blockscale(a, w, sa, sw)
    lin._bpreshuffle_onfly_enabled.cache_clear()
    assert compute_snr(y_ck.float(), y_bp.float()) > 40


@_CUDA
def test_trainable_weight_not_cached():
    """Cache must NOT engage for trainable weights (only frozen base)."""
    from atomrl.lumen.quantize import disable, enable
    from atomrl.lumen.quantize.config import QuantConfig

    m = nn.Linear(256, 256, bias=False).cuda().to(torch.bfloat16)  # requires_grad=True
    enable(
        m,
        config=QuantConfig.from_str(
            scaling="blockwise2d", block_size=128, cache_frozen_weight=True
        ),
    )
    m(torch.randn(128, 256, device="cuda", dtype=torch.bfloat16))
    assert getattr(m, "_fp8_weight_data", None) is None
    disable(m)
