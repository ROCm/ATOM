# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Backend selection of the V4.1 delayed mHC seam, on CPU with the seam kernels
stubbed, so the non-GPU unit-test job covers the gfx942 ASM branch that the GPU
guard test (tests/models/deepseek_v41/test_compilation.py) skips. The module is
loaded through atom_without_aiter so it imports where AITER is not installed."""

import pytest
import torch
from aiter_stub import atom_without_aiter


@pytest.fixture(scope="module")
def mhc():
    with atom_without_aiter() as load:
        yield load("atom.model_ops.deepseek_v41.mhc_pre_delayed")


@pytest.mark.parametrize(
    "env, with_post, supported, expected",
    [
        ("auto", True, True, "mega"),  # gfx942 post seam where AITER supports it
        ("triton", True, True, "triton"),  # forced Triton, even where ASM is supported
        ("auto", True, False, "triton"),  # AITER does not support the ASM seam here
        (
            "auto",
            False,
            True,
            "triton",
        ),  # a seam without a post never takes the ASM seam
    ],
)
def test_v41_mhc_pre_delayed_selects_backend(
    env, with_post, supported, expected, monkeypatch, request, mhc
):
    """Which seam kernel v41_mhc_pre_delayed picks for each backend/support case."""
    monkeypatch.setenv("ATOM_DSV41_MHC_BACKEND", env)
    monkeypatch.setattr(
        mhc,
        "mhc_fused_post_pre_delayed_asm_supported",
        lambda hidden, hc_mult: supported,
    )
    # Env and support are patched per case; clear before and after so no
    # result cached under the patches reaches a later test in the session.
    mhc.use_asm_seam.cache_clear()
    request.addfinalizer(mhc.use_asm_seam.cache_clear)
    rows, hc, hidden = 2, 4, 5120
    calls = []

    def stub(name):
        def kernel(residual, *args, **kwargs):
            calls.append(name)
            return (
                residual.clone(),
                torch.zeros(rows, hc),
                torch.zeros(rows, hc, hc),
                torch.zeros(rows, hidden),
                torch.zeros(rows, hc),
            )

        return kernel

    monkeypatch.setattr(mhc, "mega_mhc", stub("mega"))
    monkeypatch.setattr(mhc, "mhc_fused_post_pre_delayed_rmsnorm", stub("triton"))

    residual = torch.randn(1, rows, hc, hidden)
    pre = torch.full((1, rows, hc), 1 / hc)
    fn = torch.randn(24, hc * hidden)
    common = {
        "rms_eps": 1e-6,
        "hc_eps": 1e-6,
        "sinkhorn_iters": 20,
        "post_mult": 2.0,
        "norm_eps": 1e-6,
    }
    post_kw = (
        {
            "sublayer_output": torch.randn(1, rows, hidden),
            "post_mix": torch.ones_like(pre),
            "combination": torch.eye(hc).expand(1, rows, hc, hc).contiguous(),
        }
        if with_post
        else {}
    )
    # The guard decorator is an identity under the AITER stub, so the public name
    # may be the raw function; unwrap it where it is registered as an op.
    seam = getattr(mhc.v41_mhc_pre_delayed, "__wrapped__", mhc.v41_mhc_pre_delayed)
    seam(
        residual,
        pre,
        fn,
        torch.ones(3),
        torch.zeros(24),
        torch.ones(hidden),
        **common,
        **post_kw,
    )
    assert calls == [expected]
