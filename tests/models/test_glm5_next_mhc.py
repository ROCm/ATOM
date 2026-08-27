# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""mHC formulation checks for GLM-5.3-Flash.

`ATOM_MHC_FORCE_TORCH=1` only proves the aiter kernels and the torch fallback
agree -- both run the SAME formulation, so it says nothing about whether that
formulation is right. These tests pin the formulation itself, and in
particular the part unique to this port: a layer DEFERS its `hc_post` so the
next layer's `hc_pre` can fuse it, which reorders the computation relative to
the naive `pre -> sublayer -> post` reading of the model.

CPU-only; the torch fallback is forced, so no GPU or aiter kernels required.
"""

import os

import torch

os.environ["ATOM_MHC_FORCE_TORCH"] = "1"

from atom.model_ops.mhc import (
    HyperConnection,
    MHCOps,
    hc_contract,
    hc_expand,
)

DIM = 256
HC = 4
TOKENS = 6


def _ops():
    return MHCOps(dim=DIM, hc_mult=HC, norm_eps=1e-5, hc_eps=1e-6, sinkhorn_iters=20)


def _hc(seed):
    torch.manual_seed(seed)
    h = HyperConnection(DIM, HC)
    with torch.no_grad():
        h.fn.normal_(0, 0.02)
        h.base.normal_(0, 0.02)
        h.scale.fill_(1.0)
    return h


def test_torch_fallback_is_actually_selected():
    """Guard the premise: these tests are meaningless on the kernel path."""
    ops = _ops()
    assert ops._pre is None and ops._post is None and ops._fused_post_pre is None


def test_expand_then_contract_is_identity():
    torch.manual_seed(0)
    x = torch.randn(TOKENS, DIM)
    torch.testing.assert_close(hc_contract(hc_expand(x, HC)), x)


def test_expand_replicates_across_streams():
    torch.manual_seed(1)
    x = torch.randn(TOKENS, DIM)
    e = hc_expand(x, HC)
    assert e.shape == (TOKENS, HC, DIM)
    for i in range(HC):
        torch.testing.assert_close(e[:, i, :], x)


def test_pre_reduces_stack_and_post_restores_it():
    ops, hc = _ops(), _hc(2)
    torch.manual_seed(3)
    residual = torch.randn(TOKENS, HC, DIM)
    y, post, comb = ops.pre(residual, hc)
    assert y.shape == (TOKENS, DIM)
    assert post.shape == (TOKENS, HC)
    assert comb.shape == (TOKENS, HC, HC)
    new_residual = ops.post(y, residual, post, comb)
    assert new_residual.shape == residual.shape
    assert torch.isfinite(new_residual).all()


def test_comb_is_doubly_stochastic():
    """Sinkhorn projects `comb` onto the Birkhoff polytope."""
    ops, hc = _ops(), _hc(4)
    torch.manual_seed(5)
    _, _, comb = ops.pre(torch.randn(TOKENS, HC, DIM), hc)
    ones = torch.ones(TOKENS, HC)
    torch.testing.assert_close(comb.sum(-1), ones, atol=2e-2, rtol=0)
    torch.testing.assert_close(comb.sum(-2), ones, atol=2e-2, rtol=0)


def test_fused_post_pre_equals_post_then_pre():
    """THE key property: deferring hc_post must not change the result.

    A layer hands its hc_post inputs to the next hc_pre instead of applying
    them itself. That is only sound if `fused_post_pre` is exactly `post`
    followed by `pre`. If the two disagree (argument order, a missing squeeze,
    a swapped return) the model still runs and still emits fluent text -- it
    just quietly computes the wrong residual stream.
    """
    ops, hc_a, hc_b = _ops(), _hc(6), _hc(7)
    torch.manual_seed(8)
    residual = torch.randn(TOKENS, HC, DIM)
    sublayer_out = torch.randn(TOKENS, DIM)

    _, post_a, comb_a = ops.pre(residual, hc_a)
    fused_res, fused_post, fused_comb, fused_y = ops.fused_post_pre(
        sublayer_out, residual, post_a, comb_a, hc_b
    )

    ref_res = ops.post(sublayer_out, residual, post_a, comb_a)
    ref_y, ref_post, ref_comb = ops.pre(ref_res, hc_b)

    torch.testing.assert_close(fused_res, ref_res, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(fused_y, ref_y, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(fused_post, ref_post, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(fused_comb, ref_comb, atol=1e-5, rtol=1e-4)


def test_four_sublayer_chain_matches_undeferred_reference():
    """End-to-end: the layer's deferred threading vs a plain sequential model."""
    ops = _ops()
    hcs = [_hc(10 + i) for i in range(4)]  # attn/ffn for two layers
    torch.manual_seed(20)
    x0 = torch.randn(TOKENS, DIM)
    sub = [torch.randn(TOKENS, DIM) for _ in range(4)]

    # Reference: strictly pre -> sublayer -> post, four times.
    r = hc_expand(x0, HC)
    for hc, s in zip(hcs, sub):
        _, p, c = ops.pre(r, hc)
        r = ops.post(s, r, p, c)
    ref = hc_contract(r)

    # Deferred: the order Glm5NextDecoderLayer.forward actually runs.
    r = hc_expand(x0, HC)
    _, post, comb = ops.pre(r, hcs[0])
    for i in range(1, 4):
        r, post, comb, _ = ops.fused_post_pre(sub[i - 1], r, post, comb, hcs[i])
    got = hc_contract(ops.post(sub[-1], r, post, comb))

    torch.testing.assert_close(got, ref, atol=1e-4, rtol=1e-3)
