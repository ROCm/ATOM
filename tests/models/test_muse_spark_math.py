# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Muse Spark's non-standard blocks, pinned to the reference arithmetic.

CPU-only. Each helper is checked against the formula the SGLang reference
implementation (rocm_muse_spark) computes, written out independently here.
"""

import math

import pytest
import torch

from atom.models.muse_spark_math import (
    muse_logits,
    muse_rms,
    muse_route,
    post_expert_combine,
    residual_gate_coeffs,
)


def test_route_selects_with_bias_but_weights_without_it():
    torch.manual_seed(0)
    logits = torch.randn(5, 16)
    bias = torch.zeros(16)
    bias[3] = 100.0  # forces expert 3 into every token's top-k
    mult = 0.17677669529663687

    scores, ids = muse_route(logits, bias, top_k=4, multiplier=mult)

    assert ids.dtype == torch.int32 and ids.shape == (5, 4)
    assert (ids == 3).any(dim=-1).all()
    raw = torch.sigmoid(logits * mult)
    picked = raw.gather(1, ids.long())
    expected = picked / (picked.sum(-1, keepdim=True) + 1e-15)
    torch.testing.assert_close(scores, expected)
    torch.testing.assert_close(scores.sum(-1), torch.ones(5))


def test_residual_gate_is_norm_preserving_shape():
    gate = torch.tensor([-1.0, 0.0, 2.0])
    alpha, beta = residual_gate_coeffs(gate, temperature=0.3)

    w = gate / 0.3
    expected_beta = torch.sigmoid(w)
    expected_alpha = torch.sqrt(
        torch.clamp(torch.sigmoid(-w) * (1 + expected_beta), min=1e-3)
    )
    torch.testing.assert_close(beta, expected_beta)
    torch.testing.assert_close(alpha, expected_alpha)


def test_rms_matches_definition_with_effective_weight():
    x = torch.randn(3, 8)
    w = torch.randn(8)
    out = muse_rms(x, w, eps=1e-5)
    ref = x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5) * w
    torch.testing.assert_close(out, ref)
    assert out.dtype == torch.float32


def test_post_expert_combine_norms_each_slot_before_the_weighted_sum():
    torch.manual_seed(1)
    tokens, top_k, dim = 3, 4, 32
    y = torch.randn(tokens, top_k, dim).to(torch.bfloat16)
    scores = torch.rand(tokens, top_k)
    w_eff = (torch.randn(dim) * 0.1 + 1).to(torch.bfloat16)

    out = post_expert_combine(y, scores, w_eff, eps=1e-8)

    scaled = (w_eff * y).float()
    normed = scaled / torch.sqrt(scaled.pow(2).mean(-1, keepdim=True) + 1e-8)
    ref = (scores.unsqueeze(-1) * normed).sum(1)
    assert out.dtype == torch.bfloat16
    torch.testing.assert_close(out.float(), ref, rtol=1e-2, atol=1e-2)


def test_post_expert_combine_keeps_zero_slots_zero():
    # Under expert parallelism a slot routed to a remote expert comes back as
    # zeros; it must contribute nothing rather than eps-scaled noise.
    y = torch.zeros(2, 8, 16, dtype=torch.bfloat16)
    y[:, 0] = 1.0
    scores = torch.full((2, 8), 0.125)
    out = post_expert_combine(y, scores, torch.ones(16, dtype=torch.bfloat16), 1e-8)
    torch.testing.assert_close(out.float(), torch.full((2, 16), 0.125))


def test_logits_are_scaled_then_softcapped():
    raw = torch.tensor([[0.0, 50.0, -1000.0, 1e6]])
    out = muse_logits(raw, multiplier=0.25, cap=20.0)
    expected = torch.tanh(raw * 0.25 / 20.0) * 20.0
    torch.testing.assert_close(out, expected)
    assert out.abs().max() <= 20.0
    assert math.isclose(out[0, 0].item(), 0.0)


@pytest.mark.parametrize("cap", [None, 0.0])
def test_logits_without_a_softcap_are_only_scaled(cap):
    raw = torch.tensor([[0.0, 50.0, -1000.0]])
    torch.testing.assert_close(muse_logits(raw, 0.25, cap), raw * 0.25)


def test_post_load_coefficients_are_bit_identical_to_the_per_step_ones():
    # The norm weights and the gate's alpha/beta are computed once after
    # loading; the forward outputs must not change by one bit.
    pytest.importorskip("aiter")
    from atom.models.muse_spark import MuseResidualGate, MuseRMSNorm

    torch.manual_seed(2)
    x = torch.randn(3, 64, dtype=torch.bfloat16)
    stream = torch.randn(3, 64)
    for gain_center, input_scale in [(1.0, False), (1.0, True), (0.0, False)]:
        norm = MuseRMSNorm(64, 1e-5, gain_center, input_scale)
        norm.weight.data.copy_(torch.randn(64) * 0.01)
        w = (norm.weight.float() + gain_center).to(norm.weight.dtype)
        ref = muse_rms(w * x, None, 1e-5) if input_scale else muse_rms(x, w, 1e-5)
        norm.process_weights_after_loading()
        assert norm.effective_weight.dtype == torch.bfloat16
        assert torch.equal(norm.effective_weight, w)
        assert torch.equal(norm(x), ref)

    gate = MuseResidualGate(64, temperature=0.3)
    gate.gate.data.copy_(torch.randn(64))
    alpha, beta = residual_gate_coeffs(gate.gate, 0.3)
    gate.process_weights_after_loading()
    assert torch.equal(gate(stream, x), alpha * stream + beta * x.float())
