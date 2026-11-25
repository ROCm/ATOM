# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Lumen-level unit tests for fused MoE router ops.

Tests 1-7 map to the spec's Testing section:
  Test 1: fused_topk_with_score_function (Component 1)
  Test 2: fused_compute_score_for_moe_aux_loss (Component 2)
  Test 3: fused_moe_aux_loss (Component 3)
  Test 4: Unsupported feature guards
  Test 5: gradcheck on all three components
  Test 6: End-to-end patch smoke test
  Test 7: AITER forward parity (7a, 7b, 7c)
"""

import pytest
import torch
from conftest import compute_snr

_requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)

_requires_aiter_softmax_topk = _requires_cuda
_requires_aiter_moe_aux_loss = _requires_cuda


# -- Reference implementations --


def _topk_routing_ref(logits, topk, use_pre_softmax, scaling_factor):
    """Unfused reference matching moe_utils.topk_routing_with_score_function(fused=False)."""
    N, E = logits.shape
    logits_fp32 = logits.float()

    if use_pre_softmax:
        scores = torch.softmax(logits_fp32, dim=-1)
        topk_weights, topk_indices = scores.topk(topk, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    else:
        topk_vals, topk_indices = logits_fp32.topk(topk, dim=-1)
        topk_weights = torch.softmax(topk_vals, dim=-1)

    if scaling_factor is not None:
        topk_weights = topk_weights * scaling_factor

    routing_probs = torch.zeros(N, E, dtype=topk_weights.dtype, device=logits.device)
    routing_probs.scatter_(1, topk_indices, topk_weights)
    routing_map = torch.zeros(N, E, dtype=torch.bool, device=logits.device)
    routing_map.scatter_(1, topk_indices, True)
    return routing_map, routing_probs


def _compute_scores_ref(logits, topk):
    """Unfused reference matching moe_utils.compute_routing_scores_for_aux_loss(fused=False)."""
    scores = torch.softmax(logits.float(), dim=-1)
    _, topk_indices = scores.topk(topk, dim=-1)
    N, E = scores.shape
    routing_map = torch.zeros(N, E, dtype=torch.bool, device=logits.device)
    routing_map.scatter_(1, topk_indices, True)
    return routing_map, scores


def _switch_load_balancing_loss_ref(probs, tokens_per_expert, T, topk, E, coeff):
    """Unfused reference matching moe_utils.switch_load_balancing_loss_func(fused=False)."""
    aggregated = probs.sum(dim=0)
    C = E * coeff / (topk * T * T)
    return (aggregated * tokens_per_expert).sum() * C


# -- Test 1 --


@_requires_cuda
@pytest.mark.parametrize(
    "N,E,topk", [(32, 8, 1), (32, 8, 2), (512, 64, 4), (4096, 128, 8)]
)
@pytest.mark.parametrize("use_pre_softmax", [True, False])
@pytest.mark.parametrize("scaling_factor", [None, 0.5])
def test_fused_topk_with_score_function(N, E, topk, use_pre_softmax, scaling_factor):
    """Spec Test 1: forward + backward vs unfused reference."""
    from atomrl.lumen.ops.moe.fused_router import fused_topk_with_score_function

    torch.manual_seed(42)
    logits = torch.randn(N, E, device="cuda", dtype=torch.float32, requires_grad=True)

    routing_map, routing_probs = fused_topk_with_score_function(
        logits,
        topk,
        use_pre_softmax,
        None,
        None,
        scaling_factor,
        "softmax",
        None,
    )

    logits_ref = logits.detach().clone().requires_grad_(True)
    routing_map_ref, routing_probs_ref = _topk_routing_ref(
        logits_ref,
        topk,
        use_pre_softmax,
        scaling_factor,
    )

    assert routing_map.dtype == torch.bool
    assert routing_map.sum(dim=-1).eq(topk).all()
    torch.testing.assert_close(
        routing_probs,
        routing_probs_ref,
        atol=1e-5,
        rtol=1e-5,
    )
    assert (routing_map == routing_map_ref).all()

    loss = routing_probs.sum()
    loss.backward()
    loss_ref = routing_probs_ref.sum()
    loss_ref.backward()

    torch.testing.assert_close(logits.grad, logits_ref.grad, atol=1e-4, rtol=1e-4)
    ref_energy = logits_ref.grad.float().norm()
    if ref_energy > 1e-3:
        snr = compute_snr(logits.grad, logits_ref.grad)
        assert snr > 15, f"Backward SNR too low: {snr:.1f} dB"


# -- Test 2 --


@_requires_cuda
@pytest.mark.parametrize("N,E,topk", [(32, 8, 2), (512, 64, 4), (4096, 128, 8)])
def test_fused_compute_score_for_moe_aux_loss(N, E, topk):
    """Spec Test 2: forward + backward vs unfused reference."""
    from atomrl.lumen.ops.moe.fused_router import fused_compute_score_for_moe_aux_loss

    torch.manual_seed(42)
    logits = torch.randn(N, E, device="cuda", dtype=torch.float32, requires_grad=True)

    routing_map, scores = fused_compute_score_for_moe_aux_loss(logits, topk, "softmax")

    logits_ref = logits.detach().clone().requires_grad_(True)
    routing_map_ref, scores_ref = _compute_scores_ref(logits_ref, topk)

    torch.testing.assert_close(scores, scores_ref, atol=1e-5, rtol=1e-5)
    assert routing_map.dtype == torch.bool
    assert (routing_map == routing_map_ref).all()

    loss = scores.sum()
    loss.backward()
    loss_ref = scores_ref.sum()
    loss_ref.backward()

    torch.testing.assert_close(logits.grad, logits_ref.grad, atol=1e-4, rtol=1e-4)
    snr = compute_snr(logits.grad, logits_ref.grad)
    assert snr > 15, f"Backward SNR too low: {snr:.1f} dB"


# -- Test 3 --


@_requires_cuda
@pytest.mark.parametrize("N,E,topk", [(32, 8, 2), (512, 64, 4), (4096, 128, 8)])
def test_fused_moe_aux_loss(N, E, topk):
    """Spec Test 3: forward + backward vs unfused reference."""
    from atomrl.lumen.ops.moe.fused_router import fused_moe_aux_loss

    torch.manual_seed(42)
    probs = torch.rand(N, E, device="cuda", dtype=torch.float32, requires_grad=True)
    tokens_per_expert = torch.randint(0, N, (E,), device="cuda", dtype=torch.float32)
    T, coeff = N, 0.01

    loss = fused_moe_aux_loss(probs, tokens_per_expert, T, E, topk, coeff)
    loss_ref = _switch_load_balancing_loss_ref(
        probs.detach(), tokens_per_expert, T, topk, E, coeff
    )
    torch.testing.assert_close(loss, loss_ref, atol=1e-5, rtol=1e-5)

    loss.backward()
    probs_ref = probs.detach().clone().requires_grad_(True)
    loss_ref2 = _switch_load_balancing_loss_ref(
        probs_ref, tokens_per_expert, T, topk, E, coeff
    )
    loss_ref2.backward()
    torch.testing.assert_close(probs.grad, probs_ref.grad, atol=1e-5, rtol=1e-5)


# -- Test 4 --


@_requires_cuda
def test_unsupported_score_function():
    """Spec Test 4: NotImplementedError for unsupported features."""
    from atomrl.lumen.ops.moe.fused_router import (
        fused_compute_score_for_moe_aux_loss,
        fused_topk_with_score_function,
    )

    logits = torch.randn(4, 8, device="cuda", dtype=torch.float32)

    with pytest.raises(NotImplementedError, match="sigmoid"):
        fused_topk_with_score_function(
            logits, 2, True, None, None, None, "sigmoid", None
        )

    with pytest.raises(NotImplementedError, match="sigmoid"):
        fused_compute_score_for_moe_aux_loss(logits, 2, "sigmoid")

    with pytest.raises(NotImplementedError, match="Group"):
        fused_topk_with_score_function(logits, 2, True, 2, None, None, "softmax", None)

    with pytest.raises(NotImplementedError, match="top-k"):
        fused_topk_with_score_function(logits, 2, True, None, 4, None, "softmax", None)

    with pytest.raises(NotImplementedError, match="expert_bias"):
        fused_topk_with_score_function(
            logits, 2, True, None, None, None, "softmax", torch.ones(8, device="cuda")
        )


# -- Test 5: gradcheck with pure PyTorch float64 wrappers --


def _pure_pytorch_topk_with_score(logits, topk):
    """Pure PyTorch reimplementation of Component 1 backward math (float64-safe)."""
    s = torch.softmax(logits, dim=-1)
    _, topk_indices = s.topk(topk, dim=-1)
    s_k = s.gather(1, topk_indices)
    V = s_k.sum(dim=-1, keepdim=True)
    w = s_k / V
    N, E = logits.shape
    routing_probs = torch.zeros(N, E, dtype=logits.dtype, device=logits.device)
    routing_probs.scatter_(1, topk_indices, w)
    return routing_probs


def _pure_pytorch_compute_scores(logits, topk):
    """Pure PyTorch reimplementation of Component 2 (float64-safe)."""
    return torch.softmax(logits, dim=-1)


def _pure_pytorch_aux_loss(probs, tokens_per_expert, T, E, topk, coeff):
    """Pure PyTorch reimplementation of Component 3 (float64-safe)."""
    C = E * coeff / (topk * T * T)
    return (probs.sum(dim=0) * tokens_per_expert).sum() * C


@_requires_cuda
def test_gradcheck_topk_with_score_function():
    """Spec Test 5: gradcheck for Component 1 math (use_pre_softmax=True, topk=2)."""
    torch.manual_seed(42)
    logits = torch.randn(4, 8, device="cuda", dtype=torch.float64, requires_grad=True)

    torch.autograd.gradcheck(
        lambda x: _pure_pytorch_topk_with_score(x, 2),
        (logits,),
        eps=1e-6,
        atol=1e-4,
    )


@_requires_cuda
def test_gradcheck_compute_scores():
    """Spec Test 5: gradcheck for Component 2 math."""
    torch.manual_seed(42)
    logits = torch.randn(4, 8, device="cuda", dtype=torch.float64, requires_grad=True)

    torch.autograd.gradcheck(
        lambda x: _pure_pytorch_compute_scores(x, 2),
        (logits,),
        eps=1e-6,
        atol=1e-4,
    )


@_requires_cuda
def test_gradcheck_aux_loss():
    """Spec Test 5: gradcheck for Component 3 math."""
    torch.manual_seed(42)
    probs = torch.rand(4, 8, device="cuda", dtype=torch.float64, requires_grad=True)
    tokens_per_expert = torch.rand(8, device="cuda", dtype=torch.float64)

    torch.autograd.gradcheck(
        lambda p: _pure_pytorch_aux_loss(p, tokens_per_expert, 4, 8, 2, 0.01),
        (probs,),
        eps=1e-6,
        atol=1e-4,
    )


# -- Test 6 --


@_requires_cuda
def test_end_to_end_patch_smoke():
    """Spec Test 6: import + patch + call all three entry points."""
    from atomrl.lumen.ops.moe.fused_router import (
        fused_compute_score_for_moe_aux_loss,
        fused_moe_aux_loss,
        fused_topk_with_score_function,
    )

    try:
        from megatron.core.transformer.moe import moe_utils
    except ImportError:
        pytest.skip("Megatron-Core not installed")

    moe_utils.fused_topk_with_score_function = fused_topk_with_score_function
    moe_utils.fused_compute_score_for_moe_aux_loss = (
        fused_compute_score_for_moe_aux_loss
    )
    moe_utils.fused_moe_aux_loss = fused_moe_aux_loss

    torch.manual_seed(42)
    logits = torch.randn(16, 8, device="cuda", dtype=torch.bfloat16, requires_grad=True)

    _routing_map, routing_probs = moe_utils.fused_topk_with_score_function(
        logits,
        2,
        True,
        None,
        None,
        None,
        "softmax",
        None,
    )
    assert routing_probs.shape == (16, 8)
    routing_probs.sum().backward()

    logits2 = torch.randn(
        16, 8, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    _r_map, scores = moe_utils.fused_compute_score_for_moe_aux_loss(
        logits2,
        2,
        "softmax",
    )
    assert scores.shape == (16, 8)

    probs = torch.rand(16, 8, device="cuda", dtype=torch.float32, requires_grad=True)
    tpe = torch.ones(8, device="cuda", dtype=torch.float32)
    loss = moe_utils.fused_moe_aux_loss(probs, tpe, 16, 8, 2, 0.01)
    assert loss.dim() == 0
    loss.backward()


# -- Test 7: AITER forward parity --


@_requires_aiter_softmax_topk
def test_aiter_softmax_topk_parity():
    """Spec Test 7a: AITER HIP softmax_topk full softmax vs PyTorch reference."""
    from atomrl.lumen.ops.moe.fused_router import _aiter_softmax_topk

    torch.manual_seed(42)
    logits = torch.randn(512, 64, device="cuda", dtype=torch.float32)

    scores_aiter, _, topk_indices_aiter = _aiter_softmax_topk(logits, 4, False)
    scores_ref = torch.softmax(logits, dim=-1)
    _, topk_indices_ref = scores_ref.topk(4, dim=-1)

    torch.testing.assert_close(scores_aiter, scores_ref, atol=1e-5, rtol=1e-5)
    for row in range(min(512, 32)):
        assert set(topk_indices_aiter[row].tolist()) == set(
            topk_indices_ref[row].tolist()
        ), f"Row {row}: expert sets differ"


@_requires_aiter_softmax_topk
def test_aiter_softmax_topk_renorm_parity():
    """Spec Test 7b: AITER softmax_topk renorm vs PyTorch reference."""
    from atomrl.lumen.ops.moe.fused_router import _aiter_softmax_topk

    torch.manual_seed(42)
    logits = torch.randn(512, 64, device="cuda", dtype=torch.float32)

    _, topk_weights, topk_indices = _aiter_softmax_topk(logits, 4, True)

    scores_ref = torch.softmax(logits, dim=-1)
    w_ref, idx_ref = scores_ref.topk(4, dim=-1)
    w_ref = w_ref / w_ref.sum(dim=-1, keepdim=True)

    for row in range(min(512, 32)):
        aiter_set = set(topk_indices[row].tolist())
        ref_set = set(idx_ref[row].tolist())
        assert aiter_set == ref_set, f"Row {row}: expert sets differ"

    torch.testing.assert_close(topk_weights, w_ref, atol=1e-4, rtol=1e-4)


def test_aiter_topk_softmax_asm_parity():
    """Spec Test 7c: existing AITER ASM topk_softmax vs PyTorch reference."""
    from aiter.ops.moe_op import topk_softmax

    torch.manual_seed(42)
    N, E, k = 512, 64, 4
    logits = torch.randn(N, E, device="cuda", dtype=torch.float32)

    topk_weights = torch.empty(N, k, device="cuda", dtype=torch.float32)
    topk_indices = torch.empty(N, k, device="cuda", dtype=torch.int32)
    token_expert_indices = torch.empty(N, k, device="cuda", dtype=torch.int32)

    topk_softmax(topk_weights, topk_indices, token_expert_indices, logits, True)

    scores_ref = torch.softmax(logits, dim=-1)
    w_ref, idx_ref = scores_ref.topk(k, dim=-1)
    w_ref = w_ref / w_ref.sum(dim=-1, keepdim=True)

    for row in range(min(N, 32)):
        assert set(topk_indices[row].tolist()) == set(
            idx_ref[row].tolist()
        ), f"Row {row}: expert sets differ"

    torch.testing.assert_close(topk_weights, w_ref, atol=1e-4, rtol=1e-4)
