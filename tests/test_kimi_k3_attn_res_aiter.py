#!/usr/bin/env python3
"""Offline gate for routing Kimi-K3 apply_attn_res to aiter, no model.

ATOM_USE_AITER_ATTN_RES swaps the in-tree Triton kernels for
aiter.ops.triton.fusions.attn_res.attn_res_gate. The env is read at import
time, so instead of re-importing under a different environment these tests
drive _apply_attn_res_impl directly and patch the module-level attn_res_gate
symbol to pick a backend, then require the two to agree.

Token counts cover both in-tree config buckets (split-H at T <= 128, the fused
two-pass kernel above it) so the comparison exercises every dispatch branch.
"""

import pytest
import torch

pytest.importorskip("aiter")

try:
    from atom.model_ops.kimi_k3 import attention_residual as ar
except Exception as _e:  # pre-existing atom.config circular import under bare pytest
    pytest.skip(f"requires full atom import env: {_e}", allow_module_level=True)

from aiter.ops.triton.fusions.attn_res import attn_res_gate

# Both backends accumulate in fp32, but aiter reduces each H row in one shot
# where the in-tree kernel tiles H (and splits it across workgroups at small T),
# so the sums land differently and the two are not bitwise equal. (The exp2-vs-exp
# softmax difference is NOT a factor: tl.exp lowers to a log2e multiply plus the
# same v_exp_f32, so both spellings emit identical arithmetic on this backend.)
# Measured worst case over T <= 1024, B <= 8, H = 2048: fp32 is
# exact at small T and drifts to 3.2e-5 absolute; bf16 lands within one ULP of
# the output magnitude (1.6e-2 at |y| ~ 2). Relative error is not a useful bound
# here because the mixed residual has near-zero channels where the weighted sum
# cancels, so these are absolute-dominated tolerances.
_TOL = {
    torch.float32: (1e-4, 1e-4),
    torch.bfloat16: (2e-2, 2e-2),
}


def _inputs(T, B, H, dtype, with_add, seed=0):
    torch.manual_seed(seed)
    prefix = torch.randn(T, H, dtype=dtype, device="cuda")
    block_residual = torch.randn(T, B, H, dtype=dtype, device="cuda")
    # score_weight is fp32 in the model (norm.weight * proj.weight, folded once).
    score_weight = torch.randn(H, dtype=torch.float32, device="cuda")
    add_hidden = torch.randn(T, H, dtype=dtype, device="cuda") if with_add else None
    return prefix, block_residual, score_weight, add_hidden


def _run(backend, prefix, block_residual, score_weight, eps, add_hidden, monkeypatch):
    monkeypatch.setattr(ar, "attn_res_gate", backend)
    return ar._apply_attn_res_impl(
        prefix, block_residual, score_weight, eps, add_hidden
    )


@pytest.mark.parametrize("T", [1, 8, 64, 128, 256, 512])
@pytest.mark.parametrize("B", [1, 8])
@pytest.mark.parametrize("with_add", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_aiter_matches_in_tree(T, B, with_add, dtype, monkeypatch):
    H, eps = 2048, 1e-6
    prefix, br, sw, add = _inputs(T, B, H, dtype, with_add)

    y_tree, pref_tree = _run(None, prefix, br, sw, eps, add, monkeypatch)
    y_aiter, pref_aiter = _run(attn_res_gate, prefix, br, sw, eps, add, monkeypatch)

    atol, rtol = _TOL[dtype]
    torch.testing.assert_close(y_aiter.float(), y_tree.float(), atol=atol, rtol=rtol)
    torch.testing.assert_close(
        pref_aiter.float(), pref_tree.float(), atol=atol, rtol=rtol
    )


def test_aiter_matches_in_tree_model_hidden_size(monkeypatch):
    """Same comparison at the real K3 hidden size."""
    T, B, H, eps = 64, 8, 7168, 1e-6
    prefix, br, sw, add = _inputs(T, B, H, torch.bfloat16, with_add=True)

    y_tree, pref_tree = _run(None, prefix, br, sw, eps, add, monkeypatch)
    y_aiter, pref_aiter = _run(attn_res_gate, prefix, br, sw, eps, add, monkeypatch)

    atol, rtol = _TOL[torch.bfloat16]
    torch.testing.assert_close(y_aiter.float(), y_tree.float(), atol=atol, rtol=rtol)
    torch.testing.assert_close(
        pref_aiter.float(), pref_tree.float(), atol=atol, rtol=rtol
    )


_BACKENDS = pytest.mark.parametrize(
    "backend", [None, attn_res_gate], ids=["in_tree", "aiter"]
)


@_BACKENDS
def test_prefix_passthrough_without_add(backend, monkeypatch):
    """Without add_hidden both backends hand the caller's prefix straight back."""
    prefix, br, sw, _ = _inputs(32, 4, 2048, torch.bfloat16, with_add=False)
    prefix_copy = prefix.clone()

    _y, prefix_out = _run(backend, prefix, br, sw, 1e-6, None, monkeypatch)

    assert prefix_out is prefix
    torch.testing.assert_close(prefix, prefix_copy, atol=0, rtol=0)


@_BACKENDS
def test_inputs_not_mutated(backend, monkeypatch):
    """The summed prefix goes to a fresh tensor; the caller's inputs are intact."""
    prefix, br, sw, add = _inputs(32, 4, 2048, torch.bfloat16, with_add=True)
    prefix_copy, br_copy, add_copy = prefix.clone(), br.clone(), add.clone()

    _y, prefix_out = _run(backend, prefix, br, sw, 1e-6, add, monkeypatch)

    assert prefix_out is not prefix
    torch.testing.assert_close(prefix, prefix_copy, atol=0, rtol=0)
    torch.testing.assert_close(br, br_copy, atol=0, rtol=0)
    torch.testing.assert_close(add, add_copy, atol=0, rtol=0)


def test_public_api_shapes(monkeypatch):
    """apply_attn_res keeps its (y, prefix_out) contract on the aiter path."""
    T, B, H = 16, 3, 2048
    prefix, br, sw, add = _inputs(T, B, H, torch.bfloat16, with_add=True)
    monkeypatch.setattr(ar, "attn_res_gate", attn_res_gate)

    y, prefix_out = ar.apply_attn_res(prefix, br, sw, 1e-6, add)

    assert y.shape == (T, H) and y.dtype == prefix.dtype
    assert prefix_out.shape == (T, H) and prefix_out.dtype == prefix.dtype
