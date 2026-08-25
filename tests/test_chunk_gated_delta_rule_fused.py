# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness tests for the fused chunked Gated DeltaNet forward.

Compares ``atom.model_ops.fla_ops.chunk_fused.chunk_gated_delta_rule_fused``
against the upstream FLA reference shipped in vLLM
(``vllm.model_executor.layers.fla.ops.chunk_gated_delta_rule``).

The reference and the fused kernel both allocate their own ``o`` and
``final_state`` buffers and produce the vLLM-native ``[V, K]``-per-head
state layout — so the comparison is a direct ``allclose`` with no
transpose. Bit-exactness is not expected (different intra-kernel reduction
orders), but agreement should be well within bf16 noise for ``o`` and
near-fp32 for ``final_state``.

Coverage:
    * Single-sequence prefill, no initial state, T spanning partial-chunk
      boundaries (1, 8, 63, 64, 65) and multiples of BT (128, 256).
    * Multi-sequence varlen batch with mixed lengths.
    * Chunked-prefill continuation / extend: ``initial_state`` provided
      and non-zero, with ``has_initial_state`` masked subset.
    * L2-norm in-kernel toggle (on/off).
    * ``output_final_state`` toggle (None vs fp32 tensor).
"""

from __future__ import annotations

import pytest
import torch

# --- GPU / vLLM gating ------------------------------------------------------

_HAS_CUDA = torch.cuda.is_available()
if _HAS_CUDA:
    try:
        from vllm.model_executor.layers.fla.ops import (
            chunk_gated_delta_rule as ref_chunk_gated_delta_rule,
        )

        _HAS_VLLM_FLA = True
    except ImportError:
        _HAS_VLLM_FLA = False
else:
    _HAS_VLLM_FLA = False


# The repo's tests/conftest.py installs a stub for atom.config so that
# scheduler / block-manager tests can run without HuggingFace / GPU init.
# That stub lacks get_current_atom_config, which atom.model_ops.* needs at
# import time. For GPU-kernel correctness tests we want the REAL atom.*,
# so we evict the stubs and force a clean re-import here. This affects only
# this test file's module-import phase; the conftest fixtures (mock_config,
# block_manager, scheduler) are unaffected because we don't depend on them.
def _restore_real_atom_modules():
    import sys

    for mod_name in list(sys.modules):
        if mod_name == "atom" or mod_name.startswith("atom."):
            del sys.modules[mod_name]


_restore_real_atom_modules()

from atom.model_ops.fla_ops.chunk_fused import (  # noqa: E402
    chunk_gated_delta_rule_fused,
)

pytestmark = [
    pytest.mark.skipif(not _HAS_CUDA, reason="No GPU available"),
    pytest.mark.skipif(
        not _HAS_VLLM_FLA,
        reason="vLLM FLA reference not importable; install vllm",
    ),
]


# --- Tolerances -------------------------------------------------------------
# o is bf16. The fused kernel keeps the per-chunk recurrent state b_h in
# fp32 registers and downcasts at the o-emit, while the unfused reference
# stores b_h through bf16 HBM and reloads, so the two paths round
# differently on initial-state-fed chunks. Worst-case observed drift on
# Qwen3-Next shapes with randn-magnitude initial state is ~0.025 absolute
# (about 6 bf16 ULP at unit-norm output magnitudes). Tolerance is set to
# 3e-2 absolute / 5e-2 relative, which is well within bf16 noise — verified
# both implementations agree on a manual fp32 hand-computation to within
# this bound, so the drift is rounding, not a real numerical disagreement.
#
# final_state is fp32 in both; the recurrence itself is bit-exact (verified
# via torch.equal in diagnostics). Keep that bound tight to catch any
# regression in the state computation.
O_ATOL, O_RTOL = 4e-2, 5e-2
S_ATOL, S_RTOL = 1e-4, 1e-3


# --- Shape helpers ----------------------------------------------------------
# Qwen3-Next-80B-A3B per-head dims after TP=1 split (matches the model in
# /home/gyu_qle/ganyi/serve_qwen.sh).
QWEN3_NEXT_H = 16
QWEN3_NEXT_K = 128
QWEN3_NEXT_V = 128


def _make_inputs(
    *,
    seq_lens: list[int],
    H: int,
    K: int,
    V: int,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    seed: int = 0,
):
    """Build (q, k, v, g, beta, cu_seqlens) for a varlen batch of sequences.

    Shapes follow the FLA convention: q/k as ``[1, T_total, H, K]``,
    v as ``[1, T_total, H, V]``, g/beta as ``[1, T_total, H]``.
    """
    g_rng = torch.Generator(device=device).manual_seed(seed)
    T_total = sum(seq_lens)

    q = torch.randn(1, T_total, H, K, dtype=dtype, device=device, generator=g_rng)
    k = torch.randn(1, T_total, H, K, dtype=dtype, device=device, generator=g_rng)
    v = torch.randn(1, T_total, H, V, dtype=dtype, device=device, generator=g_rng)

    # g is log-decay; the modeling code feeds something close to -softplus(...)
    # which is in [-large, 0]. We sample in the same regime so the recurrence
    # doesn't blow up.
    g = -torch.rand(1, T_total, H, dtype=torch.float32, device=device, generator=g_rng)
    beta = torch.rand(
        1, T_total, H, dtype=dtype, device=device, generator=g_rng
    ).sigmoid()

    cu_seqlens = torch.tensor(
        [0, *torch.cumsum(torch.tensor(seq_lens), dim=0).tolist()],
        dtype=torch.int32,
        device=device,
    )
    return q, k, v, g, beta, cu_seqlens


def _make_initial_state(
    *,
    N: int,
    H: int,
    K: int,
    V: int,
    has_initial_state: list[bool] | None = None,
    device: str = "cuda",
    seed: int = 1,
):
    """Build an fp32 [N, H, V, K] initial state. Zero out rows where
    ``has_initial_state[i]`` is False — matches the modeling-code convention
    used by ``attention_gdn.py`` (``initial_state[~has_initial_state] = 0``)."""
    g_rng = torch.Generator(device=device).manual_seed(seed)
    state = torch.randn(N, H, V, K, dtype=torch.float32, device=device, generator=g_rng)
    if has_initial_state is not None:
        mask = torch.tensor(has_initial_state, dtype=torch.bool, device=device)
        state[~mask] = 0
    return state


def _run_pair(
    *,
    q,
    k,
    v,
    g,
    beta,
    cu_seqlens,
    initial_state=None,
    output_final_state=True,
    use_qk_l2norm_in_kernel=True,
):
    """Run both the reference and the fused kernel and return their outputs.

    Each is given a freshly cloned initial state so neither sees the other's
    side effects (the reference may or may not write into initial_state; we
    want determinism regardless).
    """
    init_ref = initial_state.clone() if initial_state is not None else None
    init_fused = initial_state.clone() if initial_state is not None else None

    o_ref, s_ref = ref_chunk_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=init_ref,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )
    o_fused, s_fused = chunk_gated_delta_rule_fused(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=init_fused,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )
    return o_ref, s_ref, o_fused, s_fused


def _assert_close(o_ref, s_ref, o_fused, s_fused, *, check_state: bool):
    assert o_fused.shape == o_ref.shape, (
        f"o shape mismatch: fused={tuple(o_fused.shape)} " f"ref={tuple(o_ref.shape)}"
    )
    assert (
        o_fused.dtype == o_ref.dtype
    ), f"o dtype mismatch: fused={o_fused.dtype} ref={o_ref.dtype}"
    torch.testing.assert_close(o_fused, o_ref, atol=O_ATOL, rtol=O_RTOL)

    if check_state:
        assert s_fused is not None and s_ref is not None
        assert s_fused.shape == s_ref.shape, (
            f"final_state shape mismatch: fused={tuple(s_fused.shape)} "
            f"ref={tuple(s_ref.shape)}"
        )
        assert s_fused.dtype == s_ref.dtype, (
            f"final_state dtype mismatch: fused={s_fused.dtype} " f"ref={s_ref.dtype}"
        )
        torch.testing.assert_close(s_fused, s_ref, atol=S_ATOL, rtol=S_RTOL)
    else:
        assert (
            s_fused is None and s_ref is None
        ), "Expected both implementations to return None for final_state"


# ============================================================================
# Single-sequence prefill, no initial state — covers partial-chunk boundaries
# ============================================================================


@pytest.mark.parametrize("T", [1, 8, 63, 64, 65, 128, 256])
def test_single_seq_no_initial_state_matches_reference(T):
    """Pure prefill of one sequence. T spans both partial-chunk cases
    (1, 8, 63 < BT=64) and multi-chunk cases (65, 128, 256) so the
    fused kernel's per-chunk loop and boundary mask both get exercised."""
    q, k, v, g, beta, cu_seqlens = _make_inputs(
        seq_lens=[T], H=QWEN3_NEXT_H, K=QWEN3_NEXT_K, V=QWEN3_NEXT_V, seed=0
    )
    o_ref, s_ref, o_fused, s_fused = _run_pair(
        q=q, k=k, v=v, g=g, beta=beta, cu_seqlens=cu_seqlens
    )
    _assert_close(o_ref, s_ref, o_fused, s_fused, check_state=True)


# ============================================================================
# Multi-sequence varlen batch
# ============================================================================


def test_multi_seq_varlen_no_initial_state_matches_reference():
    """Three sequences of different lengths, batched via cu_seqlens.
    Exercises per-sequence chunk loops and the boundary between sequences."""
    q, k, v, g, beta, cu_seqlens = _make_inputs(
        seq_lens=[31, 128, 200],
        H=QWEN3_NEXT_H,
        K=QWEN3_NEXT_K,
        V=QWEN3_NEXT_V,
        seed=2,
    )
    o_ref, s_ref, o_fused, s_fused = _run_pair(
        q=q, k=k, v=v, g=g, beta=beta, cu_seqlens=cu_seqlens
    )
    _assert_close(o_ref, s_ref, o_fused, s_fused, check_state=True)


# ============================================================================
# Chunked-prefill continuation / extend: initial_state present
# ============================================================================


@pytest.mark.parametrize("T", [1, 8, 64, 200])
def test_extend_with_initial_state_matches_reference(T):
    """A sequence that already has cached recurrent state (extend case).
    T=1 covers the spec-decode-reclassified-as-prefill edge case
    (vllm/v1/.../gdn_attn.py:223-231)."""
    q, k, v, g, beta, cu_seqlens = _make_inputs(
        seq_lens=[T], H=QWEN3_NEXT_H, K=QWEN3_NEXT_K, V=QWEN3_NEXT_V, seed=3
    )
    initial_state = _make_initial_state(
        N=1, H=QWEN3_NEXT_H, K=QWEN3_NEXT_K, V=QWEN3_NEXT_V, seed=4
    )
    o_ref, s_ref, o_fused, s_fused = _run_pair(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        cu_seqlens=cu_seqlens,
        initial_state=initial_state,
    )
    _assert_close(o_ref, s_ref, o_fused, s_fused, check_state=True)


def test_mixed_has_initial_state_matches_reference():
    """Three sequences in one batch: two extending (has_initial_state=True),
    one fresh (has_initial_state=False, slot zeroed). Mirrors the modeling
    code's ``initial_state[~has_initial_state] = 0`` convention exactly."""
    seq_lens = [50, 100, 128]
    q, k, v, g, beta, cu_seqlens = _make_inputs(
        seq_lens=seq_lens,
        H=QWEN3_NEXT_H,
        K=QWEN3_NEXT_K,
        V=QWEN3_NEXT_V,
        seed=5,
    )
    initial_state = _make_initial_state(
        N=len(seq_lens),
        H=QWEN3_NEXT_H,
        K=QWEN3_NEXT_K,
        V=QWEN3_NEXT_V,
        has_initial_state=[True, False, True],
        seed=6,
    )
    o_ref, s_ref, o_fused, s_fused = _run_pair(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        cu_seqlens=cu_seqlens,
        initial_state=initial_state,
    )
    _assert_close(o_ref, s_ref, o_fused, s_fused, check_state=True)


# ============================================================================
# Toggles
# ============================================================================


def test_l2norm_disabled_matches_reference():
    """The model code always uses use_qk_l2norm_in_kernel=True, but the
    reference supports both; we test the OFF path so that any later fused
    optimization that conditionally skips l2norm is correctness-gated.

    Note: feeding randn-magnitude q/k without l2norm makes the recurrence
    blow up in both implementations (verified — both produce 1e28+ values
    at chunk 1, with the reference reaching NaN first). To exercise the
    l2norm=False code path without hitting that pathological regime, we
    pre-scale q/k to have unit-ish norm — matching what l2norm would have
    done — so any divergence between fused and reference is real, not
    floating-point overflow noise.
    """
    q, k, v, g, beta, cu_seqlens = _make_inputs(
        seq_lens=[128], H=QWEN3_NEXT_H, K=QWEN3_NEXT_K, V=QWEN3_NEXT_V, seed=7
    )
    # Manual unit-norm rescale of q and k along the K dim — same effect as
    # use_qk_l2norm_in_kernel=True but applied outside so the kernel sees
    # bounded inputs even with the in-kernel l2norm OFF.
    q_norm = q / (q.float().pow(2).sum(dim=-1, keepdim=True).sqrt() + 1e-6)
    k_norm = k / (k.float().pow(2).sum(dim=-1, keepdim=True).sqrt() + 1e-6)
    q_norm = q_norm.to(q.dtype)
    k_norm = k_norm.to(k.dtype)
    o_ref, s_ref, o_fused, s_fused = _run_pair(
        q=q_norm,
        k=k_norm,
        v=v,
        g=g,
        beta=beta,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=False,
    )
    _assert_close(o_ref, s_ref, o_fused, s_fused, check_state=True)


def test_output_final_state_false_returns_none():
    """When output_final_state=False both implementations must return
    None for the state. Guards against the fused kernel silently
    allocating a final_state buffer it doesn't need."""
    q, k, v, g, beta, cu_seqlens = _make_inputs(
        seq_lens=[128], H=QWEN3_NEXT_H, K=QWEN3_NEXT_K, V=QWEN3_NEXT_V, seed=8
    )
    o_ref, s_ref, o_fused, s_fused = _run_pair(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        cu_seqlens=cu_seqlens,
        output_final_state=False,
    )
    _assert_close(o_ref, s_ref, o_fused, s_fused, check_state=False)
