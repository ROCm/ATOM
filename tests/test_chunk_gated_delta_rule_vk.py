# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness tests for the ATOM-vendored vk-layout chunk_gated_delta_rule.

Compares ``atom.model_ops.fla_ops.chunk_vk.chunk_gated_delta_rule_vk``
against the upstream FLA reference shipped in vLLM
(``vllm.model_executor.layers.fla.ops.chunk_gated_delta_rule``).

Because the ATOM file is a **verbatim port** of the vLLM source — same
kernel code, same prologue, same algorithm, only the import lines and
public symbol names differ — we expect **bit-exact** output agreement on
the same inputs. Both implementations should produce identical `o` and
identical `final_state` byte-for-byte. A non-zero difference would
indicate a port mistake.

Coverage mirrors the earlier Phase-1 fused-kernel test suite so that any
regression of the additive port can be caught in the same shapes the model
actually exercises (single-seq prefill across partial-chunk boundaries,
multi-seq varlen, chunked-prefill continuation / extend with initial_state,
l2norm toggle, output_final_state toggle).
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
# so we evict the stubs and force a clean re-import here.
def _restore_real_atom_modules():
    import sys

    for mod_name in list(sys.modules):
        if mod_name == "atom" or mod_name.startswith("atom."):
            del sys.modules[mod_name]


_restore_real_atom_modules()

from atom.model_ops.fla_ops.chunk_vk import (  # noqa: E402
    chunk_gated_delta_rule_vk,
)

pytestmark = [
    pytest.mark.skipif(not _HAS_CUDA, reason="No GPU available"),
    pytest.mark.skipif(
        not _HAS_VLLM_FLA,
        reason="vLLM FLA reference not importable; install vllm",
    ),
]


# --- Tolerances -------------------------------------------------------------
# This is a verbatim port of the vLLM kernel — same algorithm, same kernel
# code, same prologue, same reduction order. Outputs should agree to the
# last bit. Use torch.equal-strength tolerances; any drift = port bug.
O_ATOL, O_RTOL = 0.0, 0.0
S_ATOL, S_RTOL = 0.0, 0.0


# --- Shape helpers ----------------------------------------------------------
# Qwen3-Next-80B-A3B per-head dims after TP=1 split.
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
    g_rng = torch.Generator(device=device).manual_seed(seed)
    T_total = sum(seq_lens)

    q = torch.randn(1, T_total, H, K, dtype=dtype, device=device, generator=g_rng)
    k = torch.randn(1, T_total, H, K, dtype=dtype, device=device, generator=g_rng)
    v = torch.randn(1, T_total, H, V, dtype=dtype, device=device, generator=g_rng)
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
    """Build an fp32 [N, H, V, K] initial state (vk layout per slot)."""
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
    """Run both reference and ATOM-vendored vk version on identical clones."""
    init_ref = initial_state.clone() if initial_state is not None else None
    init_atom = initial_state.clone() if initial_state is not None else None

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
    o_atom, s_atom = chunk_gated_delta_rule_vk(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=init_atom,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )
    return o_ref, s_ref, o_atom, s_atom


def _assert_bit_equal(o_ref, s_ref, o_atom, s_atom, *, check_state: bool):
    assert o_atom.shape == o_ref.shape, (
        f"o shape mismatch: atom={tuple(o_atom.shape)} " f"ref={tuple(o_ref.shape)}"
    )
    assert (
        o_atom.dtype == o_ref.dtype
    ), f"o dtype mismatch: atom={o_atom.dtype} ref={o_ref.dtype}"
    # Verbatim port → bit-exact expected. torch.equal returns False on any
    # difference (NaN-aware: NaN != NaN per IEEE).
    if not torch.equal(o_atom, o_ref):
        diff = (o_atom.float() - o_ref.float()).abs()
        raise AssertionError(
            f"o not bit-exact: max abs diff = {diff.max().item():.6g}, "
            f"mismatched = {(o_atom != o_ref).sum().item()}/{o_ref.numel()}"
        )

    if check_state:
        assert s_atom is not None and s_ref is not None
        assert s_atom.shape == s_ref.shape, (
            f"final_state shape mismatch: atom={tuple(s_atom.shape)} "
            f"ref={tuple(s_ref.shape)}"
        )
        assert s_atom.dtype == s_ref.dtype, (
            f"final_state dtype mismatch: atom={s_atom.dtype} " f"ref={s_ref.dtype}"
        )
        if not torch.equal(s_atom, s_ref):
            diff = (s_atom - s_ref).abs()
            raise AssertionError(
                f"final_state not bit-exact: max abs diff = "
                f"{diff.max().item():.6g}, mismatched = "
                f"{(s_atom != s_ref).sum().item()}/{s_ref.numel()}"
            )
    else:
        assert (
            s_atom is None and s_ref is None
        ), "Expected both implementations to return None for final_state"


# ============================================================================
# Single-sequence prefill, no initial state
# ============================================================================


@pytest.mark.parametrize("T", [1, 8, 63, 64, 65, 128, 256])
def test_single_seq_no_initial_state_bit_exact(T):
    q, k, v, g, beta, cu_seqlens = _make_inputs(
        seq_lens=[T], H=QWEN3_NEXT_H, K=QWEN3_NEXT_K, V=QWEN3_NEXT_V, seed=0
    )
    o_ref, s_ref, o_atom, s_atom = _run_pair(
        q=q, k=k, v=v, g=g, beta=beta, cu_seqlens=cu_seqlens
    )
    _assert_bit_equal(o_ref, s_ref, o_atom, s_atom, check_state=True)


# ============================================================================
# Multi-sequence varlen batch
# ============================================================================


def test_multi_seq_varlen_no_initial_state_bit_exact():
    q, k, v, g, beta, cu_seqlens = _make_inputs(
        seq_lens=[31, 128, 200],
        H=QWEN3_NEXT_H,
        K=QWEN3_NEXT_K,
        V=QWEN3_NEXT_V,
        seed=2,
    )
    o_ref, s_ref, o_atom, s_atom = _run_pair(
        q=q, k=k, v=v, g=g, beta=beta, cu_seqlens=cu_seqlens
    )
    _assert_bit_equal(o_ref, s_ref, o_atom, s_atom, check_state=True)


# ============================================================================
# Chunked-prefill continuation / extend
# ============================================================================


@pytest.mark.parametrize("T", [1, 8, 64, 200])
def test_extend_with_initial_state_bit_exact(T):
    q, k, v, g, beta, cu_seqlens = _make_inputs(
        seq_lens=[T], H=QWEN3_NEXT_H, K=QWEN3_NEXT_K, V=QWEN3_NEXT_V, seed=3
    )
    initial_state = _make_initial_state(
        N=1, H=QWEN3_NEXT_H, K=QWEN3_NEXT_K, V=QWEN3_NEXT_V, seed=4
    )
    o_ref, s_ref, o_atom, s_atom = _run_pair(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        cu_seqlens=cu_seqlens,
        initial_state=initial_state,
    )
    _assert_bit_equal(o_ref, s_ref, o_atom, s_atom, check_state=True)


def test_mixed_has_initial_state_bit_exact():
    """Mirrors the modeling-code convention: some slots have prior state,
    some are freshly-allocated (zeroed via `initial_state[~mask] = 0`)."""
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
    o_ref, s_ref, o_atom, s_atom = _run_pair(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        cu_seqlens=cu_seqlens,
        initial_state=initial_state,
    )
    _assert_bit_equal(o_ref, s_ref, o_atom, s_atom, check_state=True)


# ============================================================================
# Toggles
# ============================================================================


def test_l2norm_disabled_bit_exact():
    """Verify the l2norm-off path. We pre-rescale q/k to unit norm because
    randn-magnitude unnormalized inputs send the GDN recurrence into a
    numerically explosive regime in both implementations — see the
    Phase-1-fused tests for the diagnostic. Pre-rescaling exercises the
    OFF code path without exposing both kernels to overflow."""
    q, k, v, g, beta, cu_seqlens = _make_inputs(
        seq_lens=[128], H=QWEN3_NEXT_H, K=QWEN3_NEXT_K, V=QWEN3_NEXT_V, seed=7
    )
    q_norm = q / (q.float().pow(2).sum(dim=-1, keepdim=True).sqrt() + 1e-6)
    k_norm = k / (k.float().pow(2).sum(dim=-1, keepdim=True).sqrt() + 1e-6)
    q_norm = q_norm.to(q.dtype)
    k_norm = k_norm.to(k.dtype)
    o_ref, s_ref, o_atom, s_atom = _run_pair(
        q=q_norm,
        k=k_norm,
        v=v,
        g=g,
        beta=beta,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=False,
    )
    _assert_bit_equal(o_ref, s_ref, o_atom, s_atom, check_state=True)


# ============================================================================
# Inplace o= parameter
# ============================================================================


def test_inplace_o_writes_into_caller_buffer():
    """When o= is passed, the returned tensor must be the SAME storage as
    the caller's buffer (data_ptr identity), and the numerics must agree
    bit-exactly with the alloc-internally path."""
    q, k, v, g, beta, cu_seqlens = _make_inputs(
        seq_lens=[128], H=QWEN3_NEXT_H, K=QWEN3_NEXT_K, V=QWEN3_NEXT_V, seed=11
    )
    # Reference path: kernel allocates o internally.
    o_alloc, s_alloc = chunk_gated_delta_rule_vk(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        cu_seqlens=cu_seqlens,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )
    # Inplace path: caller provides a buffer of matching shape/dtype.
    o_buf = torch.empty_like(v)
    o_buf_ptr = o_buf.data_ptr()
    o_inplace, s_inplace = chunk_gated_delta_rule_vk(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        cu_seqlens=cu_seqlens,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        o=o_buf,
    )
    # Storage identity: returned tensor must be backed by caller's buffer.
    assert o_inplace.data_ptr() == o_buf_ptr, (
        f"inplace o broken: returned data_ptr={o_inplace.data_ptr():x} "
        f"!= caller buffer data_ptr={o_buf_ptr:x}"
    )
    # Numerical identity: same bytes as the alloc-internally version.
    if not torch.equal(o_inplace, o_alloc):
        diff = (o_inplace.float() - o_alloc.float()).abs()
        raise AssertionError(
            f"inplace o numerically differs from alloc-internally: "
            f"max abs diff = {diff.max().item():.6g}"
        )
    # final_state should agree too.
    assert torch.equal(s_inplace, s_alloc), (
        "final_state changed depending on whether o was inplace — "
        "shouldn't happen: o is only the last kernel's output, the "
        "recurrence runs identically."
    )


def test_inplace_o_rejects_non_contiguous():
    """A non-contiguous o would be silently cloned by input_guard's
    .contiguous() call, defeating the inplace contract. The kernel host
    must catch this with an explicit assert so the failure mode is loud."""
    q, k, v, g, beta, cu_seqlens = _make_inputs(
        seq_lens=[128], H=QWEN3_NEXT_H, K=QWEN3_NEXT_K, V=QWEN3_NEXT_V, seed=12
    )
    # Build a non-contiguous view of a buffer twice the V dim — slicing
    # along the last dim breaks the (H*V, 1) stride contract.
    backing = torch.empty(
        v.shape[0],
        v.shape[1],
        v.shape[2],
        v.shape[3] * 2,
        dtype=v.dtype,
        device=v.device,
    )
    non_contig_o = backing[..., : v.shape[3]]
    assert (
        not non_contig_o.is_contiguous()
    ), "test setup wrong: expected non-contiguous slice"
    with pytest.raises(AssertionError, match="contiguous"):
        chunk_gated_delta_rule_vk(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            cu_seqlens=cu_seqlens,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            o=non_contig_o,
        )


def test_modeling_code_integration_pattern_bit_exact():
    """Mirrors the exact prefill call pattern used by
    atom/plugin/vllm/attention_backend/attention_gdn.py:

      # allocate output buffer shaped like the modeling-code's core_attn_out
      core_attn_out = torch.empty(num_tokens_total, H, V, dtype=bf16)
      ...
      # OLD path (allocate + copy):
      o_alloc, _ = chunk_gated_delta_rule_vk(..., o=None)
      core_attn_out[:N] = o_alloc.squeeze(0)
      # NEW path (inplace):
      o_buf = core_attn_out[:N].unsqueeze(0)
      _, _ = chunk_gated_delta_rule_vk(..., o=o_buf)

    Both must leave core_attn_out byte-identical AND final_state byte-identical.
    This is the integration-level safety net for the modeling-code wiring."""
    N_actual = 128  # tokens for the lone "sequence"
    N_padding = 32  # padding rows past num_actual_tokens
    num_tokens_total = N_actual + N_padding

    q, k, v, g, beta, cu_seqlens = _make_inputs(
        seq_lens=[N_actual],
        H=QWEN3_NEXT_H,
        K=QWEN3_NEXT_K,
        V=QWEN3_NEXT_V,
        seed=21,
    )
    initial_state = _make_initial_state(
        N=1, H=QWEN3_NEXT_H, K=QWEN3_NEXT_K, V=QWEN3_NEXT_V, seed=22
    )

    # --- OLD path: allocate output internally, then copy slice into a
    # caller-managed buffer shaped like the modeling-code's core_attn_out.
    core_attn_out_old = torch.empty(
        num_tokens_total,
        QWEN3_NEXT_H,
        QWEN3_NEXT_V,
        dtype=torch.bfloat16,
        device="cuda",
    )
    # The modeling code does NOT initialize the padding tail before the
    # kernel call — it zeroes it afterward — so don't pre-init here either.
    o_alloc, s_alloc = chunk_gated_delta_rule_vk(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=initial_state.clone(),
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=True,
    )
    core_attn_out_old[:N_actual] = o_alloc.squeeze(0)

    # --- NEW path: pass the slice as the inplace target. Skip the copy.
    core_attn_out_new = torch.empty(
        num_tokens_total,
        QWEN3_NEXT_H,
        QWEN3_NEXT_V,
        dtype=torch.bfloat16,
        device="cuda",
    )
    o_buf = core_attn_out_new[:N_actual].unsqueeze(0)
    o_inplace, s_inplace = chunk_gated_delta_rule_vk(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=initial_state.clone(),
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=True,
        o=o_buf,
    )
    # The returned o must be backed by the caller's buffer (not a clone).
    assert o_inplace.data_ptr() == o_buf.data_ptr(), (
        f"inplace contract broken: returned data_ptr "
        f"{o_inplace.data_ptr():x} != caller buffer {o_buf.data_ptr():x}"
    )

    # --- Bit-exact agreement on the actual-token slice. We don't compare
    # the padding tail because OLD leaves it uninitialized while NEW
    # leaves it uninitialized too (modeling code zeroes it at end of
    # function, outside the kernel) — so neither has defined contents there.
    if not torch.equal(core_attn_out_old[:N_actual], core_attn_out_new[:N_actual]):
        diff = (
            core_attn_out_old[:N_actual].float() - core_attn_out_new[:N_actual].float()
        ).abs()
        raise AssertionError(
            f"integration pattern broke bit-equality: max abs diff = "
            f"{diff.max().item():.6g}"
        )
    assert torch.equal(
        s_inplace, s_alloc
    ), "final_state differs between allocate+copy and inplace paths"


def test_output_final_state_false_returns_none():
    q, k, v, g, beta, cu_seqlens = _make_inputs(
        seq_lens=[128], H=QWEN3_NEXT_H, K=QWEN3_NEXT_K, V=QWEN3_NEXT_V, seed=8
    )
    o_ref, s_ref, o_atom, s_atom = _run_pair(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        cu_seqlens=cu_seqlens,
        output_final_state=False,
    )
    _assert_bit_equal(o_ref, s_ref, o_atom, s_atom, check_state=False)


# ============================================================================
# Fused SSM-state gather/scatter (continuous-batching path)
# ============================================================================
#
# Bit-exactness caveat: comparing the fused indexed path against the
# explicit `gather → call → scatter` reference calls the SAME wrapper twice
# with two *different specializations* of the underlying Triton kernel
# (different constexpr / dtype-list keys → potentially different
# autotune-selected BV / num_warps). The kernel's bf16 matmul tile-shape
# changes with BV, which produces small but non-zero numerical drift in
# the intermediate `h` buffer and consequently in `o`. The final fp32 state
# converges symmetrically so its rounding noise stays near machine epsilon.
#
# We assert:
#   - bit-exact agreement on the *final cache buffer* (fp32-equivalent path),
#   - tight numerical (not bit-exact) agreement on `o`,
#   - bit-exact agreement on untouched slots (the scatter must not corrupt
#     unrelated rows of the cache).


def _per_slot_diff(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Max-abs diff per slot index for a `[num_slots, ...]` tensor."""
    return (a.float() - b.float()).abs().flatten(1).max(dim=1).values


def test_fused_ssm_state_indexed_path_matches_explicit():
    """The new ssm_state_indices / has_initial_state / inplace_final_state
    code path fuses what the plugin's GDN backend used to do in Python:

        initial_state = ssm_state[non_spec_state_indices_tensor].contiguous()
        initial_state[~has_initial_state, ...] = 0
        o, final_state = chunk_gated_delta_rule_vk(..., initial_state=initial_state, ...)
        ssm_state[non_spec_state_indices_tensor] = final_state.to(ssm_state.dtype)

    Run that explicit gather→call→scatter on a CLONE of the cache buffer
    (the OLD path) and the fused indexed path on the ORIGINAL (the NEW
    path). Verify (a) numerical agreement on o (within bf16 autotune
    noise), (b) bit-exact agreement on the written cache slots (final
    state converges in fp32), and (c) bit-exact agreement on UNTOUCHED
    slots — the scatter must not corrupt unrelated rows of the cache.
    """
    seq_lens = [50, 100, 128]
    H, K, V = QWEN3_NEXT_H, QWEN3_NEXT_K, QWEN3_NEXT_V

    q, k, v, g, beta, cu_seqlens = _make_inputs(
        seq_lens=seq_lens, H=H, K=K, V=V, seed=30
    )

    # Build a `[num_slots, H, V, K]` cache larger than the batch's sequence
    # count, so untouched rows can be checked for non-corruption. Use bf16
    # to match the real cache dtype (forces the kernel's fp32→bf16 cast on
    # the inplace scatter, mirroring `.to(ssm_state.dtype)` semantics).
    num_slots = 16
    g_rng = torch.Generator(device="cuda").manual_seed(31)
    cache = torch.randn(
        num_slots, H, V, K, dtype=torch.bfloat16, device="cuda", generator=g_rng
    )

    # Pick non-contiguous slot ids out of order to make sure the kernel
    # really uses the lookup table and not e.g. arange(N).
    slot_ids = torch.tensor([7, 2, 11], dtype=torch.int32, device="cuda")
    has_init = torch.tensor([True, False, True], dtype=torch.bool, device="cuda")

    # --- OLD path: explicit gather + masked-fill + call + scatter ---
    cache_old = cache.clone()
    initial_state_old = (
        cache_old[slot_ids.to(torch.long)].contiguous().to(torch.float32)
    )
    initial_state_old[~has_init, ...] = 0
    o_old, s_old = chunk_gated_delta_rule_vk(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=initial_state_old,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=True,
    )
    cache_old[slot_ids.to(torch.long)] = s_old.to(cache_old.dtype)

    # --- NEW path: fused indexed call writes straight into the cache ---
    cache_new = cache.clone()
    o_new, s_new = chunk_gated_delta_rule_vk(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=cache_new,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=True,
        ssm_state_indices=slot_ids,
        has_initial_state=has_init,
        inplace_final_state=True,
    )

    # (0) final_state IS the cache buffer under inplace_final_state.
    assert s_new.data_ptr() == cache_new.data_ptr(), (
        "inplace_final_state=True must return the cache buffer as final_state, "
        f"got data_ptr {s_new.data_ptr():x} != cache {cache_new.data_ptr():x}"
    )

    # (a) o numerical agreement within bf16 autotune-config noise.
    o_diff = (o_new.float() - o_old.float()).abs().max().item()
    assert o_diff < 0.5, (
        f"fused indexed o diff {o_diff:.6g} too large vs explicit gather/scatter "
        f"(bf16 autotune-config noise expected to be O(0.1), large diff indicates "
        f"a real correctness bug in the indexed path)"
    )

    # (b) written-slot bit-exactness: both paths must write the same fp32
    #     final state to the cache (cast to bf16 once, on the kernel store).
    written = slot_ids.to(torch.long)
    if not torch.equal(cache_new[written], cache_old[written]):
        diff = (cache_new[written].float() - cache_old[written].float()).abs()
        raise AssertionError(
            f"written cache slots not bit-exact: max abs diff = "
            f"{diff.max().item():.6g}; the fused-scatter kernel store and the "
            f"Python-side `cache[ids] = state.to(bf16)` should produce the "
            f"same bf16 result."
        )

    # (c) untouched slots must be byte-identical — scatter cannot corrupt
    #     unrelated rows.
    all_idx = torch.arange(num_slots, device="cuda")
    untouched = all_idx[~torch.isin(all_idx, written)]
    if not torch.equal(cache_new[untouched], cache_old[untouched]):
        diff = _per_slot_diff(cache_new[untouched], cache_old[untouched])
        bad = untouched[diff > 0].tolist()
        raise AssertionError(
            f"untouched cache slots corrupted by fused scatter: "
            f"slots {bad} differ from the original (max abs diff "
            f"{diff.max().item():.6g})"
        )


def test_fused_ssm_state_indexed_matches_vllm_reference():
    """Cross-check the indexed path against the vLLM upstream reference
    (which has no fused gather/scatter — caller does it). This is the
    authoritative correctness test: any drift here means the indexed
    kernel's algorithm itself disagrees with upstream FLA, not just
    autotune noise vs the OLD wrapper path."""
    seq_lens = [50, 100, 128]
    H, K, V = QWEN3_NEXT_H, QWEN3_NEXT_K, QWEN3_NEXT_V
    q, k, v, g, beta, cu_seqlens = _make_inputs(
        seq_lens=seq_lens, H=H, K=K, V=V, seed=33
    )

    num_slots = 16
    g_rng = torch.Generator(device="cuda").manual_seed(34)
    cache = torch.randn(
        num_slots, H, V, K, dtype=torch.bfloat16, device="cuda", generator=g_rng
    )
    slot_ids = torch.tensor([3, 9, 14], dtype=torch.int32, device="cuda")
    has_init = torch.tensor([True, True, False], dtype=torch.bool, device="cuda")

    # Reference: vLLM upstream, given the pre-gathered + zeroed initial_state.
    init_ref = cache[slot_ids.to(torch.long)].contiguous().to(torch.float32)
    init_ref[~has_init, ...] = 0
    o_ref, s_ref = ref_chunk_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=init_ref,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=True,
    )

    # Indexed path.
    cache_idx = cache.clone()
    o_idx, _ = chunk_gated_delta_rule_vk(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=cache_idx,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=True,
        ssm_state_indices=slot_ids,
        has_initial_state=has_init,
        inplace_final_state=True,
    )

    # o: bit-exact when both kernels happen to be compiled with the same
    # autotune config (verbatim port + same algorithm). Different configs
    # produce O(0.5) bf16 noise — we allow a small tolerance.
    o_diff = (o_idx.float() - o_ref.float()).abs().max().item()
    assert (
        o_diff < 1.0
    ), f"indexed path o diverges from vLLM reference: max abs diff {o_diff:.6g}"

    # final state: indexed-path scatter cast to bf16 vs reference fp32 → bf16.
    # Should match bit-exactly per slot id.
    s_idx_gather = cache_idx[slot_ids.to(torch.long)]
    s_ref_bf16 = s_ref.to(torch.bfloat16)
    if not torch.equal(s_idx_gather, s_ref_bf16):
        diff = (s_idx_gather.float() - s_ref_bf16.float()).abs()
        raise AssertionError(
            f"indexed-path final cache vs vLLM reference (cast bf16) not bit-exact: "
            f"max abs diff {diff.max().item():.6g}"
        )


def test_fused_ssm_state_indexed_no_initial_state_mask():
    """has_initial_state=None means every slot is loaded as-is — no
    masked-fill. The cache (final state) must agree bit-exactly with the
    explicit gather (without zeroing) + call + scatter, even though `o`
    drifts within bf16 autotune-config noise."""
    seq_lens = [64, 128]
    H, K, V = QWEN3_NEXT_H, QWEN3_NEXT_K, QWEN3_NEXT_V
    q, k, v, g, beta, cu_seqlens = _make_inputs(
        seq_lens=seq_lens, H=H, K=K, V=V, seed=40
    )

    num_slots = 8
    g_rng = torch.Generator(device="cuda").manual_seed(41)
    cache = torch.randn(
        num_slots, H, V, K, dtype=torch.bfloat16, device="cuda", generator=g_rng
    )
    slot_ids = torch.tensor([5, 1], dtype=torch.int32, device="cuda")

    # OLD path
    cache_old = cache.clone()
    initial_state_old = (
        cache_old[slot_ids.to(torch.long)].contiguous().to(torch.float32)
    )
    o_old, s_old = chunk_gated_delta_rule_vk(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=initial_state_old,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=True,
    )
    cache_old[slot_ids.to(torch.long)] = s_old.to(cache_old.dtype)

    # NEW path, has_initial_state=None
    cache_new = cache.clone()
    o_new, _ = chunk_gated_delta_rule_vk(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=cache_new,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=True,
        ssm_state_indices=slot_ids,
        has_initial_state=None,
        inplace_final_state=True,
    )

    # o within bf16 autotune noise.
    o_diff = (o_new.float() - o_old.float()).abs().max().item()
    assert o_diff < 0.5, f"o diff {o_diff:.6g} too large with has_initial_state=None"

    # Written slots: bit-exact (same fp32 final state, same bf16 cast).
    written = slot_ids.to(torch.long)
    if not torch.equal(cache_new[written], cache_old[written]):
        diff = (cache_new[written].float() - cache_old[written].float()).abs()
        raise AssertionError(
            f"written cache slots not bit-exact (has_initial_state=None): "
            f"max abs diff {diff.max().item():.6g}"
        )

    # Untouched slots must be byte-identical.
    all_idx = torch.arange(num_slots, device="cuda")
    untouched = all_idx[~torch.isin(all_idx, written)]
    assert torch.equal(cache_new[untouched], cache_old[untouched]), (
        "untouched slots changed under has_initial_state=None — fused scatter "
        "corrupted unrelated rows"
    )
