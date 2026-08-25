# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Numerical parity between ATOM's fused_sigmoid_gating decode kernel and
fla's fused_recurrent_kda, for the Kimi-K3 KDA gate.

Kimi-K3 uses a *lower-bounded sigmoid* forget gate
    g = lower_bound * sigmoid(exp(A_log) * (a + dt_bias))
with a *per-K-channel* (diagonal) decay. fla's `fused_recurrent_kda`
implements this via its USE_LOWER_BOUND branch. ATOM's
`fused_sigmoid_gating_delta_rule_update` historically only had the GDN gate
    g = -exp(A_log) * softplus(a + dt_bias)
which is a DIFFERENT function of the same inputs.

These tests assert:
  * WITH lower_bound=-5.0 + is_kda=True, the ATOM kernel matches fla
    (the newly added USE_LOWER_BOUND branch is numerically correct).
  * WITHOUT lower_bound (the old GDN gate), the ATOM kernel visibly
    diverges from fla-KDA — i.e. the swap the user asked about would be
    wrong if the lower-bound branch did not exist.
  * lower_bound=None still runs the original GDN path unchanged.
"""

from __future__ import annotations

import pytest
import torch

_HAS_CUDA = torch.cuda.is_available()


# Evict any conftest stubs of atom.* / fla.* so the real GPU kernels import.
def _restore_real_modules():
    import sys

    for mod_name in list(sys.modules):
        if (
            mod_name == "atom"
            or mod_name.startswith("atom.")
            or mod_name == "fla"
            or mod_name.startswith("fla.")
        ):
            del sys.modules[mod_name]


_restore_real_modules()

pytestmark = [
    pytest.mark.skipif(not _HAS_CUDA, reason="No GPU available"),
]

LOWER_BOUND = -5.0


def _make_inputs(N, H, HV, K, V, dtype=torch.bfloat16, seed=0):
    """One decode token per sequence (T=1), flattened as B=1 varlen.

    Shapes follow the fla/vLLM decode convention:
        q, k : [1, N, H,  K]
        v    : [1, N, HV, V]
        a(=g): [1, N, HV, K]   (per-K-channel KDA gate input)
        b    : [1, N, HV]      (raw write-strength logits)
        A_log   : [HV]
        dt_bias : [HV * K]     (per head, per channel)
        h0      : [N, HV, V, K]  (state_v_first / V-first layout)
        cu_seqlens : [0, 1, 2, ..., N]
    """
    g_dev = torch.Generator(device="cuda").manual_seed(seed)

    def rnd(*shape):
        return torch.randn(*shape, device="cuda", dtype=dtype, generator=g_dev)

    q = rnd(1, N, H, K)
    k = rnd(1, N, H, K)
    v = rnd(1, N, HV, V)
    a = rnd(1, N, HV, K)  # KDA gate input (per-channel)
    b = rnd(1, N, HV)  # write-strength logits
    A_log = torch.randn(HV, device="cuda", dtype=torch.float32, generator=g_dev)
    dt_bias = torch.randn(HV * K, device="cuda", dtype=torch.float32, generator=g_dev)
    h0 = torch.randn(N, HV, V, K, device="cuda", dtype=torch.float32, generator=g_dev)
    cu_seqlens = torch.arange(N + 1, device="cuda", dtype=torch.int32)
    return q, k, v, a, b, A_log, dt_bias, h0, cu_seqlens


def _run_fla(q, k, v, a, b, A_log, dt_bias, h0, cu_seqlens, lower_bound):
    """fla reference. `a` is the raw gate input `g`; `b` is the raw beta."""
    from fla.ops.kda import fused_recurrent_kda

    o, ht = fused_recurrent_kda(
        q=q,
        k=k,
        v=v,
        g=a,
        beta=b.float(),
        A_log=A_log,
        dt_bias=dt_bias,
        initial_state=h0.clone(),
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        lower_bound=lower_bound,
        state_v_first=True,  # V-first [N, HV, V, K], matches ATOM kernel
        cu_seqlens=cu_seqlens,
    )
    return o, ht


def _run_atom(q, k, v, a, b, A_log, dt_bias, h0, cu_seqlens, lower_bound):
    """ATOM kernel under test. inplace update into a fresh state buffer."""
    from atom.model_ops.fla_ops.fused_sigmoid_gating import (
        fused_sigmoid_gating_delta_rule_update,
    )

    ssm_state = h0.clone()  # [N, HV, V, K]
    N = h0.shape[0]
    ssm_state_indices = torch.arange(N, device="cuda", dtype=torch.int32)
    o, _ = fused_sigmoid_gating_delta_rule_update(
        A_log=A_log,
        a=a,
        b=b,
        dt_bias=dt_bias,
        q=q,
        k=k,
        v=v,
        initial_state=ssm_state,
        inplace_final_state=True,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
        use_qk_l2norm_in_kernel=True,
        is_kda=True,
        lower_bound=lower_bound,
    )
    return o, ssm_state


@pytest.mark.parametrize("N", [1, 4])
def test_kda_lower_bound_matches_fla(N):
    """With the lower-bound branch, ATOM == fla-KDA (output and final state)."""
    H = HV = 4
    K = V = 64
    inp = _make_inputs(N, H, HV, K, V)

    o_ref, ht_ref = _run_fla(*inp, lower_bound=LOWER_BOUND)
    o_atom, ht_atom = _run_atom(*inp, lower_bound=LOWER_BOUND)

    # Two independent triton kernels, both fp32-accumulating the same math.
    torch.testing.assert_close(o_atom.float(), o_ref.float(), rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(ht_atom.float(), ht_ref.float(), rtol=2e-2, atol=2e-2)


def test_gdn_gate_diverges_from_kda():
    """Without the lower-bound branch (old GDN gate), ATOM != fla-KDA.

    This is the whole reason the swap needs USE_LOWER_BOUND: feeding the
    Kimi inputs through the softplus gate produces a materially different
    result, which is what would silently break accuracy.
    """
    N, H, HV, K, V = 4, 4, 4, 64, 64
    inp = _make_inputs(N, H, HV, K, V)

    o_ref, _ = _run_fla(*inp, lower_bound=LOWER_BOUND)
    o_gdn, _ = _run_atom(*inp, lower_bound=None)  # softplus gate

    max_abs_diff = (o_gdn.float() - o_ref.float()).abs().max().item()
    assert max_abs_diff > 1e-2, (
        "GDN softplus gate unexpectedly matched the KDA lower-bound gate; "
        f"max|diff|={max_abs_diff:.3e}. The two gates should differ."
    )


def test_gdn_path_still_runs():
    """Regression guard: lower_bound=None + is_kda=False (the original GDN
    decode path) still executes and returns finite outputs."""
    from atom.model_ops.fla_ops.fused_sigmoid_gating import (
        fused_sigmoid_gating_delta_rule_update,
    )

    N, HV, K, V = 4, 4, 64, 64
    H = HV
    g_dev = torch.Generator(device="cuda").manual_seed(1)
    q = torch.randn(1, N, H, K, device="cuda", dtype=torch.bfloat16, generator=g_dev)
    k = torch.randn(1, N, H, K, device="cuda", dtype=torch.bfloat16, generator=g_dev)
    v = torch.randn(1, N, HV, V, device="cuda", dtype=torch.bfloat16, generator=g_dev)
    a = torch.randn(1, N, HV, device="cuda", dtype=torch.bfloat16, generator=g_dev)
    b = torch.randn(1, N, HV, device="cuda", dtype=torch.bfloat16, generator=g_dev)
    A_log = torch.randn(HV, device="cuda", dtype=torch.float32, generator=g_dev)
    dt_bias = torch.randn(HV, device="cuda", dtype=torch.float32, generator=g_dev)
    h0 = torch.randn(N, HV, V, K, device="cuda", dtype=torch.float32, generator=g_dev)
    cu_seqlens = torch.arange(N + 1, device="cuda", dtype=torch.int32)
    ssm_state_indices = torch.arange(N, device="cuda", dtype=torch.int32)

    o, _ = fused_sigmoid_gating_delta_rule_update(
        A_log=A_log,
        a=a,
        b=b,
        dt_bias=dt_bias,
        q=q,
        k=k,
        v=v,
        initial_state=h0,
        inplace_final_state=True,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
        use_qk_l2norm_in_kernel=True,
        is_kda=False,
        lower_bound=None,
    )
    assert torch.isfinite(o.float()).all()
