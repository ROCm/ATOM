# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Parity and checkpoint tests for ATOM's local KDA inference path."""

from __future__ import annotations

import pytest
import torch

_HAS_CUDA = torch.cuda.is_available()
if _HAS_CUDA:
    try:
        from fla.ops.kda import chunk_kda as upstream_chunk_kda

        _HAS_FLA = True
    except ImportError:
        _HAS_FLA = False
else:
    _HAS_FLA = False


def _restore_real_atom_modules():
    import sys

    for mod_name in list(sys.modules):
        if mod_name == "atom" or mod_name.startswith("atom."):
            del sys.modules[mod_name]


_restore_real_atom_modules()

from atom.model_ops.fla_ops.kda import chunk_kda as atom_chunk_kda  # noqa: E402

pytestmark = [
    pytest.mark.skipif(not _HAS_CUDA, reason="No GPU available"),
    pytest.mark.skipif(not _HAS_FLA, reason="FLA KDA reference not importable"),
]


def _inputs(lengths: list[int], *, initial: bool):
    torch.manual_seed(19)
    T = sum(lengths)
    B, H, K, V = 1, 2, 64, 80
    q = torch.randn(B, T, H, K, device="cuda", dtype=torch.bfloat16) / 8
    k = torch.randn_like(q) / 8
    v = torch.randn(B, T, H, V, device="cuda", dtype=torch.bfloat16) / 8
    g = torch.randn(B, T, H, K, device="cuda", dtype=torch.bfloat16) / 8
    beta = torch.randn(B, T, H, device="cuda", dtype=torch.float32)
    A_log = torch.randn(H, device="cuda", dtype=torch.float32)
    dt_bias = torch.randn(H * K, device="cuda", dtype=torch.float32)
    cu_seqlens = None
    if len(lengths) > 1:
        cu_seqlens = torch.tensor(
            [0, *torch.tensor(lengths).cumsum(0).tolist()],
            device="cuda",
            dtype=torch.int32,
        )
    h0 = None
    if initial:
        h0 = torch.randn(len(lengths), H, V, K, device="cuda", dtype=torch.float32) / 8
    return dict(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        A_log=A_log,
        dt_bias=dt_bias,
        initial_state=h0,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        safe_gate=True,
        lower_bound=-5.0,
        state_v_first=True,
        cu_seqlens=cu_seqlens,
    )


@pytest.mark.parametrize("lengths", [[512], [65], [128, 193]])
@pytest.mark.parametrize("initial", [False, True])
def test_local_kda_is_bit_exact_with_upstream(lengths, initial):
    """Porting the scan must not alter the model's existing KDA result."""
    kwargs = _inputs(lengths, initial=initial)
    with torch.inference_mode():
        ref_o, ref_ht = upstream_chunk_kda(**kwargs)
        got_o, got_ht = atom_chunk_kda(**kwargs)
    assert torch.equal(got_o, ref_o)
    assert torch.equal(got_ht, ref_ht)


def test_local_kda_scatter_does_not_perturb_output():
    """Checkpoint side-writes leave full KDA output and final state unchanged."""
    kwargs = _inputs([512], initial=True)
    H, V, K = kwargs["v"].shape[2], kwargs["v"].shape[3], kwargs["q"].shape[3]
    ckpt = torch.zeros(8, H, V, K, device="cuda", dtype=torch.float32)
    slots = torch.tensor([[0, 1, 2, -1]], device="cuda", dtype=torch.long)
    base = torch.zeros(1, device="cuda", dtype=torch.long)

    with torch.inference_mode():
        ref_o, ref_ht, h = atom_chunk_kda(**kwargs, return_intermediate_states=True)
        got_o, got_ht = atom_chunk_kda(
            **kwargs,
            ckpt=ckpt,
            ckpt_slots=slots,
            ckpt_base=base,
            ckpt_every=128,
        )

    assert torch.equal(got_o, ref_o)
    assert torch.equal(got_ht, ref_ht)
    for i, chunk in enumerate((2, 4, 6)):
        assert torch.equal(ckpt[i].to(torch.bfloat16), h[0, chunk])
    assert ckpt[3].count_nonzero().item() == 0
    assert ckpt[4].count_nonzero().item() == 0
