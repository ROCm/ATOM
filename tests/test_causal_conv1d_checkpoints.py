# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Boundary checkpoint tests for the causal-conv prefill kernel."""

import sys

import pytest
import torch

for mod_name in list(sys.modules):
    if mod_name == "atom" or mod_name.startswith("atom."):
        del sys.modules[mod_name]

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="No GPU available"
)

from atom.model_ops.mamba_ops.causal_conv1d import causal_conv1d_fn  # noqa: E402


def test_conv_checkpoint_matches_boundary_window_without_perturbing_output():
    torch.manual_seed(31)
    dim, length = 48, 192
    x = torch.randn(dim, length, device="cuda", dtype=torch.bfloat16)
    state = torch.zeros(1, dim, 3, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(dim, 4, device="cuda", dtype=torch.bfloat16)
    query_start = torch.tensor([0, length], device="cuda", dtype=torch.int32)
    indices = torch.tensor([0], device="cuda", dtype=torch.int32)
    has_initial = torch.tensor([False], device="cuda")

    ref_state = state.clone()
    ref = causal_conv1d_fn(
        x,
        weight,
        None,
        ref_state,
        query_start,
        dim // 3,
        dim // 3,
        cache_indices=indices,
        has_initial_state=has_initial,
    )

    got_state = state.clone()
    checkpoints = torch.zeros(4, dim, 3, device="cuda", dtype=torch.bfloat16)
    slots = torch.tensor([[0, 2, -1]], device="cuda", dtype=torch.long)
    got = causal_conv1d_fn(
        x,
        weight,
        None,
        got_state,
        query_start,
        dim // 3,
        dim // 3,
        cache_indices=indices,
        has_initial_state=has_initial,
        ckpt_conv=checkpoints,
        ckpt_slots=slots,
        ckpt_base=torch.zeros(1, device="cuda", dtype=torch.long),
        ckpt_every=64,
    )

    for actual, expected in zip(got, ref):
        assert torch.equal(actual, expected), (
            (actual.float() - expected.float()).abs().max().item(),
            (actual != expected).sum().item(),
        )
    assert torch.equal(got_state, ref_state)
    assert torch.equal(checkpoints[0], x[:, 61:64])
    assert torch.equal(checkpoints[2], x[:, 125:128])
    assert checkpoints[1].count_nonzero().item() == 0
    assert checkpoints[3].count_nonzero().item() == 0
