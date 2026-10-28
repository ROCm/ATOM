# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Which kernel routes renormalized softmax top-k, checked without a GPU.

`test_triton_topk_softmax_gpu.py` compares the two kernels bit for bit but
skips on a GPU-less runner. These pin the choice itself: the Triton kernel
takes exactly the shapes AITER's fused `topkGatingSoftmax` handles (its
dispatch stops at 512 experts), never a layout it cannot store to, and eager
and captured passes choose the same kernel.
"""

import pytest
import torch

from atom.model_ops.topk_softmax_dispatch import (
    triton_topk_softmax_supported,
    use_triton_topk_softmax,
)


def _launch(experts, k, shared, rows=4):
    """(topk_weights, topk_ids, gating_output) in the fused-shared layout."""
    logits = torch.zeros(rows, experts + shared, dtype=torch.bfloat16)
    w = torch.empty(rows, k + shared)
    ids = torch.empty(rows, k + shared, dtype=torch.int32)
    return w, ids[:, :k], logits


@pytest.mark.parametrize("shared", [0, 1, 2, 4, 8])
@pytest.mark.parametrize("k", [1, 8, 10, 32])
@pytest.mark.parametrize("experts", [32, 64, 128, 256, 512])
def test_takes_aiter_fused_kernel_shapes(experts, k, shared):
    assert triton_topk_softmax_supported(*_launch(experts, k, shared), True, shared)


@pytest.mark.parametrize(
    "experts,k,shared",
    [
        (1024, 8, 0),  # AITER: moeSoftmax + moeTopK
        (1024, 10, 1),
        (2048, 8, 0),
        (16, 1, 0),
        (384, 8, 0),
        (513, 10, 0),
        (512, 33, 0),
        (512, 10, 3),
    ],
)
def test_refuses_other_shapes(experts, k, shared):
    assert not triton_topk_softmax_supported(*_launch(experts, k, shared), True, shared)


def test_refuses_unnormalized_and_strided_launches():
    w, ids, logits = _launch(512, 10, 1)
    assert triton_topk_softmax_supported(w, ids, logits, True, 1)
    assert not triton_topk_softmax_supported(w, ids, logits, False, 1)
    wide = torch.zeros(4, 2 * 513, dtype=torch.bfloat16)
    assert not triton_topk_softmax_supported(w, ids, wide[:, ::2], True, 1)
    w_t = torch.empty(11, 4).t()
    ids_t = torch.empty(11, 4, dtype=torch.int32).t()[:, :10]
    assert not triton_topk_softmax_supported(w_t, ids, logits, True, 1)
    assert not triton_topk_softmax_supported(w, ids_t, logits, True, 1)


@pytest.mark.parametrize("experts", [512, 1024])
def test_eager_and_captured_passes_choose_alike(experts, monkeypatch):
    """A token's route must not depend on whether its batch is replayed from
    a graph or run eagerly."""
    monkeypatch.setenv("ATOM_TRITON_TOPK_SOFTMAX", "1")
    launch = _launch(experts, 10, 1)
    choices = set()
    for capturing in (False, True):
        monkeypatch.setattr(
            torch.cuda, "is_current_stream_capturing", lambda c=capturing: c
        )
        choices.add(use_triton_topk_softmax(*launch, True, 1, "sigmoid"))
    assert choices == {experts <= 512}


@pytest.mark.parametrize("scoring_func,shared", [("", 0), ("sigmoid", 1)])
def test_switch_and_scoring_func(scoring_func, shared, monkeypatch):
    launch = _launch(512, 10, shared)
    monkeypatch.setenv("ATOM_TRITON_TOPK_SOFTMAX", "1")
    assert use_triton_topk_softmax(*launch, True, shared, scoring_func)
    assert not use_triton_topk_softmax(*launch, True, shared, "softplus")
    monkeypatch.setenv("ATOM_TRITON_TOPK_SOFTMAX", "0")
    assert not use_triton_topk_softmax(*launch, True, shared, scoring_func)
