# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Which kernel routes renormalized softmax top-k, checked without a GPU.

`test_triton_topk_softmax_gpu.py` compares the two kernels bit for bit but
skips on a GPU-less runner. These pin the choice itself: the Triton kernel
takes exactly the shapes AITER's fused `topkGatingSoftmax` handles (its
dispatch stops at 512 experts), never a layout it cannot store to, only
launches of up to 20 rows per CU, and eager and captured passes choose the
same kernel.
"""

import pytest
import torch

from atom.model_ops import topk_softmax_dispatch
from atom.model_ops.topk_softmax_dispatch import (
    triton_topk_softmax_max_rows,
    triton_topk_softmax_supported,
    use_triton_topk_softmax,
)

# MI355X: 256 CUs, so launches of up to 20 * 256 rows take the Triton kernel.
MI355X_CUS, MI355X_MAX_ROWS = 256, 5120


@pytest.fixture(autouse=True)
def cu_num(monkeypatch):
    """The CU count, which the dispatch asks AITER for (absent on this CI)."""

    def set_cu_num(n):
        monkeypatch.setattr(topk_softmax_dispatch, "_cu_num", lambda: n)

    set_cu_num(MI355X_CUS)
    return set_cu_num


def _launch(experts, k, shared, rows=4):
    """(topk_weights, topk_ids, gating_output) in the fused-shared layout."""
    logits = torch.empty(rows, experts + shared, dtype=torch.bfloat16)
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


def test_row_cutoff_scales_with_the_cu_count(cu_num):
    assert triton_topk_softmax_max_rows() == MI355X_MAX_ROWS
    cu_num(304)  # MI300X
    assert triton_topk_softmax_max_rows() == 6080
    cu_num(80)
    assert triton_topk_softmax_max_rows() == 1600


# The softmax top-k shapes of the models that route through it: Qwen3.8 /
# Qwen3.5-397B / Qwen3-Next (512 / top-10, a fused shared expert or not),
# Qwen3.5-35B (256 / top-8 + shared), Qwen3 MoE (128 / top-8), gpt-oss-120b /
# -20b (128 and 32 / top-4).
MODEL_SHAPES = [
    (512, 10, 1),
    (512, 10, 0),
    (256, 8, 1),
    (256, 8, 0),
    (128, 8, 0),
    (128, 4, 0),
    (32, 4, 0),
]


@pytest.mark.parametrize("cus,max_rows", [(MI355X_CUS, MI355X_MAX_ROWS), (304, 6080)])
@pytest.mark.parametrize("experts,k,shared", MODEL_SHAPES)
def test_long_launches_take_aiter(
    experts, k, shared, cus, max_rows, cu_num, monkeypatch
):
    monkeypatch.setenv("ATOM_TRITON_TOPK_SOFTMAX", "1")
    cu_num(cus)
    scoring_func = "sigmoid" if shared else ""
    for rows, triton in (
        (1, True),
        (max_rows, True),
        (max_rows + 1, False),
        (16384, False),
    ):
        launch = _launch(experts, k, shared, rows)
        assert use_triton_topk_softmax(*launch, True, shared, scoring_func) == triton


@pytest.mark.parametrize("rows", [4, MI355X_MAX_ROWS, MI355X_MAX_ROWS + 1])
@pytest.mark.parametrize("experts", [512, 1024])
def test_eager_and_captured_passes_choose_alike(experts, rows, monkeypatch):
    """A pass of a given size runs the same kernel whether it is replayed from
    a graph or run eagerly."""
    monkeypatch.setenv("ATOM_TRITON_TOPK_SOFTMAX", "1")
    launch = _launch(experts, 10, 1, rows)
    choices = set()
    for capturing in (False, True):
        monkeypatch.setattr(
            torch.cuda, "is_current_stream_capturing", lambda c=capturing: c
        )
        choices.add(use_triton_topk_softmax(*launch, True, 1, "sigmoid"))
    assert choices == {experts <= 512 and rows <= MI355X_MAX_ROWS}


@pytest.mark.parametrize("rows", [4, MI355X_MAX_ROWS, MI355X_MAX_ROWS + 1])
@pytest.mark.parametrize("scoring_func,shared", [("", 0), ("sigmoid", 1)])
def test_switch_and_scoring_func(scoring_func, shared, rows, monkeypatch):
    launch = _launch(512, 10, shared, rows)
    monkeypatch.setenv("ATOM_TRITON_TOPK_SOFTMAX", "1")
    triton = rows <= MI355X_MAX_ROWS
    assert use_triton_topk_softmax(*launch, True, shared, scoring_func) == triton
    assert not use_triton_topk_softmax(*launch, True, shared, "softplus")
    monkeypatch.setenv("ATOM_TRITON_TOPK_SOFTMAX", "0")
    assert not use_triton_topk_softmax(*launch, True, shared, scoring_func)
