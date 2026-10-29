# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Which kernel routes renormalized softmax top-k, checked without a GPU.

`test_triton_topk_softmax_gpu.py` compares the two kernels bit for bit but
skips on a GPU-less runner. These pin the choice itself, which a MoE layer
makes once when it is built (`triton_topk_softmax_rows`): the Triton kernel
takes exactly the shapes AITER's fused `topkGatingSoftmax` handles (its
dispatch stops at 512 experts), only in launches of up to 20 rows per CU, and
only with the switch on when the layer is built.
"""

import pytest

from atom.model_ops import topk_softmax_dispatch
from atom.model_ops.topk_softmax_dispatch import (
    triton_topk_softmax_max_rows,
    triton_topk_softmax_rows,
    triton_topk_softmax_supported,
)
from atom.utils import envs

# MI355X: 256 CUs, so launches of up to 20 * 256 rows take the Triton kernel.
MI355X_CUS, MI355X_MAX_ROWS = 256, 5120


@pytest.fixture(autouse=True)
def cu_num(monkeypatch):
    """The CU count, which the dispatch asks AITER for (absent on this CI)."""

    def set_cu_num(n):
        monkeypatch.setattr(topk_softmax_dispatch, "_cu_num", lambda: n)

    set_cu_num(MI355X_CUS)
    return set_cu_num


def _scoring(shared):
    return "sigmoid" if shared else ""


@pytest.mark.parametrize("shared", [0, 1, 2, 4, 8])
@pytest.mark.parametrize("k", [1, 8, 10, 32])
@pytest.mark.parametrize("experts", [32, 64, 128, 256, 512])
def test_takes_aiter_fused_kernel_shapes(experts, k, shared):
    assert triton_topk_softmax_supported(experts, k, True, shared, _scoring(shared))


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
        (512, 0, 0),
        (512, 10, 3),
    ],
)
def test_refuses_other_shapes(experts, k, shared):
    assert not triton_topk_softmax_supported(experts, k, True, shared, _scoring(shared))


def test_refuses_unnormalized_and_other_scoring_funcs():
    assert triton_topk_softmax_supported(512, 10, True, 1, "sigmoid")
    assert not triton_topk_softmax_supported(512, 10, False, 1, "sigmoid")
    assert not triton_topk_softmax_supported(512, 10, True, 1, "softplus")


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
def test_model_layers_take_triton_up_to_the_cutoff(
    experts, k, shared, cus, max_rows, cu_num, monkeypatch
):
    monkeypatch.setenv("ATOM_TRITON_TOPK_SOFTMAX", "1")
    cu_num(cus)
    scoring_func = "sigmoid" if shared else None
    assert triton_topk_softmax_rows(experts, k, True, shared, scoring_func) == max_rows


@pytest.mark.parametrize(
    "args",
    [
        (512, 10, False, 1, "sigmoid"),  # not renormalized
        (512, 10, True, 1, "softplus"),
        (1024, 10, True, 1, "sigmoid"),  # AITER: moeSoftmax + moeTopK
        (8, 2, True, 0, None),  # Mixtral
        (160, 8, True, 0, None),  # Qwen3-Coder-480B
        (512, 10, True, 3, "sigmoid"),
    ],
)
def test_other_layers_never_take_triton(args, monkeypatch):
    monkeypatch.setenv("ATOM_TRITON_TOPK_SOFTMAX", "1")
    assert triton_topk_softmax_rows(*args) is None


def test_unscored_shared_experts_are_not_the_kernels(monkeypatch):
    """Without a scoring func the kernels leave the shared columns alone (they
    are prefilled), so any shared count routes like none."""
    monkeypatch.setenv("ATOM_TRITON_TOPK_SOFTMAX", "1")
    assert triton_topk_softmax_rows(512, 10, True, 3, None) == MI355X_MAX_ROWS
    assert triton_topk_softmax_rows(512, 10, True, 3, "sigmoid") is None


@pytest.mark.parametrize("experts,k,shared", MODEL_SHAPES)
def test_switch_off_puts_every_layer_on_aiter(experts, k, shared, monkeypatch):
    monkeypatch.setenv("ATOM_TRITON_TOPK_SOFTMAX", "0")
    scoring_func = "sigmoid" if shared else None
    assert triton_topk_softmax_rows(experts, k, True, shared, scoring_func) is None


def test_decides_once_from_the_switch_and_cu_count(monkeypatch, cu_num):
    """The layer reads the switch and the CU count when it decides; nothing a
    launch does reads them again, so changing them later moves no layer."""
    reads = []
    get_env = envs.__getattr__

    def counting_getattr(name):
        reads.append(name)
        return get_env(name)

    monkeypatch.setattr(envs, "__getattr__", counting_getattr)
    monkeypatch.setattr(
        topk_softmax_dispatch, "_cu_num", lambda: reads.append("cu") or MI355X_CUS
    )
    monkeypatch.setenv("ATOM_TRITON_TOPK_SOFTMAX", "1")
    max_rows = triton_topk_softmax_rows(512, 10, True, 1, "sigmoid")
    assert max_rows == MI355X_MAX_ROWS
    assert reads == ["ATOM_TRITON_TOPK_SOFTMAX", "cu"]
    # The switch off: no CU count needed.
    reads.clear()
    monkeypatch.setenv("ATOM_TRITON_TOPK_SOFTMAX", "0")
    assert triton_topk_softmax_rows(512, 10, True, 1, "sigmoid") is None
    assert reads == ["ATOM_TRITON_TOPK_SOFTMAX"]
