# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Which layers take `triton_topk_softmax` and up to how many rows, on the
GPU-less CI (through `test_comm_fused_moe`'s AITER / Triton stubs)."""

import importlib
import sys
from types import SimpleNamespace

import pytest
from test_comm_fused_moe import _stubbed_external_kernel_modules


@pytest.fixture(scope="module")
def topK():
    saved = {n: m for n, m in sys.modules.items() if n.split(".")[0] == "atom"}
    with _stubbed_external_kernel_modules():
        try:
            yield importlib.import_module("atom.model_ops.topK")
        finally:
            for name in list(sys.modules):
                if name.split(".")[0] == "atom" and name not in saved:
                    sys.modules.pop(name)
            sys.modules.update(saved)


@pytest.fixture
def gpu(topK, monkeypatch):
    cfg = SimpleNamespace(enable_rapidserve=False, disagg_constrained=False)
    monkeypatch.setattr(topK, "get_current_atom_config", lambda: cfg)

    def set_gpu(arch="gfx950", cus=256):
        monkeypatch.setattr(topK, "get_gfx_runtime", lambda: arch)
        monkeypatch.setattr(topK, "get_cu_num", lambda: cus)
        topK.triton_topk_softmax_rows.cache_clear()

    set_gpu()
    yield set_gpu
    topK.triton_topk_softmax_rows.cache_clear()


@pytest.mark.parametrize(
    "experts,k,shared,scoring,renormalize",
    [
        (1024, 8, 0, None, True),  # AITER: moeSoftmax + moeTopK
        (1024, 10, 1, "sigmoid", True),
        (8, 2, 0, None, True),  # Mixtral
        (16, 4, 0, None, True),
        (160, 8, 0, None, True),  # Qwen3-Coder-480B
        (513, 10, 0, None, True),
        (512, 33, 0, None, True),
        (512, 0, 0, None, True),
        (128, 1, 0, None, True),  # Triton slower near the cutoff
        (512, 2, 0, None, True),
        (512, 10, 3, "sigmoid", True),
        (512, 10, 1, "softplus", True),
        (512, 10, 1, "", True),  # AITER refuses shared columns unscored by ""
        (512, 10, 1, "sigmoid", False),
    ],
)
def test_other_layers_stay_on_aiter(
    experts, k, shared, scoring, renormalize, topK, gpu
):
    rows = topK.triton_topk_softmax_rows(experts, k, renormalize, shared, scoring)
    assert rows is None


def test_unscored_shared_experts_route_like_none(topK, gpu):
    # Without a scoring func the shared columns are prefilled, not routed.
    assert topK.triton_topk_softmax_rows(512, 10, True, 3, None) == 5120


@pytest.mark.parametrize(
    "arch,cus,max_rows",
    [
        ("gfx950", 256, 5120),
        ("gfx950", 128, 2560),
        ("gfx942", 304, None),
        ("gfx1250", 256, None),
    ],
)
def test_cutoff_counts_the_cus(arch, cus, max_rows, topK, gpu):
    gpu(arch, cus)
    assert topK.triton_topk_softmax_rows(512, 10, True, 1, "sigmoid") == max_rows
    assert topK.triton_topk_softmax_rows(32, 4, True, 0, None) == max_rows
