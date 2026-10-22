# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Muse Spark must refuse the FusedMoE paths that cannot run its expert slots.

CPU-only. The MoE runs top-1 FusedMoE over ``tokens * top_k`` slot rows whose
expert ids are preselected (``_slot_routing``). The Triton TP path routes from
logits and ignores the callback; DP attention gathers one row per token. Both
must fail at init with an actionable message instead of misrouting or asserting
in warmup.
"""

from types import SimpleNamespace

import pytest

pytest.importorskip("aiter")

from atom.models.muse_spark import _slot_moe_unsupported


def _experts(use_triton=None, dp_size=1):
    quant_method = SimpleNamespace()
    if use_triton is not None:
        quant_method.use_triton = use_triton
    return SimpleNamespace(quant_method=quant_method, dp_size=dp_size)


@pytest.mark.parametrize(
    "experts, dp_attention",
    [
        (_experts(use_triton=False), False),  # FlyDSL / CK MXFP4 path
        (_experts(), False),  # a method without a Triton path
        (_experts(use_triton=False, dp_size=1), True),  # DP attention, one rank
        (_experts(use_triton=False, dp_size=2), False),  # DP without DP attention
    ],
)
def test_supported_paths(experts, dp_attention):
    assert _slot_moe_unsupported(experts, dp_attention) is None


def test_triton_tp_path_is_rejected():
    msg = _slot_moe_unsupported(_experts(use_triton=True), False)
    assert msg is not None and "ATOM_USE_TRITON_MOE=0" in msg


def test_dp_attention_is_rejected():
    msg = _slot_moe_unsupported(_experts(use_triton=False, dp_size=2), True)
    assert msg is not None and "DP attention" in msg
