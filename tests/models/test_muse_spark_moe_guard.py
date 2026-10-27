# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Muse Spark must refuse the FusedMoE paths that cannot run its expert slots.

CPU-only. The MoE runs top-1 FusedMoE over ``tokens * top_k`` slot rows whose
expert ids are preselected (``_slot_routing``). The Triton, DP attention, DP +
EP all2all, mega, fake-EPLB and context-parallel paths cannot run them, and must fail at init with
an actionable message instead of misrouting or asserting in warmup.
"""

from types import SimpleNamespace

import pytest

pytest.importorskip("aiter")

from atom.models.muse_spark import _slot_moe_unsupported


def _experts(use_triton=None, dp_size=1, use_ep=False, use_triton_ep=False):
    quant_method = SimpleNamespace(use_triton_ep=use_triton_ep)
    if use_triton is not None:
        quant_method.use_triton = use_triton
    return SimpleNamespace(quant_method=quant_method, dp_size=dp_size, use_ep=use_ep)


def _config(dp_attention=False, moe_backend="standard", fake_eplb=False, pcp=1, dcp=1):
    return SimpleNamespace(
        enable_dp_attention=dp_attention,
        moe_backend=moe_backend,
        fake_eplb=fake_eplb,
        prefill_context_parallel_size=pcp,
        decode_context_parallel_size=dcp,
    )


@pytest.mark.parametrize(
    "experts, dp_attention",
    [
        (_experts(use_triton=False), False),  # FlyDSL / CK MXFP4 path
        (_experts(), False),  # a method without a Triton path
        (_experts(use_triton=False, use_ep=True), False),  # TP + EP
        (_experts(use_triton=False, dp_size=1), True),  # DP attention, one rank
        (_experts(use_triton=False, dp_size=2), False),  # DP without EP
    ],
)
def test_supported_paths(experts, dp_attention):
    assert _slot_moe_unsupported(experts, _config(dp_attention)) is None


@pytest.mark.parametrize(
    "experts, config, hint",
    [
        (_experts(use_triton=True), _config(), "ATOM_USE_TRITON_MOE=0"),
        (_experts(use_ep=True, use_triton_ep=True), _config(), "ATOM_USE_TRITON_MOE=0"),
        (_experts(dp_size=2), _config(dp_attention=True), "DP attention"),
        (_experts(dp_size=2, use_ep=True), _config(), "DP without expert parallel"),
        (_experts(use_ep=True), _config(moe_backend="mega"), "--moe-backend standard"),
        (_experts(), _config(fake_eplb=True), "--fake-eplb"),
        (_experts(), _config(pcp=2), "-pcp 1"),
        (_experts(use_ep=True), _config(dcp=2), "-dcp 1"),
    ],
)
def test_unsupported_paths_are_rejected(experts, config, hint):
    msg = _slot_moe_unsupported(experts, config)
    assert msg is not None and hint in msg
