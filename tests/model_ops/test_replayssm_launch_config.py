# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""ReplaySSM verify launch rules apply only to the shapes they were measured on."""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip(
        "importing replayssm loads the Triton and AITER kernels; needs a real GPU",
        allow_module_level=True,
    )

from atom.model_ops.fla_ops import replayssm


def _bv(monkeypatch, gfx, N, HV, V=128, T=4, is_kda=False):
    monkeypatch.setattr(replayssm, "get_gfx", lambda: gfx)
    monkeypatch.setattr(replayssm, "get_gfx_runtime", lambda: gfx)
    return replayssm._serial_launch_config(N, HV, V, T, is_kda)[0]


@pytest.mark.parametrize(("N", "bv"), [(1, 16), (4, 16), (8, 64), (16, 32), (128, 32)])
def test_gfx1250_ladder_on_the_measured_qwen38_shape(monkeypatch, N, bv):
    assert _bv(monkeypatch, "gfx1250", N, HV=48) == bv


@pytest.mark.parametrize(
    "shape",
    [
        {"HV": 32},  # another head count
        {"HV": 48, "V": 64},  # another value width
        {"HV": 48, "T": 2},  # another verify length
        {"HV": 48, "is_kda": True},
    ],
)
def test_gfx1250_unmeasured_shapes_keep_the_large_batch_config(monkeypatch, shape):
    assert _bv(monkeypatch, "gfx1250", 1, **shape) == min(shape.get("V", 128), 64)


def test_gfx950_small_batch_rule_is_unchanged(monkeypatch):
    assert _bv(monkeypatch, "gfx950", 1, HV=16) == 16
    assert _bv(monkeypatch, "gfx950", 64, HV=16) == 64
