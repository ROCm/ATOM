# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""ReplaySSM verify launch rules apply only to the shapes they were measured on."""

import pytest
import torch
from aiter_stub import atom_under_stubs


@pytest.fixture
def replay(monkeypatch):
    with atom_under_stubs() as load:
        module = load("atom.model_ops.fla_ops.replayssm")
        monkeypatch.setattr(
            module.triton, "next_power_of_2", lambda n: 1 << (n - 1).bit_length()
        )
        yield module


def _bv(
    replay,
    monkeypatch,
    gfx,
    N,
    HV,
    V=128,
    T=4,
    is_kda=False,
    H=8,
    K=128,
    cache_len=16,
    input_dtype=torch.bfloat16,
    record_dtypes=(torch.bfloat16, torch.bfloat16, torch.float32),
    state_dtype=torch.float32,
):
    monkeypatch.setattr(replay, "get_gfx", lambda: gfx)
    monkeypatch.setattr(replay, "get_gfx_runtime", lambda: gfx)
    return replay._serial_launch_config(
        N,
        HV,
        V,
        T,
        is_kda,
        H=H,
        K=K,
        cache_len=cache_len,
        input_dtype=input_dtype,
        record_dtypes=record_dtypes,
        state_dtype=state_dtype,
    )[0]


@pytest.mark.parametrize(("N", "bv"), [(1, 16), (4, 16), (8, 64), (16, 32), (128, 32)])
def test_gfx1250_ladder_on_the_measured_qwen38_shape(replay, monkeypatch, N, bv):
    assert _bv(replay, monkeypatch, "gfx1250", N, HV=48, H=16) == bv


@pytest.mark.parametrize(
    "shape",
    [
        {"HV": 32},  # another head count
        {"HV": 48, "V": 64},  # another value width
        {"HV": 48, "T": 2},  # another verify length
        {"HV": 48, "is_kda": True},
    ],
)
def test_gfx1250_unmeasured_shapes_keep_the_large_batch_config(
    replay, monkeypatch, shape
):
    assert _bv(replay, monkeypatch, "gfx1250", 1, **shape) == min(
        shape.get("V", 128), 64
    )


def test_gfx950_small_batch_rule_is_unchanged(replay, monkeypatch):
    assert _bv(replay, monkeypatch, "gfx950", 1, HV=16, H=4) == 16
    assert _bv(replay, monkeypatch, "gfx950", 64, HV=16, H=4) == 64


def test_gfx942_measured_shape_uses_sixteen_wide_tiles(replay, monkeypatch):
    assert _bv(replay, monkeypatch, "gfx942", 1, HV=24) == 16


@pytest.mark.parametrize(
    "delta",
    [
        {"N": 2},
        {"H": 4},
        {"HV": 48},
        {"K": 64},
        {"V": 64},
        {"T": 2},
        {"is_kda": True},
        {"cache_len": 32},
        {"input_dtype": torch.float16},
        {"record_dtypes": (torch.float32, torch.bfloat16, torch.float32)},
        {"record_dtypes": (torch.bfloat16, torch.float32, torch.float32)},
        {"record_dtypes": (torch.bfloat16, torch.bfloat16, torch.bfloat16)},
        {"state_dtype": torch.bfloat16},
    ],
)
def test_gfx942_unmeasured_shapes_keep_existing_tiles(replay, monkeypatch, delta):
    args = {"N": 1, "HV": 24, **delta}
    assert _bv(replay, monkeypatch, "gfx942", **args) == 64


@pytest.mark.parametrize("gfx", ["gfx950", "gfx1250", "gfx1200"])
def test_new_shape_does_not_change_other_architectures(replay, monkeypatch, gfx):
    assert _bv(replay, monkeypatch, gfx, 1, HV=24) == (16 if gfx == "gfx950" else 64)
