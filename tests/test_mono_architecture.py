# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU-only contracts for mono's architecture-separated device substrate."""

from types import SimpleNamespace

import pytest

from atom.mono.plan.architecture import (
    GFX950,
    GFX1250,
    Architecture,
    Bf16MmaContract,
    MatrixIsa,
    architecture_for,
)


def test_wave_and_bf16_matrix_contracts_cover_one_tile():
    assert GFX950.wave_size == 64
    assert GFX950.reduction_offsets == (32, 16, 8, 4, 2, 1)
    assert GFX950.bf16_mma.isa is MatrixIsa.MFMA
    assert GFX950.bf16_mma.accumulator_elements_per_lane == 4
    assert GFX950.max_workgroup_lds_bytes == 160 * 1024

    assert GFX1250.wave_size == 32
    assert GFX1250.reduction_offsets == (16, 8, 4, 2, 1)
    assert GFX1250.bf16_mma.isa is MatrixIsa.WMMA
    assert GFX1250.bf16_mma.accumulator_elements_per_lane == 8
    assert GFX1250.max_workgroup_lds_bytes == 327680

    for architecture in (GFX950, GFX1250):
        mma = architecture.bf16_mma
        distributed = architecture.wave_size * mma.accumulator_elements_per_lane
        assert distributed == mma.m * mma.n


def test_unknown_architecture_is_not_inferred():
    assert architecture_for("gfx950") is GFX950
    assert architecture_for("gfx1250") is GFX1250
    with pytest.raises(ValueError, match="unsupported mono architecture"):
        architecture_for("gfx9999")


def test_invalid_distributed_accumulator_contract_is_refused():
    with pytest.raises(ValueError, match="distributed accumulator"):
        Architecture(
            gfx="bad",
            wave_size=32,
            bf16_mma=Bf16MmaContract(
                isa=MatrixIsa.WMMA,
                m=16,
                n=16,
                k=32,
                accumulator_elements_per_lane=4,
            ),
            native_permlane_exchange=False,
        )


def test_existing_execution_model_remains_explicitly_gfx950():
    from atom.mono.plan import execution

    assert execution.ARCHITECTURE is GFX950
    assert execution.WAVE_SIZE == 64
    assert execution.WAVES == execution.THREADS // GFX950.wave_size
    assert execution.BLOCKS == 256
    assert execution.LDS_BYTES == 160 * 1024


def _device_ops():
    return pytest.importorskip("atom.mono.device.ops")


def test_gfx1250_wave_sum_uses_only_wave32_offsets(monkeypatch):
    ops = _device_ops()
    seen = []

    def xred(value, offset, operation, architecture):
        seen.append((offset, architecture))
        return value + 1

    monkeypatch.setattr(ops, "xred", xred)
    assert ops.wave_sum(0, GFX1250) == 5
    assert seen == [(offset, GFX1250) for offset in (16, 8, 4, 2, 1)]


def test_gfx1250_cross_row_reduction_avoids_permlane(monkeypatch):
    ops = _device_ops()
    value, partner = object(), object()

    def no_permlane(*_args):
        pytest.fail("wave32 reduction must not emit the gfx950 permlane exchange")

    monkeypatch.setattr(ops, "permlane_swap", no_permlane)
    monkeypatch.setattr(
        ops,
        "xshfl",
        lambda got, offset, architecture: (
            partner
            if (got, offset, architecture) == (value, 16, GFX1250)
            else pytest.fail("unexpected shuffle")
        ),
    )

    assert ops.xred(value, 16, lambda lhs, rhs: (lhs, rhs), GFX1250) == (
        value,
        partner,
    )


@pytest.mark.parametrize(
    ("architecture", "intrinsic_name", "accumulator_elements"),
    [
        (GFX950, "mfma_f32_16x16x32_bf16", 4),
        (GFX1250, "wmma_f32_16x16x32_bf16", 8),
    ],
)
def test_bf16_mma_routes_to_the_architecture_intrinsic(
    monkeypatch, architecture, intrinsic_name, accumulator_elements
):
    ops = _device_ops()
    calls = []

    class Vector:
        def __new__(cls, value):
            return ("vector", value)

        @staticmethod
        def make_type(width, dtype):
            return width, dtype

    def mfma(result_type, operands):
        calls.append(("mfma", result_type, operands))
        return "mfma-result"

    def wmma(result_type, *operands, **modifiers):
        calls.append(("wmma", result_type, operands, modifiers))
        return SimpleNamespace(result="wmma-result")

    monkeypatch.setattr(
        ops,
        "rocdl",
        SimpleNamespace(
            mfma_f32_16x16x32_bf16=mfma,
            wmma_f32_16x16x32_bf16=wmma,
        ),
    )
    monkeypatch.setattr(
        ops,
        "T",
        SimpleNamespace(f32="f32", vec=lambda width, dtype: (width, dtype)),
    )
    monkeypatch.setattr(ops, "as_ir_value", lambda value: f"ir:{value}")
    monkeypatch.setattr(
        ops,
        "fx",
        SimpleNamespace(Float32="f32", Vector=Vector),
    )

    assert ops.mma_bf16("a", "b", "c", architecture) == (
        "vector",
        f"{architecture.bf16_mma.isa.value}-result",
    )
    assert calls[0][0] == architecture.bf16_mma.isa.value
    assert calls[0][1] == (accumulator_elements, "f32")
    if intrinsic_name.startswith("wmma"):
        assert calls[0][2] == ("ir:a", "ir:b", "ir:c")
        assert calls[0][3] == {"reuseA": False, "reuseB": False}
    else:
        assert calls[0][2] == ["a", "b", "c"]
