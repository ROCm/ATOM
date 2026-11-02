# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU contracts for architecture-explicit Kimi expert scale binding."""

from functools import lru_cache
from types import SimpleNamespace

import pytest
import torch

from atom.models.kimi_k3.mono.config import EXPERTS, LATENT, Dims
from atom.models.kimi_k3.mono.expert_scales import (
    ExpertScaleCache,
    materialize_expert_scale,
)
from atom.models.kimi_k3.mono.kernels.stages.moe_expert_layout import (
    ExpertScaleLayout,
    ExpertScaleSpec,
)
from atom.models.kimi_k3.mono.weights import bind_k2_expert_scales
from atom.mono.plan.architecture import GFX950, GFX1250
from atom.mono.runtime.consensus import MonoUnsupported


def _legacy_source(spec: ExpertScaleSpec) -> torch.Tensor:
    source = torch.full(
        spec.storage_shape(ExpertScaleLayout.GFX950_SHUFFLE),
        0x7F,
        dtype=torch.uint8,
    )
    flat = source.view(-1)
    for row in range(spec.rows):
        for group in range(spec.groups):
            flat[spec.packing.gfx950_scale_index(row, group)] = (
                row * spec.groups + group
            ) % 251
    return source


def _small_spec(
    name: str = "scale",
    *,
    rank: int = 0,
) -> ExpertScaleSpec:
    return ExpertScaleSpec(name, 2, 64, 256, rank, 2)


def test_legacy_to_n32k4_matches_every_reference_coordinate():
    spec = _small_spec()
    source = _legacy_source(spec)
    handle = materialize_expert_scale(
        source,
        spec=spec,
        source_layout=ExpertScaleLayout.GFX950_SHUFFLE,
        architecture=GFX1250,
        cache=ExpertScaleCache(),
    )
    handle.validate(GFX1250, spec)
    output = handle.tensor.view(-1)
    for row in range(spec.rows):
        for group in range(spec.groups):
            old = spec.packing.gfx950_scale_index(row, group)
            new = spec.packing.gfx1250_scale_index(row, group)
            assert output[new] == source.view(-1)[old]


def test_gate_up_and_down_row_bands_survive_repacking():
    spec = _small_spec("w13_scale")
    source = _legacy_source(spec)
    output = materialize_expert_scale(
        source,
        spec=spec,
        source_layout=ExpertScaleLayout.GFX950_SHUFFLE,
        architecture=GFX1250,
        cache=ExpertScaleCache(),
    ).tensor.view(-1)
    # Two experts, with gate/up represented by the two 32-row halves.
    for row in (0, 31, 32, 63, 64, 95, 96, 127):
        for group in range(spec.groups):
            expected = (row * spec.groups + group) % 251
            assert output[spec.packing.gfx1250_scale_index(row, group)] == expected

    down = ExpertScaleSpec("w2_scale", 2, 96, 256, 0, 2)
    down_source = _legacy_source(down)
    down_output = materialize_expert_scale(
        down_source,
        spec=down,
        source_layout=ExpertScaleLayout.GFX950_SHUFFLE,
        architecture=GFX1250,
        cache=ExpertScaleCache(),
    ).tensor.view(-1)
    for row in range(down.rows):
        for group in range(down.groups):
            old = down.packing.gfx950_scale_index(row, group)
            new = down.packing.gfx1250_scale_index(row, group)
            assert down_output[new] == down_source.view(-1)[old]


def test_native_n32k4_is_zero_copy_and_cache_stable():
    spec = _small_spec()
    source = torch.arange(
        spec.rows * spec.groups,
        dtype=torch.int32,
    ).to(torch.uint8)
    source = source.reshape(spec.storage_shape(ExpertScaleLayout.GFX1250_N32K4))
    cache = ExpertScaleCache()
    first = materialize_expert_scale(
        source,
        spec=spec,
        source_layout=ExpertScaleLayout.GFX1250_N32K4,
        architecture=GFX1250,
        cache=cache,
    )
    second = materialize_expert_scale(
        source,
        spec=spec,
        source_layout=ExpertScaleLayout.GFX1250_N32K4,
        architecture=GFX1250,
        cache=cache,
    )
    assert first is second
    assert first.tensor.data_ptr() == source.data_ptr()


def test_cache_isolates_architecture_layout_and_tp_rank():
    source0 = _legacy_source(_small_spec(rank=0))
    cache = ExpertScaleCache()
    gfx950 = materialize_expert_scale(
        source0,
        spec=_small_spec(rank=0),
        source_layout=ExpertScaleLayout.GFX950_SHUFFLE,
        architecture=GFX950,
        cache=cache,
    )
    gfx1250 = materialize_expert_scale(
        source0,
        spec=_small_spec(rank=0),
        source_layout=ExpertScaleLayout.GFX950_SHUFFLE,
        architecture=GFX1250,
        cache=cache,
    )
    rank1 = materialize_expert_scale(
        source0,
        spec=_small_spec(rank=1),
        source_layout=ExpertScaleLayout.GFX950_SHUFFLE,
        architecture=GFX1250,
        cache=cache,
    )
    assert gfx950 is not gfx1250
    assert gfx1250 is not rank1
    assert gfx950.tensor.data_ptr() == source0.data_ptr()
    assert gfx950.layout is ExpertScaleLayout.GFX950_SHUFFLE
    assert gfx1250.layout is ExpertScaleLayout.GFX1250_N32K4


def test_stale_and_wrong_layout_materializations_are_rejected():
    spec = _small_spec()
    source = _legacy_source(spec)
    handle = materialize_expert_scale(
        source,
        spec=spec,
        source_layout=ExpertScaleLayout.GFX950_SHUFFLE,
        architecture=GFX1250,
        cache=ExpertScaleCache(),
    )
    source[0, 0] ^= 1
    with pytest.raises(ValueError, match="stale"):
        handle.validate(GFX1250)
    with pytest.raises(ValueError, match="not gfx950"):
        handle.validate(GFX950)
    wrong_shape = torch.zeros(spec.rows, spec.groups, dtype=torch.uint8)
    with pytest.raises(ValueError, match="shape"):
        materialize_expert_scale(
            wrong_shape,
            spec=spec,
            source_layout=ExpertScaleLayout.GFX1250_N32K4,
            architecture=GFX1250,
            cache=ExpertScaleCache(),
        )


@pytest.mark.parametrize("tp", [1, 2, 4, 8])
def test_production_specs_cover_all_tp_slices_without_allocation(tp):
    dims = Dims(tp)
    for name, rows, k in (
        ("w13_scale", 2 * dims.inter, LATENT),
        ("w2_scale", LATENT, dims.inter),
    ):
        for rank in range(tp):
            spec = ExpertScaleSpec(name, EXPERTS, rows, k, rank, tp)
            assert spec.rows == EXPERTS * rows
            assert spec.storage_shape(ExpertScaleLayout.GFX1250_N32K4) == (
                EXPERTS,
                rows // 32,
                (k // 32) * 32,
            )
            assert spec.packing.gfx1250_scale_bytes == EXPERTS * rows * k // 32


@lru_cache(maxsize=1)
def _production_handles():
    dims = Dims(8)
    w13_shape = (
        EXPERTS,
        2 * dims.inter // 32,
        (LATENT // 32) * 32,
    )
    w2_shape = (
        EXPERTS,
        LATENT // 32,
        (dims.inter // 32) * 32,
    )
    experts = SimpleNamespace(
        quant_method=SimpleNamespace(
            is_gfx1250=True,
            is_guinterleave=False,
        ),
        w13_weight_scale=torch.nn.Parameter(
            torch.zeros(w13_shape, dtype=torch.uint8),
            requires_grad=False,
        ),
        w2_weight_scale=torch.nn.Parameter(
            torch.zeros(w2_shape, dtype=torch.uint8),
            requires_grad=False,
        ),
    )
    moe = SimpleNamespace(experts=experts)
    first = bind_k2_expert_scales(moe, 1, dims, 3, GFX1250)
    second = bind_k2_expert_scales(moe, 1, dims, 3, GFX1250)
    assert first.w13 is second.w13
    assert first.w2 is second.w2
    return first


def test_runtime_binding_reuses_cache_and_exposes_native_handles():
    handles = _production_handles()
    handles.validate()
    assert handles.w13.layout is ExpertScaleLayout.GFX1250_N32K4
    assert handles.w2.layout is ExpertScaleLayout.GFX1250_N32K4


def test_runtime_binding_requires_explicit_layout_metadata():
    dims = Dims(8)
    experts = SimpleNamespace(quant_method=SimpleNamespace(is_guinterleave=False))
    with pytest.raises(MonoUnsupported, match="metadata"):
        bind_k2_expert_scales(
            SimpleNamespace(experts=experts),
            1,
            dims,
            0,
            GFX1250,
        )
