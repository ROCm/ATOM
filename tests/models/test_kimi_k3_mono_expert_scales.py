# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU contracts for native gfx1250 Kimi expert-scale binding."""

from types import SimpleNamespace

import pytest
import torch

from atom.models.kimi_k3.mono import weights as weights_module
from atom.models.kimi_k3.mono.config import EXPERTS, LATENT, TOPK, Dims
from atom.models.kimi_k3.mono.expert_scales import bind_expert_scale
from atom.models.kimi_k3.mono.kernels.stages.moe_expert_layout import (
    ExpertScaleLayout,
    ExpertScaleSpec,
)
from atom.models.kimi_k3.mono.weights import bind_k2_expert_scales
from atom.mono.plan.architecture import GFX950, GFX1250
from atom.mono.runtime.consensus import MonoUnsupported


def _small_spec(name: str = "scale") -> ExpertScaleSpec:
    return ExpertScaleSpec(name, 2, 64, 256, 0, 2)


def _native_source(spec: ExpertScaleSpec) -> torch.Tensor:
    return (
        torch.arange(
            spec.rows * spec.groups,
            dtype=torch.int32,
        )
        .to(torch.uint8)
        .reshape(spec.storage_shape(ExpertScaleLayout.GFX1250_N32K4))
    )


def test_native_n32k4_binding_is_zero_copy():
    spec = _small_spec()
    source = _native_source(spec)
    handle = bind_expert_scale(
        source,
        spec=spec,
        source_layout=ExpertScaleLayout.GFX1250_N32K4,
        architecture=GFX1250,
    )
    handle.validate(GFX1250, spec)
    assert handle.tensor.data_ptr() == source.data_ptr()


def test_legacy_scale_layout_is_refused():
    spec = _small_spec()
    source = torch.zeros(
        spec.storage_shape(ExpertScaleLayout.GFX950_SHUFFLE),
        dtype=torch.uint8,
    )
    with pytest.raises(ValueError, match="native"):
        bind_expert_scale(
            source,
            spec=spec,
            source_layout=ExpertScaleLayout.GFX950_SHUFFLE,
            architecture=GFX1250,
        )


def test_wrong_architecture_shape_dtype_and_stride_are_refused():
    spec = _small_spec()
    source = _native_source(spec)
    with pytest.raises(ValueError, match="unsupported"):
        bind_expert_scale(
            source,
            spec=spec,
            source_layout=ExpertScaleLayout.GFX1250_N32K4,
            architecture=GFX950,
        )
    with pytest.raises(ValueError, match="shape"):
        bind_expert_scale(
            torch.zeros(spec.rows, spec.groups, dtype=torch.uint8),
            spec=spec,
            source_layout=ExpertScaleLayout.GFX1250_N32K4,
            architecture=GFX1250,
        )
    with pytest.raises(ValueError, match="one-byte"):
        bind_expert_scale(
            torch.zeros(
                spec.storage_shape(ExpertScaleLayout.GFX1250_N32K4),
                dtype=torch.int32,
            ),
            spec=spec,
            source_layout=ExpertScaleLayout.GFX1250_N32K4,
            architecture=GFX1250,
        )
    with pytest.raises(ValueError, match="contiguous"):
        bind_expert_scale(
            source.transpose(1, 2),
            spec=spec,
            source_layout=ExpertScaleLayout.GFX1250_N32K4,
            architecture=GFX1250,
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


def _production_experts(*, is_gfx1250=True, gu_interleave=False):
    dims = Dims(8)
    w13_shape = (EXPERTS, 2 * dims.inter // 32, (LATENT // 32) * 32)
    w2_shape = (EXPERTS, LATENT // 32, (dims.inter // 32) * 32)
    return SimpleNamespace(
        quant_method=SimpleNamespace(
            is_gfx1250=is_gfx1250,
            is_guinterleave=gu_interleave,
        ),
        w13_weight_scale=torch.empty(w13_shape, dtype=torch.uint8, device="meta"),
        w2_weight_scale=torch.empty(w2_shape, dtype=torch.uint8, device="meta"),
    )


def _bound_production_experts(*, gu_interleave):
    dims = Dims(8)

    def shuffled(nbytes):
        tensor = torch.empty(nbytes, dtype=torch.uint8, device="meta")
        tensor.is_shuffled = True
        return tensor

    experts = _production_experts(gu_interleave=gu_interleave)
    experts.top_k = TOPK
    experts.quant_method.fp8_activations = False
    experts.w13_weight = shuffled(EXPERTS * 2 * dims.inter * LATENT // 2)
    experts.w2_weight = shuffled(EXPERTS * LATENT * dims.inter // 2)
    experts.w13_swizzle_layout = None
    experts._mega_w1 = None
    return experts


@pytest.mark.parametrize("gu_interleave", [False, True])
def test_runtime_binding_exposes_native_zero_copy_handles(gu_interleave):
    dims = Dims(8)
    experts = _production_experts(gu_interleave=gu_interleave)
    handles = bind_k2_expert_scales(
        SimpleNamespace(experts=experts), 1, dims, 3, GFX1250
    )
    handles.validate()
    assert handles.w13.tensor.is_meta
    assert handles.w2.tensor.is_meta
    assert handles.w13.tensor.shape == experts.w13_weight_scale.shape
    assert handles.w2.tensor.shape == experts.w2_weight_scale.shape


@pytest.mark.parametrize("gu_interleave", [False, True])
def test_expert_binding_accepts_production_gfx1250_layouts(gu_interleave):
    result = weights_module._experts(
        SimpleNamespace(
            experts=_bound_production_experts(
                gu_interleave=gu_interleave,
            )
        ),
        1,
        Dims(8),
    )
    assert len(result) == 4


def test_gfx950_binding_still_refuses_interleaved_gate_up():
    experts = _bound_production_experts(gu_interleave=True)
    experts.quant_method.is_gfx1250 = False
    with pytest.raises(MonoUnsupported, match="requires the gfx1250"):
        weights_module._experts(
            SimpleNamespace(experts=experts),
            1,
            Dims(8),
        )


def test_runtime_binding_requires_explicit_native_layout_metadata():
    dims = Dims(8)
    experts = SimpleNamespace(quant_method=SimpleNamespace(is_guinterleave=False))
    with pytest.raises(MonoUnsupported, match="metadata"):
        bind_k2_expert_scales(SimpleNamespace(experts=experts), 1, dims, 0, GFX1250)
    with pytest.raises(MonoUnsupported, match="native"):
        bind_k2_expert_scales(
            SimpleNamespace(experts=_production_experts(is_gfx1250=False)),
            1,
            dims,
            0,
            GFX1250,
        )
