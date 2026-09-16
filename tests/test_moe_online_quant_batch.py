# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Numerical tests for batched MoE online re-quantization."""

from types import SimpleNamespace

import pytest
import torch

aiter = pytest.importorskip("aiter", reason="needs the AITER GPU kernel library")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a ROCm GPU"
)

from aiter import QuantType, dtypes

from atom.model_ops.moe import FusedMoE
from atom.quantization.quark.utils import (
    dequant_moe_weight_online,
    dequant_weight_online,
    quant_weight_online,
)


def _quantize_expertwise(weight, scale):
    """Reference the old w1/w3-per-expert implementation."""
    experts, rows, _ = weight.shape
    half_rows = rows // 2
    scale_half_rows = scale.shape[1] // 2
    quantized = []
    quant_scales = []
    for expert in range(experts):
        expert_q = []
        expert_s = []
        for row_slice, scale_slice in (
            (slice(0, half_rows), slice(0, scale_half_rows)),
            (slice(half_rows, rows), slice(scale_half_rows, scale.shape[1])),
        ):
            bf16 = dequant_weight_online(
                weight[expert, row_slice].contiguous(),
                scale[expert, scale_slice].contiguous(),
                QuantType.per_1x128,
                torch.float8_e4m3fnuz,
            )
            q_weight, q_scale = quant_weight_online(
                bf16,
                online_quant_type=QuantType.per_1x32,
                online_quant_dtype=dtypes.fp4x2,
            )
            expert_q.append(q_weight)
            expert_s.append(q_scale)
        quantized.append(torch.cat(expert_q))
        quant_scales.append(torch.cat(expert_s))
    return torch.stack(quantized), torch.stack(quant_scales)


def test_batched_dequant_quant_matches_expertwise():
    torch.manual_seed(7)
    experts, rows, cols = 3, 512, 256
    weight = torch.randn(experts, rows, cols, device="cuda").to(torch.float8_e4m3fnuz)
    scale = torch.rand(experts, rows // 128, cols // 128, device="cuda") + 0.125

    bf16 = dequant_moe_weight_online(
        weight,
        scale,
        QuantType.per_1x128,
        torch.float8_e4m3fnuz,
    )
    batch_q, batch_s = quant_weight_online(
        bf16,
        online_quant_type=QuantType.per_1x32,
        online_quant_dtype=dtypes.fp4x2,
    )
    batch_q = batch_q.reshape(experts, rows, cols // 2)
    batch_s = batch_s.reshape(experts, rows, cols // 32)
    ref_q, ref_s = _quantize_expertwise(weight, scale)
    torch.cuda.synchronize()

    assert torch.equal(batch_q.view(torch.uint8), ref_q.view(torch.uint8))
    assert torch.equal(batch_s.view(torch.uint8), ref_s.view(torch.uint8))


@pytest.mark.parametrize(
    ("quant_type", "scale_shape"),
    [
        (QuantType.per_Token, lambda e, n, k: (e, n)),
        (QuantType.per_1x32, lambda e, n, k: (e, n, k // 32)),
    ],
)
def test_batched_dequant_matches_expertwise_for_row_local_sources(
    quant_type, scale_shape
):
    torch.manual_seed(9)
    experts, rows, cols = 3, 64, 256
    weight = torch.randn(experts, rows, cols, device="cuda").to(torch.float8_e4m3fnuz)
    if quant_type == QuantType.per_Token:
        scale = torch.rand(scale_shape(experts, rows, cols), device="cuda") + 0.125
    else:
        # Raw E8M0 code 127 represents a scale of one.
        scale = torch.full(
            scale_shape(experts, rows, cols),
            127,
            dtype=torch.uint8,
            device="cuda",
        )

    batched = dequant_moe_weight_online(
        weight,
        scale,
        quant_type,
        torch.float8_e4m3fnuz,
    )
    reference = torch.cat(
        [
            dequant_weight_online(
                weight[expert].contiguous(),
                scale[expert].contiguous(),
                quant_type,
                torch.float8_e4m3fnuz,
            )
            for expert in range(experts)
        ]
    )
    torch.cuda.synchronize()

    assert torch.equal(batched, reference)


def test_batched_dequant_unquantized_is_zero_copy_view():
    weight = torch.randn(3, 8, 32, device="cuda")
    flat = dequant_moe_weight_online(weight, None, QuantType.No)

    assert flat.shape == (24, 32)
    assert flat.untyped_storage().data_ptr() == weight.untyped_storage().data_ptr()


@pytest.mark.parametrize(
    "target_quant_type",
    [
        QuantType.per_Token,
        QuantType.per_1x32,
        QuantType.per_1x128,
    ],
)
def test_batched_path_matches_expertwise_fp8_targets(monkeypatch, target_quant_type):
    torch.manual_seed(10)
    monkeypatch.setenv("ATOM_ONLINE_QUANT_MOE_EXPERT_BATCH_SIZE", "2")
    experts, intermediate, hidden = 3, 256, 256
    old_w13 = torch.randn(
        experts,
        2 * intermediate,
        hidden,
        dtype=torch.bfloat16,
        device="cuda",
    )
    old_w2 = torch.randn(
        experts,
        hidden,
        intermediate,
        dtype=torch.bfloat16,
        device="cuda",
    )

    target_w13 = torch.empty(
        experts,
        2 * intermediate,
        hidden,
        dtype=dtypes.fp8,
        device="cuda",
    )
    target_w2 = torch.empty(
        experts,
        hidden,
        intermediate,
        dtype=dtypes.fp8,
        device="cuda",
    )
    if target_quant_type == QuantType.per_Token:
        target_w13_scale = torch.zeros(
            experts, 2 * intermediate, dtype=torch.float32, device="cuda"
        )
        target_w2_scale = torch.zeros(
            experts, hidden, dtype=torch.float32, device="cuda"
        )
    elif target_quant_type == QuantType.per_1x32:
        target_w13_scale = torch.empty(
            experts,
            2 * intermediate,
            hidden // 32,
            dtype=dtypes.fp8_e8m0,
            device="cuda",
        )
        target_w2_scale = torch.empty(
            experts,
            hidden,
            intermediate // 32,
            dtype=dtypes.fp8_e8m0,
            device="cuda",
        )
        target_w13_scale.view(torch.uint8).zero_()
        target_w2_scale.view(torch.uint8).zero_()
    else:
        target_w13_scale = torch.zeros(
            experts,
            2 * (intermediate // 128),
            hidden // 128,
            dtype=torch.float32,
            device="cuda",
        )
        target_w2_scale = torch.zeros(
            experts,
            hidden // 128,
            intermediate // 128,
            dtype=torch.float32,
            device="cuda",
        )

    layer = SimpleNamespace(
        local_num_experts=experts,
        w13_weight=SimpleNamespace(data=target_w13),
        w2_weight=SimpleNamespace(data=target_w2),
        w13_weight_scale=SimpleNamespace(data=target_w13_scale),
        w2_weight_scale=SimpleNamespace(data=target_w2_scale),
    )
    FusedMoE._online_quant_row_local_batched(
        layer,
        old_w13_data=old_w13,
        old_w2_data=old_w2,
        old_w13_scale=None,
        old_w2_scale=None,
        source_quant_type=QuantType.No,
        source_quant_dtype=torch.bfloat16,
        online_quant_type=target_quant_type,
        online_quant_dtype=dtypes.fp8,
    )

    ref_w13 = []
    ref_w13_scale = []
    ref_w2 = []
    ref_w2_scale = []
    for expert in range(experts):
        w13_parts = []
        w13_scale_parts = []
        for rows in (
            slice(0, intermediate),
            slice(intermediate, 2 * intermediate),
        ):
            q_weight, q_scale = quant_weight_online(
                old_w13[expert, rows],
                online_quant_type=target_quant_type,
                online_quant_dtype=dtypes.fp8,
            )
            w13_parts.append(q_weight)
            w13_scale_parts.append(q_scale)
        ref_w13.append(torch.cat(w13_parts))
        ref_w13_scale.append(torch.cat(w13_scale_parts))

        q_weight, q_scale = quant_weight_online(
            old_w2[expert],
            online_quant_type=target_quant_type,
            online_quant_dtype=dtypes.fp8,
        )
        ref_w2.append(q_weight)
        ref_w2_scale.append(q_scale)

    ref_w13 = torch.stack(ref_w13)
    ref_w13_scale = torch.stack(ref_w13_scale)
    ref_w2 = torch.stack(ref_w2)
    ref_w2_scale = torch.stack(ref_w2_scale)
    torch.cuda.synchronize()

    assert torch.equal(
        target_w13.view(torch.uint8),
        ref_w13.view(torch.uint8),
    )
    assert torch.equal(
        target_w2.view(torch.uint8),
        ref_w2.view(torch.uint8),
    )
    if target_quant_type == QuantType.per_Token:
        ref_w13_scale = ref_w13_scale.squeeze(-1)
        ref_w2_scale = ref_w2_scale.squeeze(-1)
    if target_quant_type == QuantType.per_1x32:
        target_w13_scale = target_w13_scale.view(torch.uint8)
        target_w2_scale = target_w2_scale.view(torch.uint8)
        ref_w13_scale = ref_w13_scale.view(torch.uint8)
        ref_w2_scale = ref_w2_scale.view(torch.uint8)
    assert torch.equal(target_w13_scale, ref_w13_scale)
    assert torch.equal(target_w2_scale, ref_w2_scale)


def test_batched_path_preserves_target_padding(monkeypatch):
    torch.manual_seed(11)
    monkeypatch.setenv("ATOM_ONLINE_QUANT_MOE_EXPERT_BATCH_SIZE", "2")
    experts, intermediate, hidden = 3, 256, 256
    padded_intermediate, padded_hidden = 512, 512

    old_w13 = torch.randn(experts, 2 * intermediate, hidden, device="cuda").to(
        torch.float8_e4m3fnuz
    )
    old_w2 = torch.randn(experts, hidden, intermediate, device="cuda").to(
        torch.float8_e4m3fnuz
    )
    old_w13_scale = (
        torch.rand(
            experts,
            2 * (intermediate // 128),
            hidden // 128,
            device="cuda",
        )
        + 0.125
    )
    old_w2_scale = (
        torch.rand(
            experts,
            hidden // 128,
            intermediate // 128,
            device="cuda",
        )
        + 0.125
    )

    target_w13 = torch.empty(
        experts,
        2 * padded_intermediate,
        padded_hidden // 2,
        dtype=dtypes.fp4x2,
        device="cuda",
    )
    target_w2 = torch.empty(
        experts,
        padded_hidden,
        padded_intermediate // 2,
        dtype=dtypes.fp4x2,
        device="cuda",
    )
    target_w13.view(torch.uint8).zero_()
    target_w2.view(torch.uint8).zero_()
    target_w13_scale = torch.zeros(
        experts,
        2 * padded_intermediate,
        padded_hidden // 32,
        dtype=torch.uint8,
        device="cuda",
    )
    target_w2_scale = torch.zeros(
        experts,
        padded_hidden,
        padded_intermediate // 32,
        dtype=torch.uint8,
        device="cuda",
    )
    layer = SimpleNamespace(
        local_num_experts=experts,
        w13_weight=SimpleNamespace(data=target_w13),
        w2_weight=SimpleNamespace(data=target_w2),
        w13_weight_scale=SimpleNamespace(data=target_w13_scale),
        w2_weight_scale=SimpleNamespace(data=target_w2_scale),
    )

    FusedMoE._online_quant_row_local_batched(
        layer,
        old_w13_data=old_w13,
        old_w2_data=old_w2,
        old_w13_scale=old_w13_scale,
        old_w2_scale=old_w2_scale,
        source_quant_type=QuantType.per_1x128,
        source_quant_dtype=torch.float8_e4m3fnuz,
        online_quant_type=QuantType.per_1x32,
        online_quant_dtype=dtypes.fp4x2,
    )
    torch.cuda.synchronize()

    ref_w13, ref_w13_scale = _quantize_expertwise(old_w13, old_w13_scale)
    ref_w2 = []
    ref_w2_scale = []
    for expert in range(experts):
        bf16 = dequant_weight_online(
            old_w2[expert].contiguous(),
            old_w2_scale[expert].contiguous(),
            QuantType.per_1x128,
            torch.float8_e4m3fnuz,
        )
        q_weight, q_scale = quant_weight_online(
            bf16,
            online_quant_type=QuantType.per_1x32,
            online_quant_dtype=dtypes.fp4x2,
        )
        ref_w2.append(q_weight)
        ref_w2_scale.append(q_scale)
    ref_w2 = torch.stack(ref_w2)
    ref_w2_scale = torch.stack(ref_w2_scale)
    torch.cuda.synchronize()

    assert torch.equal(
        target_w13[:, :intermediate, : hidden // 2].view(torch.uint8),
        ref_w13[:, :intermediate].view(torch.uint8),
    )
    assert torch.equal(
        target_w13[
            :,
            padded_intermediate : padded_intermediate + intermediate,
            : hidden // 2,
        ].view(torch.uint8),
        ref_w13[:, intermediate:].view(torch.uint8),
    )
    assert torch.equal(
        target_w2[:, :hidden, : intermediate // 2].view(torch.uint8),
        ref_w2.view(torch.uint8),
    )
    assert torch.equal(
        target_w13_scale[:, :intermediate, : hidden // 32],
        ref_w13_scale[:, :intermediate].view(torch.uint8),
    )
    assert torch.equal(
        target_w13_scale[
            :,
            padded_intermediate : padded_intermediate + intermediate,
            : hidden // 32,
        ],
        ref_w13_scale[:, intermediate:].view(torch.uint8),
    )
    assert torch.equal(
        target_w2_scale[:, :hidden, : intermediate // 32],
        ref_w2_scale.view(torch.uint8),
    )

    # All target-only padding remains at the zero initialization expected by
    # Mxfp4MoEMethod.create_weights.
    target_w13_bytes = target_w13.view(torch.uint8)
    target_w2_bytes = target_w2.view(torch.uint8)
    assert (
        target_w13_bytes[:, intermediate:padded_intermediate].count_nonzero().item()
        == 0
    )
    assert (
        target_w13_bytes[:, padded_intermediate + intermediate :].count_nonzero().item()
        == 0
    )
    assert target_w2_bytes[:, hidden:].count_nonzero().item() == 0
    assert target_w2_bytes[:, :hidden, intermediate // 2 :].count_nonzero().item() == 0
