# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Numerical tests for batched MoE online re-quantization."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

aiter = pytest.importorskip("aiter", reason="needs the AITER GPU kernel library")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a ROCm GPU"
)

from aiter import QuantType, dtypes

import atom.model_ops.moe as moe_mod
from atom.model_ops.moe import FusedMoE
from atom.quant_spec import LayerQuantConfig
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
        intermediate_size_per_partition=intermediate,
        hidden_size=hidden,
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
        intermediate_size_per_partition=intermediate,
        hidden_size=hidden,
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


def test_batched_path_trims_the_source_intermediate_padding(monkeypatch):
    """An MXFP4 source pads the intermediate rows; each converted half (and
    w2's columns) is the unpadded rows' quantization."""
    torch.manual_seed(12)
    monkeypatch.setenv("ATOM_ONLINE_QUANT_MOE_EXPERT_BATCH_SIZE", "2")
    experts, inter, padded, hidden = 3, 192, 256, 256
    old_w13 = torch.randn(
        experts, 2 * padded, hidden, dtype=torch.bfloat16, device="cuda"
    )
    old_w2 = torch.randn(experts, hidden, padded, dtype=torch.bfloat16, device="cuda")
    w13 = torch.empty(experts, 2 * inter, hidden, dtype=dtypes.fp8, device="cuda")
    w2 = torch.empty(experts, hidden, inter, dtype=dtypes.fp8, device="cuda")
    w13_s = torch.zeros(experts, 2 * inter, dtype=torch.float32, device="cuda")
    w2_s = torch.zeros(experts, hidden, dtype=torch.float32, device="cuda")
    layer = SimpleNamespace(
        local_num_experts=experts,
        intermediate_size_per_partition=inter,
        hidden_size=hidden,
        w13_weight=SimpleNamespace(data=w13),
        w2_weight=SimpleNamespace(data=w2),
        w13_weight_scale=SimpleNamespace(data=w13_s),
        w2_weight_scale=SimpleNamespace(data=w2_s),
    )
    FusedMoE._online_quant_row_local_batched(
        layer,
        old_w13_data=old_w13,
        old_w2_data=old_w2,
        old_w13_scale=None,
        old_w2_scale=None,
        source_quant_type=QuantType.No,
        source_quant_dtype=torch.bfloat16,
        online_quant_type=QuantType.per_Token,
        online_quant_dtype=dtypes.fp8,
    )

    def quant(x):
        q, s = quant_weight_online(
            x.contiguous(),
            online_quant_type=QuantType.per_Token,
            online_quant_dtype=dtypes.fp8,
        )
        return q.view(torch.uint8), s.squeeze(-1)

    for e in range(experts):
        for rows, src in (
            (slice(0, inter), old_w13[e, :inter]),
            (slice(inter, 2 * inter), old_w13[e, padded : padded + inter]),
        ):
            q, s = quant(src)
            assert torch.equal(w13[e, rows].view(torch.uint8), q)
            assert torch.equal(w13_s[e, rows], s)
        q, s = quant(old_w2[e, :, :inter])
        assert torch.equal(w2[e].view(torch.uint8), q)
        assert torch.equal(w2_s[e], s)


def test_per_expert_path_trims_the_source_intermediate_padding(monkeypatch):
    """A per-token target at TP 2 gathers w2, so the conversion runs the
    per-expert loop: it quantizes an MXFP4 source's unpadded w1/w3 rows and w2
    columns."""
    torch.manual_seed(13)
    experts, inter, padded, hidden = 2, 64, 128, 64
    source = LayerQuantConfig(quant_type=QuantType.per_1x32, quant_dtype=dtypes.fp4x2)
    target = LayerQuantConfig(quant_type=QuantType.per_Token, quant_dtype=dtypes.fp8)

    def mxfp4(rows, cols):
        bytes_ = torch.randint(0, 256, (experts, rows, cols // 2), dtype=torch.uint8)
        scale = torch.randint(120, 131, (experts, rows, cols // 32), dtype=torch.uint8)
        return bytes_.view(dtypes.fp4x2), scale.view(dtypes.fp8_e8m0)

    w13, w13_s = mxfp4(2 * padded, hidden)
    w2, w2_s = mxfp4(hidden, padded)

    class Target:
        def create_weights(self, layer, **_kwargs):
            pass  # the stubbed loaders below take the quantized tensors

    monkeypatch.setattr(moe_mod, "Fp8MoEMethod", lambda *_a, **_k: Target())
    quantized = []
    monkeypatch.setattr(
        moe_mod,
        "quant_weight_online",
        lambda w, **_k: quantized.append(w.clone()) or (w, None),
    )
    monkeypatch.setattr(
        moe_mod, "get_tp_group", lambda: SimpleNamespace(all_gather=lambda x, dim: x)
    )
    layer = SimpleNamespace(
        online_quant=True,
        quant_config=SimpleNamespace(get_layer_quant_config=lambda *_a, **_k: target),
        layer_name="model.layers.0.ffn.experts",
        layer_quant_config=source,
        source_is_nvfp4=False,
        params_dtype=dtypes.fp4x2,
        quant_method=SimpleNamespace(
            hidden_pad=0,
            intermediate_pad=padded - inter,
            need_normalize_e4m3fn_to_e4m3fnuz=False,
        ),
        w13_bias=None,
        has_bias=False,
        moe_config=SimpleNamespace(),
        moe_quant_params={"params_dtype": dtypes.fp4x2},
        _stream_online_quant=False,
        local_num_experts=experts,
        intermediate_size_per_partition=inter,
        hidden_size=hidden,
        use_ep=False,
        tp_size=2,
        tp_rank=0,
        w13_weight=nn.Parameter(w13, requires_grad=False),
        w2_weight=nn.Parameter(w2, requires_grad=False),
        w13_weight_scale=nn.Parameter(w13_s, requires_grad=False),
        w2_weight_scale=nn.Parameter(w2_s, requires_grad=False),
        _copy_quant_storage=FusedMoE._copy_quant_storage,
        _load_model_weight_or_group_weight_scale=lambda **_k: None,
        _load_quant_weight_scale=lambda **_k: None,
    )
    FusedMoE._online_quant(layer)

    assert len(quantized) == 3 * experts
    for e in range(experts):
        full13 = dequant_weight_online(
            w13[e], w13_s[e], source.quant_type, source.quant_dtype
        )
        full2 = dequant_weight_online(
            w2[e], w2_s[e], source.quant_type, source.quant_dtype
        )
        w1, w3, w2_q = quantized[3 * e : 3 * e + 3]
        assert torch.equal(w1, full13[:inter])
        assert torch.equal(w3, full13[padded : padded + inter])
        assert torch.equal(w2_q, full2[:, :inter])


@pytest.mark.parametrize(("hidden_pad", "bias"), [(64, None), (0, torch.zeros(1))])
def test_mxfp4_source_with_hidden_padding_or_bias_is_refused(hidden_pad, bias):
    """The converted experts carry neither an MXFP4 source's hidden padding nor
    its expert bias, which their consumers keep."""
    target = LayerQuantConfig(quant_type=QuantType.per_Token, quant_dtype=dtypes.fp8)
    layer = SimpleNamespace(
        online_quant=True,
        quant_config=SimpleNamespace(get_layer_quant_config=lambda *_a, **_k: target),
        layer_name="model.layers.0.ffn.experts",
        layer_quant_config=LayerQuantConfig(
            quant_type=QuantType.per_1x32, quant_dtype=dtypes.fp4x2
        ),
        source_is_nvfp4=False,
        params_dtype=dtypes.fp4x2,
        quant_method=SimpleNamespace(hidden_pad=hidden_pad),
        w13_bias=bias,
    )
    with pytest.raises(NotImplementedError, match="hidden padding or expert bias"):
        FusedMoE._online_quant(layer)
