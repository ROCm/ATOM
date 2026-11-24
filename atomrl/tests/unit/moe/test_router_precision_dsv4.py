# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

import pytest
import torch
from atomrl.moe.router_precision import (
    _is_router,
    enable_fp32_moe_router,
    vllm_quantization_config,
)


@pytest.mark.parametrize("container_name", ["ffn", "mlp"])
def test_dsv4_gate_names_are_detected_and_return_fp32(
    container_name,
    monkeypatch,
):
    monkeypatch.setenv("ATOMRL_FP32_MOE_ROUTER", "1")
    model = torch.nn.Module()
    block = torch.nn.Module()
    container = torch.nn.Module()
    gate = torch.nn.Linear(3, 2, bias=False, dtype=torch.bfloat16)
    container.add_module("gate", gate)
    block.add_module(container_name, container)
    model.add_module("model", block)

    assert enable_fp32_moe_router(model) == 1
    assert gate.weight.dtype == torch.bfloat16

    output = gate(torch.ones(2, 3, dtype=torch.bfloat16))
    assert output.dtype == torch.float32


@pytest.mark.parametrize(
    "name",
    [
        "model.ffn.gatekeeper",
        "model.ffn.gate.projection",
        "model.mlp.gated",
        "model.attention.gate",
    ],
)
def test_similarly_named_non_router_modules_are_rejected(name):
    assert not _is_router(name, torch.nn.Linear(3, 2, bias=False))


def test_fp32_router_refuses_quantized_gate(monkeypatch):
    monkeypatch.setenv("ATOMRL_FP32_MOE_ROUTER", "1")
    model = torch.nn.Module()
    model.add_module("model", torch.nn.Module())
    model.model.add_module("mlp", torch.nn.Module())
    gate = torch.nn.Linear(3, 2, bias=False)
    gate.weight = torch.nn.Parameter(
        gate.weight.to(torch.float8_e4m3fn), requires_grad=False
    )
    model.model.mlp.add_module("gate", gate)

    with pytest.raises(RuntimeError, match="mlp.gate is quantized"):
        enable_fp32_moe_router(model)
    monkeypatch.setenv("ATOMRL_FP32_MOE_ROUTER", "0")
    assert enable_fp32_moe_router(model) == 0


@pytest.mark.parametrize(
    ("quantization", "fp32", "ignored"),
    [
        ("fp8_per_block", "1", True),
        ("fp8_per_block", "0", False),
        ("fp8", "1", False),
        ("", "1", False),
    ],
)
def test_online_quantization_keeps_routers_unquantized_for_fp32_router(
    monkeypatch, quantization, fp32, ignored
):
    pytest.importorskip("vllm", exc_type=ImportError)
    from vllm.model_executor.layers.quantization.compressed_tensors.utils import (
        should_ignore_layer,
    )

    monkeypatch.setenv("ATOMRL_FP32_MOE_ROUTER", fp32)
    quant_cfg = vllm_quantization_config(quantization)

    assert (quant_cfg is not None) == ignored
    if ignored:
        ignore = quant_cfg["ignore"]
        assert should_ignore_layer("model.layers.3.mlp.gate", ignore=ignore)
        assert should_ignore_layer("model.layers.3.ffn.gate", ignore=ignore)
        for name in (
            "model.layers.3.mlp.gate_up_proj",
            "model.layers.3.mlp.experts",
            "model.layers.3.self_attn.qkv_proj",
        ):
            assert not should_ignore_layer(
                name,
                ignore=ignore,
                fused_mapping={
                    "gate_up_proj": ["gate_proj", "up_proj"],
                    "qkv_proj": ["q_proj", "k_proj", "v_proj"],
                },
            )


@pytest.mark.parametrize("name", ["model.ffn.gate", "model.mlp.gate"])
def test_router_names_with_non_matrix_weights_are_rejected(name):
    module = torch.nn.Module()
    module.weight = torch.nn.Parameter(torch.ones(3))

    assert not _is_router(name, module)
