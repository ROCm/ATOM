# SPDX-License-Identifier: MIT

import sys
import types
import enum

import pytest
import torch


def _prepare_real_linear_import():
    atom_pkg = sys.modules.get("atom")
    atom_root = atom_pkg.__path__[0]
    for mod_name in list(sys.modules):
        if mod_name == "atom.model_ops" or mod_name.startswith("atom.model_ops."):
            del sys.modules[mod_name]

    model_ops_pkg = types.ModuleType("atom.model_ops")
    model_ops_pkg.__path__ = [f"{atom_root}/model_ops"]
    model_ops_pkg.__package__ = "atom.model_ops"
    sys.modules["atom.model_ops"] = model_ops_pkg

    atom_config = sys.modules.get("atom.config")
    atom_config.QuantizationConfig = getattr(
        atom_config, "QuantizationConfig", type("QuantizationConfig", (), {})
    )
    atom_config.get_current_atom_config = lambda: types.SimpleNamespace(
        torch_dtype=torch.bfloat16,
    )


class _QuantType(enum.IntEnum):
    No = 0
    per_Token = 1
    per_Tensor = 2
    per_1x32 = 3
    per_1x128 = 4


def _install_minimal_aiter_stubs(monkeypatch):
    aiter = types.ModuleType("aiter")
    aiter.__path__ = []
    aiter.ActivationType = enum.Enum(
        "ActivationType",
        {"Silu": "silu", "Swiglu": "swiglu"},
    )
    aiter.QuantType = _QuantType
    monkeypatch.setitem(sys.modules, "aiter", aiter)

    jit_chip_info = types.ModuleType("aiter.jit.utils.chip_info")
    jit_chip_info.get_gfx = lambda: ""
    monkeypatch.setitem(sys.modules, "aiter.jit.utils.chip_info", jit_chip_info)

    for name in (
        "aiter.ops",
        "aiter.ops.triton",
        "aiter.ops.triton.fusions",
    ):
        mod = types.ModuleType(name)
        mod.__path__ = []
        monkeypatch.setitem(sys.modules, name, mod)
    routing_mod = types.ModuleType("aiter.ops.triton.fusions.fused_routing_from_topk")
    routing_mod.fused_routing_from_topk = None
    monkeypatch.setitem(
        sys.modules,
        "aiter.ops.triton.fusions.fused_routing_from_topk",
        routing_mod,
    )
    act_mod = types.ModuleType("aiter.ops.triton.fusions.fused_clamp_act_mul")
    act_mod.fused_clamp_act_mul = None
    monkeypatch.setitem(
        sys.modules,
        "aiter.ops.triton.fusions.fused_clamp_act_mul",
        act_mod,
    )

    utils_mod = types.ModuleType("atom.model_ops.utils")
    utils_mod.has_triton_kernels = lambda: False
    monkeypatch.setitem(sys.modules, "atom.model_ops.utils", utils_mod)


def test_linear_static_lora_requires_lora_x_for_scaled_quantized_input(monkeypatch):
    aiter = pytest.importorskip("aiter")
    _prepare_real_linear_import()
    import atom.model_ops.linear as linear_mod
    from atom.model_ops.linear import ReplicatedLinear

    class FakeTPGroup:
        rank_in_group = 0
        world_size = 1

    monkeypatch.setattr(linear_mod, "get_tp_group", lambda: FakeTPGroup())
    quant_config = types.SimpleNamespace(
        get_layer_quant_config=lambda _prefix: types.SimpleNamespace(
            quant_type=aiter.QuantType.per_Tensor,
            quant_dtype=torch.float8_e4m3fn,
            is_dynamic=True,
        )
    )
    layer = ReplicatedLinear(
        2,
        2,
        quant_config=quant_config,
        prefix="test.linear",
    )
    layer.add_lora_adapter(
        torch.ones(1, 2, dtype=torch.bfloat16),
        torch.ones(2, 1, dtype=torch.bfloat16),
        1.0,
    )

    with pytest.raises(ValueError, match="requires unquantized lora_x"):
        layer.forward(
            torch.zeros(1, 2, dtype=torch.float8_e4m3fn),
            x_scale=torch.ones(1, 1),
        )


def test_build_static_lora_routing_maps_only_valid_blocks(monkeypatch):
    _prepare_real_linear_import()
    _install_minimal_aiter_stubs(monkeypatch)
    import atom.model_ops.fused_moe_triton as triton_mod

    class FakeOps:
        @staticmethod
        def moe_lora_align_block_size(
            _topk_ids,
            _token_lora_mapping,
            _num_experts,
            _block_size,
            _max_loras,
            _max_num_tokens_padded,
            _max_num_m_blocks,
            _sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            _adapter_enabled,
            _lora_ids,
        ):
            expert_ids.copy_(torch.tensor([2, 99, -7], dtype=torch.int32))
            num_tokens_post_padded.fill_(2)

    monkeypatch.setitem(sys.modules, "vllm", types.SimpleNamespace(_custom_ops=FakeOps))

    _, expert_ids, _, _, _, _ = triton_mod._build_static_lora_routing(
        torch.zeros(3, 1, dtype=torch.int32),
        num_experts=3,
        block_size=2,
        expert_map=torch.tensor([0, 1, 4], dtype=torch.int32),
    )

    assert expert_ids.view(-1).tolist() == [4, -1, -1]
