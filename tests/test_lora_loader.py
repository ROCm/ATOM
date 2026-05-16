# SPDX-License-Identifier: MIT

import json

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from atom.model_loader.lora import (
    _dequantize_fp8_blocks,
    apply_lora_adapters,
    load_lora_tensors,
    parse_lora_module_entry,
    parse_lora_tensor_name,
    resolve_lora_target,
    slice_lora_tensors_for_module,
    validate_lora_adapters_supported,
)


class FakeLinear(nn.Module):
    def __init__(
        self,
        input_size=4,
        output_size=4,
        tp_dim=None,
        tp_rank=0,
        tp_size=1,
        output_sizes=None,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.tp_dim = tp_dim
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.output_sizes = output_sizes or [output_size]
        self.weight = nn.Parameter(
            torch.empty(output_size, input_size, dtype=torch.bfloat16)
        )
        self.loaded_loras = []

    def add_lora_adapter(self, lora_a, lora_b, scaling, output_offset=None, **kwargs):
        self.loaded_loras.append((lora_a, lora_b, scaling, output_offset, kwargs))


class FakeQKVLinear(FakeLinear):
    def __init__(self):
        super().__init__(input_size=3, output_size=4, tp_dim=0, tp_rank=1, tp_size=2)
        self.num_heads = 2
        self.head_size = 1
        self.num_kv_heads = 1
        self.v_head_size = 1
        self.num_kv_head_replicas = 1
        self.output_partition_sizes = [2, 1, 1]


class FakeQKVGLinear(FakeQKVLinear):
    def __init__(self):
        super().__init__()
        self.output_partition_sizes = [2, 2, 1, 1]

    def _deinterleave(self, weight, head_stride=None):
        hs = head_stride if head_stride is not None else self.head_size
        return (
            weight.view(self.num_heads, 2, hs, -1)
            .transpose(0, 1)
            .reshape(self.num_heads * 2 * hs, -1)
        )


class FakeFusedMoE(nn.Module):
    def __init__(self, dtype=torch.bfloat16, tp_rank=1, tp_size=2):
        super().__init__()
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.intermediate_size_per_partition = 2
        self.expert_map = None
        self.quant_method = None
        self.w13_weight = nn.Parameter(
            torch.zeros(2, 4, 3, dtype=dtype),
            requires_grad=False,
        )
        self.w2_weight = nn.Parameter(
            torch.zeros(2, 3, 2, dtype=dtype),
            requires_grad=False,
        )


class FakeFp8BlockFusedMoE(nn.Module):
    def __init__(self):
        super().__init__()
        self.tp_rank = 0
        self.tp_size = 1
        self.intermediate_size_per_partition = 2
        self.expert_map = None
        self.quant_method = object()
        self.w13_weight = nn.Parameter(
            torch.zeros(1, 4, 4).to(torch.float8_e4m3fn),
            requires_grad=False,
        )
        self.w2_weight = nn.Parameter(
            torch.zeros(1, 4, 2).to(torch.float8_e4m3fn),
            requires_grad=False,
        )
        self.w13_weight_scale = nn.Parameter(
            torch.ones(1, 2, 2),
            requires_grad=False,
        )
        self.w2_weight_scale = nn.Parameter(
            torch.ones(1, 2, 1),
            requires_grad=False,
        )


def _write_adapter(path, tensors, r=2, alpha=4):
    path.mkdir()
    (path / "adapter_config.json").write_text(
        json.dumps({"r": r, "lora_alpha": alpha, "target_modules": ["q_proj"]}),
        encoding="utf-8",
    )
    save_file(tensors, path / "adapter_model.safetensors")


def test_parse_lora_module_entry_supports_name_equals_path():
    assert parse_lora_module_entry("adapter=/tmp/lora").name == "adapter"
    assert parse_lora_module_entry("/tmp/my-lora").name == "my-lora"


def test_parse_lora_tensor_name_strips_peft_prefix_and_default_slot():
    parsed = parse_lora_tensor_name(
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight"
    )
    assert parsed == ("model.layers.0.self_attn.q_proj", "A")


def test_lora_scaling_respects_zero_alpha_pattern(tmp_path):
    adapter_path = tmp_path / "adapter"
    lora_a = torch.ones(2, 3)
    lora_b = torch.ones(4, 2)
    _write_adapter(
        adapter_path,
        {
            "base_model.model.q_proj.lora_A.weight": lora_a,
            "base_model.model.q_proj.lora_B.weight": lora_b,
        },
    )
    config_path = adapter_path / "adapter_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["alpha_pattern"] = {"q_proj": 0}
    config_path.write_text(json.dumps(config), encoding="utf-8")

    pairs = load_lora_tensors(str(adapter_path))

    assert pairs[0].scaling == 0.0


def test_slice_row_parallel_lora_shards_input_dim():
    module = FakeLinear(input_size=3, output_size=2, tp_dim=1, tp_rank=1, tp_size=2)
    lora_a = torch.arange(12, dtype=torch.float32).reshape(2, 6)
    lora_b = torch.arange(4, dtype=torch.float32).reshape(2, 2)

    local_a, local_b, output_offset = slice_lora_tensors_for_module(
        module, lora_a, lora_b, shard_id=None
    )

    assert torch.equal(local_a, lora_a[:, 3:6])
    assert torch.equal(local_b, lora_b)
    assert output_offset is None


def test_slice_merged_column_lora_shards_output_dim():
    module = FakeLinear(
        input_size=3,
        output_size=7,
        tp_dim=0,
        tp_rank=1,
        tp_size=2,
        output_sizes=[6, 8],
    )
    lora_a = torch.ones(1, 3)
    lora_b = torch.arange(8, dtype=torch.float32).reshape(8, 1)

    local_a, local_b, output_offset = slice_lora_tensors_for_module(
        module, lora_a, lora_b, shard_id=1
    )

    assert torch.equal(local_a, lora_a)
    assert torch.equal(local_b, lora_b[4:8])
    assert output_offset == 3


def test_apply_lora_adapters_loads_packed_qkv_adapter(tmp_path):
    adapter_path = tmp_path / "adapter"
    lora_a = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    lora_b = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    _write_adapter(
        adapter_path,
        {
            "base_model.model.q_proj.lora_A.weight": lora_a,
            "base_model.model.q_proj.lora_B.weight": lora_b,
        },
    )
    model = nn.Module()
    model.qkv_proj = FakeQKVLinear()

    apply_lora_adapters(
        model,
        [f"test={adapter_path}"],
        packed_modules_mapping={"q_proj": ("qkv_proj", "q")},
    )

    loaded_a, loaded_b, scaling, output_offset, kwargs = model.qkv_proj.loaded_loras[0]
    assert torch.equal(loaded_a.cpu(), lora_a.to(torch.bfloat16))
    assert torch.equal(loaded_b.cpu(), lora_b[2:4].to(torch.bfloat16))
    assert scaling == 2.0
    assert output_offset == 0
    assert kwargs["adapter_name"] == "test"


def test_qkvg_q_lora_targets_q_slice_after_gate_slice():
    module = FakeQKVGLinear()
    lora_a = torch.ones(2, 3)
    lora_b = torch.arange(16, dtype=torch.float32).reshape(8, 2)

    _, local_b, output_offset = slice_lora_tensors_for_module(
        module, lora_a, lora_b, shard_id="q"
    )

    assert torch.equal(
        local_b,
        torch.cat([lora_b[5:6], lora_b[7:8], lora_b[4:5], lora_b[6:7]], dim=0),
    )
    assert output_offset == 0


def test_apply_lora_adapters_maps_glm_shared_expert_down_proj(tmp_path):
    adapter_path = tmp_path / "adapter"
    lora_a = torch.arange(12, dtype=torch.float32).reshape(2, 6)
    lora_b = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    _write_adapter(
        adapter_path,
        {
            "base_model.model.model.layers.10.mlp.shared_experts.down_proj."
            "lora_A.weight": lora_a,
            "base_model.model.model.layers.10.mlp.shared_experts.down_proj."
            "lora_B.weight": lora_b,
        },
    )
    model = nn.Module()
    model.model = nn.Module()
    model.model.layers = nn.ModuleList(nn.Module() for _ in range(11))
    model.model.layers[10].mlp = nn.Module()
    model.model.layers[10].mlp.shared_experts = nn.Module()
    model.model.layers[10].mlp.shared_experts.w2 = FakeLinear(
        input_size=3,
        output_size=4,
        tp_dim=1,
        tp_rank=1,
        tp_size=2,
    )

    apply_lora_adapters(
        model,
        [f"test={adapter_path}"],
        packed_modules_mapping={
            "shared_experts.down_proj": ("shared_experts.w2", None)
        },
    )

    loaded_a, loaded_b, scaling, output_offset, _ = (
        model.model.layers[10].mlp.shared_experts.w2.loaded_loras[0]
    )
    assert torch.equal(loaded_a.cpu(), lora_a[:, 3:6].to(torch.bfloat16))
    assert torch.equal(loaded_b.cpu(), lora_b.to(torch.bfloat16))
    assert scaling == 2.0
    assert output_offset is None


def test_apply_lora_adapters_merges_routed_expert_lora_into_bf16_moe(tmp_path):
    adapter_path = tmp_path / "adapter"
    gate_a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    gate_b = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    down_a = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    down_b = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    _write_adapter(
        adapter_path,
        {
            "base_model.model.model.layers.10.mlp.experts.1.gate_proj."
            "lora_A.weight": gate_a,
            "base_model.model.model.layers.10.mlp.experts.1.gate_proj."
            "lora_B.weight": gate_b,
            "base_model.model.model.layers.10.mlp.experts.1.down_proj."
            "lora_A.weight": down_a,
            "base_model.model.model.layers.10.mlp.experts.1.down_proj."
            "lora_B.weight": down_b,
        },
    )
    model = nn.Module()
    model.model = nn.Module()
    model.model.layers = nn.ModuleList(nn.Module() for _ in range(11))
    model.model.layers[10].mlp = nn.Module()
    model.model.layers[10].mlp.experts = FakeFusedMoE()

    apply_lora_adapters(model, [f"test={adapter_path}"])

    experts = model.model.layers[10].mlp.experts
    assert torch.allclose(
        experts.w13_weight[1, :2].float(),
        (gate_b[2:4] @ gate_a * 2.0).bfloat16().float(),
    )
    assert torch.allclose(
        experts.w2_weight[1].float(),
        (down_b @ down_a[:, 2:4] * 2.0).bfloat16().float(),
    )


def test_apply_lora_adapters_merges_routed_expert_lora_into_fp8_block_moe(
    tmp_path,
):
    adapter_path = tmp_path / "adapter"
    lora_a = torch.ones(1, 4)
    lora_b = torch.ones(2, 1)
    _write_adapter(
        adapter_path,
        {
            "base_model.model.model.layers.0.mlp.experts.0.gate_proj."
            "lora_A.weight": lora_a,
            "base_model.model.model.layers.0.mlp.experts.0.gate_proj."
            "lora_B.weight": lora_b,
            "base_model.model.model.layers.0.mlp.experts.0.up_proj."
            "lora_A.weight": lora_a.clone(),
            "base_model.model.model.layers.0.mlp.experts.0.up_proj."
            "lora_B.weight": lora_b * 2,
        },
        r=1,
        alpha=1,
    )
    model = nn.Module()
    model.model = nn.Module()
    model.model.layers = nn.ModuleList([nn.Module()])
    model.model.layers[0].mlp = nn.Module()
    model.model.layers[0].mlp.experts = FakeFp8BlockFusedMoE()

    apply_lora_adapters(model, [f"test={adapter_path}"])

    experts = model.model.layers[0].mlp.experts
    dequant = _dequantize_fp8_blocks(
        experts.w13_weight[0, :2],
        experts.w13_weight_scale[0, :1],
        block_n=2,
        block_k=2,
    )
    assert torch.allclose(dequant, torch.ones(2, 4), atol=0.125)
    dequant = _dequantize_fp8_blocks(
        experts.w13_weight[0, 2:],
        experts.w13_weight_scale[0, 1:],
        block_n=2,
        block_k=2,
    )
    assert torch.allclose(dequant, torch.full((2, 4), 2.0), atol=0.25)


def test_resolve_lora_target_rejects_routed_fused_moe_experts():
    with pytest.raises(NotImplementedError, match="routed FusedMoE expert"):
        resolve_lora_target(
            "model.layers.10.mlp.experts.0.down_proj",
            {},
            {},
        )


def test_validate_lora_adapters_accepts_routed_fused_moe_experts(tmp_path):
    adapter_path = tmp_path / "adapter"
    _write_adapter(
        adapter_path,
        {
            "base_model.model.model.layers.10.mlp.experts.0.down_proj."
            "lora_A.weight": torch.ones(2, 3),
            "base_model.model.model.layers.10.mlp.experts.0.down_proj."
            "lora_B.weight": torch.ones(4, 2),
        },
    )

    validate_lora_adapters_supported([f"user={adapter_path}"])
