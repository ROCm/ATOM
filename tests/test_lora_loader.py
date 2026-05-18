# SPDX-License-Identifier: MIT

import json
import os
import sys
import types

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from atom.model_loader.lora import (
    _dequantize_fp8_blocks,
    any_module_has_static_lora_adapters,
    apply_lora_adapters,
    load_lora_tensors,
    mark_static_routed_lora_targets,
    module_has_static_lora_adapters,
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

    def add_static_routed_lora(
        self, expert_id, proj, lora_a, lora_b, scaling, adapter_name
    ):
        if not hasattr(self, "runtime_loras"):
            self.runtime_loras = []
        self.runtime_loras.append(
            (expert_id, proj, lora_a.clone(), lora_b.clone(), scaling, adapter_name)
        )


class FakeParallelLMHead(nn.Module):
    def __init__(self, tp_rank=1, tp_size=2):
        super().__init__()
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.num_embeddings = 6
        self.num_embeddings_per_partition = self.num_embeddings // self.tp_size
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition
        self.weight = nn.Parameter(
            torch.zeros(self.num_embeddings_per_partition, 4, dtype=torch.bfloat16),
            requires_grad=False,
        )
        self.loaded_loras = []

    def add_lora_adapter(self, lora_a, lora_b, scaling, **kwargs):
        self.loaded_loras.append((lora_a, lora_b, scaling, kwargs))


def _write_adapter(path, tensors, r=2, alpha=4):
    path.mkdir()
    (path / "adapter_config.json").write_text(
        json.dumps({"r": r, "lora_alpha": alpha, "target_modules": ["q_proj"]}),
        encoding="utf-8",
    )
    save_file(tensors, path / "adapter_model.safetensors")


def _prepare_real_model_ops_import():
    atom_pkg = sys.modules.get("atom")
    if atom_pkg is not None:
        atom_root = atom_pkg.__path__[0]
    else:
        atom_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), "atom")

    for mod_name in list(sys.modules):
        if mod_name == "atom.model_ops" or mod_name.startswith("atom.model_ops."):
            del sys.modules[mod_name]

    model_ops_pkg = types.ModuleType("atom.model_ops")
    model_ops_pkg.__path__ = [os.path.join(atom_root, "model_ops")]
    model_ops_pkg.__package__ = "atom.model_ops"
    sys.modules["atom.model_ops"] = model_ops_pkg

    atom_config = sys.modules.get("atom.config")
    if atom_config is not None:
        atom_config.QuantizationConfig = getattr(
            atom_config, "QuantizationConfig", type("QuantizationConfig", (), {})
        )
        atom_config.CompilationConfig = getattr(
            atom_config, "CompilationConfig", type("CompilationConfig", (), {})
        )
        atom_config.CompilationLevel = getattr(
            atom_config,
            "CompilationLevel",
            types.SimpleNamespace(
                NO_COMPILATION=0,
                DYNAMO_AS_IS=1,
                DYNAMO_ONCE=2,
                PIECEWISE=3,
            ),
        )
        atom_config.LayerQuantConfig = getattr(
            atom_config, "LayerQuantConfig", type("LayerQuantConfig", (), {})
        )
        fake_quant_config = types.SimpleNamespace(
            exclude_layers=[],
            quant_dtype="bf16",
            get_layer_quant_config=lambda _name: types.SimpleNamespace(
                quant_dtype="bf16"
            ),
        )
        atom_config.get_current_atom_config = lambda: types.SimpleNamespace(
            enable_dp_attention=False,
            enable_expert_parallel=False,
            max_num_batched_tokens=64,
            torch_dtype=torch.bfloat16,
            compilation_config=types.SimpleNamespace(static_forward_context={}),
            quant_config=fake_quant_config,
            parallel_config=types.SimpleNamespace(data_parallel_size=1),
            lora_modules=None,
        )


def test_parse_lora_module_entry_supports_name_equals_path():
    assert parse_lora_module_entry("adapter=/tmp/lora").name == "adapter"
    assert parse_lora_module_entry("/tmp/my-lora").name == "my-lora"


def test_static_lora_helper_detects_registered_adapters():
    without_lora = nn.Module()
    with_lora = nn.Module()
    with_lora._static_lora_adapters = [("a", "b", 1.0, None)]

    assert not module_has_static_lora_adapters(None)
    assert not module_has_static_lora_adapters(without_lora)
    assert module_has_static_lora_adapters(with_lora)
    assert any_module_has_static_lora_adapters(without_lora, with_lora)


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


def test_load_lora_tensors_rejects_non_2d_tensors(tmp_path):
    adapter_path = tmp_path / "adapter"
    _write_adapter(
        adapter_path,
        {
            "base_model.model.q_proj.lora_A.weight": torch.ones(2, 3, 1),
            "base_model.model.q_proj.lora_B.weight": torch.ones(4, 2),
        },
    )

    with pytest.raises(ValueError, match="q_proj.*must be 2D"):
        load_lora_tensors(str(adapter_path))


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


def test_slice_lora_tensors_rejects_b_output_mismatch():
    module = FakeLinear(input_size=3, output_size=4)

    with pytest.raises(ValueError, match="LoRA B output mismatch"):
        slice_lora_tensors_for_module(
            module,
            torch.ones(2, 3),
            torch.ones(5, 2),
            shard_id=None,
        )


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


def test_apply_lora_adapters_registers_routed_expert_lora_for_fp8_block_moe(
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
    assert [(e, p, s, name) for e, p, _, _, s, name in experts.runtime_loras] == [
        (0, "gate_proj", 1.0, "test"),
        (0, "up_proj", 1.0, "test"),
    ]
    assert torch.equal(
        experts.w13_weight[0, :2], torch.zeros_like(experts.w13_weight[0, :2])
    )


def test_mark_static_routed_lora_targets_preserves_unshuffled_moe_weights(tmp_path):
    adapter_path = tmp_path / "adapter"
    _write_adapter(
        adapter_path,
        {
            "base_model.model.model.layers.0.mlp.experts.0.gate_proj."
            "lora_A.weight": torch.ones(1, 4),
            "base_model.model.model.layers.0.mlp.experts.0.gate_proj."
            "lora_B.weight": torch.ones(2, 1),
            "base_model.model.model.layers.0.self_attn.q_a_proj."
            "lora_A.weight": torch.ones(1, 4),
            "base_model.model.model.layers.0.self_attn.q_a_proj."
            "lora_B.weight": torch.ones(2, 1),
        },
        r=1,
        alpha=1,
    )
    model = nn.Module()
    model.model = nn.Module()
    model.model.layers = nn.ModuleList([nn.Module(), nn.Module()])
    model.model.layers[0].mlp = nn.Module()
    model.model.layers[0].mlp.experts = FakeFp8BlockFusedMoE()
    model.model.layers[1].mlp = nn.Module()
    model.model.layers[1].mlp.experts = FakeFp8BlockFusedMoE()

    mark_static_routed_lora_targets(model, [f"test={adapter_path}"])

    assert model.model.layers[0].mlp.experts._static_routed_lora_requires_unshuffled
    assert not hasattr(
        model.model.layers[1].mlp.experts,
        "_static_routed_lora_requires_unshuffled",
    )


def test_apply_lora_adapters_registers_routed_expert_lora_when_supported(tmp_path):
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
    assert len(experts.runtime_loras) == 1
    expert_id, proj, loaded_a, loaded_b, scaling, adapter_name = experts.runtime_loras[
        0
    ]
    assert expert_id == 0
    assert proj == "gate_proj"
    assert adapter_name == "test"
    assert scaling == 1.0
    assert torch.equal(loaded_a, lora_a)
    assert torch.equal(loaded_b, lora_b)
    assert torch.equal(
        experts.w13_weight[0, :2], torch.zeros_like(experts.w13_weight[0, :2])
    )


def test_static_routed_lora_reference_matches_manual_moe():
    pytest.importorskip("aiter")
    _prepare_real_model_ops_import()
    from atom.model_ops.moe import _apply_static_routed_lora_reference

    layer = nn.Module()
    layer.intermediate_size_per_partition = 2
    layer.local_num_experts = 2
    layer.quant_method = None
    layer._static_routed_lora_dtype = torch.float32
    layer.w13_weight = nn.Parameter(
        torch.tensor(
            [
                [[0.2, -0.1, 0.3], [0.4, 0.0, -0.2], [0.1, 0.5, 0.2], [-0.3, 0.2, 0.4]],
                [[-0.2, 0.3, 0.1], [0.5, -0.4, 0.2], [0.3, 0.1, -0.5], [0.2, 0.4, 0.1]],
            ],
            dtype=torch.float32,
        ),
        requires_grad=False,
    )
    layer.w2_weight = nn.Parameter(
        torch.tensor(
            [
                [[0.2, 0.3], [-0.1, 0.4], [0.5, -0.2]],
                [[0.1, -0.3], [0.2, 0.2], [-0.4, 0.6]],
            ],
            dtype=torch.float32,
        ),
        requires_grad=False,
    )
    layer.register_buffer("gate_a", torch.tensor([[0.5, -0.2, 0.1]]))
    layer.register_buffer("gate_b", torch.tensor([[0.3], [-0.4]]))
    layer.register_buffer("up_a", torch.tensor([[-0.1, 0.4, 0.2]]))
    layer.register_buffer("up_b", torch.tensor([[0.6], [0.2]]))
    layer.register_buffer("down_a", torch.tensor([[0.7, -0.3]]))
    layer.register_buffer("down_b", torch.tensor([[0.2], [-0.5], [0.1]]))
    layer._static_routed_lora = {
        1: {
            "gate_proj": [("gate_a", "gate_b", 0.5)],
            "up_proj": [("up_a", "up_b", 0.25)],
            "down_proj": [("down_a", "down_b", 0.75)],
        }
    }

    x = torch.tensor([[1.0, -0.5, 0.25], [0.3, 0.2, -0.7]], dtype=torch.float32)
    topk_ids = torch.tensor([[1, 0], [1, 1]], dtype=torch.int64)
    topk_weights = torch.tensor([[0.75, 0.25], [0.6, 0.4]], dtype=torch.float32)

    actual = _apply_static_routed_lora_reference(
        layer,
        x,
        topk_weights,
        topk_ids,
        expert_map=None,
        apply_router_weight_on_input=False,
    )

    expected = torch.zeros_like(x)
    for token_idx in range(x.shape[0]):
        for route_idx in range(topk_ids.shape[1]):
            expert_id = int(topk_ids[token_idx, route_idx].item())
            token_x = x[token_idx : token_idx + 1]
            w13 = layer.w13_weight[expert_id]
            gate = token_x @ w13[:2].t()
            up = token_x @ w13[2:].t()
            if expert_id == 1:
                gate = gate + (token_x @ layer.gate_a.t() @ layer.gate_b.t()) * 0.5
                up = up + (token_x @ layer.up_a.t() @ layer.up_b.t()) * 0.25
            hidden = torch.nn.functional.silu(gate) * up
            down = hidden @ layer.w2_weight[expert_id].t()
            if expert_id == 1:
                down = down + (hidden @ layer.down_a.t() @ layer.down_b.t()) * 0.75
            expected[token_idx] += down.squeeze(0) * topk_weights[token_idx, route_idx]

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_fused_moe_static_routed_lora_keeps_vllm_stacked_layout(monkeypatch):
    pytest.importorskip("aiter")
    _prepare_real_model_ops_import()
    import atom.model_ops.moe as moe_mod
    from atom.model_ops.moe import FusedMoE

    class FakeDPGroup:
        world_size = 1
        rank_in_group = 0

    monkeypatch.setattr(moe_mod, "get_dp_group", lambda: FakeDPGroup())

    layer = FusedMoE(
        num_experts=2,
        top_k=2,
        hidden_size=3,
        intermediate_size=2,
        params_dtype=torch.bfloat16,
        tp_size=1,
        dp_size=1,
    )
    gate_a = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
    gate_b = torch.tensor([[4.0], [5.0]], dtype=torch.float32)
    up_a = torch.tensor([[6.0, 7.0, 8.0]], dtype=torch.float32)
    up_b = torch.tensor([[9.0], [10.0]], dtype=torch.float32)
    down_a = torch.tensor([[11.0, 12.0]], dtype=torch.float32)
    down_b = torch.tensor([[13.0], [14.0], [15.0]], dtype=torch.float32)

    layer.add_static_routed_lora(1, "gate_proj", gate_a, gate_b, 0.5, "test")
    layer.add_static_routed_lora(1, "up_proj", up_a, up_b, 1.0, "test")
    layer.add_static_routed_lora(1, "down_proj", down_a, down_b, 0.25, "test")

    assert layer._static_routed_lora_w13_a_stacked.shape == (2, 2, 1, 3)
    assert layer._static_routed_lora_w13_b_stacked.shape == (2, 2, 2, 1)
    assert layer._static_routed_lora_w2_a_stacked.shape == (1, 2, 1, 2)
    assert layer._static_routed_lora_w2_b_stacked.shape == (1, 2, 3, 1)
    assert torch.equal(
        layer._static_routed_lora_w13_a_stacked[0, 1].cpu().float(),
        gate_a,
    )
    assert torch.equal(
        layer._static_routed_lora_w13_b_stacked[0, 1].cpu().float(),
        gate_b * 0.5,
    )
    assert torch.equal(
        layer._static_routed_lora_w13_a_stacked[1, 1].cpu().float(),
        up_a,
    )
    assert torch.equal(
        layer._static_routed_lora_w13_b_stacked[1, 1].cpu().float(),
        up_b,
    )
    assert torch.equal(
        layer._static_routed_lora_w2_a_stacked[0, 1].cpu().float(),
        down_a,
    )
    assert torch.equal(
        layer._static_routed_lora_w2_b_stacked[0, 1].cpu().float(),
        down_b * 0.25,
    )
    assert layer._static_routed_lora_expert_enabled.cpu().tolist() == [0, 1]


def test_fp8_moe_static_routed_lora_uses_vllm_triton_path(monkeypatch):
    pytest.importorskip("aiter")
    _prepare_real_model_ops_import()
    import atom.model_ops.moe as moe_mod
    from aiter import ActivationType, QuantType

    sentinel = torch.full((2, 3), 7.0)
    captured = {}

    monkeypatch.setattr(
        moe_mod.FusedMoE,
        "select_experts",
        staticmethod(
            lambda **_kwargs: (
                torch.ones(2, 1, dtype=torch.float32),
                torch.zeros(2, 1, dtype=torch.int64),
            )
        ),
    )

    def fake_vllm_triton_path(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return sentinel

    monkeypatch.setattr(
        moe_mod,
        "_apply_static_routed_lora_vllm_triton_fp8",
        fake_vllm_triton_path,
    )
    monkeypatch.setattr(
        moe_mod,
        "_apply_static_routed_lora_reference",
        lambda *_args, **_kwargs: pytest.fail("reference fallback was used"),
    )

    quant_config = types.SimpleNamespace(
        quant_type=QuantType.per_1x128,
        quant_dtype=torch.float8_e4m3fn,
    )
    method = moe_mod.Fp8MoEMethod(quant_config, moe=types.SimpleNamespace())
    layer = nn.Module()
    layer._static_routed_lora = {0: {"gate_proj": []}}
    layer.num_fused_shared_experts = 0
    layer.routed_scaling_factor = 1.0

    result = method.apply(
        layer,
        torch.zeros(2, 3, dtype=torch.bfloat16),
        torch.zeros(2, 4, dtype=torch.float32),
        top_k=1,
        renormalize=True,
        activation=ActivationType.Silu,
    )

    assert result is sentinel
    assert captured["args"][0] is layer
    assert captured["kwargs"]["block_shape"] == [128, 128]
    assert captured["kwargs"]["per_channel_quant"] is False


def test_apply_lora_adapters_registers_vocab_parallel_lm_head(tmp_path):
    adapter_path = tmp_path / "adapter"
    lora_a = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    lora_b = torch.arange(12, dtype=torch.float32).reshape(6, 2)
    _write_adapter(
        adapter_path,
        {
            "base_model.model.lm_head.lora_A.weight": lora_a,
            "base_model.model.lm_head.lora_B.weight": lora_b,
        },
    )
    model = nn.Module()
    model.lm_head = FakeParallelLMHead()

    apply_lora_adapters(model, [f"test={adapter_path}"])

    assert torch.count_nonzero(model.lm_head.weight) == 0
    loaded_a, loaded_b, scaling, kwargs = model.lm_head.loaded_loras[0]
    assert torch.equal(loaded_a.cpu(), lora_a.to(torch.bfloat16))
    assert torch.equal(loaded_b.cpu(), lora_b[3:6].to(torch.bfloat16))
    assert scaling == 2.0
    assert kwargs["adapter_name"] == "test"


def test_parallel_lm_head_lora_applies_delta_before_gather(monkeypatch):
    import importlib

    pytest.importorskip("aiter")
    _prepare_real_model_ops_import()
    try:
        embed_head = importlib.import_module("atom.model_ops.embed_head")
    except ImportError as exc:
        pytest.skip(f"ParallelLMHead dependencies are not importable: {exc}")

    class FakeTPGroup:
        rank_in_group = 0
        world_size = 1

    monkeypatch.setattr(embed_head, "get_tp_group", lambda: FakeTPGroup())
    lm_head = embed_head.ParallelLMHead(num_embeddings=3, embedding_dim=4)
    lora_a = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        dtype=torch.bfloat16,
    )
    lora_b = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        dtype=torch.bfloat16,
    )
    lm_head.add_lora_adapter(lora_a, lora_b, scaling=0.5, adapter_name="test")
    x = torch.tensor([[2.0, 4.0, 8.0, 16.0]], dtype=torch.bfloat16)
    logits = torch.zeros(1, 3, dtype=torch.bfloat16)

    out = lm_head._apply_lora_adapters(x, logits)

    expected = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.bfloat16)
    assert torch.equal(out, expected)


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


def test_validate_lora_adapters_requires_config(tmp_path):
    adapter_path = tmp_path / "adapter"
    adapter_path.mkdir()
    save_file(
        {
            "base_model.model.q_proj.lora_A.weight": torch.ones(2, 3),
            "base_model.model.q_proj.lora_B.weight": torch.ones(4, 2),
        },
        adapter_path / "adapter_model.safetensors",
    )

    with pytest.raises(FileNotFoundError, match="adapter_config.json"):
        validate_lora_adapters_supported([f"user={adapter_path}"])


def test_validate_lora_adapters_rejects_invalid_config_json(tmp_path):
    adapter_path = tmp_path / "adapter"
    adapter_path.mkdir()
    save_file(
        {
            "base_model.model.q_proj.lora_A.weight": torch.ones(2, 3),
            "base_model.model.q_proj.lora_B.weight": torch.ones(4, 2),
        },
        adapter_path / "adapter_model.safetensors",
    )
    (adapter_path / "adapter_config.json").write_text("{not json")

    with pytest.raises(json.JSONDecodeError):
        validate_lora_adapters_supported([f"user={adapter_path}"])


def test_apply_lora_adapters_does_not_count_nonlocal_routed_experts(
    tmp_path,
    caplog,
):
    adapter_path = tmp_path / "adapter"
    _write_adapter(
        adapter_path,
        {
            "base_model.model.model.layers.0.mlp.experts.0.down_proj."
            "lora_A.weight": torch.ones(2, 3),
            "base_model.model.model.layers.0.mlp.experts.0.down_proj."
            "lora_B.weight": torch.ones(4, 2),
        },
    )
    model = nn.Module()
    model.model = nn.Module()
    model.model.layers = nn.ModuleList([nn.Module()])
    model.model.layers[0].mlp = nn.Module()
    experts = nn.Module()
    experts.expert_map = torch.tensor([-1])
    model.model.layers[0].mlp.experts = experts

    with caplog.at_level("INFO", logger="atom"):
        apply_lora_adapters(model, [f"test={adapter_path}"])

    assert "Loaded 0 static LoRA tensor pairs" in caplog.text
