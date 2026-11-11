# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""The Qwen3.5 MTP drafter on a dense target (jgong5/ATOM#1204).

The drafter builds its MTP layer from the target's text config, so on a dense
target (Qwen3.8-27B, `qwen3_5_text`) the layer has a plain MLP and the config
has no expert count. The loader still asks every model that has
`get_expert_mapping` for one, so the mapping must be empty there and unchanged
on a MoE target.
"""

import pytest
import torch
from aiter_stub import atom_under_stubs
from torch import nn

from atom.model_config.qwen3_5 import Qwen3_5TextConfig
from atom.model_config.qwen3_5_moe import Qwen3_5MoeTextConfig
from atom.model_loader.loading_core import load_weights_into_model

# Qwen3.8-27B's MTP block, as its model.safetensors.index.json names it.
DENSE_MTP_CHECKPOINT = [
    "mtp.fc.weight",
    "mtp.layers.0.input_layernorm.weight",
    "mtp.layers.0.mlp.down_proj.weight",
    "mtp.layers.0.mlp.gate_proj.weight",
    "mtp.layers.0.mlp.up_proj.weight",
    "mtp.layers.0.post_attention_layernorm.weight",
    "mtp.layers.0.self_attn.k_norm.weight",
    "mtp.layers.0.self_attn.k_proj.weight",
    "mtp.layers.0.self_attn.o_proj.weight",
    "mtp.layers.0.self_attn.q_norm.weight",
    "mtp.layers.0.self_attn.q_proj.weight",
    "mtp.layers.0.self_attn.v_proj.weight",
    "mtp.norm.weight",
    "mtp.pre_fc_norm_embedding.weight",
    "mtp.pre_fc_norm_hidden.weight",
]
# The parameters `Qwen3_5MTP` builds for that config, less the embedding and
# LM head it shares with the target.
DENSE_DRAFTER_PARAMS = [
    "model.fc.weight",
    "model.layers.0.self_attn.qkv_proj.weight",
    "model.layers.0.self_attn.o_proj.weight",
    "model.layers.0.self_attn.q_norm.weight",
    "model.layers.0.self_attn.k_norm.weight",
    "model.layers.0.mlp.gate_up_proj.weight",
    "model.layers.0.mlp.down_proj.weight",
    "model.layers.0.input_layernorm.weight",
    "model.layers.0.post_attention_layernorm.weight",
    "model.norm.weight",
    "model.pre_fc_norm_hidden.weight",
    "model.pre_fc_norm_embedding.weight",
]


@pytest.fixture(scope="module")
def Qwen3_5MTP():
    with atom_under_stubs() as load:
        yield load("atom.models.qwen3_5_mtp").Qwen3_5MTP


def _drafter(cls, config, param_names=()):
    """The real class's loader hooks, with a recording parameter per name in
    place of its GPU modules."""
    model = object.__new__(cls)
    nn.Module.__init__(model)
    model.config = config
    model.writes = []
    for name in param_names:
        *path, leaf = name.split(".")
        module = model
        for part in path:
            if not hasattr(module, part):
                module.add_module(part, nn.Module())
            module = getattr(module, part)
        param = nn.Parameter(torch.empty(0), requires_grad=False)
        param.weight_loader = lambda _p, _t, *shard, n=name: model.writes.append(
            (n, *shard)
        )
        module.register_parameter(leaf, param)
    return model


def test_dense_target_has_no_expert_mapping(Qwen3_5MTP):
    assert _drafter(Qwen3_5MTP, Qwen3_5TextConfig()).get_expert_mapping() == []


def test_dense_drafter_loads_its_mtp_block(Qwen3_5MTP):
    config = Qwen3_5TextConfig(num_hidden_layers=64)
    model = _drafter(Qwen3_5MTP, config, DENSE_DRAFTER_PARAMS)

    loaded = load_weights_into_model(
        model,
        "<synthetic>",
        config,
        spec_decode=True,
        default_weight_loader=None,
        fuse_shared_expert=lambda *_: False,
        is_rank0=lambda: True,
        weights_iterator=lambda _path, _mmap, wanted: (
            (name, torch.empty(0)) for name in DENSE_MTP_CHECKPOINT if wanted(name)
        ),
    )

    assert loaded == set(DENSE_DRAFTER_PARAMS)
    qkv = "model.layers.0.self_attn.qkv_proj.weight"
    gate_up = "model.layers.0.mlp.gate_up_proj.weight"
    packed = [(qkv, "q"), (qkv, "k"), (qkv, "v"), (gate_up, 0), (gate_up, 1)]
    unpacked = [(n,) for n in DENSE_DRAFTER_PARAMS if n not in (qkv, gate_up)]
    assert sorted(model.writes, key=str) == sorted(packed + unpacked, key=str)


def test_moe_target_maps_routed_and_shared_experts(Qwen3_5MTP):
    config = Qwen3_5MoeTextConfig(num_experts=4)
    # What ATOM derives from `num_experts` on the text config the drafter shares.
    config.n_routed_experts, config.n_shared_experts = config.num_experts, 1

    mapping = _drafter(Qwen3_5MTP, config).get_expert_mapping()

    assert len(mapping) == 3 * (4 + 1)
    assert ("experts.w13_", "experts.0.gate_proj.", 0, "w1") in mapping
    assert ("experts.w2_", "experts.4.down_proj.", 4, "w2") in mapping
