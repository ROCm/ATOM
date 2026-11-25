# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

from collections.abc import Mapping

import atomrl.engine.training.bridges.dsv4 as dsv4_bridge
import torch
from atomrl.engine.training.bridges.dsv4 import (
    DSV4Dims,
    _normalize_hf_keys,
    hf_to_megatron,
    megatron_to_hf,
)


def _tiny_dsv4_checkpoint(num_layers=43):
    dims = DSV4Dims(
        num_layers=num_layers,
        hidden=2,
        num_heads=1,
        num_kv_groups=1,
        head_dim=2,
        ffn=2,
        vocab=3,
        num_experts=4,
        moe_ffn=1,
        shared_expert_ffn=0,
        compress_ratios=[0] * num_layers,
    )
    state = {
        "model.embed_tokens.weight": torch.tensor([[1.0]]),
        "model.norm.weight": torch.tensor([2.0]),
        "lm_head.weight": torch.tensor([[3.0]]),
    }
    for layer in range(num_layers):
        hp = f"model.layers.{layer}."
        marker = torch.tensor([float(layer)])
        state.update(
            {
                hp + "input_layernorm.weight": marker,
                hp + "self_attn.wq_a.weight": marker,
                hp + "self_attn.q_norm.weight": marker,
                hp + "self_attn.wq_b.weight": marker,
                hp + "self_attn.wkv.weight": marker,
                hp + "self_attn.kv_norm.weight": marker,
                hp + "self_attn.wo_a.weight": marker,
                hp + "self_attn.wo_b.weight": marker,
                hp + "post_attention_layernorm.weight": marker,
                hp + "mlp.gate.weight": torch.full((4, 1), float(layer)),
                hp
                + "mlp.gate.e_score_correction_bias": torch.arange(
                    4, dtype=torch.float32
                ),
            }
        )
        if layer < 3:
            state[hp + "mlp.topk.tid2eid"] = torch.tensor(
                [[0, 1], [2, 3], [3, 0]], dtype=torch.int32
            )
        for expert in range(dims.num_experts):
            ep = hp + f"mlp.experts.{expert}."
            value = float(layer * 100 + expert * 10)
            state[ep + "gate_proj.weight"] = torch.tensor([[value + 1]])
            state[ep + "up_proj.weight"] = torch.tensor([[value + 2]])
            state[ep + "down_proj.weight"] = torch.tensor([[value + 3]])
    return dims, state


def test_normalizes_redhat_layer_keys_in_mixed_state_dict():
    layer_weight = object()
    hc_weight = object()
    state = {
        "layers.22.attn_norm.weight": layer_weight,
        "model.hc_head_fn": hc_weight,
    }

    normalized = _normalize_hf_keys(state)

    assert normalized["model.layers.22.input_layernorm.weight"] is layer_weight
    assert normalized["model.hc_head_fn"] is hc_weight


def test_megatron_to_hf_uses_lookup_mapping_without_materializing_it():
    tensors = {
        "embedding.word_embeddings.weight": torch.ones(2, 3),
        "decoder.final_layernorm.weight": torch.ones(3),
        "output_layer.weight": torch.ones(2, 3),
    }

    class LookupOnlyMapping(Mapping):
        def __getitem__(self, key):
            return tensors[key]

        def __contains__(self, key):
            return key in tensors

        def __iter__(self):
            raise AssertionError("streaming mapping must not be materialized")

        def __len__(self):
            return len(tensors)

    converted = dict(
        megatron_to_hf(
            LookupOnlyMapping(),
            DSV4Dims(num_layers=0, num_experts=0, compress_ratios=[]),
        )
    )

    assert list(converted) == [
        "model.embed_tokens.weight",
        "model.norm.weight",
        "lm_head.weight",
    ]


def test_redhat_key_conversion_is_inverse_of_input_normalization():
    assert hasattr(dsv4_bridge, "denormalize_redhat_key")
    cases = {
        "embed.weight": "model.embed_tokens.weight",
        "head.weight": "lm_head.weight",
        "norm.weight": "model.norm.weight",
        "hc_head_fn": "model.hc_head_fn",
        "layers.0.attn_norm.weight": "model.layers.0.input_layernorm.weight",
        "layers.0.ffn_norm.weight": "model.layers.0.post_attention_layernorm.weight",
        "layers.0.attn.wq_a.weight": "model.layers.0.self_attn.wq_a.weight",
        "layers.0.ffn.gate.weight": "model.layers.0.mlp.gate.weight",
        "layers.0.ffn.gate.bias": "model.layers.0.mlp.gate.e_score_correction_bias",
        "layers.0.ffn.gate.tid2eid": "model.layers.0.mlp.topk.tid2eid",
        "layers.0.ffn.experts.3.w1.weight": "model.layers.0.mlp.experts.3.gate_proj.weight",
        "layers.0.ffn.experts.3.w2.weight": "model.layers.0.mlp.experts.3.down_proj.weight",
        "layers.0.ffn.experts.3.w3.weight": "model.layers.0.mlp.experts.3.up_proj.weight",
    }

    for redhat_key, official_key in cases.items():
        assert dsv4_bridge.denormalize_redhat_key(official_key) == redhat_key
        normalized = _normalize_hf_keys({redhat_key: object()})
        assert list(normalized) == [official_key]


def test_missing_expert_bias_defaults_to_fp32_zeros():
    assert hasattr(dsv4_bridge, "_expert_bias_or_zeros")
    router_weight = torch.ones(4, 3, dtype=torch.bfloat16)

    bias = dsv4_bridge._expert_bias_or_zeros(
        {"model.layers.0.mlp.gate.weight": router_weight},
        "model.layers.0.",
        4,
    )

    assert bias.dtype == torch.float32
    torch.testing.assert_close(bias, torch.zeros(4))


def test_asymmetric_pipeline_round_trip_preserves_global_layers_and_hash_tables():
    dims, checkpoint = _tiny_dsv4_checkpoint()
    layers_per_rank = [11, 11, 11, 10]
    offsets = [0, 11, 22, 33]

    for pp_rank, (offset, local_count) in enumerate(zip(offsets, layers_per_rank)):
        megatron = hf_to_megatron(
            checkpoint,
            dims,
            pp_rank=pp_rank,
            pp_size=4,
            layers_per_pp_rank=layers_per_rank,
        )
        local_layer_ids = {
            int(name.split(".")[2])
            for name in megatron
            if name.startswith("decoder.layers.")
        }
        assert local_layer_ids == set(range(local_count))

        restored = dict(
            megatron_to_hf(
                megatron,
                dims,
                pp_rank=pp_rank,
                pp_size=4,
                layers_per_pp_rank=layers_per_rank,
            )
        )
        restored_layers = {
            int(name.split(".")[2])
            for name in restored
            if name.startswith("model.layers.")
        }
        assert restored_layers == set(range(offset, offset + local_count))

        for local_layer, global_layer in enumerate(range(offset, offset + local_count)):
            torch.testing.assert_close(
                megatron[f"decoder.layers.{local_layer}.input_layernorm.weight"],
                checkpoint[f"model.layers.{global_layer}.input_layernorm.weight"],
            )
            hash_key = f"model.layers.{global_layer}.mlp.topk.tid2eid"
            if global_layer < 3:
                torch.testing.assert_close(restored[hash_key], checkpoint[hash_key])
            else:
                assert hash_key not in restored


def test_ep_rank_loads_global_experts_into_local_slots_without_remapping_hash_ids():
    dims, checkpoint = _tiny_dsv4_checkpoint(num_layers=1)

    megatron = hf_to_megatron(
        checkpoint,
        dims,
        ep_rank=1,
        ep_size=2,
    )

    torch.testing.assert_close(
        megatron["decoder.layers.0.mlp.experts.linear_fc1.weight0"],
        torch.cat(
            [
                checkpoint["model.layers.0.mlp.experts.2.gate_proj.weight"],
                checkpoint["model.layers.0.mlp.experts.2.up_proj.weight"],
            ],
            dim=0,
        ),
    )
    torch.testing.assert_close(
        megatron["decoder.layers.0.mlp.experts.linear_fc1.weight1"],
        torch.cat(
            [
                checkpoint["model.layers.0.mlp.experts.3.gate_proj.weight"],
                checkpoint["model.layers.0.mlp.experts.3.up_proj.weight"],
            ],
            dim=0,
        ),
    )
    torch.testing.assert_close(
        megatron["decoder.layers.0.mlp.router.tid2eid"],
        checkpoint["model.layers.0.mlp.topk.tid2eid"],
    )


def test_is_dsv4_recognises_both_spellings_and_rejects_qwen3():
    # MegatronNativeEngine calls this for every model it builds, so it has to
    # answer for a non-DSv4 config too rather than not exist.
    assert dsv4_bridge.is_dsv4({"model_type": "deepseek_v4"})
    assert dsv4_bridge.is_dsv4({"architectures": ["DeepseekV4ForCausalLM"]})
    assert not dsv4_bridge.is_dsv4({"model_type": "qwen3_moe"})
    assert not dsv4_bridge.is_dsv4({"architectures": ["Qwen3MoeForCausalLM"]})
    assert not dsv4_bridge.is_dsv4({})
