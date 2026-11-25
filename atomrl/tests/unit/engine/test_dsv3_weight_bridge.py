# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""MLA weight bridge: HF <-> Megatron round-trip on a real constructed model.

Rather than asserting a hand-written name table, this builds a tiny DSv3-shaped
Megatron model, exports its actual parameters to HF names, converts back, and
requires every tensor to survive. That catches a wrong mapping, a missed
parameter and a bad gate/up split at once -- and it is the only way to be sure
of the two non-obvious spellings (LoRA norms fused into the up-projections; the
post-attention norm named differently on dense and MoE layers).
"""

import os

import pytest
import torch
from atomrl.engine.training.bridges import dsv3

TINY = {
    "architectures": ["DeepseekV3ForCausalLM"],
    "num_hidden_layers": 3,
    "hidden_size": 128,
    "num_attention_heads": 4,
    "num_key_value_heads": 4,
    "intermediate_size": 256,
    "vocab_size": 512,
    "rms_norm_eps": 1e-6,
    "q_lora_rank": 32,
    "kv_lora_rank": 16,
    "qk_nope_head_dim": 16,
    "qk_rope_head_dim": 8,
    "v_head_dim": 16,
    "n_routed_experts": 4,
    "n_shared_experts": 1,
    "num_experts_per_tok": 2,
    "moe_intermediate_size": 64,
    "first_k_dense_replace": 1,  # layer 0 dense, layers 1-2 MoE
    "routed_scaling_factor": 2.5,
    "topk_method": "noaux_tc",
    "scoring_func": "sigmoid",
}


@pytest.fixture(scope="module")
def tiny_model():
    """A real Megatron DSv3 model at TP=PP=EP=1, or skip if we cannot build one."""
    torch.distributed = torch.distributed
    import torch.distributed as dist

    if not dist.is_available():
        pytest.skip("torch.distributed unavailable")
    if not torch.cuda.is_available():
        pytest.skip("the TE layer spec and its CUDA RNG need a GPU")

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29591")
    created = False
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo", rank=0, world_size=1)
        created = True
    try:
        from megatron.core import parallel_state as mpu
        from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
        from megatron.core.models.gpt.gpt_model import GPTModel
        from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

        if not mpu.is_initialized():
            mpu.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(0)

        cfg = dsv3.build_dsv3_config(TINY, {"moe_grouped_gemm": True})
        spec = get_gpt_decoder_block_spec(cfg, use_transformer_engine=True)
        model = GPTModel(
            config=cfg,
            transformer_layer_spec=spec,
            vocab_size=TINY["vocab_size"],
            max_sequence_length=64,
            pre_process=True,
            post_process=True,
            share_embeddings_and_output_weights=False,
            position_embedding_type="rope",
            parallel_output=False,
        )
        yield model
    finally:
        if created and dist.is_initialized():
            dist.destroy_process_group()


def _export(model):
    import itertools

    named = itertools.chain(model.named_parameters(), dsv3.router_bias_buffers(model))
    return dict(dsv3.megatron_to_hf(named))


# --- export -------------------------------------------------------------------


def test_export_produces_deepseek_hf_names(tiny_model):
    hf = _export(tiny_model)
    for key in (
        "model.embed_tokens.weight",
        "model.norm.weight",
        "lm_head.weight",
        "model.layers.0.self_attn.q_a_proj.weight",
        "model.layers.0.self_attn.q_b_proj.weight",
        "model.layers.0.self_attn.kv_a_proj_with_mqa.weight",
        "model.layers.0.self_attn.kv_b_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.input_layernorm.weight",
    ):
        assert key in hf, f"missing {key}"


def test_export_emits_the_fused_lora_layernorms(tiny_model):
    """These live on the up-projections in Megatron, not as separate modules."""
    hf = _export(tiny_model)
    assert hf["model.layers.0.self_attn.q_a_layernorm.weight"].shape == (32,)
    assert hf["model.layers.0.self_attn.kv_a_layernorm.weight"].shape == (16,)


def test_post_attention_norm_emitted_for_both_dense_and_moe_layers(tiny_model):
    """Dense fuses it into linear_fc1; MoE keeps pre_mlp_layernorm. Same HF key."""
    hf = _export(tiny_model)
    assert "model.layers.0.post_attention_layernorm.weight" in hf  # dense
    assert "model.layers.1.post_attention_layernorm.weight" in hf  # MoE


def test_export_splits_gate_and_up(tiny_model):
    hf = _export(tiny_model)
    # dense layer: ffn=256
    assert hf["model.layers.0.mlp.gate_proj.weight"].shape == (256, 128)
    assert hf["model.layers.0.mlp.up_proj.weight"].shape == (256, 128)
    assert hf["model.layers.0.mlp.down_proj.weight"].shape == (128, 256)
    # routed expert: moe_ffn=64
    assert hf["model.layers.1.mlp.experts.0.gate_proj.weight"].shape == (64, 128)
    assert hf["model.layers.1.mlp.experts.0.up_proj.weight"].shape == (64, 128)


def test_export_covers_every_routed_and_shared_expert(tiny_model):
    hf = _export(tiny_model)
    for layer in (1, 2):
        for e in range(TINY["n_routed_experts"]):
            assert f"model.layers.{layer}.mlp.experts.{e}.down_proj.weight" in hf
        assert f"model.layers.{layer}.mlp.shared_experts.down_proj.weight" in hf


def test_export_includes_the_router_and_its_bias_buffer(tiny_model):
    """noaux_tc reads the bias; a rollout without it picks different experts."""
    hf = _export(tiny_model)
    assert hf["model.layers.1.mlp.gate.weight"].shape == (4, 128)
    assert hf["model.layers.1.mlp.gate.e_score_correction_bias"].shape == (4,)


def test_dense_layers_have_no_router(tiny_model):
    hf = _export(tiny_model)
    assert "model.layers.0.mlp.gate.weight" not in hf


# --- round trip ---------------------------------------------------------------


def test_round_trip_restores_every_megatron_parameter(tiny_model):
    """megatron -> HF -> megatron must return every parameter bit-identical."""
    original = {n: p.detach().clone() for n, p in tiny_model.named_parameters()}
    hf = _export(tiny_model)
    back = dsv3.hf_to_megatron(hf, dsv3.build_dsv3_dims(TINY), use_grouped_mlp=True)

    missing = sorted(set(original) - set(back))
    assert not missing, f"round trip lost {len(missing)} params, e.g. {missing[:5]}"

    for name, want in original.items():
        got = back[name]
        assert got.shape == want.shape, f"{name}: {got.shape} != {want.shape}"
        assert torch.equal(got, want), f"{name}: values changed"


def test_round_trip_introduces_no_tensor_the_model_cannot_hold(tiny_model):
    """Every produced name must correspond to a real slot on the model.

    "Slot" means parameter *or* buffer: the router bias is a buffer, so it is
    absent from named_parameters() yet is a legitimate thing to restore. Anything
    outside both sets would be written nowhere, or worse, silently dropped by a
    tolerant loader.
    """
    slots = set(dict(tiny_model.named_parameters())) | set(
        dict(tiny_model.named_buffers())
    )
    back = dsv3.hf_to_megatron(
        _export(tiny_model), dsv3.build_dsv3_dims(TINY), use_grouped_mlp=True
    )
    assert not sorted(set(back) - slots)


def test_round_trip_restores_the_router_bias_buffer(tiny_model):
    """It is a buffer, so the parameter-only round-trip check cannot see it."""
    back = dsv3.hf_to_megatron(
        _export(tiny_model), dsv3.build_dsv3_dims(TINY), use_grouped_mlp=True
    )
    buffers = dict(tiny_model.named_buffers())
    for layer in (1, 2):
        key = f"decoder.layers.{layer}.mlp.router.expert_bias"
        assert key in back, f"{key} lost in round trip"
        assert torch.equal(back[key], buffers[key])


def test_sequential_expert_layout_round_trips_too(tiny_model):
    """Non-grouped MoE uses local_experts.{E}; both spellings must convert."""
    hf = _export(tiny_model)
    back = dsv3.hf_to_megatron(hf, dsv3.build_dsv3_dims(TINY), use_grouped_mlp=False)
    assert "decoder.layers.1.mlp.experts.local_experts.0.linear_fc1.weight" in back
    assert "decoder.layers.1.mlp.experts.linear_fc1.weight0" not in back
