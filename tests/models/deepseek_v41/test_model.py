# SPDX-License-Identifier: MIT
"""Text backbone assembly, native parameter layout, and tensor-parallel MoE."""

from types import SimpleNamespace

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm GPU required")
def test_text_backbone_native_weights_and_single_pass_mhc(
    reference, single_rank, small_config, attention_contract
):
    from atom.models.deepseek_v41.model import DeepseekV41ForCausalLM

    captured, check_attention = attention_contract
    config = small_config
    config.vocab_size = 128
    config.n_routed_experts, config.num_experts_per_tok = 8, 2
    config.moe_intermediate_size = 64
    config.routed_scaling_factor, config.swiglu_limit = 1.5, 10.0
    config.engram_layer_ids = ()
    config.hc_mult, config.hc_eps, config.hc_sinkhorn_iters = 4, 1e-6, 20
    config.qk_rope_head_dim, config.rope_theta, config.compress_rope_theta = (
        32,
        10000,
        160000,
    )
    config.rope_scaling = {
        "original_max_position_embeddings": 65536,
        "factor": 16,
        "beta_fast": 32,
        "beta_slow": 1,
    }
    args = reference.ModelArgs(
        dim=64,
        vocab_size=128,
        n_heads=config.num_attention_heads,
        head_dim=config.head_dim,
        q_lora_rank=32,
        o_groups=config.o_groups,
        o_lora_rank=32,
        window_size=4,
        index_n_heads=32,
        index_head_dim=32,
        index_topk=4,
        candidate_block_size=2,
        candidate_topk_blocks=4,
        compress_ratios=(0, 2, 2, 1, 1),
        kv_source_layers=(1, 3),
        index_source_layers=(1, 3, 4),
        candidate_source_layer=3,
        n_layers=5,
        max_batch_size=1,
        max_seq_len=32,
        rope_head_dim=32,
        n_mtp_layers=0,
        dspark_block_size=0,
        vision_n_layers=0,
        n_routed_experts=8,
        n_activated_experts=2,
        moe_inter_dim=64,
        swiglu_limit=10.0,
        route_scale=1.5,
        temperature=0,
        original_seq_len=65536,
        rope_factor=16,
        compress_rope_theta=160000,
    )
    torch.manual_seed(315)
    with reference.set_dtype(torch.bfloat16):
        target = DeepseekV41ForCausalLM(config, max_length=32).cuda()
        source = reference.Transformer(args)
        source_params = dict(source.named_parameters())
        for name, parameter in target.named_parameters():
            if parameter.dtype == torch.float4_e2m1fn_x2:
                value = torch.randint(0, 256, parameter.shape, dtype=torch.uint8).view(
                    parameter.dtype
                )
            elif parameter.dtype == torch.float8_e8m0fnu:
                value = torch.full(parameter.shape, 2**-5).to(parameter.dtype)
            elif parameter.dtype == torch.float8_e4m3fn:
                value = (torch.randn(parameter.shape) * 4).to(parameter.dtype)
            elif name.endswith("norm.weight"):
                value = torch.ones(parameter.shape, dtype=parameter.dtype)
            else:
                value = (torch.randn(parameter.shape) * 0.05).to(parameter.dtype)
            source_name = name.replace(".weight_scale", ".scale")
            if name.endswith(".gate.bias_vl"):
                # The text-only upstream fixture omits vision bias; it is not
                # read without an image mask. The target preserves the schema.
                parameter.data.copy_(value)
                continue
            src = source_params[source_name]
            if parameter.dtype == torch.float4_e2m1fn_x2:
                parameter.data.view(torch.uint8).copy_(value.view(torch.uint8))
                src.data.view(torch.uint8).copy_(value.view(torch.uint8))
            else:
                parameter.data.copy_(value)
                src.data.copy_(value)
        target.process_weights_after_loading()
        tokens = torch.tensor([[3, 7, 11, 2, 5, 9]])
        cache = target.new_cache(1)
        for position, length in ((0, 3), (3, 1), (4, 1), (5, 1)):
            chunk = tokens[:, position : position + length]
            captured.clear()
            expected = source(chunk, position)[1]
            with check_attention(captured):
                actual = target(chunk.cuda(), cache).cpu()
            torch.testing.assert_close(
                actual,
                expected,
                rtol=1 / 64,
                atol=2**-10,
                msg=lambda message, position=position: f"position={position}: {message}",
            )
        assert cache.position == 6

        # Projection cropping must preserve all-token execution and cache state.
        # Use the actual V4 attention path for both forwards.
        full_cache, suffix_cache = target.new_cache(1), target.new_cache(1)
        full_cache.pool.fill_(torch.nan)
        for values in full_cache.index.values():
            values.fill_(torch.nan)
        full = target(tokens.cuda(), full_cache, full_logits=True)
        suffix = target(tokens.cuda(), suffix_cache, full_logits=True, logits_start=3)
        torch.testing.assert_close(suffix, full[:, 3:], rtol=1e-5, atol=1e-6)
        assert full_cache.position == suffix_cache.position == 6
        next_token = torch.tensor([[13]], device="cuda")
        torch.testing.assert_close(
            target(next_token, suffix_cache),
            target(next_token, full_cache),
            rtol=0,
            atol=0,
        )


@pytest.mark.parametrize("rank", [0, 7])
def test_full_checkpoint_runtime_parameter_layout(single_rank, reference, rank):
    import json
    from pathlib import Path

    from atom.models.deepseek_v41.config import normalize_hf_config
    from atom.models.deepseek_v41.model import DeepseekV41ForCausalLM
    from atom.models.deepseek_v41.weights import (
        build_weight_manifest,
        checkpoint_schema,
    )

    config = normalize_hf_config(
        json.loads((Path(__file__).parent / "fixtures/config.json").read_text())
    )
    single_rank.world_size, single_rank.rank_in_group = 8, rank
    with torch.device("meta"), reference.set_dtype(torch.bfloat16):
        model = DeepseekV41ForCausalLM(config, max_length=128)
    manifest = build_weight_manifest(
        checkpoint_schema(config), tp_rank=rank, tp_size=8, ep_rank=rank, ep_size=8
    )
    parameters = dict(model.named_parameters())
    assert set(parameters) == {
        entry.target for entry in manifest if entry.action == "load"
    }
    dtypes = {
        "BF16": torch.bfloat16,
        "F32": torch.float32,
        "F8_E4M3": torch.float8_e4m3fn,
        "F8_E8M0": torch.float8_e8m0fnu,
        "I8": torch.float4_e2m1fn_x2,
    }
    for entry in manifest:
        if entry.action == "load":
            parameter = parameters[entry.target]
            assert parameter.shape == entry.shape, entry.target
            expected_dtype = (
                torch.bfloat16
                if entry.source.dequantize
                else dtypes[entry.source.dtype]
            )
            assert parameter.dtype == expected_dtype, entry.target


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm GPU required")
def test_shared_expert_keeps_tp_partials_fp32(reference, single_rank):
    from atom.model_ops.blockscale import native_quant_linear
    from atom.model_ops.deepseek_v41.moe import weighted_swiglu
    from atom.models.deepseek_v41.moe import MoE

    torch.manual_seed(198)
    single_rank.world_size = 2
    config = SimpleNamespace(
        hidden_size=32,
        moe_intermediate_size=64,
        n_routed_experts=2,
        num_experts_per_tok=1,
        routed_scaling_factor=1.5,
        swiglu_limit=10.0,
    )
    with reference.set_dtype(torch.bfloat16):
        target = MoE(config).cuda()
        source = reference.Expert(32, 64, dtype=torch.float8_e4m3fn)
        for parameter in target.parameters():
            if parameter.dtype == torch.float4_e2m1fn_x2:
                parameter.data.view(torch.uint8).zero_()
            else:
                parameter.data.copy_(torch.zeros_like(parameter, dtype=torch.float32))
        for expert in target.experts.values():
            for projection in (expert.w1, expert.w2, expert.w3):
                projection.weight_scale.data.copy_(
                    torch.ones(projection.weight_scale.shape)
                )
        for name in ("w1", "w2", "w3"):
            src = getattr(source, name)
            src.weight.data.copy_(torch.randint(-8, 9, src.weight.shape).float())
            src.scale.data.copy_(torch.full(src.scale.shape, 0.25))
            dst = getattr(target.shared_experts, name)
            weight = src.weight[:, :32] if name == "w2" else src.weight[:32]
            scale = src.scale[:, :1] if name == "w2" else src.scale[:1]
            dst.weight.data.copy_(weight)
            dst.weight_scale.data.copy_(scale)
        for module in target.modules():
            if hasattr(module, "process_weights_after_loading"):
                module.process_weights_after_loading()
        hidden = torch.randn(7, 32, dtype=torch.bfloat16) * 0.25
        with torch.no_grad():
            expected = source(hidden)
            activation = weighted_swiglu(source.w1(hidden), source.w3(hidden))
        remote = native_quant_linear(
            activation[:, 32:].contiguous().cuda(),
            source.w2.weight[:, 32:].contiguous().cuda(),
            source.w2.scale[:, 1:].contiguous().cuda(),
            dtype=torch.float32,
        )
        reductions = []

        def all_reduce(partial, **_):
            assert partial.dtype == torch.float32
            reductions.append(partial)
            # All routed weights are zero; the other shared shard remains real.
            return partial if len(reductions) == 1 else partial + remote

        single_rank.all_reduce = all_reduce
        with torch.no_grad():
            actual = target(hidden.cuda()).cpu()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert len(reductions) == 2
        early_rounding = (
            (reductions[1].bfloat16().float() + remote.bfloat16().float())
            .bfloat16()
            .cpu()
        )
        assert not torch.equal(early_rounding, expected)


def test_capture_and_replay_offer_the_execution_policy_the_same_stages():
    """A decode step routes the FFN through `run`; a prefill step does not.

    The stage set is what a graph capture records, so a capture built on the
    wrong step kind records a different set than the replay runs and the
    difference is silent -- the missing stage just falls back to eager. This
    pins the routing so `build_for_cudagraph_capture`'s step kind has something
    to be wrong against.
    """
    from atom.models.deepseek_v41.model import Block

    block = Block.__new__(Block)
    state = SimpleNamespace(residual="residual", pre_mix="pre_mix")
    five = ("hidden", "residual", "pre", "post", "comb")
    block.prepare_attention = lambda *args: five
    block.attn = lambda *args: "attn_out"
    block.prepare_ffn = lambda *args: five
    block.decode_ffn = lambda *args: ("decode_out",)
    block.ffn = lambda *args: "prefill_out"
    block.finish_ffn = lambda *args: ("residual", "pre_mix")

    def stages_for(decode):
        seen = []

        def run(function, *args):
            seen.append(function)
            return function(*args)

        block.forward(state, None, SimpleNamespace(decode=decode), None, execution=run)
        return seen

    assert block.decode_ffn in stages_for(decode=True)
    assert block.decode_ffn not in stages_for(decode=False)
