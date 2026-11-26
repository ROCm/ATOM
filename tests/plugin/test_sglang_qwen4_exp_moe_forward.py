"""Qwen MoE numerics and serving dispatch on one shared GPU fixture.

This exercises one rank's experts, without launching a model or TP collectives.
The direct AITER call and ATOM forward use the same weights/Torch reference.
SGLang currently leaves Context.scheduled_tokens_gpu unset: padded MoE rows
may be computed, but must not affect live rows or escape the replay output slice.
"""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("aiter", exc_type=ImportError)
pytest.importorskip("sglang")

from aiter import ActivationType, QuantType, dtypes

from atom.model_ops import moe, topK
from atom.models import qwen4_exp
from atom.quant_spec import LayerQuantConfig
from atom.utils import forward_context as fc


def _make_block(monkeypatch, mode, tp, hidden):
    """Real block construction/forward and MoE methods, with local group fixtures."""
    policy = LayerQuantConfig(
        quant_dtype=dtypes.fp8 if mode == "fp8" else torch.bfloat16,
        quant_type=QuantType.per_Token if mode == "fp8" else QuantType.No,
        quant_method="compressed-tensors" if mode == "fp8" else "",
    )
    hf = SimpleNamespace(
        architectures=["Qwen4ExpForConditionalGeneration"],
        model_type="qwen4_exp_text",
        hidden_size=hidden,
        num_experts=512,
        num_experts_per_tok=10,
        moe_intermediate_size=640,
        shared_expert_intermediate_size=640,
        hidden_act="silu",
        norm_topk_prob=True,
    )
    config = SimpleNamespace(
        hf_config=hf,
        parallel_config=SimpleNamespace(data_parallel_size=1, data_parallel_rank=0),
        compilation_config=SimpleNamespace(static_forward_context={}),
        enable_dp_attention=False,
        enable_expert_parallel=False,
        enable_tbo=False,
        moe_all2all_backend="none",
    )
    monkeypatch.setattr(moe, "get_current_atom_config", lambda: config)
    monkeypatch.setattr(topK, "get_current_atom_config", lambda: config)
    monkeypatch.setattr(qwen4_exp, "get_current_atom_config", lambda: config)
    monkeypatch.setattr(moe, "get_dp_group", lambda: SimpleNamespace(world_size=1))
    monkeypatch.setattr(qwen4_exp, "get_tensor_model_parallel_world_size", lambda: tp)
    monkeypatch.setattr(qwen4_exp, "get_gfx_runtime", lambda: "gfx942")

    def make_experts(**kwargs):
        # Avoid engine/distributed initialization, but keep FusedMoE.forward,
        # both custom-op boundaries, routing, and method.apply unmodified.
        layer = object.__new__(moe.FusedMoE)
        torch.nn.Module.__init__(layer)
        layer.moe_parallel_config = SimpleNamespace(
            tp_size=tp,
            tp_rank=0,
            ep_size=1,
            ep_rank=0,
            dp_size=1,
            use_ep=False,
            pcp_merge=False,
            dp_logical_ratio=1,
        )
        layer.expert_layout = SimpleNamespace(
            num_routed=512, num_fused_shared_experts=1
        )
        layer.global_num_experts = 512
        layer.expert_map = layer.expert_mask = None
        layer.intermediate_size_per_partition = 640 // tp
        layer.prefix = layer.layer_name = kwargs["prefix"]
        layer.quant_config = kwargs["quant_config"]
        layer.top_k = kwargs["top_k"]
        layer.renormalize = kwargs["renormalize"]
        layer.use_grouped_topk = False
        layer.topk_group = layer.num_expert_group = None
        layer.custom_routing_function = layer.e_score_correction_bias = None
        layer.scoring_func = "softmax"
        layer.shared_expert_scoring_func = kwargs["shared_expert_scoring_func"]
        layer.routed_scaling_factor = 1.0
        layer.activation = ActivationType.Silu
        layer.apply_router_weight_on_input = layer.reduce_results = False
        layer.sp_mega_moe = layer.balance_router_logits = None
        method_config = SimpleNamespace(in_dtype=torch.bfloat16, hidden_dim=hidden)
        layer.quant_method = (
            moe.CompressedTensorsFp8MoEMethod(policy, method_config)
            if mode == "fp8"
            else moe.UnquantizedFusedMoEMethod(method_config)
        )
        layer.quant_method.pad_align = kwargs["pad_align"]
        layer.quant_method.create_weights(
            layer,
            513,
            hidden,
            640 // tp,
            policy.quant_dtype,
            weight_loader=layer.weight_loader,
        )
        config.compilation_config.static_forward_context[layer.layer_name] = layer
        return layer

    monkeypatch.setattr(qwen4_exp, "FusedMoE", make_experts)
    monkeypatch.setattr(
        qwen4_exp,
        "MergedReplicatedLinear",
        lambda h, outputs, **_: torch.nn.Linear(
            h, sum(outputs), bias=False, dtype=torch.bfloat16
        ),
    )
    block = qwen4_exp.Qwen4ExpSparseMoeBlock(
        hf,
        SimpleNamespace(get_layer_quant_config=lambda *_a, **_k: policy),
        prefix="mtp.layers.0.mlp" if mode == "bf16" else "model.layers.0.mlp",
        fuse_shared_expert=True,
    )
    assert block.use_gfx942_tuned_moe
    assert block.fuse_shared
    return block, config


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm GPU required")
@pytest.mark.parametrize("tp", [2, 4])
@pytest.mark.parametrize("mode", ["fp8", "bf16"])
def test_gpu_sglang_moe_forward_and_padded_graph(monkeypatch, mode, tp):
    from aiter.fused_moe import fused_moe
    from aiter.jit.utils.chip_info import get_cu_num, get_gfx_runtime
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput
    from sglang.srt.model_executor.forward_batch_info import ForwardMode
    from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
        DecodeCudaGraphRunner,
    )

    from atom.model_ops.utils import normalize_e4m3fn_to_e4m3fnuz
    from atom.plugin.sglang.runtime.forward_context import _set_atom_forward_context

    if torch.version.hip is None or get_gfx_runtime() != "gfx942" or get_cu_num() != 80:
        pytest.skip("gfx942/80 CU Qwen MoE contract")
    monkeypatch.setattr(fc, "_forward_context", fc.ForwardContext())
    monkeypatch.setattr(topK, "aiter_topK_meta_data", None)
    with torch.device("cuda"), torch.inference_mode():
        torch.manual_seed(29)
        block, config = _make_block(monkeypatch, mode, tp, hidden=2560)
        experts = block.experts
        logical, physical = 640 // tp, experts.w2_weight.shape[-1]
        selected = [0, 3, 11, 49, 87, 131, 207, 311, 402, 511]
        block.gate.weight.zero_()
        for i, expert in enumerate(selected):
            block.gate.weight[expert, 0] = (i + 1) * 0.25
        block.gate.weight[512, 0] = 0.5
        experts.w13_weight.data.view(torch.uint8).zero_()
        experts.w2_weight.data.view(torch.uint8).zero_()
        reference_weights = {}
        source_dtype = torch.float8_e4m3fn if mode == "fp8" else torch.bfloat16
        for expert in selected + [512]:
            gate = (torch.randn(logical, 2560) * 0.02).to(source_dtype)
            up = (torch.randn(logical, 2560) * 0.02).to(source_dtype)
            down = (torch.randn(2560, logical) * 0.02).to(source_dtype)
            experts._copy_quant_storage(experts.w13_weight[expert, :logical], gate)
            experts._copy_quant_storage(
                experts.w13_weight[expert, physical : physical + logical], up
            )
            experts._copy_quant_storage(experts.w2_weight[expert, :, :logical], down)
            if mode == "fp8":
                w13, s13, _ = normalize_e4m3fn_to_e4m3fnuz(
                    experts.w13_weight[expert].clone(),
                    experts.w13_weight_scale[expert].clone(),
                )
                w2, s2, _ = normalize_e4m3fn_to_e4m3fnuz(
                    experts.w2_weight[expert].clone(),
                    experts.w2_weight_scale[expert].clone(),
                )
                w13, w2 = w13.float() * s13, w2.float() * s2
                reference_weights[expert] = (
                    w13[:logical],
                    w13[physical : physical + logical],
                    w2[:, :logical],
                )
            else:
                reference_weights[expert] = (gate.float(), up.float(), down.float())
        experts.quant_method.process_weights_after_loading(experts)

        meta_weights = torch.empty((4, 11), dtype=torch.float32)
        meta_ids = torch.full((4, 11), -1, dtype=torch.int32)
        meta_ids[:, 10] = 512  # Only the shared column is prefilled, as in serving.
        topK.aiter_topK_meta_data = (meta_weights, meta_ids)

        def forbid_legacy(*_):
            pytest.fail("single-token serving bypassed ATOM's MoE forward")

        monkeypatch.setattr(block, "_single_token_experts", forbid_legacy)

        def context(rows):
            fb = SimpleNamespace(
                forward_mode=ForwardMode.DECODE,
                batch_size=rows,
                seq_lens_cpu=[1] * rows,
                seq_lens=torch.ones(rows, dtype=torch.int32),
            )
            _set_atom_forward_context(config, fb, torch.zeros(rows, dtype=torch.int64))
            assert fc.get_forward_context().context.scheduled_tokens_gpu is None
            fc.get_forward_context().context.is_draft = mode == "bf16"

        def routing(x):
            logits = torch.nn.functional.linear(x.float(), block.gate.weight.float())
            probs = torch.softmax(logits[:, :512], -1)
            weights, ids = probs.topk(10, dim=-1)
            weights /= weights.sum(-1, keepdim=True)
            return (
                torch.cat((weights, logits[:, 512:].sigmoid()), dim=-1).contiguous(),
                torch.cat((ids, torch.full_like(ids[:, :1], 512)), dim=-1)
                .to(torch.int32)
                .contiguous(),
            )

        def reference(x):
            weights, ids = routing(x)
            expected = torch.zeros(x.shape, dtype=torch.float32)
            for expert, (gate, up, down) in reference_weights.items():
                score = (weights * (ids == expert)).sum(-1)
                intermediate = torch.nn.functional.silu(x.float() @ gate.T) * (
                    x.float() @ up.T
                )
                expected += (intermediate @ down.T) * score[:, None]
            return expected

        def check(actual, expected):
            assert torch.isfinite(actual).all()
            relative_l2 = (actual.float() - expected).norm(dim=-1) / expected.norm(
                dim=-1
            ).clamp_min(1e-6)
            assert relative_l2.max().item() < 0.05

        hidden = torch.randn((4, 2560), dtype=torch.bfloat16) * 0.1
        hidden[:, 0] = 1
        # The former standalone AITER test now shares this fixture/reference.
        # Its explicit top-k call remains covered alongside ATOM's real router.
        weights, ids = routing(hidden)
        direct = fused_moe(
            hidden,
            experts.w13_weight,
            experts.w2_weight,
            weights,
            ids,
            activation=ActivationType.Silu,
            quant_type=QuantType.per_Token if mode == "fp8" else QuantType.No,
            w1_scale=getattr(experts, "w13_weight_scale", None),
            w2_scale=getattr(experts, "w2_weight_scale", None),
        )
        check(direct, reference(hidden))
        context(1)
        block(hidden[:1], reduce_results=False)  # JIT/warmup outside profiler/capture.
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU]
        ) as profile:
            single = block(hidden[:1], reduce_results=False)
        torch.cuda.synchronize()
        ops = {event.key for event in profile.key_averages()}
        assert "aiter::moe_forward" in ops
        if mode == "fp8":
            assert "aiter::rocm_aiter_fused_moe" in ops
        check(single, reference(hidden[:1]))

        context(4)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                block(hidden, reduce_results=False)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = block(hidden, reduce_results=False)
        torch.cuda.synchronize()

        # Use SGLang's real replay output slicing, with a one-MoE graph backend.
        # No fabricated scheduled_tokens_gpu mask: that is not this plugin's ABI.
        runner = DecodeCudaGraphRunner.__new__(DecodeCudaGraphRunner)
        runner.model_runner = SimpleNamespace(
            device_timer=None, is_draft_worker=mode == "bf16"
        )
        runner.is_dllm = False
        runner.ragged_verify_mode = False
        runner._replay_graph_key = SimpleNamespace(size=4)
        runner._replay_attn_backend = lambda: None
        runner._resolve_shared_read_ends = lambda *_: None

        def replay(_key, _batch):
            graph.replay()
            return LogitsProcessorOutput(next_token_logits=None, hidden_states=captured)

        runner.backend = SimpleNamespace(replay_session=nullcontext, replay=replay)
        runner.load_batch = lambda batch, _proxy: setattr(
            runner, "raw_num_token", batch.batch_size
        )
        original = hidden.clone()
        for live in (1, 4, 3, 0, 1):
            for pad_value in (0.0, float("nan"), 9.0):
                hidden.copy_(original)
                hidden[live:].fill_(pad_value)
                batch = SimpleNamespace(
                    forward_mode=ForwardMode.DECODE, batch_size=live
                )
                output = runner.execute(batch).hidden_states
                torch.cuda.synchronize()
                assert output.shape == (live, 2560)
                if live:
                    check(output, reference(original[:live]))
