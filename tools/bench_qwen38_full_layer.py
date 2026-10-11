# SPDX-License-Identifier: MIT
"""Offline, synthetic-weight Qwen3.8 decode-layer benchmark on TP4 + EP4.

Example::

    AITER_JIT_DIR=/path/to/working/aiter-jit \
      torchrun --nproc_per_node=4 tools/bench_qwen38_full_layer.py \
      --model /path/to/local/config --layer qsa --rows 8 --context 262144

``--model`` must be a local config directory. ``load_dummy='xavier'`` skips
checkpoint tensor reads; this benchmark never downloads model weights.
The output reports one complete decoder-layer graph, component graphs, and
errors from the same synthetic inputs. A QSA layer includes HC, attention,
both TP reductions, and MoE. A GDN layer includes recurrent state updates.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from aiter.dist.communication_op import tensor_model_parallel_all_reduce
from aiter.dist.parallel_state import (
    destroy_model_parallel,
    init_distributed_environment,
    initialize_model_parallel,
)

from atom.config import Config, KVCacheTensor, set_current_atom_config
from atom.distributed.qwen_qsa_cp import warmup_qsa_cp_exchange
from atom.model_loader.loader import load_model
from atom.model_ops.attentions.gdn_attn import GDNAttentionMetadata
from atom.model_ops.attentions.qwen4_exp_attn import Qwen4ExpQSAMetadata
from atom.model_ops.fla_ops.gdn_flydsl import GDNFlyDSLPolicy, select_policy
from atom.models.qwen4_exp import (
    Qwen4ExpDecoderLayer,
    Qwen4ExpForConditionalGeneration,
    _Qwen4ExpQuantizationConfig,
)
from atom.utils.forward_context import (
    AttentionMetaData,
    Context,
    get_forward_context,
)


def _error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    diff = (actual - expected).float()
    return float(diff.norm() / expected.float().norm().clamp_min(1e-8))


def _time_graph(fn, repeats: int) -> tuple[float, object, torch.cuda.CUDAGraph]:
    fn()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = fn()
    graph.replay()
    torch.cuda.synchronize()
    dist.barrier()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        graph.replay()
    end.record()
    end.synchronize()
    return round(start.elapsed_time(end) * 1000 / repeats, 3), result, graph


def _setup_qsa(layer: Qwen4ExpDecoderLayer, rows: int, context: int) -> None:
    if context < 2048 or context % 64:
        raise ValueError("QSA context must be >= 2048 and divisible by 64")
    page, ratio = 64, 4
    pages = context // page
    total_pages = pages * rows
    table = torch.arange(total_pages, device="cuda", dtype=torch.int32).view(
        rows, pages
    )
    slots = table[:, -1].long() * page + page - 1
    compressed_slots = table[:, -1].long() * (page // ratio) + page // ratio - 1
    positions = torch.full((rows,), context - 1, device="cuda", dtype=torch.int64)
    metadata = Qwen4ExpQSAMetadata(
        table,
        slots,
        compressed_slots,
        torch.arange(rows, device="cuda", dtype=torch.int32),
        positions,
        torch.full((rows,), context, device="cuda", dtype=torch.int32),
        context,
    )
    kv_heads = layer.self_attn.num_kv_heads
    keys = torch.empty(
        (total_pages, page, kv_heads, 256), device="cuda", dtype=torch.bfloat16
    ).normal_(0, 0.1)
    values = torch.empty_like(keys).normal_(0, 0.1)
    raw = torch.empty(
        (total_pages, page, 1, 128), device="cuda", dtype=torch.bfloat16
    ).normal_(0, 0.1)
    compressed = torch.empty(
        (total_pages, page // ratio, 1, 128),
        device="cuda",
        dtype=torch.bfloat16,
    ).normal_(0, 0.1)
    layer.self_attn.bind_caches(keys, values, raw, compressed)
    get_forward_context().attn_metadata.qsa_metadata = metadata


def _setup_gdn(
    layer: Qwen4ExpDecoderLayer, rows: int, world: int
) -> tuple[torch.Tensor, torch.Tensor]:
    reads = torch.arange(rows, device="cuda", dtype=torch.int32)
    writes = reads + rows
    starts = torch.arange(rows + 1, device="cuda", dtype=torch.int32)
    conv_dim = layer.linear_attn.conv_dim // world
    conv_state = torch.zeros(
        (2 * rows, 3, conv_dim), device="cuda", dtype=torch.bfloat16
    )
    state = (
        torch.randn(
            (2 * rows, 48 // world, 128, 128), device="cuda", dtype=torch.float32
        )
        * 0.01
    )
    # Match the recurrent kernel's strided state layout.
    state = state.transpose(-1, -2).contiguous().transpose(-1, -2)
    get_forward_context().attn_metadata.gdn_metadata = GDNAttentionMetadata(
        num_prefills=0,
        num_prefill_tokens=0,
        num_decodes=rows,
        num_decode_tokens=rows,
        num_spec_decodes=0,
        num_spec_decode_tokens=0,
        num_actual_tokens=rows,
        has_initial_state=torch.ones(rows, device="cuda", dtype=torch.bool),
        non_spec_query_start_loc=starts,
        non_spec_state_indices_tensor=writes,
        non_spec_state_indices_in_tensor=reads,
    )
    get_forward_context().kv_cache_data = {
        "layer_0": KVCacheTensor(layer_num=0, k_cache=conv_state, v_cache=state)
    }
    return conv_state, state


def run(args: argparse.Namespace) -> dict:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    if world != 4 or args.rows not in (4, 8, 16):
        raise ValueError("This measured setup requires TP4 + EP4 and 4, 8 or 16 rows")
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    init_distributed_environment(local_rank=local_rank)
    initialize_model_parallel(tensor_model_parallel_size=world)

    layer_type = "full_attention" if args.layer == "qsa" else "linear_attention"
    config = Config(
        model=args.model,
        tensor_parallel_size=world,
        max_model_len=args.context,
        max_num_seqs=args.rows,
        max_num_batched_tokens=max(2048, args.rows),
        load_dummy="xavier",
        enable_expert_parallel=True,
        hf_overrides={
            "num_hidden_layers": 1,
            "layer_types": [layer_type],
            "ple_layer_ids": [],
        },
    )
    set_current_atom_config(config)
    config.quant_config.remap_layer_name(
        config.hf_config,
        packed_modules_mapping=Qwen4ExpForConditionalGeneration.packed_modules_mapping,
        quant_exclude_name_mapping=getattr(
            Qwen4ExpForConditionalGeneration, "quant_exclude_name_mapping", {}
        ),
    )
    quant = _Qwen4ExpQuantizationConfig(config.quant_config)
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(config.torch_dtype)
    torch.set_default_device("cuda")
    torch.manual_seed(2026)
    try:
        layer = Qwen4ExpDecoderLayer(
            config,
            layer_type,
            prefix="model.layers.0",
            layer_num=0,
            quant_config=quant,
            fused_hc=True,
            fuse_shared_expert=True,
        )
        load_model(layer, args.model, config.hf_config, load_dummy="xavier")
    finally:
        torch.set_default_device(None)
        torch.set_default_dtype(previous_dtype)

    with torch.no_grad():
        torch.manual_seed(6107)
        context = get_forward_context()
        context.attn_metadata = AttentionMetaData()
        positions = torch.full(
            (args.rows,), args.context - 1, device="cuda", dtype=torch.int64
        )
        context.context = Context(
            positions,
            is_prefill=False,
            scheduled_bs=args.rows,
            scheduled_tokens=args.rows,
            running_bs=args.rows,
            running_tokens=args.rows,
        )
        context.main_stream = torch.cuda.current_stream()
        context.kv_cache_data = {}
        if args.layer == "qsa":
            _setup_qsa(layer, args.rows, args.context)
            state_pair = None
        else:
            state_pair = _setup_gdn(layer, args.rows, world)
        hidden = (
            torch.randn(args.rows, 10240, device="cuda", dtype=torch.bfloat16) * 0.02
        )
        pending = (
            torch.randn(args.rows, 2560, device="cuda", dtype=torch.bfloat16) * 0.02,
            torch.randn(args.rows, 4, device="cuda", dtype=torch.bfloat16) * 0.02,
        )
        state_seed = (
            tuple(tensor.clone() for tensor in state_pair)
            if state_pair is not None
            else None
        )

        def reset_state() -> None:
            if state_pair is not None:
                for target, seed in zip(state_pair, state_seed):
                    target.copy_(seed)

        def decode():
            return layer(positions, hidden, None, pending)

        os.environ["ATOM_QWEN38_MONO_HC"] = "0"
        os.environ["ATOM_QWEN38_QSA_CP"] = "0"
        os.environ["ATOM_QWEN38_QSA_SHORT"] = "0"
        os.environ["ATOM_QWEN38_QSA_FUSED_PROJ"] = "0"
        if args.layer == "qsa":
            layer.self_attn.qsa_index_cp = False
        else:
            impl = layer.linear_attn.attn.impl
            impl.gdn_flydsl_policy = GDNFlyDSLPolicy()
        reset_state()
        base_us, reference, base_graph = _time_graph(decode, args.repeats)
        reference_values = tuple(
            tensor.clone() for tensor in (reference[0], *reference[1])
        )
        reference_state = state_pair[1].clone() if state_pair is not None else None

        os.environ["ATOM_QWEN38_MONO_HC"] = "1"
        os.environ["ATOM_QWEN38_QSA_CP"] = "1"
        os.environ["ATOM_QWEN38_QSA_SHORT"] = "1"
        os.environ["ATOM_QWEN38_QSA_FUSED_PROJ"] = "1"
        if args.layer == "qsa":
            layer.self_attn.qsa_index_cp = True
            warmup_qsa_cp_exchange(torch.device("cuda", local_rank))
        else:
            tuned_policy = select_policy(
                allowed=True,
                replayssm=False,
                lossy_decode=False,
                state=state_pair[1],
                activation_dtype=layer.linear_attn.dt_bias.dtype,
            )
            impl.gdn_flydsl_policy = tuned_policy
        reset_state()
        tuned_us, actual, tuned_graph = _time_graph(decode, args.repeats)
        kernel_profile = None
        if args.profile:
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ]
            ) as profiler:
                tuned_graph.replay()
                torch.cuda.synchronize()
            kernel_profile = sorted(
                (
                    (
                        event.key.split("<", 1)[0][:96],
                        round(event.self_device_time_total, 3),
                    )
                    for event in profiler.key_averages()
                    if event.self_device_time_total > 0
                ),
                key=lambda event: event[1],
                reverse=True,
            )[:20]
        actual_values = (actual[0], *actual[1])
        errors = [_error(a, b) for a, b in zip(actual_values, reference_values)]
        state_error = (
            _error(state_pair[1], reference_state) if state_pair is not None else None
        )

        mixed, (streams, raw) = layer.attn_hyper_connection.mix_fused(hidden, pending)
        sub = (
            layer.self_attn(positions, mixed)
            if args.layer == "qsa"
            else layer.linear_attn(mixed)
        )
        reduced = tensor_model_parallel_all_reduce(sub)
        moe_input, _ = layer.mlp_hyper_connection.mix_fused(streams, (reduced, raw))
        attention = (
            (lambda: layer.self_attn(positions, mixed))
            if args.layer == "qsa"
            else (lambda: layer.linear_attn(mixed))
        )
        stages = {
            "attention_hc_us": _time_graph(
                lambda: layer.attn_hyper_connection.mix_fused(hidden, pending),
                args.repeats,
            )[0],
            "attention_us": _time_graph(attention, args.repeats)[0],
            "attention_tp_us": _time_graph(
                lambda: tensor_model_parallel_all_reduce(sub), args.repeats
            )[0],
            "mlp_hc_us": _time_graph(
                lambda: layer.mlp_hyper_connection.mix_fused(streams, (reduced, raw)),
                args.repeats,
            )[0],
            "moe_us": _time_graph(lambda: layer.mlp(moe_input), args.repeats)[0],
        }
        projection_errors = None
        qsa_selection_equal = None
        gdn_stage_errors = None
        if args.layer == "qsa":
            fused_qkv, fused_index = layer.self_attn._project_qkv_index(mixed, True)
            os.environ["ATOM_QWEN38_QSA_FUSED_PROJ"] = "0"
            baseline_qkv = layer.self_attn.qkv_proj(mixed)
            baseline_index = layer.self_attn.indexer.index_qk_proj(mixed)
            projection_errors = [
                _error(fused_qkv, baseline_qkv),
                _error(fused_index, baseline_index),
            ]
            os.environ["ATOM_QWEN38_QSA_FUSED_PROJ"] = "1"
            metadata = context.attn_metadata.qsa_metadata.for_tokens(args.rows)
            layer.self_attn.qsa_index_cp = False
            selected_base = layer.self_attn._select_tokens(
                mixed, positions, metadata, fused_index
            )
            layer.self_attn.qsa_index_cp = True
            selected_tuned = layer.self_attn._select_tokens(
                mixed, positions, metadata, fused_index
            )
            qsa_selection_equal = bool(torch.equal(selected_base, selected_tuned))
        else:

            def gdn_stages(tuned: bool):
                reset_state()
                os.environ["ATOM_QWEN38_MONO_HC"] = "1" if tuned else "0"
                impl.gdn_flydsl_policy = tuned_policy if tuned else GDNFlyDSLPolicy()
                stage_mixed, (stage_streams, stage_raw) = (
                    layer.attn_hyper_connection.mix_fused(hidden, pending)
                )
                stage_attention = layer.linear_attn(stage_mixed)
                stage_reduced = tensor_model_parallel_all_reduce(stage_attention)
                stage_moe_input, _ = layer.mlp_hyper_connection.mix_fused(
                    stage_streams, (stage_reduced, stage_raw)
                )
                stage_logits = layer.mlp.gate(stage_moe_input)
                stage_moe = layer.mlp(stage_moe_input)
                return tuple(
                    tensor.clone()
                    for tensor in (
                        stage_mixed,
                        stage_attention,
                        stage_moe_input,
                        stage_logits,
                        stage_moe,
                    )
                )

            baseline_stages = gdn_stages(False)
            tuned_stages = gdn_stages(True)
            ids_before = baseline_stages[3][:, :512].topk(10, dim=-1).indices
            ids_after = tuned_stages[3][:, :512].topk(10, dim=-1).indices
            route_overlap = (
                (ids_before.unsqueeze(-1) == ids_after.unsqueeze(-2))
                .any(dim=-1)
                .sum(dim=-1)
                .tolist()
            )
            gdn_stage_errors = {
                "mixed_nrmse": _error(tuned_stages[0], baseline_stages[0]),
                "attention_nrmse": _error(tuned_stages[1], baseline_stages[1]),
                "moe_input_nrmse": _error(tuned_stages[2], baseline_stages[2]),
                "router_nrmse": _error(tuned_stages[3], baseline_stages[3]),
                "moe_output_nrmse": _error(tuned_stages[4], baseline_stages[4]),
                "top10_route_overlap": route_overlap,
            }
        result = {
            "layer": args.layer,
            "rows": args.rows,
            "context": args.context if args.layer == "qsa" else None,
            "tp": world,
            "ep": world,
            "base_us": base_us,
            "tuned_us": tuned_us,
            "speedup": round(base_us / tuned_us, 3),
            "stream_and_pending_nrmse": errors,
            "gdn_state_nrmse": state_error,
            "qsa_projection_nrmse": projection_errors,
            "qsa_selection_equal": qsa_selection_equal,
            "gdn_stage_errors": gdn_stage_errors,
            "stages": stages,
            "kernel_profile": kernel_profile,
        }
        del base_graph, tuned_graph
        dist.barrier()
    destroy_model_parallel()
    dist.destroy_process_group()
    return result if rank == 0 else {}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="local config directory")
    parser.add_argument("--layer", choices=("qsa", "gdn"), required=True)
    parser.add_argument("--rows", type=int, choices=(4, 8, 16), required=True)
    parser.add_argument("--context", type=int, default=2048)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    if not Path(args.model, "config.json").is_file():
        parser.error("--model must point to a local directory with config.json")
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    output = run(args)
    if output:
        print(json.dumps(output), flush=True)


if __name__ == "__main__":
    main()
