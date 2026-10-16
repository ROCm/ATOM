# SPDX-License-Identifier: MIT
"""Compare native SP4 MoE and AITER #5937 with synthetic weights and inputs.

torchrun --nproc-per-node=4 -m atom.benchmarks.benchmark_sp_mega_moe
Uses the checkpoint's configuration only; no model weights or server are loaded.
--list-cases prints the matrix without importing torch or initializing a GPU.
"""

import argparse
import importlib.metadata
import json
import math
import os
import statistics
import time
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/home/models/MiniMax-M3-MXFP4")
    parser.add_argument(
        "--tokens", nargs="+", type=int, default=[64, 128, 8192, 16384, 32768]
    )
    parser.add_argument("--max-local-tokens", type=int, default=1024)
    parser.add_argument("--act-dtype", choices=["fp4", "fp8"], default="fp4")
    parser.add_argument("--comm-dtype", choices=["fp8", "bf16"], default="fp8")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cooldown", type=float, default=0)
    parser.add_argument("--allow-activation-change", action="store_true")
    parser.add_argument("--max-relative-l2", type=float, default=0.05)
    parser.add_argument("--max-fusion-relative-l2", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=20261009)
    parser.add_argument("--graph", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output", type=Path, default=Path("sp-mega-moe.json"))
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument(
        "--full-fusion",
        action="store_true",
        help="Include residual/norm, gate/topk and next norm/quant",
    )
    parser.add_argument(
        "--fp8-output", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--trace", action="store_true", help="Profile one fused 2K call per rank"
    )
    args = parser.parse_args()
    if (
        min(
            *args.tokens,
            args.max_local_tokens,
            args.warmup,
            args.iterations,
            args.repeats,
        )
        < 1
    ):
        parser.error(
            "token counts, tile size, warmup, iterations and repeats must be positive"
        )
    if args.max_local_tokens > 8192:
        parser.error("SP4 MegaMoE supports at most 8192 local tokens per launch")
    if args.act_dtype == "fp8" and args.max_local_tokens > 64:
        parser.error("MXFP8 activation requires --max-local-tokens <=64")
    if args.max_relative_l2 <= 0 or args.max_fusion_relative_l2 <= 0:
        parser.error("relative L2 limits must be positive")
    if args.cooldown < 0:
        parser.error("--cooldown must be nonnegative")
    return args


def case_matrix(args):
    return [
        {
            "global_tokens": tokens,
            "padded_global_tokens": math.ceil(tokens / 4) * 4,
            "local_tokens": math.ceil(tokens / 4),
            "mega_launches_per_layer": math.ceil(
                math.ceil(tokens / 4) / args.max_local_tokens
            ),
            "candidate_activation": "MX" + args.act_dtype.upper(),
            "candidate_communication": args.comm_dtype,
        }
        for tokens in args.tokens
    ]


def main():
    args = parse_args()
    if args.list_cases:
        print(json.dumps(case_matrix(args), indent=2))
        return 0

    # Set before AITER import/dispatch caches these choices. Keep both arms on
    # the same AITER build and ordinary FlyDSL weight layout.
    os.environ.update(
        ATOM_USE_TRITON_MOE="0",
        ATOM_USE_TRITON_MOE_DECODE="0",
        ATOM_MOE_GU_ITLV="0",
        ATOM_SP_MOE_MEGA="1",
        ATOM_SP_MOE_MEGA_MAX_LOCAL_TOKENS=str(args.max_local_tokens),
        ATOM_SP_MOE_MEGA_ACT_DTYPE=args.act_dtype,
        ATOM_SP_MOE_MEGA_COMM_DTYPE=args.comm_dtype,
        ATOM_SP_MOE_MEGA_ALLOW_ACTIVATION_CHANGE=(
            "1" if args.allow_activation_change else "0"
        ),
    )
    os.environ.setdefault("HSA_ENABLE_IPC_MODE_LEGACY", "1")
    os.environ.setdefault("AITER_QUICK_REDUCE_QUANTIZATION", "INT4")
    os.environ.setdefault("AITER_QUICK_REDUCE_CAST_BF16_TO_FP16", "1")
    import torch
    import torch.distributed as dist
    from aiter.dist.parallel_state import (
        get_pcp_group,
        graph_capture,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from aiter.fused_moe import resolve_activation_dtype
    from aiter.ops.flydsl.moe_common import GateMode

    from atom.config import Config, set_current_atom_config
    from atom.distributed.ulysses_sp import set_sp_world_size, sp_graph_capture
    from atom.model_ops.layernorm import GemmaRMSNorm
    from atom.model_ops.sp_mega_moe import _norm_input, initialize_sp_mega_moe
    from atom.models.minimax_m3.model import MiniMaxM3DecoderLayer, MiniMaxM3MoE
    from atom.utils.forward_context import reset_forward_context

    if int(os.environ.get("WORLD_SIZE", "0")) != 4:
        raise ValueError("Launch with torchrun --nproc-per-node=4")
    rank, local_rank = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    init_distributed_environment(world_size=4, rank=rank, local_rank=local_rank)
    initialize_model_parallel(
        tensor_model_parallel_size=1, prefill_context_model_parallel_size=4
    )
    set_sp_world_size(4)
    group = get_pcp_group().cpu_group
    maximum = max(row["padded_global_tokens"] for row in case_matrix(args))
    config = Config(
        model=args.model,
        trust_remote_code=True,
        tensor_parallel_size=1,
        sequence_parallel_size=4,
        max_num_batched_tokens=maximum,
        max_model_len=max(32768, maximum),
        moe_all2all_backend="none",
        enforce_eager=True,
    )
    set_current_atom_config(config)
    reset_forward_context()
    torch.manual_seed(args.seed + rank)
    with torch.device(device), torch.no_grad():
        block = MiniMaxM3MoE(
            config.hf_config,
            layer_id=3,
            quant_config=config.quant_config,
            params_dtype=torch.bfloat16,
            prefix="model.layers.3.block_sparse_moe",
        )
        layer = block.experts
        # All MXFP4 nibble values are finite. E8M0 bytes 120/119 give moderate
        # synthetic weights; both arms use exactly these same stored bytes.
        layer.w13_weight.view(torch.uint8).random_(0, 256)
        layer.w2_weight.view(torch.uint8).random_(0, 256)
        layer.w13_weight_scale.view(torch.uint8).fill_(120)
        layer.w2_weight_scale.view(torch.uint8).fill_(119)
        if layer.e_score_correction_bias is not None:
            layer.e_score_correction_bias.zero_()
        layer.quant_method.process_weights_after_loading(layer)
        layer.quant_method.init_prepare_finalize(layer)
        if args.full_fusion:
            # A real decoder container binds the fusion after weight loading;
            # attention is the input boundary of this synthetic region test.
            holder = MiniMaxM3DecoderLayer.__new__(MiniMaxM3DecoderLayer)
            torch.nn.Module.__init__(holder)
            holder.is_moe_layer = True
            holder.block_sparse_moe = block
            holder.input_layernorm = GemmaRMSNorm(6144).to(dtype=torch.bfloat16)
            holder.post_attention_layernorm = GemmaRMSNorm(6144).to(
                dtype=torch.bfloat16
            )
            # Router/norm weights are replicated; expert intermediate slices
            # intentionally differ on each rank.
            torch.manual_seed(args.seed)
            block.gate.to(dtype=torch.bfloat16)
            block.gate.weight.normal_(std=0.01)
            block.e_score_correction_bias.normal_(std=0.02)
            holder.input_layernorm.weight.normal_(std=0.1)
            holder.post_attention_layernorm.weight.normal_(std=0.1)
            initialize_sp_mega_moe(holder, config)
        else:
            initialize_sp_mega_moe(block, config)
    backend = layer._sp_mega_moe

    def all_rank_max(value):
        values = [None] * 4
        dist.all_gather_object(values, value, group=group)
        return max(values)

    def relative_error(actual, expected):
        error = (actual.float() - expected.float()).square().sum().item()
        magnitude = expected.float().square().sum().item()
        sums = [None] * 4
        dist.all_gather_object(sums, (error, magnitude), group=group)
        return math.sqrt(sum(v[0] for v in sums) / max(1e-30, sum(v[1] for v in sums)))

    def observed(output):
        if not args.full_fusion:
            return output
        normalized, scale, residual = output
        normalized = normalized.float()
        if args.fp8_output:
            normalized = normalized * scale
        return torch.cat((normalized, residual.float()), dim=1)

    def measure(fn):
        if args.cooldown:
            torch.cuda.synchronize()
            dist.barrier(group=group)
            time.sleep(args.cooldown)
        for _ in range(args.warmup):
            fn()
        torch.cuda.synchronize()
        dist.barrier(group=group)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
            enable_timing=True
        )
        start.record()
        for _ in range(args.iterations):
            fn()
        end.record()
        end.synchronize()
        backend.runtime.check_errors()
        return all_rank_max(start.elapsed_time(end) * 1000 / args.iterations)

    results = {
        "scope": "synthetic single M3 MoE layer, native routing and SP collectives; excludes router GEMM",
        "aiter": importlib.metadata.version("amd-aiter"),
        "flydsl": importlib.metadata.version("flydsl"),
        "settings": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "environment": {
            name: os.environ.get(name)
            for name in (
                "HIP_VISIBLE_DEVICES",
                "GPU_ARCHS",
                "AITER_QUICK_REDUCE_QUANTIZATION",
                "AITER_QUICK_REDUCE_CAST_BF16_TO_FP16",
                "GPTOSS_SWIGLU_MXFP4_BF16_BOUND",
                "AITER_MEGAMOE_TP_LB_MT",
            )
        },
        "rows": [],
    }
    if args.full_fusion:
        results["scope"] = (
            "SP-local attention output + residual -> norm/gate/topk -> MoE -> residual/next norm/quant"
        )

    def save_results():
        if rank == 0:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(results, indent=2) + "\n")

    passed = True
    with torch.no_grad():
        for row in case_matrix(args):
            m = row["local_tokens"]
            x = torch.randn((m, 6144), dtype=torch.bfloat16, device=device)
            logits = torch.randn((m, 128), dtype=torch.bfloat16, device=device)
            residual = torch.randn_like(x)
            live = max(0, min(m, row["global_tokens"] - rank * m))
            x[live:].zero_()

            def run(use_mega, x=x, logits=logits, residual=residual):
                layer._sp_mega_moe = backend if use_mega else None
                if args.full_fusion:
                    if use_mega:
                        return torch.ops.aiter.minimax_m3_sp_mega_block(
                            x,
                            residual,
                            holder.post_attention_layernorm.weight,
                            block.gate.weight,
                            block.e_score_correction_bias,
                            holder.input_layernorm.weight,
                            layer.layer_name,
                            1e-6,
                            args.fp8_output,
                        )
                    return split_region(False)
                return layer.forward_impl(x, logits)

            def split_region(use_mega, x=x, residual=residual):
                layer._sp_mega_moe = backend if use_mega else None
                normalized, _, post_res = _norm_input(
                    x, residual, holder.post_attention_layernorm.weight, 1e-6, False
                )
                router = torch.nn.functional.linear(normalized, block.gate.weight)
                out = layer.forward_impl(normalized, router)
                return _norm_input(
                    out, post_res, holder.input_layernorm.weight, 1e-6, args.fp8_output
                )

            expected = observed(run(False)).clone()
            before_x, before_res = x.clone(), residual.clone()
            candidate = observed(run(True)).clone()
            torch.testing.assert_close(x, before_x, atol=0, rtol=0)
            torch.testing.assert_close(residual, before_res, atol=0, rtol=0)
            if args.full_fusion:
                split_mega = observed(split_region(True)).clone()
                row["relative_l2_vs_split_mega"] = relative_error(
                    candidate[:live], split_mega[:live]
                )
                error = row["relative_l2_vs_split_mega"]
                row["fusion_gate_passed"] = (
                    math.isfinite(error) and error <= args.max_fusion_relative_l2
                )
                passed = passed and row["fusion_gate_passed"]
                assert backend.prefix.poll_errors() == 0
            torch.cuda.synchronize()
            backend.runtime.check_errors()
            relative_l2 = relative_error(candidate[:live], expected[:live])
            row["relative_l2_vs_native"] = relative_l2
            row["numerical_gate_passed"] = (
                math.isfinite(relative_l2) and relative_l2 <= args.max_relative_l2
            )
            passed = passed and row["numerical_gate_passed"]
            row["native_activation"] = str(
                resolve_activation_dtype(
                    layer.quant_method.quant_type,
                    layer.w13_weight.dtype,
                    activation=layer.activation,
                    gate_mode=GateMode.SEPARATED,
                    M=row["padded_global_tokens"],
                    hidden_dtype=x.dtype,
                )
            )
            selected = backend.supports(layer, x)
            row["selected_backend"] = "mega" if selected else "native"
            row["mega_launch_config"] = (
                str(backend.runtime.launch_config(min(m, args.max_local_tokens)))
                if selected
                else None
            )
            if not selected:
                row["mega_launches_per_layer"] = 0
                row["candidate_activation"] = row["native_activation"]
                row["candidate_communication"] = "native"
            results["rows"].append(row)
            save_results()
            functions = [lambda: run(False), lambda: run(True)]
            graphs, graph_outputs = [], []
            if args.graph:
                for fn in functions:
                    for _ in range(args.warmup):
                        fn()
                    torch.cuda.synchronize()
                    dist.barrier(group=group)
                    graph = torch.cuda.CUDAGraph()
                    with (
                        graph_capture() as ctx,
                        sp_graph_capture(ctx),
                        torch.cuda.graph(graph, stream=ctx.stream),
                    ):
                        out = fn()
                    graphs.append(graph)
                    graph_outputs.append(out)
                # Replaying with changed input detects stale IPC state and
                # captured pointers accidentally bound to an earlier tile.
                x.mul_(0.75)
                logits.neg_()
                row["graph_errors"] = {}
                for use_mega in (False, True):
                    fresh = observed(run(use_mega)).clone()
                    repeated = observed(run(use_mega)).clone()
                    graphs[int(use_mega)].replay()
                    torch.cuda.synchronize()
                    eager_noise = relative_error(repeated, fresh)
                    graph_error = relative_error(
                        observed(graph_outputs[int(use_mega)]), fresh
                    )
                    # Native BF16 MoE uses atomic accumulation: two eager runs
                    # already differ by ~0.2%. Bound graph error by measured
                    # eager noise and an independent 1% ceiling.
                    tolerance = min(0.01, max(1e-3, 3 * eager_noise))
                    close = (
                        math.isfinite(graph_error)
                        and math.isfinite(eager_noise)
                        and eager_noise <= 0.01
                        and graph_error <= tolerance
                    )
                    row["graph_errors"]["mega" if use_mega else "native"] = {
                        "relative_l2": graph_error,
                        "eager_repeat_relative_l2": eager_noise,
                        "tolerance": tolerance,
                        "passed": close,
                    }
                    save_results()
                    if not close:
                        raise AssertionError(
                            f"graph replay differs from eager: mega={use_mega}"
                        )
                backend.runtime.check_errors()
                functions = [g.replay for g in graphs]
                row["changed_input_graph_check"] = "passed"
            timings = [[], []]
            for repeat in range(args.repeats):
                for index in ((0, 1) if repeat % 2 == 0 else (1, 0)):
                    timings[index].append(measure(functions[index]))
            row["native_us"] = statistics.median(timings[0])
            row["mega_us"] = statistics.median(timings[1])
            row["speedup"] = row["native_us"] / row["mega_us"]
            row["rank_max_timing_samples_us"] = timings
            save_results()
            if rank == 0:
                print(json.dumps(row), flush=True)
            if args.trace and args.full_fusion and row["global_tokens"] == 2048:
                with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ]
                ) as profile:
                    run(True)
                    torch.cuda.synchronize()
                profile.export_chrome_trace(
                    str(args.output.with_suffix(f".rank{rank}.trace.json"))
                )
            del graphs, graph_outputs, functions
    dist.destroy_process_group()
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
