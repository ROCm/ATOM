#!/usr/bin/env python3
"""Search launch/tile parameters for the experimental V4 FP8 Triton decode."""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from atom.model_ops.v4_kernels.paged_decode import (
    _sparse_attn_v4_paged_decode_asm,
    sparse_attn_v4_paged_decode,
)
from atom.model_ops.v4_kernels.paged_decode_fp8_triton import (
    sparse_attn_v4_paged_decode_fp8_triton,
    sparse_attn_v4_paged_decode_fp8_triton_query_group,
    sparse_attn_v4_paged_decode_fp8_triton_twopass,
)
from atom.model_ops.v4_kernels.v4_quant import quantize_bf16_to_v4_2buff_triton
from scripts.performance.bench_v4_kv_cache_dtype import (
    HEAD_DIM,
    LOCAL_HEADS,
    ROPE_HEAD_DIM,
    SOFTMAX_SCALE,
    _parse_int_list,
)
from scripts.performance.bench_v4_kv_cache_scenarios import (
    _make_request_shared_indices,
    make_scenario,
)


def _parse_matrix_instr_list(value: str) -> list[int]:
    values = [int(part) for part in value.split(",")]
    if not values or any(item not in (0, 16, 32) for item in values):
        raise argparse.ArgumentTypeError("expected a comma-separated subset of 0,16,32")
    return values


@dataclass(frozen=True)
class Config:
    block_h: int
    block_k: int
    kv_splits: int
    num_stages: int
    num_warps: int
    waves_per_eu: int
    matrix_instr_nonkdim: int
    reduce_d_chunk: int
    reduce_num_warps: int
    bf16_partials: bool = False
    fp16_partials: bool = False
    use_mxfp8_qk: bool = True
    use_native_bf16_v: bool = False
    schedule_hint: str = "none"


def _time(
    fn: Callable[[], torch.Tensor],
    flush: torch.Tensor,
    *,
    warmup: int,
    iterations: int,
) -> float:
    for _ in range(warmup):
        flush.add_(1.0)
        fn()
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(iterations):
        flush.add_(1.0)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = fn()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)) * 1000.0)
        del out
    return statistics.median(samples)


def _capture_runner(fn: Callable[[], torch.Tensor]) -> Callable[[], torch.Tensor]:
    """Capture one candidate so tuning reflects production graph replay."""
    side_stream = torch.cuda.Stream()
    side_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side_stream):
        for _ in range(3):
            fn()
    side_stream.synchronize()
    torch.cuda.current_stream().wait_stream(side_stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = fn()

    def replay() -> torch.Tensor:
        graph.replay()
        return output

    return replay


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefill-length", type=int, required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--verify-width", type=int, default=4)
    parser.add_argument(
        "--local-heads",
        type=int,
        default=LOCAL_HEADS,
        help="query heads resident on this rank (16 for TP8, 128 for DP attention)",
    )
    parser.add_argument("--pattern", choices=("hca", "csa"), required=True)
    parser.add_argument("--block-h", type=_parse_int_list, default=[16])
    parser.add_argument("--block-k", type=_parse_int_list, default=[8, 16, 32, 64])
    parser.add_argument("--kv-splits", type=_parse_int_list, default=[1, 2, 4, 8])
    parser.add_argument("--num-stages", type=_parse_int_list, default=[1, 2, 3, 4])
    parser.add_argument("--num-warps", type=_parse_int_list, default=[4, 8])
    parser.add_argument("--waves-per-eu", type=_parse_int_list, default=[1, 2])
    parser.add_argument(
        "--matrix-instr-nonkdim", type=_parse_matrix_instr_list, default=[0]
    )
    parser.add_argument(
        "--schedule-hint",
        choices=(
            "none",
            "attention",
            "memory-bound-attention",
            "attention,memory-bound-attention",
        ),
        default="none",
    )
    parser.add_argument("--reduce-d-chunk", type=_parse_int_list, default=[512])
    parser.add_argument("--reduce-num-warps", type=_parse_int_list, default=[4])
    parser.add_argument("--bf16-partials", action="store_true")
    parser.add_argument("--fp16-partials", action="store_true")
    parser.add_argument("--no-mxfp8-qk", action="store_true")
    parser.add_argument(
        "--native-bf16-v",
        action="store_true",
        help="use the gfx950 native FP8-to-BF16 conversion path for V",
    )
    parser.add_argument("--two-pass", action="store_true")
    parser.add_argument(
        "--cuda-graph",
        action="store_true",
        help="time each candidate through CUDA Graph replay",
    )
    parser.add_argument(
        "--query-group",
        "--mtp-group",
        dest="query_group",
        type=int,
        choices=(0, 2, 4, 7),
        default=0,
        help="legacy query-fused path; leave 0 for the generic head-tiled kernel",
    )
    parser.add_argument(
        "--fused-query-group",
        "--mtp-fused-group",
        dest="fused_query_group",
        type=int,
        choices=(0, 2, 4),
        default=0,
        help="queries fused per program; 0 uses the full --query-group",
    )
    parser.add_argument("--scan-warmup", type=int, default=1)
    parser.add_argument("--scan-iterations", type=int, default=5)
    parser.add_argument("--final-warmup", type=int, default=5)
    parser.add_argument("--final-iterations", type=int, default=30)
    parser.add_argument("--top", type=int, default=8)
    parser.add_argument("--l2-flush-mib", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if args.bf16_partials and args.fp16_partials:
        parser.error("--bf16-partials and --fp16-partials are mutually exclusive")
    if args.local_heads <= 0:
        parser.error("--local-heads must be positive")

    scenario = make_scenario(args.prefill_length, args.concurrency, args.verify_width)
    kv_len = scenario.hca_kv_len if args.pattern == "hca" else scenario.csa_kv_len
    tokens = scenario.tokens
    pages = args.concurrency * kv_len
    device = torch.device("cuda", torch.cuda.current_device())
    torch.manual_seed(args.seed + args.prefill_length + args.concurrency + kv_len)

    q = torch.randn(
        (tokens, args.local_heads, HEAD_DIM), dtype=torch.bfloat16, device=device
    )
    kv = torch.randn((pages, HEAD_DIM), dtype=torch.bfloat16, device=device)
    indices, indptr, qo_indptr = _make_request_shared_indices(
        args.concurrency,
        args.verify_width,
        kv_len,
        pattern=args.pattern,
        device=device,
    )
    sink = torch.randn((args.local_heads,), dtype=torch.float32, device=device)
    q_packed, q_rope = quantize_bf16_to_v4_2buff_triton(q)
    kv_packed, kv_rope = quantize_bf16_to_v4_2buff_triton(kv.view(pages, 1, HEAD_DIM))
    kv_packed = kv_packed.view(pages, HEAD_DIM)
    kv_rope = kv_rope.view(pages, ROPE_HEAD_DIM)
    flush = torch.zeros(
        args.l2_flush_mib * 1024 * 1024 // 4,
        dtype=torch.float32,
        device=device,
    )

    def run_bf16() -> torch.Tensor:
        return sparse_attn_v4_paged_decode(q, kv, indices, indptr, sink, SOFTMAX_SCALE)

    def run_aiter() -> torch.Tensor:
        return _sparse_attn_v4_paged_decode_asm(
            kv_packed,
            indices,
            indptr,
            sink,
            SOFTMAX_SCALE,
            kv_rope,
            q_packed,
            q_rope,
            qo_indptr=qo_indptr,
        )

    def make_runner(config: Config) -> Callable[[], torch.Tensor]:
        def run() -> torch.Tensor:
            if args.query_group:
                return sparse_attn_v4_paged_decode_fp8_triton_query_group(
                    q_packed,
                    q_rope,
                    kv_packed,
                    kv_rope,
                    indices,
                    indptr,
                    sink,
                    SOFTMAX_SCALE,
                    query_group=args.query_group,
                    fused_query_group=args.fused_query_group or None,
                    block_h=config.block_h,
                    block_k=config.block_k,
                    kv_splits=config.kv_splits,
                    num_stages=config.num_stages,
                    num_warps=config.num_warps,
                    waves_per_eu=config.waves_per_eu,
                    matrix_instr_nonkdim=config.matrix_instr_nonkdim,
                    use_mxfp8_qk=config.use_mxfp8_qk,
                    use_native_bf16_v=config.use_native_bf16_v,
                    reduce_d_chunk=config.reduce_d_chunk,
                    reduce_num_warps=config.reduce_num_warps,
                    bf16_partials=config.bf16_partials,
                    fp16_partials=config.fp16_partials,
                )
            if args.two_pass:
                return sparse_attn_v4_paged_decode_fp8_triton_twopass(
                    q_packed,
                    q_rope,
                    kv_packed,
                    kv_rope,
                    indices,
                    indptr,
                    sink,
                    SOFTMAX_SCALE,
                    qk_block_k=config.block_k,
                    qk_num_stages=config.num_stages,
                    pv_block_k=config.block_k,
                    pv_num_stages=config.num_stages,
                    num_warps=config.num_warps,
                    waves_per_eu=config.waves_per_eu,
                )
            return sparse_attn_v4_paged_decode_fp8_triton(
                q_packed,
                q_rope,
                kv_packed,
                kv_rope,
                indices,
                indptr,
                sink,
                SOFTMAX_SCALE,
                block_h=config.block_h,
                block_k=config.block_k,
                kv_splits=config.kv_splits,
                num_stages=config.num_stages,
                num_warps=config.num_warps,
                waves_per_eu=config.waves_per_eu,
                matrix_instr_nonkdim=config.matrix_instr_nonkdim,
                schedule_hint=config.schedule_hint,
                reduce_d_chunk=config.reduce_d_chunk,
                reduce_num_warps=config.reduce_num_warps,
                bf16_partials=config.bf16_partials,
                fp16_partials=config.fp16_partials,
                use_mxfp8_qk=config.use_mxfp8_qk,
                use_native_bf16_v=config.use_native_bf16_v,
            )

        return run

    if args.cuda_graph:
        run_bf16 = _capture_runner(run_bf16)
        run_aiter = _capture_runner(run_aiter)

    bf16_us = _time(
        run_bf16,
        flush,
        warmup=args.final_warmup,
        iterations=args.final_iterations,
    )
    aiter_us = _time(
        run_aiter,
        flush,
        warmup=args.final_warmup,
        iterations=args.final_iterations,
    )
    print(
        f"shape L={args.prefill_length} C={args.concurrency} T={tokens} "
        f"{args.pattern.upper()} K={kv_len}; BF16={bf16_us:.1f}us "
        f"AITER={aiter_us:.1f}us",
        flush=True,
    )

    configs = [
        Config(
            *values,
            bf16_partials=args.bf16_partials,
            fp16_partials=args.fp16_partials,
            use_mxfp8_qk=not args.no_mxfp8_qk,
            use_native_bf16_v=args.native_bf16_v,
            schedule_hint=args.schedule_hint,
        )
        for values in itertools.product(
            args.block_h,
            args.block_k,
            args.kv_splits,
            args.num_stages,
            args.num_warps,
            args.waves_per_eu,
            args.matrix_instr_nonkdim,
            args.reduce_d_chunk,
            args.reduce_num_warps,
        )
    ]
    scanned: list[tuple[float, Config]] = []
    for index, config in enumerate(configs, 1):
        try:
            runner = make_runner(config)
            if args.cuda_graph:
                runner = _capture_runner(runner)
            latency = _time(
                runner,
                flush,
                warmup=args.scan_warmup,
                iterations=args.scan_iterations,
            )
        except Exception as exc:  # noqa: BLE001 - tuner must continue the sweep
            print(f"FAIL {config}: {exc}", flush=True)
            continue
        scanned.append((latency, config))
        print(
            f"[{index:>3}/{len(configs)}] {latency:8.1f}us "
            f"bh={config.block_h} bk={config.block_k} split={config.kv_splits} "
            f"stage={config.num_stages} nw={config.num_warps} "
            f"waves={config.waves_per_eu} mi={config.matrix_instr_nonkdim} "
            f"rd={config.reduce_d_chunk}/rw={config.reduce_num_warps}",
            flush=True,
        )

    finalists = sorted(scanned, key=lambda item: item[0])[: args.top]
    final: list[tuple[float, Config, float, float]] = []
    reference = run_aiter().float()
    for _, config in finalists:
        runner = make_runner(config)
        correctness_runner = runner
        if args.cuda_graph:
            runner = _capture_runner(runner)
        latency = _time(
            runner,
            flush,
            warmup=args.final_warmup,
            iterations=args.final_iterations,
        )
        output = correctness_runner().float()
        cosine = torch.nn.functional.cosine_similarity(
            reference.flatten(), output.flatten(), dim=0
        ).item()
        relative_rmse = (
            (output - reference).square().mean().sqrt()
            / reference.square().mean().sqrt()
        ).item()
        final.append((latency, config, cosine, relative_rmse))

    final.sort(key=lambda item: item[0])
    print("finalists:", flush=True)
    for latency, config, cosine, relative_rmse in final:
        print(
            f"  {latency:8.1f}us vsAITER={aiter_us / latency - 1:+.1%} "
            f"vsBF16={bf16_us / latency - 1:+.1%} cos={cosine:.7f} "
            f"rrmse={relative_rmse:.4%} {config}",
            flush=True,
        )

    if args.json is not None:
        payload = {
            "shape": {
                "prefill_length": args.prefill_length,
                "concurrency": args.concurrency,
                "tokens": tokens,
                "local_heads": args.local_heads,
                "pattern": args.pattern,
                "kv_len": kv_len,
            },
            "bf16_us": bf16_us,
            "aiter_us": aiter_us,
            "finalists": [
                {
                    "latency_us": latency,
                    "config": asdict(config),
                    "cosine_vs_aiter": cosine,
                    "relative_rmse_vs_aiter": relative_rmse,
                }
                for latency, config, cosine, relative_rmse in final
            ],
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
