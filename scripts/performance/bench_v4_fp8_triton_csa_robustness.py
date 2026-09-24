#!/usr/bin/env python3
"""Validate the tuned H=128 q7 CSA Triton schedule across nearby shapes."""

from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
from collections.abc import Callable
from pathlib import Path

import torch

from atom.model_ops.v4_kernels.paged_decode import _sparse_attn_v4_paged_decode_asm
from atom.model_ops.v4_kernels.paged_decode_fp8_flydsl import (
    sparse_attn_v4_paged_decode_fp8_flydsl_graphsafe,
)
from atom.model_ops.v4_kernels.paged_decode_fp8_triton import (
    sparse_attn_v4_paged_decode_fp8_triton,
    sparse_attn_v4_paged_decode_fp8_triton_auto,
)
from atom.model_ops.v4_kernels.v4_quant import quantize_bf16_to_v4_2buff_triton
from scripts.performance.bench_v4_kv_cache_dtype import (
    HEAD_DIM,
    ROPE_HEAD_DIM,
    SOFTMAX_SCALE,
)
from scripts.performance.bench_v4_kv_cache_scenarios import (
    _make_request_shared_indices,
)

VERIFY_WIDTH = 7
LOCAL_HEADS = 128


def _parse_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",")]


def _parse_flydsl_graphsafe_config(value: str) -> tuple[int, int, int, int, int, int]:
    """Parse CAP,LONG,SHORT,SHORT_MAX,HEAD_GROUP,WAVES_PER_EU."""
    try:
        config = tuple(int(item) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("FlyDSL config must contain integers") from exc
    if len(config) != 6:
        raise argparse.ArgumentTypeError(
            "FlyDSL config must be CAP,LONG,SHORT,SHORT_MAX,HEAD_GROUP,WAVES_PER_EU"
        )
    cap, long_tiles, short_tiles, short_max, head_group, waves_per_eu = config
    if (
        cap <= 1
        or long_tiles <= 0
        or short_tiles < 0
        or short_max < 0
        or head_group not in (1, 2, 4, 8)
        or waves_per_eu < 0
    ):
        raise argparse.ArgumentTypeError("invalid FlyDSL graph-safe config")
    return config  # type: ignore[return-value]


def _capture(fn: Callable[[], torch.Tensor]) -> Callable[[], torch.Tensor]:
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(5):
            fn()
    side.synchronize()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = fn()

    def replay() -> torch.Tensor:
        graph.replay()
        return output

    return replay


def _sample(
    fn: Callable[[], torch.Tensor], flush: torch.Tensor, count: int
) -> list[float]:
    values: list[float] = []
    for _ in range(count):
        flush.add_(1.0)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        values.append(float(start.elapsed_time(end)) * 1000.0)
    return values


def _stats(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "p50_us": statistics.median(ordered),
        "p90_us": ordered[math.ceil(0.9 * len(ordered)) - 1],
    }


def _benchmark_case(
    *,
    batch: int,
    kv_len: int,
    seed: int,
    iterations: int,
    flydsl_graphsafe_config: tuple[int, int, int, int, int, int] | None,
) -> dict[str, object]:
    tokens = batch * VERIFY_WIDTH
    pages = batch * kv_len
    device = torch.device("cuda", torch.cuda.current_device())
    torch.manual_seed(seed + batch + kv_len)
    q = torch.randn(
        (tokens, LOCAL_HEADS, HEAD_DIM), dtype=torch.bfloat16, device=device
    )
    kv = torch.randn((pages, HEAD_DIM), dtype=torch.bfloat16, device=device)
    indices, indptr, qo_indptr = _make_request_shared_indices(
        batch, VERIFY_WIDTH, kv_len, pattern="csa", device=device
    )
    sink = torch.randn((LOCAL_HEADS,), dtype=torch.float32, device=device)
    q_packed, q_rope = quantize_bf16_to_v4_2buff_triton(q)
    kv_packed, kv_rope = quantize_bf16_to_v4_2buff_triton(kv.view(pages, 1, HEAD_DIM))
    kv_packed = kv_packed.view(pages, HEAD_DIM)
    kv_rope = kv_rope.view(pages, ROPE_HEAD_DIM)
    empty_indptr = torch.zeros_like(indptr)
    flydsl_out = torch.empty(
        (tokens, LOCAL_HEADS, HEAD_DIM), dtype=torch.bfloat16, device=device
    )
    flush = torch.zeros(64 * 1024 * 1024 // 4, dtype=torch.float32, device=device)

    def aiter() -> torch.Tensor:
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

    splits = 3 if batch <= 12 else 1

    def triton_runner(schedule_hint: str) -> Callable[[], torch.Tensor]:
        def run() -> torch.Tensor:
            return sparse_attn_v4_paged_decode_fp8_triton(
                q_packed,
                q_rope,
                kv_packed,
                kv_rope,
                indices,
                indptr,
                sink,
                SOFTMAX_SCALE,
                block_h=64,
                block_k=64,
                kv_splits=splits,
                num_stages=2,
                num_warps=4,
                waves_per_eu=1,
                matrix_instr_nonkdim=16,
                schedule_hint=schedule_hint,
                reduce_d_chunk=512,
                reduce_num_warps=1,
                fp16_partials=True,
                use_mxfp8_qk=True,
                use_native_bf16_v=True,
            )

        return run

    eager = {
        "aiter": aiter,
        "triton_none": triton_runner("none"),
        "triton_attention": triton_runner("attention"),
        "triton_auto": lambda: sparse_attn_v4_paged_decode_fp8_triton_auto(
            q_packed,
            q_rope,
            kv_packed,
            kv_rope,
            indices,
            indptr,
            sink,
            SOFTMAX_SCALE,
            query_group=VERIFY_WIDTH,
            kv_kind="csa",
            gfx950_native_v=True,
        ),
    }
    if flydsl_graphsafe_config is not None:
        cap, long_tiles, short_tiles, short_max, head_group, waves_per_eu = (
            flydsl_graphsafe_config
        )
        eager["flydsl_graphsafe"] = lambda: (
            sparse_attn_v4_paged_decode_fp8_flydsl_graphsafe(
                q_packed,
                q_rope,
                kv_packed,
                kv_rope,
                indices,
                indptr,
                empty_indptr,
                sink,
                SOFTMAX_SCALE,
                max_splits=cap,
                split_tiles=long_tiles,
                split_tiles_short=short_tiles,
                split_short_max_tiles=short_max,
                reduce_head_group=head_group,
                waves_per_eu=waves_per_eu,
                out=flydsl_out,
            )
        )
    reference = eager["aiter"]().float()
    correctness = {}
    for name, runner in eager.items():
        output = runner().float()
        correctness[name] = {
            "cosine": float(
                torch.nn.functional.cosine_similarity(
                    reference.flatten(), output.flatten(), dim=0
                ).item()
            ),
            "relative_rmse": float(
                (
                    (output - reference).square().mean().sqrt()
                    / reference.square().mean().sqrt()
                ).item()
            ),
            "max_abs": float((output - reference).abs().max().item()),
            "nonfinite": int((~torch.isfinite(output)).sum().item()),
        }

    runners = {name: _capture(runner) for name, runner in eager.items()}
    half = iterations // 2
    forward_order = list(eager)
    order = forward_order + list(reversed(forward_order))
    samples = {name: [] for name in runners}
    for name in order:
        samples[name].extend(_sample(runners[name], flush, half))
    results = {name: _stats(values) for name, values in samples.items()}
    aiter_p50 = results["aiter"]["p50_us"]
    for name in results.keys() - {"aiter"}:
        results[name]["speedup_vs_aiter_pct"] = (
            aiter_p50 / results[name]["p50_us"] - 1.0
        ) * 100.0
        results[name]["speedup_vs_triton_auto_pct"] = (
            results["triton_auto"]["p50_us"] / results[name]["p50_us"] - 1.0
        ) * 100.0
    flydsl_text = ""
    if flydsl_graphsafe_config is not None:
        flydsl_text = (
            f" flydsl-gs={results['flydsl_graphsafe']['p50_us']:7.3f}us "
            f"(vs-auto "
            f"{results['flydsl_graphsafe']['speedup_vs_triton_auto_pct']:+6.2f}%)"
        )
    print(
        f"seed={seed} K={kv_len:4d} B={batch:2d} split={splits} "
        f"AITER={aiter_p50:7.3f}us "
        f"none={results['triton_none']['p50_us']:7.3f}us "
        f"({results['triton_none']['speedup_vs_aiter_pct']:+6.2f}%) "
        f"attention={results['triton_attention']['p50_us']:7.3f}us "
        f"({results['triton_attention']['speedup_vs_aiter_pct']:+6.2f}%) "
        f"auto={results['triton_auto']['p50_us']:7.3f}us "
        f"({results['triton_auto']['speedup_vs_aiter_pct']:+6.2f}%)" + flydsl_text,
        flush=True,
    )
    return {
        "seed": seed,
        "kv_len": kv_len,
        "batch": batch,
        "tokens": tokens,
        "kv_splits": splits,
        "results": results,
        "correctness": correctness,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", type=_parse_ints, default=list(range(10, 17)))
    parser.add_argument("--kv-lens", type=_parse_ints, default=[384, 640, 1152])
    parser.add_argument("--seeds", type=_parse_ints, default=[20260917, 20260918])
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument(
        "--flydsl-graphsafe-config",
        type=_parse_flydsl_graphsafe_config,
        help=(
            "also benchmark graph-safe FlyDSL as "
            "CAP,LONG,SHORT,SHORT_MAX,HEAD_GROUP,WAVES_PER_EU"
        ),
    )
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    if args.iterations % 2:
        parser.error("--iterations must be even")

    cases = []
    for seed in args.seeds:
        for kv_len in args.kv_lens:
            for batch in args.batches:
                cases.append(
                    _benchmark_case(
                        batch=batch,
                        kv_len=kv_len,
                        seed=seed,
                        iterations=args.iterations,
                        flydsl_graphsafe_config=args.flydsl_graphsafe_config,
                    )
                )
                gc.collect()
                torch.cuda.empty_cache()

    wins = sum(
        case["results"]["triton_attention"]["speedup_vs_aiter_pct"] > 0
        for case in cases
    )
    hint_wins = sum(
        case["results"]["triton_attention"]["p50_us"]
        < case["results"]["triton_none"]["p50_us"]
        for case in cases
    )
    flydsl_graphsafe_wins = (
        sum(
            case["results"]["flydsl_graphsafe"]["speedup_vs_triton_auto_pct"] > 0
            for case in cases
        )
        if args.flydsl_graphsafe_config is not None
        else None
    )
    payload = {
        "iterations": args.iterations,
        "cases": cases,
        "summary": {
            "case_count": len(cases),
            "triton_attention_wins_vs_aiter": wins,
            "attention_hint_wins_vs_none": hint_wins,
            "flydsl_graphsafe_wins_vs_triton_auto": flydsl_graphsafe_wins,
        },
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        f"summary: Triton attention beat AITER in {wins}/{len(cases)} cases; "
        f"hint beat default Triton in {hint_wins}/{len(cases)} cases"
        + (
            "; graph-safe FlyDSL beat Triton auto in "
            f"{flydsl_graphsafe_wins}/{len(cases)} cases"
            if args.flydsl_graphsafe_config is not None
            else ""
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
