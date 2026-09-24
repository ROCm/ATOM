#!/usr/bin/env python3
"""Tune DP-attention q7 HCA decode on heterogeneous request lengths.

The ordinary tuner gives every request the same context length.  Agentic
serving does not: one DP rank can decode a long request beside several much
shorter requests.  This benchmark gives each request its own physical HCA
cache and repeats that request's indices for the seven target verification
rows.  HCA stores the 128-row local window plus one compressed row per 128
context tokens, so command-line vectors contain context lengths rather than
the already-compressed kernel row counts.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from atom.model_ops.v4_kernels.paged_decode import _sparse_attn_v4_paged_decode_asm
from atom.model_ops.v4_kernels.paged_decode_fp8_flydsl import (
    sparse_attn_v4_paged_decode_fp8_flydsl_graphsafe,
)
from atom.model_ops.v4_kernels.paged_decode_fp8_triton import (
    sparse_attn_v4_paged_decode_fp8_triton,
    sparse_attn_v4_paged_decode_fp8_triton_auto,
    sparse_attn_v4_paged_decode_fp8_triton_query_group,
)
from atom.model_ops.v4_kernels.v4_quant import quantize_bf16_to_v4_2buff_triton
from scripts.performance.bench_v4_kv_cache_dtype import (
    HEAD_DIM,
    ROPE_HEAD_DIM,
    SOFTMAX_SCALE,
)

VERIFY_WIDTH = 7
LOCAL_HEADS = 128
WINDOW = 128
COMPRESS_RATIO = 128


@dataclass(frozen=True)
class Config:
    name: str
    mode: str
    block_h: int
    block_k: int
    kv_splits: int
    num_stages: int
    num_warps: int
    matrix_instr_nonkdim: int
    use_mxfp8_qk: bool = True
    use_mxfp8_v: bool = False
    use_native_bf16_v: bool = False
    fused_query_group: int = 4
    split_tiles: int = 0
    waves_per_eu: int = 1
    reduce_d_chunk: int = 512
    reduce_num_warps: int = 1
    reduce_head_group: int = 1
    split_tiles_short: int = 0
    split_short_max_tiles: int = 0
    split_tiles_mid: int = 0
    split_mid_max_tiles: int = 0
    packed_query_group: int = 0
    packed_uniform_splits: int = 0
    sequential_reduce: bool = False
    schedule_hint: str = "none"


CONFIGS = (
    Config(
        "native-adaptive-short7-long9-packed-q7",
        "fused",
        64,
        64,
        11,
        2,
        4,
        16,
        use_native_bf16_v=True,
        split_tiles=9,
        split_tiles_short=7,
        split_short_max_tiles=32,
        packed_query_group=VERIFY_WIDTH,
    ),
    Config(
        "native-adaptive-uniform3-cap11-sequential-rd512-packed-q7",
        "fused",
        64,
        64,
        11,
        2,
        4,
        16,
        use_native_bf16_v=True,
        split_tiles=9,
        split_tiles_short=7,
        split_short_max_tiles=32,
        packed_query_group=VERIFY_WIDTH,
        packed_uniform_splits=3,
        sequential_reduce=True,
    ),
    Config(
        "native-b9-adaptive-cap9-short2-mid3-long9-packed-q7",
        "fused",
        64,
        64,
        9,
        2,
        4,
        16,
        use_native_bf16_v=True,
        split_tiles=9,
        split_tiles_short=2,
        split_short_max_tiles=3,
        split_tiles_mid=3,
        split_mid_max_tiles=6,
        packed_query_group=VERIFY_WIDTH,
    ),
)

DEFAULT_CONTEXT_VECTORS = (
    (8192, 32768, 86016, 262144, 524288, 700000),
    (32768, 65536, 98304, 131072, 262144, 524288),
    (8192, 8192, 32768, 86016, 262144, 700000),
)


def _parse_vectors(value: str) -> list[tuple[int, ...]]:
    try:
        vectors = [
            tuple(int(item) for item in vector.split(","))
            for vector in value.split(";")
        ]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("vectors must contain integers") from exc
    if not vectors or any(
        not vector or any(item <= 0 for item in vector) for vector in vectors
    ):
        raise argparse.ArgumentTypeError(
            "vectors and context lengths must be non-empty and positive"
        )
    return vectors


def _parse_flydsl_graphsafe_config(value: str) -> Config:
    """Parse a graph-safe FlyDSL schedule, optionally with a middle K tier."""
    fields = value.split(",")
    if len(fields) not in (7, 9):
        raise argparse.ArgumentTypeError(
            "FlyDSL config must be "
            "NAME,CAP,LONG,SHORT,SHORT_MAX[,MID,MID_MAX],HEAD_GROUP,WAVES_PER_EU"
        )
    name = fields[0]
    try:
        values = tuple(int(item) for item in fields[1:])
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "FlyDSL config fields after NAME must be integers"
        ) from exc
    if len(values) == 6:
        cap, long_tiles, short_tiles, short_max, head_group, waves_per_eu = values
        mid_tiles = mid_max = 0
    else:
        (
            cap,
            long_tiles,
            short_tiles,
            short_max,
            mid_tiles,
            mid_max,
            head_group,
            waves_per_eu,
        ) = values
    if (
        cap <= 1
        or long_tiles <= 0
        or short_tiles < 0
        or short_max < 0
        or mid_tiles < 0
        or mid_max < 0
        or head_group not in (1, 2, 4, 8)
        or waves_per_eu < 0
    ):
        raise argparse.ArgumentTypeError("invalid FlyDSL graph-safe config")
    return Config(
        name,
        "flydsl_graphsafe",
        128,
        32,
        cap,
        2,
        8,
        0,
        split_tiles=long_tiles,
        split_tiles_short=short_tiles,
        split_short_max_tiles=short_max,
        split_tiles_mid=mid_tiles,
        split_mid_max_tiles=mid_max,
        waves_per_eu=waves_per_eu,
        reduce_head_group=head_group,
    )


def _hca_rows(context_len: int) -> int:
    return WINDOW + (context_len + VERIFY_WIDTH) // COMPRESS_RATIO


def _make_heterogeneous_indices(
    kv_lens: Sequence[int], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    request_indices: list[torch.Tensor] = []
    physical_offset = 0
    for kv_len in kv_lens:
        rows = torch.arange(
            physical_offset,
            physical_offset + kv_len,
            dtype=torch.int32,
            device=device,
        )
        request_indices.extend([rows] * VERIFY_WIDTH)
        physical_offset += kv_len
    indices = torch.cat(request_indices).contiguous()
    query_kv_lens = torch.tensor(
        [kv_len for kv_len in kv_lens for _ in range(VERIFY_WIDTH)],
        dtype=torch.int32,
        device=device,
    )
    indptr = torch.empty(query_kv_lens.numel() + 1, dtype=torch.int32, device=device)
    indptr[0] = 0
    torch.cumsum(query_kv_lens, dim=0, out=indptr[1:])
    return indices, indptr


def _capture(fn: Callable[[], torch.Tensor]) -> Callable[[], torch.Tensor]:
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


def _time(
    fn: Callable[[], torch.Tensor],
    flush: torch.Tensor,
    *,
    warmup: int,
    iterations: int,
) -> tuple[float, float, float]:
    for _ in range(warmup):
        flush.add_(1.0)
        fn()
    torch.cuda.synchronize()
    samples: list[float] = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(iterations):
        flush.add_(1.0)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)) * 1000.0)
    samples.sort()
    return (
        statistics.median(samples),
        samples[max(0, int(iterations * 0.1) - 1)],
        samples[min(iterations - 1, int(iterations * 0.9))],
    )


def _make_runner(
    config: Config,
    q_packed: torch.Tensor,
    q_rope: torch.Tensor,
    kv_packed: torch.Tensor,
    kv_rope: torch.Tensor,
    indices: torch.Tensor,
    indptr: torch.Tensor,
    sink: torch.Tensor,
    empty_indptr: torch.Tensor | None = None,
    flydsl_out: torch.Tensor | None = None,
) -> Callable[[], torch.Tensor]:
    def run() -> torch.Tensor:
        if config.mode == "flydsl_graphsafe":
            if empty_indptr is None or flydsl_out is None:
                raise RuntimeError("graph-safe FlyDSL buffers were not provided")
            return sparse_attn_v4_paged_decode_fp8_flydsl_graphsafe(
                q_packed,
                q_rope,
                kv_packed,
                kv_rope,
                indices,
                indptr,
                empty_indptr,
                sink,
                SOFTMAX_SCALE,
                max_splits=config.kv_splits,
                block_k=config.block_k,
                split_tiles=config.split_tiles,
                split_tiles_short=config.split_tiles_short,
                split_short_max_tiles=config.split_short_max_tiles,
                split_tiles_mid=config.split_tiles_mid,
                split_mid_max_tiles=config.split_mid_max_tiles,
                reduce_head_group=config.reduce_head_group,
                waves_per_eu=config.waves_per_eu,
                out=flydsl_out,
            )
        if config.mode == "auto":
            return sparse_attn_v4_paged_decode_fp8_triton_auto(
                q_packed,
                q_rope,
                kv_packed,
                kv_rope,
                indices,
                indptr,
                sink,
                SOFTMAX_SCALE,
                query_group=VERIFY_WIDTH,
                kv_kind="hca",
            )
        if config.mode == "group":
            return sparse_attn_v4_paged_decode_fp8_triton_query_group(
                q_packed,
                q_rope,
                kv_packed,
                kv_rope,
                indices,
                indptr,
                sink,
                SOFTMAX_SCALE,
                query_group=VERIFY_WIDTH,
                fused_query_group=config.fused_query_group,
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
                fp16_partials=True,
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
            use_mxfp8_qk=config.use_mxfp8_qk,
            use_mxfp8_v=config.use_mxfp8_v,
            use_native_bf16_v=config.use_native_bf16_v,
            split_tiles=config.split_tiles,
            split_tiles_short=config.split_tiles_short,
            split_short_max_tiles=config.split_short_max_tiles,
            split_tiles_mid=config.split_tiles_mid,
            split_mid_max_tiles=config.split_mid_max_tiles,
            packed_query_group=config.packed_query_group,
            packed_uniform_splits=config.packed_uniform_splits,
            sequential_reduce=config.sequential_reduce,
            reduce_d_chunk=config.reduce_d_chunk,
            reduce_num_warps=config.reduce_num_warps,
            fp16_partials=True,
        )

    return run


def benchmark_vector(
    context_lens: Sequence[int],
    *,
    warmup: int,
    iterations: int,
    l2_flush_mib: int,
    seed: int,
    device: torch.device,
) -> dict[str, object]:
    kv_lens = [_hca_rows(context_len) for context_len in context_lens]
    requests = len(context_lens)
    tokens = requests * VERIFY_WIDTH
    pages = sum(kv_lens)
    torch.manual_seed(seed + sum(context_lens) + requests)
    q = torch.randn(
        (tokens, LOCAL_HEADS, HEAD_DIM), dtype=torch.bfloat16, device=device
    )
    kv = torch.randn((pages, HEAD_DIM), dtype=torch.bfloat16, device=device)
    indices, indptr = _make_heterogeneous_indices(kv_lens, device)
    sink = torch.randn((LOCAL_HEADS,), dtype=torch.float32, device=device)
    q_packed, q_rope = quantize_bf16_to_v4_2buff_triton(q)
    kv_packed, kv_rope = quantize_bf16_to_v4_2buff_triton(kv.view(pages, 1, HEAD_DIM))
    kv_packed = kv_packed.view(pages, HEAD_DIM)
    kv_rope = kv_rope.view(pages, ROPE_HEAD_DIM)
    empty_indptr = torch.zeros_like(indptr)
    flydsl_out = torch.empty(
        (tokens, LOCAL_HEADS, HEAD_DIM), dtype=torch.bfloat16, device=device
    )
    qo_indptr = torch.arange(tokens + 1, dtype=torch.int32, device=device)
    flush = torch.zeros(
        l2_flush_mib * 1024 * 1024 // 4, dtype=torch.float32, device=device
    )

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

    configs = CONFIGS
    eager_runners = {
        config.name: _make_runner(
            config,
            q_packed,
            q_rope,
            kv_packed,
            kv_rope,
            indices,
            indptr,
            sink,
            empty_indptr,
            flydsl_out,
        )
        for config in configs
    }
    reference = run_aiter().float()
    aiter_runner = _capture(run_aiter)
    runners = {name: _capture(fn) for name, fn in eager_runners.items()}
    results: list[dict[str, object]] = []
    current_us = 0.0
    aiter_us, aiter_p10_us, aiter_p90_us = _time(
        aiter_runner, flush, warmup=warmup, iterations=iterations
    )
    print(
        f"             AITER: {aiter_us:7.2f} us "
        f"p10/p90={aiter_p10_us:.2f}/{aiter_p90_us:.2f}",
        flush=True,
    )
    for config in configs:
        output = eager_runners[config.name]().float()
        cosine = torch.nn.functional.cosine_similarity(
            reference.flatten(), output.flatten(), dim=0
        ).item()
        relative_rmse = (
            (output - reference).square().mean().sqrt()
            / reference.square().mean().sqrt()
        ).item()
        median_us, p10_us, p90_us = _time(
            runners[config.name], flush, warmup=warmup, iterations=iterations
        )
        if config is CONFIGS[0]:
            current_us = median_us
        result = {
            "config": asdict(config),
            "median_us": median_us,
            "p10_us": p10_us,
            "p90_us": p90_us,
            "speedup_vs_current_pct": (current_us / median_us - 1.0) * 100.0,
            "speedup_vs_aiter_pct": (aiter_us / median_us - 1.0) * 100.0,
            "cosine_vs_aiter": cosine,
            "relative_rmse_vs_aiter": relative_rmse,
            "max_abs_vs_aiter": float((output - reference).abs().max().item()),
            "nonfinite": int((~torch.isfinite(output)).sum().item()),
        }
        results.append(result)
        print(
            f"  {config.name:>16}: {median_us:7.2f} us "
            f"p10/p90={p10_us:.2f}/{p90_us:.2f} "
            f"vs-current={result['speedup_vs_current_pct']:+6.2f}% "
            f"vs-AITER={result['speedup_vs_aiter_pct']:+6.2f}% "
            f"cos={cosine:.7f} rrmse={relative_rmse:.4%}",
            flush=True,
        )
    results.sort(key=lambda item: float(item["median_us"]))
    return {
        "context_lens": list(context_lens),
        "hca_kv_lens": kv_lens,
        "requests": requests,
        "tokens": tokens,
        "aiter_us": aiter_us,
        "aiter_p10_us": aiter_p10_us,
        "aiter_p90_us": aiter_p90_us,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--context-vectors",
        type=_parse_vectors,
        default=list(DEFAULT_CONTEXT_VECTORS),
        help="semicolon-separated vectors of comma-separated context lengths",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--l2-flush-mib", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument(
        "--configs",
        help="comma-separated config names; default benchmarks every config",
    )
    parser.add_argument(
        "--flydsl-graphsafe-config",
        type=_parse_flydsl_graphsafe_config,
        help=(
            "benchmark one graph-safe FlyDSL config as "
            "NAME,CAP,LONG,SHORT,SHORT_MAX[,MID,MID_MAX],HEAD_GROUP,WAVES_PER_EU"
        ),
    )
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if args.warmup < 0 or args.iterations <= 0 or args.l2_flush_mib <= 0:
        parser.error("warmup must be >=0; iterations and l2-flush-mib must be >0")
    if not torch.cuda.is_available():
        raise RuntimeError("this benchmark requires a ROCm GPU")

    global CONFIGS
    if args.flydsl_graphsafe_config is not None:
        CONFIGS = (CONFIGS[0], args.flydsl_graphsafe_config)
    elif args.configs:
        requested = args.configs.split(",")
        configs_by_name = {config.name: config for config in CONFIGS}
        missing = [name for name in requested if name not in configs_by_name]
        if missing:
            parser.error(f"unknown configs: {','.join(missing)}")
        CONFIGS = tuple(configs_by_name[name] for name in requested)

    device = torch.device("cuda", torch.cuda.current_device())
    payload: dict[str, object] = {
        "metadata": {
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "device": torch.cuda.get_device_properties(device).name,
            "verify_width": VERIFY_WIDTH,
            "local_heads": LOCAL_HEADS,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "l2_flush_mib": args.l2_flush_mib,
            "seed": args.seed,
            "cuda_graph": True,
        },
        "vectors": [],
    }
    for index, context_lens in enumerate(args.context_vectors, 1):
        print(
            f"vector {index}: contexts={list(context_lens)} "
            f"HCA-rows={[_hca_rows(item) for item in context_lens]}",
            flush=True,
        )
        result = benchmark_vector(
            context_lens,
            warmup=args.warmup,
            iterations=args.iterations,
            l2_flush_mib=args.l2_flush_mib,
            seed=args.seed,
            device=device,
        )
        payload["vectors"].append(result)  # type: ignore[union-attr]
        torch.cuda.empty_cache()

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"wrote {args.json}", flush=True)


if __name__ == "__main__":
    main()
