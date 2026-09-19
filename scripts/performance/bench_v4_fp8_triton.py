#!/usr/bin/env python3
"""Compare BF16 Triton, AITER FP8 ASM, and experimental FP8 Triton decode.

Uses the same production-shaped multi-row verification cases as
``bench_v4_kv_cache_scenarios.py``. This benchmark times attention only; QK,
the FP4 indexer, and the rest of the model are outside the measured span.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from atom.model_ops.v4_kernels.paged_decode import (
    _sparse_attn_v4_paged_decode_asm,
    sparse_attn_v4_paged_decode,
)
from atom.model_ops.v4_kernels.paged_decode_fp8_triton import (
    _q7_auto_config,
    _q7_dp_auto_config,
    sparse_attn_v4_paged_decode_fp8_triton,
    sparse_attn_v4_paged_decode_fp8_triton_query_group,
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
    Scenario,
    _make_request_shared_indices,
    make_scenario,
)


@dataclass
class KernelConfig:
    block_k: int
    kv_splits: int
    num_stages: int
    num_warps: int = 4
    use_mxfp8_qk: bool = True
    use_mxfp8_v: bool = False
    query_group: int = 1
    fused_query_group: int = 0
    reduce_d_chunk: int = 512
    reduce_num_warps: int = 4
    waves_per_eu: int = 1
    matrix_instr_nonkdim: int = 0
    fp16_partials: bool = False


@dataclass
class Result:
    scenario: Scenario
    pattern: str
    kv_len: int
    config: KernelConfig
    bf16_us: float
    aiter_fp8_us: float
    triton_fp8_us: float
    triton_vs_aiter_pct: float
    triton_vs_bf16_pct: float
    cosine_vs_aiter: float
    relative_rmse_vs_aiter: float
    cosine_vs_bf16: float
    relative_rmse_vs_bf16: float


def choose_config(tokens: int, kv_len: int) -> KernelConfig:
    """Best simple static choices found by the initial gfx950 sweep."""
    if tokens <= 64:
        if kv_len >= 1024:
            return KernelConfig(
                block_k=16,
                kv_splits=16,
                num_stages=3,
                num_warps=4,
                use_mxfp8_qk=False,
                query_group=4,
                reduce_num_warps=1,
                matrix_instr_nonkdim=16,
                fp16_partials=True,
            )
        return KernelConfig(
            block_k=16,
            kv_splits=4,
            num_stages=3,
            num_warps=8,
            use_mxfp8_qk=False,
            matrix_instr_nonkdim=16,
            reduce_num_warps=1,
            fp16_partials=True,
        )
    if tokens == 256:
        return KernelConfig(
            block_k=16,
            kv_splits=4,
            num_stages=3,
            num_warps=4,
            use_mxfp8_qk=False,
            query_group=4,
            reduce_num_warps=1,
            matrix_instr_nonkdim=16,
            fp16_partials=True,
        )
    return KernelConfig(
        block_k=16,
        kv_splits=1,
        num_stages=3,
        num_warps=8,
        use_mxfp8_qk=False,
        query_group=4,
        waves_per_eu=2 if tokens == 1024 and kv_len >= 1024 else 1,
    )


def _time_one(
    fn: Callable[[], object],
    l2_flush: torch.Tensor,
    *,
    warmup: int,
    iterations: int,
) -> float:
    for _ in range(warmup):
        l2_flush.add_(1.0)
        fn()
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(iterations):
        l2_flush.add_(1.0)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = fn()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)) * 1000.0)
        del result
    return statistics.median(samples)


def _capture_runner(fn: Callable[[], object]) -> Callable[[], object]:
    """Capture one attention call and return a stable-output graph replay."""
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

    def replay() -> object:
        graph.replay()
        return output

    return replay


def benchmark_case(
    scenario: Scenario,
    *,
    pattern: str,
    kv_len: int,
    warmup: int,
    iterations: int,
    l2_flush_mib: int,
    seed: int,
    device: torch.device,
    local_heads: int,
    use_mxfp8_v: bool,
    auto_dispatch: bool,
    cuda_graph: bool,
) -> Result:
    torch.manual_seed(seed + scenario.prefill_len + scenario.concurrency + kv_len)
    T = scenario.tokens
    pages = scenario.concurrency * kv_len
    q = torch.randn((T, local_heads, HEAD_DIM), dtype=torch.bfloat16, device=device)
    kv = torch.randn((pages, HEAD_DIM), dtype=torch.bfloat16, device=device)
    indices, indptr, qo_indptr = _make_request_shared_indices(
        scenario.concurrency,
        scenario.verify_width,
        kv_len,
        pattern=pattern,
        device=device,
    )
    sink = torch.randn((local_heads,), dtype=torch.float32, device=device)
    q_packed, q_rope = quantize_bf16_to_v4_2buff_triton(q)
    kv_packed, kv_rope = quantize_bf16_to_v4_2buff_triton(kv.view(pages, 1, HEAD_DIM))
    kv_packed = kv_packed.view(pages, HEAD_DIM)
    kv_rope = kv_rope.view(pages, ROPE_HEAD_DIM)
    config = choose_config(T, kv_len)
    if auto_dispatch and scenario.verify_width == 7:
        if local_heads == 128:
            block_k, kv_splits, stages = _q7_dp_auto_config(T, pattern)
            config = KernelConfig(
                block_k=block_k,
                kv_splits=kv_splits,
                num_stages=stages,
                num_warps=4,
                use_mxfp8_qk=True,
                query_group=7,
                fused_query_group=4,
                reduce_num_warps=1,
                matrix_instr_nonkdim=16,
                fp16_partials=True,
            )
        else:
            fused_q, block_k, kv_splits, stages, reduce_warps = _q7_auto_config(T)
            config = KernelConfig(
                block_k=block_k,
                kv_splits=kv_splits,
                num_stages=stages,
                num_warps=4,
                use_mxfp8_qk=True,
                query_group=7,
                fused_query_group=fused_q,
                reduce_num_warps=reduce_warps,
                matrix_instr_nonkdim=16,
                fp16_partials=True,
            )
    config.use_mxfp8_v = use_mxfp8_v
    l2_flush = torch.zeros(
        l2_flush_mib * 1024 * 1024 // 4,
        dtype=torch.float32,
        device=device,
    )

    def run_bf16() -> object:
        return sparse_attn_v4_paged_decode(q, kv, indices, indptr, sink, SOFTMAX_SCALE)

    def run_aiter() -> object:
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

    def run_triton() -> object:
        if auto_dispatch:
            return sparse_attn_v4_paged_decode(
                None,
                kv_packed,
                indices,
                indptr,
                sink,
                SOFTMAX_SCALE,
                unified_kv_rope=kv_rope,
                q_packed_in=q_packed,
                q_rope_in=q_rope,
                qo_indptr=qo_indptr,
                query_group=scenario.verify_width,
                kv_kind=pattern,
            )
        if config.query_group > 1:
            return sparse_attn_v4_paged_decode_fp8_triton_query_group(
                q_packed,
                q_rope,
                kv_packed,
                kv_rope,
                indices,
                indptr,
                sink,
                SOFTMAX_SCALE,
                query_group=config.query_group,
                fused_query_group=config.fused_query_group or None,
                block_k=config.block_k,
                kv_splits=config.kv_splits,
                num_stages=config.num_stages,
                num_warps=config.num_warps,
                waves_per_eu=config.waves_per_eu,
                matrix_instr_nonkdim=config.matrix_instr_nonkdim,
                use_mxfp8_qk=config.use_mxfp8_qk,
                reduce_d_chunk=config.reduce_d_chunk,
                reduce_num_warps=config.reduce_num_warps,
                fp16_partials=config.fp16_partials,
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
            block_k=config.block_k,
            kv_splits=config.kv_splits,
            num_stages=config.num_stages,
            num_warps=config.num_warps,
            waves_per_eu=config.waves_per_eu,
            matrix_instr_nonkdim=config.matrix_instr_nonkdim,
            reduce_d_chunk=config.reduce_d_chunk,
            reduce_num_warps=config.reduce_num_warps,
            fp16_partials=config.fp16_partials,
            use_mxfp8_qk=config.use_mxfp8_qk,
            use_mxfp8_v=config.use_mxfp8_v,
        )

    bf16_out = run_bf16().float()
    aiter_out = run_aiter().float()
    triton_out = run_triton().float()
    torch.cuda.synchronize()
    aiter_diff = triton_out - aiter_out
    bf16_diff = triton_out - bf16_out
    aiter_rms = aiter_out.square().mean().sqrt()
    bf16_rms = bf16_out.square().mean().sqrt()
    cosine_aiter = torch.nn.functional.cosine_similarity(
        aiter_out.flatten(), triton_out.flatten(), dim=0
    ).item()
    cosine_bf16 = torch.nn.functional.cosine_similarity(
        bf16_out.flatten(), triton_out.flatten(), dim=0
    ).item()
    relative_rmse_aiter = (aiter_diff.square().mean().sqrt() / aiter_rms).item()
    relative_rmse_bf16 = (bf16_diff.square().mean().sqrt() / bf16_rms).item()
    del bf16_out, aiter_out, triton_out, aiter_diff, bf16_diff

    if cuda_graph:
        run_bf16 = _capture_runner(run_bf16)
        run_aiter = _capture_runner(run_aiter)
        run_triton = _capture_runner(run_triton)

    bf16_us = _time_one(run_bf16, l2_flush, warmup=warmup, iterations=iterations)
    aiter_us = _time_one(run_aiter, l2_flush, warmup=warmup, iterations=iterations)
    triton_us = _time_one(run_triton, l2_flush, warmup=warmup, iterations=iterations)
    return Result(
        scenario=scenario,
        pattern=pattern,
        kv_len=kv_len,
        config=config,
        bf16_us=bf16_us,
        aiter_fp8_us=aiter_us,
        triton_fp8_us=triton_us,
        triton_vs_aiter_pct=(aiter_us / triton_us - 1.0) * 100.0,
        triton_vs_bf16_pct=(bf16_us / triton_us - 1.0) * 100.0,
        cosine_vs_aiter=cosine_aiter,
        relative_rmse_vs_aiter=relative_rmse_aiter,
        cosine_vs_bf16=cosine_bf16,
        relative_rmse_vs_bf16=relative_rmse_bf16,
    )


def _print_result(result: Result) -> None:
    s = result.scenario
    print(
        f"L={s.prefill_len:>5} C={s.concurrency:>4} T={s.tokens:>4} "
        f"{result.pattern.upper():>3} K={result.kv_len:>4} | "
        f"BF16={result.bf16_us:7.1f} AITER={result.aiter_fp8_us:7.1f} "
        f"TRITON={result.triton_fp8_us:7.1f} us | "
        f"vs AITER={result.triton_vs_aiter_pct:+6.1f}% "
        f"vs BF16={result.triton_vs_bf16_pct:+6.1f}% | "
        f"cosA={result.cosine_vs_aiter:.7f} "
        f"rrmseA={result.relative_rmse_vs_aiter:.4%} "
        f"cosB={result.cosine_vs_bf16:.7f} "
        f"rrmseB={result.relative_rmse_vs_bf16:.4%} | "
        f"bk={result.config.block_k} split={result.config.kv_splits} "
        f"qg={result.config.query_group}x{result.config.fused_query_group or result.config.query_group} "
        f"mxqk={int(result.config.use_mxfp8_qk)} "
        f"mi={result.config.matrix_instr_nonkdim}",
        flush=True,
    )


def _markdown(results: Sequence[Result]) -> str:
    lines = [
        "| Prefill | C | T | Kind | K | BF16 us | AITER FP8 us | Triton FP8 us | Triton vs AITER | Triton vs BF16 | RRMSE vs AITER | RRMSE vs BF16 | Config |",
        "|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for result in results:
        s = result.scenario
        c = result.config
        lines.append(
            f"| {s.prefill_len} | {s.concurrency} | {s.tokens} | "
            f"{result.pattern.upper()} | {result.kv_len} | {result.bf16_us:.1f} | "
            f"{result.aiter_fp8_us:.1f} | {result.triton_fp8_us:.1f} | "
            f"{result.triton_vs_aiter_pct:+.1f}% | {result.triton_vs_bf16_pct:+.1f}% | "
            f"{result.relative_rmse_vs_aiter:.4%} | "
            f"{result.relative_rmse_vs_bf16:.4%} | "
            f"qg{c.query_group}x{c.fused_query_group or c.query_group}/"
            f"bk{c.block_k}/s{c.kv_splits}/stage{c.num_stages}/"
            f"mi{c.matrix_instr_nonkdim} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prefill-lengths", type=_parse_int_list, default=[8192, 16384, 32768]
    )
    parser.add_argument(
        "--concurrencies", type=_parse_int_list, default=[16, 64, 256, 1024]
    )
    parser.add_argument("--verify-width", type=int, default=4)
    parser.add_argument(
        "--local-heads",
        type=int,
        default=LOCAL_HEADS,
        help="query heads resident on this rank (16 for TP8, 128 for DP attention)",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--l2-flush-mib", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument(
        "--mxfp8-v",
        action="store_true",
        help="try grouped BF16 x raw-FP8 V dots (experimental; normally slower)",
    )
    parser.add_argument(
        "--auto-dispatch",
        action="store_true",
        help="benchmark the production native-FP8 Triton dispatcher",
    )
    parser.add_argument(
        "--cuda-graph",
        action="store_true",
        help="time CUDA Graph replay, matching the production decode path",
    )
    parser.add_argument("--json", type=Path)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args()

    if (
        args.verify_width <= 0
        or args.local_heads <= 0
        or args.iterations <= 0
        or args.l2_flush_mib <= 0
    ):
        parser.error(
            "verify-width, local-heads, iterations, and l2-flush-mib must be positive"
        )
    if args.warmup < 0:
        parser.error("warmup must be non-negative")
    if not torch.cuda.is_available():
        raise RuntimeError("this benchmark requires a ROCm GPU")

    device = torch.device("cuda", torch.cuda.current_device())
    scenarios = [
        make_scenario(prefill_len, concurrency, args.verify_width)
        for prefill_len in args.prefill_lengths
        for concurrency in args.concurrencies
    ]
    results: list[Result] = []
    for scenario in scenarios:
        for pattern, kv_len in (
            ("hca", scenario.hca_kv_len),
            ("csa", scenario.csa_kv_len),
        ):
            result = benchmark_case(
                scenario,
                pattern=pattern,
                kv_len=kv_len,
                warmup=args.warmup,
                iterations=args.iterations,
                l2_flush_mib=args.l2_flush_mib,
                seed=args.seed,
                device=device,
                local_heads=args.local_heads,
                use_mxfp8_v=args.mxfp8_v,
                auto_dispatch=args.auto_dispatch,
                cuda_graph=args.cuda_graph,
            )
            results.append(result)
            _print_result(result)
            torch.cuda.empty_cache()

    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    payload = {
        "metadata": {
            "timestamp_unix": time.time(),
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "device": props.name,
            "compute_units": props.multi_processor_count,
            "local_heads": args.local_heads,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "l2_flush_mib": args.l2_flush_mib,
            "mxfp8_v": args.mxfp8_v,
            "auto_dispatch": args.auto_dispatch,
            "cuda_graph": args.cuda_graph,
        },
        "results": [asdict(result) for result in results],
    }
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"wrote {args.json}", flush=True)
    if args.markdown is not None:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(_markdown(results))
        print(f"wrote {args.markdown}", flush=True)


if __name__ == "__main__":
    main()
