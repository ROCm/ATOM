#!/usr/bin/env python3
"""Microbenchmark the production DeepSeek-V4 BF16 and FP8 KV-cache paths.

This intentionally benchmarks the public wrappers used by the model rather
than isolated implementation kernels.  Changing the V4 KV-cache dtype changes
two production dispatches:

* QK RMSNorm + RoPE + fused SWA write
  * BF16: FlyDSL fused QK/RoPE, writing a BF16 SWA row
  * FP8: AITER fused QK/RoPE/group-quant, writing native V4 2-buffer rows
* Sparse paged decode attention
  * BF16: ATOM Triton paged decode
  * FP8: AITER ``mla_decode_fwd_v4_nm`` assembly kernel

The V4 indexer is deliberately absent.  It remains FP4 in both full-model
configurations, so including it would add the same term to both sides and hide
the KV-cache-specific difference this benchmark is designed to measure.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from aiter import dtypes

from atom.model_ops.v4_kernels.paged_decode import sparse_attn_v4_paged_decode
from atom.model_ops.v4_kernels.qk_norm_rope_maybe_quant import (
    qk_norm_rope_maybe_quant,
)
from atom.model_ops.v4_kernels.v4_quant import quantize_bf16_to_v4_2buff_triton

HEAD_DIM = 512
ROPE_HEAD_DIM = 64
LOCAL_HEADS = 16  # DeepSeek-V4 128 heads / TP=8.
WINDOW = 128
EPS = 1.0e-6
SOFTMAX_SCALE = HEAD_DIM**-0.5


@dataclass
class Timing:
    median_us: float
    p10_us: float
    p90_us: float
    mean_us: float
    samples: int


@dataclass
class Comparison:
    section: str
    tokens: int
    kv_len: int | None
    bf16: Timing
    fp8: Timing
    fp8_over_bf16: float
    fp8_speedup_pct: float
    winner: str


def _parse_int_list(value: str) -> list[int]:
    result = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError(
            "expected a comma-separated list of positive ints"
        )
    return result


def _percentile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        raise ValueError("cannot compute a percentile of an empty sample")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = q * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)


def _summarize(samples_ms: Sequence[float]) -> Timing:
    samples_us = sorted(value * 1000.0 for value in samples_ms)
    return Timing(
        median_us=statistics.median(samples_us),
        p10_us=_percentile(samples_us, 0.10),
        p90_us=_percentile(samples_us, 0.90),
        mean_us=statistics.fmean(samples_us),
        samples=len(samples_us),
    )


def _time_interleaved(
    bf16_fn: Callable[[], object],
    fp8_fn: Callable[[], object],
    *,
    warmup: int,
    iterations: int,
    before_each: Callable[[], None] | None = None,
) -> tuple[Timing, Timing]:
    """Time both variants with alternating order to reduce clock/order bias."""
    for iteration in range(warmup):
        if iteration % 2:
            if before_each is not None:
                before_each()
            fp8_fn()
            if before_each is not None:
                before_each()
            bf16_fn()
        else:
            if before_each is not None:
                before_each()
            bf16_fn()
            if before_each is not None:
                before_each()
            fp8_fn()
    torch.cuda.synchronize()

    samples: dict[str, list[float]] = {"bf16": [], "fp8": []}
    fns = {"bf16": bf16_fn, "fp8": fp8_fn}
    for iteration in range(iterations):
        order = ("bf16", "fp8") if iteration % 2 == 0 else ("fp8", "bf16")
        for name in order:
            if before_each is not None:
                before_each()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            result = fns[name]()
            end.record()
            end.synchronize()
            samples[name].append(float(start.elapsed_time(end)))
            # Keep the result alive until the timed work is complete.  This is
            # important because both production wrappers allocate outputs.
            del result

    return _summarize(samples["bf16"]), _summarize(samples["fp8"])


def _comparison(
    section: str,
    tokens: int,
    kv_len: int | None,
    bf16: Timing,
    fp8: Timing,
) -> Comparison:
    ratio = fp8.median_us / bf16.median_us
    speedup = (bf16.median_us / fp8.median_us - 1.0) * 100.0
    if math.isclose(bf16.median_us, fp8.median_us, rel_tol=0.002):
        winner = "tie"
    else:
        winner = "fp8" if fp8.median_us < bf16.median_us else "bf16"
    return Comparison(
        section=section,
        tokens=tokens,
        kv_len=kv_len,
        bf16=bf16,
        fp8=fp8,
        fp8_over_bf16=ratio,
        fp8_speedup_pct=speedup,
        winner=winner,
    )


def _build_rope_cache(
    max_position: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    # The model stores [max_pos, 1, 1, rotary_dim/2].  The BF16 kernel accepts
    # this shape directly and the FP8 wrapper squeezes it to AITER's 2-D table.
    half = ROPE_HEAD_DIM // 2
    position = torch.arange(max_position, device=device, dtype=torch.float32)[:, None]
    frequency = 1.0 / (
        10000.0
        ** (
            torch.arange(0, ROPE_HEAD_DIM, 2, device=device, dtype=torch.float32)
            / ROPE_HEAD_DIM
        )
    )
    phase = position * frequency[None, :]
    return (
        phase.cos().to(torch.bfloat16).view(max_position, 1, 1, half),
        phase.sin().to(torch.bfloat16).view(max_position, 1, 1, half),
    )


def benchmark_qk(
    tokens: int,
    *,
    local_heads: int = LOCAL_HEADS,
    warmup: int,
    iterations: int,
    device: torch.device,
    seed: int,
) -> Comparison:
    torch.manual_seed(seed + tokens)
    q = torch.randn(
        (tokens, local_heads * HEAD_DIM), dtype=torch.bfloat16, device=device
    )
    kv = torch.randn((tokens, HEAD_DIM), dtype=torch.bfloat16, device=device)
    kv_weight = torch.randn((HEAD_DIM,), dtype=torch.bfloat16, device=device)
    cos_cache, sin_cache = _build_rope_cache(max(tokens, 2048), device)
    positions = torch.arange(tokens, dtype=torch.int64, device=device)

    # Give every token a distinct row, matching the decode scatter without
    # introducing write contention.  The plane is larger than one window when
    # needed solely so the T sweep remains valid.
    plane_rows = max(WINDOW, tokens)
    swa_dest_rows = torch.arange(tokens, dtype=torch.int32, device=device)
    batch_ids = torch.zeros(tokens, dtype=torch.int32, device=device)
    swa_bf16 = torch.empty((plane_rows, HEAD_DIM), dtype=torch.bfloat16, device=device)
    swa_fp8 = torch.empty((plane_rows, HEAD_DIM), dtype=dtypes.fp8, device=device)
    swa_fp8_rope = torch.empty(
        (plane_rows, ROPE_HEAD_DIM), dtype=torch.bfloat16, device=device
    )

    def run_bf16() -> object:
        return qk_norm_rope_maybe_quant(
            q,
            kv,
            kv_weight,
            cos_cache,
            sin_cache,
            positions,
            local_heads,
            HEAD_DIM,
            ROPE_HEAD_DIM,
            EPS,
            quant_q=False,
            quant_k=False,
            fp8_2buff=False,
            swa_kv=swa_bf16,
            swa_dest_rows=swa_dest_rows,
            batch_id_per_q_token=batch_ids,
        )

    def run_fp8() -> object:
        return qk_norm_rope_maybe_quant(
            q,
            kv,
            kv_weight,
            cos_cache,
            sin_cache,
            positions,
            local_heads,
            HEAD_DIM,
            ROPE_HEAD_DIM,
            EPS,
            quant_q=False,
            quant_k=False,
            fp8_2buff=True,
            swa_nope_scale_buff=swa_fp8,
            swa_rope_buff=swa_fp8_rope,
            swa_dest_rows=swa_dest_rows,
            batch_id_per_q_token=batch_ids,
        )

    bf16, fp8 = _time_interleaved(
        run_bf16, run_fp8, warmup=warmup, iterations=iterations
    )
    return _comparison("qk_rope_swa_write", tokens, None, bf16, fp8)


def benchmark_attention(
    tokens: int,
    kv_len: int,
    *,
    local_heads: int = LOCAL_HEADS,
    warmup: int,
    iterations: int,
    device: torch.device,
    seed: int,
) -> Comparison:
    torch.manual_seed(seed + tokens * 10000 + kv_len)
    q = torch.randn(
        (tokens, local_heads, HEAD_DIM), dtype=torch.bfloat16, device=device
    )

    # Each query gets a disjoint, contiguous KV region.  Reusing one region for
    # every query would make the later rows artificially L2-hot and overstate
    # the benefit of either implementation.
    pages = tokens * kv_len
    kv_bf16 = torch.randn((pages, HEAD_DIM), dtype=torch.bfloat16, device=device)
    kv_indices = torch.arange(pages, dtype=torch.int32, device=device)
    kv_indptr = torch.arange(tokens + 1, dtype=torch.int32, device=device) * kv_len
    qo_indptr = torch.arange(tokens + 1, dtype=torch.int32, device=device)
    attn_sink = torch.randn((local_heads,), dtype=torch.float32, device=device)

    # Packing/quantization is deliberately outside the timed region: in the
    # model, Q arrives prepacked from QK/RoPE and KV is already stored in native
    # 2-buffer format.  Its cost is measured in the QK section instead.
    kv_fp8, kv_fp8_rope = quantize_bf16_to_v4_2buff_triton(
        kv_bf16.view(pages, 1, HEAD_DIM)
    )
    kv_fp8 = kv_fp8.view(pages, HEAD_DIM)
    kv_fp8_rope = kv_fp8_rope.view(pages, ROPE_HEAD_DIM)
    q_fp8, q_fp8_rope = quantize_bf16_to_v4_2buff_triton(q)
    torch.cuda.synchronize()

    def run_bf16() -> object:
        return sparse_attn_v4_paged_decode(
            q,
            kv_bf16,
            kv_indices,
            kv_indptr,
            attn_sink,
            SOFTMAX_SCALE,
        )

    def run_fp8() -> object:
        return sparse_attn_v4_paged_decode(
            None,
            kv_fp8,
            kv_indices,
            kv_indptr,
            attn_sink,
            SOFTMAX_SCALE,
            unified_kv_rope=kv_fp8_rope,
            q_packed_in=q_fp8,
            q_rope_in=q_fp8_rope,
            qo_indptr=qo_indptr,
        )

    bf16, fp8 = _time_interleaved(
        run_bf16, run_fp8, warmup=warmup, iterations=iterations
    )
    return _comparison("paged_decode", tokens, kv_len, bf16, fp8)


def _print_header() -> None:
    print(
        f"{'section':<20} {'T':>5} {'K':>6} "
        f"{'BF16 med us':>12} {'FP8 med us':>11} {'FP8/BF16':>10} "
        f"{'FP8 speedup':>12} {'winner':>7}",
        flush=True,
    )
    print("-" * 96, flush=True)


def _print_result(result: Comparison) -> None:
    kv_len = "-" if result.kv_len is None else str(result.kv_len)
    print(
        f"{result.section:<20} {result.tokens:>5} {kv_len:>6} "
        f"{result.bf16.median_us:>12.2f} {result.fp8.median_us:>11.2f} "
        f"{result.fp8_over_bf16:>10.3f} {result.fp8_speedup_pct:>+11.2f}% "
        f"{result.winner:>7}",
        flush=True,
    )


def _markdown(results: Sequence[Comparison]) -> str:
    lines = [
        "| section | T | K | BF16 median us | FP8 median us | FP8/BF16 | FP8 speedup | winner |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for result in results:
        kv_len = "-" if result.kv_len is None else str(result.kv_len)
        lines.append(
            f"| {result.section} | {result.tokens} | {kv_len} | "
            f"{result.bf16.median_us:.2f} | {result.fp8.median_us:.2f} | "
            f"{result.fp8_over_bf16:.3f} | {result.fp8_speedup_pct:+.2f}% | "
            f"{result.winner} |"
        )
    return "\n".join(lines) + "\n"


def _metadata(args: argparse.Namespace) -> dict[str, object]:
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    return {
        "timestamp_unix": time.time(),
        "hostname": platform.node(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "device": props.name,
        "device_index": torch.cuda.current_device(),
        "compute_units": props.multi_processor_count,
        "hip_visible_devices": os.environ.get("HIP_VISIBLE_DEVICES"),
        "warmup": args.warmup,
        "iterations": args.iterations,
        "t_values": args.t_values,
        "k_values": args.k_values,
        "section": args.section,
        "fixed": {
            "local_heads": args.local_heads,
            "head_dim": HEAD_DIM,
            "rope_head_dim": ROPE_HEAD_DIM,
            "window": WINDOW,
            "indexer_dtype": "fp4 (not timed; unchanged between variants)",
            "kv_row_bytes": {"bf16": 1024, "fp8_2buff": 640},
            "production_dispatch": {
                "qk_bf16": "FlyDSL fused QK/RoPE + BF16 SWA scatter",
                "qk_fp8": "AITER fused QK/RoPE/group-quant + 2-buffer SWA scatter",
                "attention_bf16": "ATOM Triton sparse paged decode",
                "attention_fp8": "AITER mla_decode_fwd_v4_nm ASM",
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--section", choices=("all", "qk", "attention"), default="all")
    parser.add_argument("--t-values", type=_parse_int_list, default=[16, 56, 112])
    parser.add_argument(
        "--k-values", type=_parse_int_list, default=[128, 640, 896, 1152]
    )
    parser.add_argument(
        "--local-heads",
        type=int,
        default=LOCAL_HEADS,
        help="query heads resident on this rank (16 for TP8, 128 for DP attention)",
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args()

    if args.warmup < 0 or args.iterations <= 0 or args.local_heads <= 0:
        parser.error(
            "--warmup must be >= 0; --iterations and --local-heads must be > 0"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("this benchmark requires a ROCm GPU")

    device = torch.device("cuda", torch.cuda.current_device())
    print(json.dumps(_metadata(args), indent=2), flush=True)
    _print_header()
    results: list[Comparison] = []

    if args.section in ("all", "qk"):
        for tokens in args.t_values:
            result = benchmark_qk(
                tokens,
                local_heads=args.local_heads,
                warmup=args.warmup,
                iterations=args.iterations,
                device=device,
                seed=args.seed,
            )
            results.append(result)
            _print_result(result)

    if args.section in ("all", "attention"):
        for tokens in args.t_values:
            for kv_len in args.k_values:
                result = benchmark_attention(
                    tokens,
                    kv_len,
                    local_heads=args.local_heads,
                    warmup=args.warmup,
                    iterations=args.iterations,
                    device=device,
                    seed=args.seed,
                )
                results.append(result)
                _print_result(result)
                torch.cuda.empty_cache()

    payload = {
        "metadata": _metadata(args),
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
