#!/usr/bin/env python3
"""ABBA benchmark for V4 FlyDSL/dispatch native-FP8 prefill vs AITER OPUS.

The generated CSR rows model long-context V4 prefill rather than dense full
attention: the prefix span is the retained SWA/compressor set and the extend
span is the causal tail from the current chunk.  ``mixed`` alternates short and
long rows so a candidate cannot win only by assuming uniform lengths.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from aiter.ops.pa_sparse_prefill_opus import pa_sparse_prefill_fp8_opus

from atom.model_ops.v4_kernels.paged_prefill_fp8_flydsl import (
    sparse_attn_v4_paged_prefill_fp8_flydsl,
)
from atom.model_ops.v4_kernels.paged_prefill import sparse_attn_v4_paged_prefill
from atom.model_ops.v4_kernels.v4_quant import quantize_bf16_to_v4_2buff_triton


@dataclass(frozen=True)
class Config:
    waves_per_eu: int

    @property
    def name(self) -> str:
        return f"weu{self.waves_per_eu}"


DEFAULT_CONFIGS = (Config(1),)


def _parse_config(value: str) -> Config:
    return Config(int(value))


def _counts(tokens: int, maximum: int, scenario: str) -> list[int]:
    if maximum <= 0:
        return [0] * tokens
    if scenario == "uniform":
        return [maximum] * tokens
    if scenario == "mixed":
        pattern = (0, 1, 8, 32, max(1, maximum // 4), max(1, maximum // 2), maximum)
        return [min(maximum, pattern[i % len(pattern)]) for i in range(tokens)]
    # Causal growth up to the requested long-context bound.
    return [min(maximum, i + 1) for i in range(tokens)]


def _csr(
    counts: list[int],
    rows: int,
    *,
    device: torch.device,
    stride: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    indptr = torch.zeros(len(counts) + 1, dtype=torch.int32)
    indptr[1:] = torch.tensor(counts, dtype=torch.int64).cumsum(0).to(torch.int32)
    pieces = []
    for token, count in enumerate(counts):
        if count:
            pieces.append((token * 131 + torch.arange(count) * stride) % rows)
    indices = (
        torch.cat(pieces).to(torch.int32)
        if pieces
        else torch.empty(0, dtype=torch.int32)
    )
    return indices.to(device), indptr.to(device)


def _compact_sentinels(
    indices: torch.Tensor, indptr: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the sentinel-free CSR that OPUS needs as a semantic reference."""
    indices_cpu = indices.cpu()
    indptr_cpu = indptr.cpu()
    pieces = []
    compact_indptr = [0]
    for row in range(indptr_cpu.numel() - 1):
        piece = indices_cpu[indptr_cpu[row] : indptr_cpu[row + 1]]
        piece = piece[piece >= 0]
        pieces.append(piece)
        compact_indptr.append(compact_indptr[-1] + piece.numel())
    compact_indices = (
        torch.cat(pieces) if pieces else torch.empty(0, dtype=indices_cpu.dtype)
    )
    return (
        compact_indices.to(indices.device),
        torch.tensor(compact_indptr, dtype=torch.int32, device=indices.device),
    )


def _time_once(fn) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end) * 1000.0)


def _summary(samples: list[float]) -> dict[str, float | list[float]]:
    ordered = sorted(samples)
    mean = statistics.fmean(ordered)
    stdev = statistics.pstdev(ordered)
    return {
        "median_us": statistics.median(ordered),
        "mean_us": mean,
        "stdev_us": stdev,
        "cv_pct": stdev / mean * 100.0,
        "min_us": ordered[0],
        "max_us": ordered[-1],
        "samples_us": samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--heads", type=int, default=128)
    parser.add_argument("--prefix-len", type=int, default=643)
    parser.add_argument("--extend-len", type=int, default=128)
    parser.add_argument(
        "--scenario", choices=("uniform", "mixed", "causal"), default="mixed"
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--input-std", type=float, default=0.125)
    parser.add_argument(
        "--prefix-sentinel-period",
        type=int,
        default=0,
        help="replace every Nth prefix index with -1 (0 disables injection)",
    )
    parser.add_argument("--assume-full-tiles", action="store_true")
    parser.add_argument(
        "--profile-target",
        choices=("both", "opus", "candidate"),
        default="both",
        help="run only one attention implementation for hardware-counter capture",
    )
    parser.add_argument(
        "--backend",
        choices=("flydsl", "dispatch"),
        default="flydsl",
    )
    parser.add_argument("--flydsl-pipeline-two", action="store_true")
    parser.add_argument("--flydsl-cluster-two", action="store_true")
    parser.add_argument("--flydsl-wave-padded-k", action="store_true")
    parser.add_argument("--flydsl-alpha-bpermute", action="store_true")
    parser.add_argument(
        "--flydsl-lds-padding",
        type=int,
        choices=(0, 1, 2, 4, 6, 8, 10, 12, 16),
        default=0,
    )
    parser.add_argument("--flydsl-rescale-once", action="store_true")
    parser.add_argument("--flydsl-fixed-softmax-ref", action="store_true")
    parser.add_argument("--flydsl-transpose-v", action="store_true")
    parser.add_argument("--flydsl-register-p", action="store_true")
    parser.add_argument(
        "--flydsl-cache-q-step", type=int, choices=(-1, 0, 1, 2), default=-1
    )
    parser.add_argument("--flydsl-reuse-q-across-n", action="store_true")
    parser.add_argument("--flydsl-broadcast-indices", action="store_true")
    parser.add_argument("--flydsl-no-sentinel", action="store_true")
    parser.add_argument("--flydsl-full-tile-fastpath", action="store_true")
    parser.add_argument("--flydsl-p-lane-layout", action="store_true")
    parser.add_argument("--flydsl-stage-q", action="store_true")
    parser.add_argument("--flydsl-cache-all-q", action="store_true")
    parser.add_argument("--flydsl-permute-k-scales", action="store_true")
    parser.add_argument("--flydsl-pairwise-pv", action="store_true")
    parser.add_argument("--flydsl-post-misched", action="store_true")
    parser.add_argument("--flydsl-machine-sink", action="store_true")
    parser.add_argument("--flydsl-setprio", action="store_true")
    parser.add_argument("--flydsl-dynamic-full-prefix", action="store_true")
    parser.add_argument(
        "--config",
        action="append",
        type=_parse_config,
        help="repeatable FlyDSL waves-per-EU value",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("this benchmark requires an AMD GPU")
    if args.tokens <= 0 or args.heads <= 0:
        raise ValueError("tokens and heads must be positive")
    if args.backend == "flydsl" and args.assume_full_tiles:
        if (
            args.scenario != "uniform"
            or args.prefix_len % 32 != 0
            or args.extend_len % 32 != 0
        ):
            raise ValueError(
                "FlyDSL --assume-full-tiles requires uniform prefix/extend "
                "lengths divisible by 32"
            )

    configs = tuple(args.config or DEFAULT_CONFIGS)
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    prefix_rows = max(4096, args.prefix_len * 8)
    extend_rows = max(args.tokens, args.extend_len, 1)

    q = (
        torch.randn(
            args.tokens,
            args.heads,
            512,
            dtype=torch.float32,
            device=device,
        )
        * args.input_std
    ).to(torch.bfloat16)
    prefix = (
        torch.randn(prefix_rows, 512, dtype=torch.float32, device=device)
        * args.input_std
    ).to(torch.bfloat16)
    extend = (
        torch.randn(extend_rows, 512, dtype=torch.float32, device=device)
        * args.input_std
    ).to(torch.bfloat16)
    q_packed, q_rope = quantize_bf16_to_v4_2buff_triton(q)
    prefix_packed, prefix_rope = quantize_bf16_to_v4_2buff_triton(prefix)
    extend_packed, extend_rope = quantize_bf16_to_v4_2buff_triton(extend)
    del q, prefix, extend

    prefix_counts = _counts(args.tokens, args.prefix_len, args.scenario)
    extend_counts = _counts(args.tokens, args.extend_len, args.scenario)
    prefix_indices, prefix_indptr = _csr(
        prefix_counts, prefix_rows, device=device, stride=127
    )
    if args.prefix_sentinel_period < 0:
        raise ValueError("prefix_sentinel_period must be non-negative")
    if args.prefix_sentinel_period and prefix_indices.numel():
        prefix_indices[:: args.prefix_sentinel_period] = -1
    opus_prefix_indices, opus_prefix_indptr = _compact_sentinels(
        prefix_indices, prefix_indptr
    )
    extend_indices, extend_indptr = _csr(
        extend_counts, extend_rows, device=device, stride=1
    )
    sink = torch.randn(args.heads, dtype=torch.float32, device=device) * 0.125
    scale = 1.0 / math.sqrt(512)
    candidate_common = (
        q_packed,
        q_rope,
        prefix_packed,
        prefix_rope,
        prefix_indices,
        prefix_indptr,
        extend_packed,
        extend_rope,
        extend_indices,
        extend_indptr,
        sink,
        scale,
    )
    opus_common = (
        q_packed,
        q_rope,
        prefix_packed,
        prefix_rope,
        opus_prefix_indices,
        opus_prefix_indptr,
        extend_packed,
        extend_rope,
        extend_indices,
        extend_indptr,
        sink,
        scale,
    )
    opus_out = torch.empty(
        args.tokens, args.heads, 512, dtype=torch.bfloat16, device=device
    )
    candidate_out = torch.empty_like(opus_out)

    def opus():
        return pa_sparse_prefill_fp8_opus(*opus_common, out=opus_out)

    records = []
    for config in configs:
        kwargs = asdict(config)

        def candidate():
            if args.backend == "dispatch":
                return sparse_attn_v4_paged_prefill(
                    None,
                    prefix_packed,
                    prefix_indices,
                    prefix_indptr,
                    None,
                    extend_indices,
                    extend_indptr,
                    sink,
                    scale,
                    out=candidate_out,
                    unified_kv_rope=prefix_rope,
                    q_packed=q_packed,
                    q_rope=q_rope,
                    k_packed=extend_packed,
                    k_rope=extend_rope,
                    prefix_has_sentinel=bool(args.prefix_sentinel_period),
                    max_seqlen_q=args.extend_len,
                    max_seqlen_k=args.prefix_len + args.extend_len,
                )
            return sparse_attn_v4_paged_prefill_fp8_flydsl(
                *candidate_common,
                out=candidate_out,
                waves_per_eu=config.waves_per_eu,
                pipeline_two=args.flydsl_pipeline_two,
                cluster_two=args.flydsl_cluster_two,
                wave_padded_k=args.flydsl_wave_padded_k,
                alpha_bpermute=args.flydsl_alpha_bpermute,
                lds_padding=args.flydsl_lds_padding,
                rescale_once=args.flydsl_rescale_once,
                fixed_softmax_ref=args.flydsl_fixed_softmax_ref,
                transpose_v=args.flydsl_transpose_v,
                register_p=args.flydsl_register_p,
                cache_q_step=args.flydsl_cache_q_step,
                reuse_q_across_n=args.flydsl_reuse_q_across_n,
                broadcast_indices=args.flydsl_broadcast_indices,
                no_sentinel=args.flydsl_no_sentinel,
                full_tile_fastpath=args.flydsl_full_tile_fastpath,
                p_lane_layout=args.flydsl_p_lane_layout,
                stage_q=args.flydsl_stage_q,
                cache_all_q=args.flydsl_cache_all_q,
                assume_full_tiles=args.assume_full_tiles,
                permute_k_scales=args.flydsl_permute_k_scales,
                pairwise_pv=args.flydsl_pairwise_pv,
                post_misched=args.flydsl_post_misched,
                machine_sink=args.flydsl_machine_sink,
                setprio=args.flydsl_setprio,
                dynamic_full_prefix=args.flydsl_dynamic_full_prefix,
            )

        if args.profile_target != "both":
            target = opus if args.profile_target == "opus" else candidate
            for _ in range(args.warmup):
                target()
            torch.cuda.synchronize()
            samples = [_time_once(target) for _ in range(args.iterations)]
            stats = _summary(samples)
            records.append(
                {
                    "backend": args.backend,
                    "profile_target": args.profile_target,
                    "config": config.name,
                    "config_values": kwargs,
                    args.profile_target: stats,
                }
            )
            print(
                f"{config.name:23} {args.profile_target}={stats['median_us']:9.2f} us",
                flush=True,
            )
            continue

        for _ in range(args.warmup):
            opus()
            candidate()
        torch.cuda.synchronize()

        opus()
        candidate()
        torch.cuda.synchronize()
        diff = (opus_out.float() - candidate_out.float()).abs()
        cosine = torch.nn.functional.cosine_similarity(
            opus_out.float().flatten(), candidate_out.float().flatten(), dim=0
        ).item()
        per_token_max = diff.reshape(args.tokens, -1).max(dim=1).values
        worst_count = min(16, args.tokens)
        worst_values, worst_indices = torch.topk(per_token_max, worst_count)

        opus_samples: list[float] = []
        candidate_samples: list[float] = []
        for iteration in range(args.iterations):
            order = ((opus, opus_samples), (candidate, candidate_samples))
            if iteration % 2:
                order = tuple(reversed(order))
            for fn, samples in order:
                samples.append(_time_once(fn))

        opus_stats = _summary(opus_samples)
        candidate_stats = _summary(candidate_samples)
        speedup = opus_stats["median_us"] / candidate_stats["median_us"]
        record = {
            "backend": args.backend,
            "flydsl_pipeline_two": args.flydsl_pipeline_two,
            "flydsl_cluster_two": args.flydsl_cluster_two,
            "flydsl_wave_padded_k": args.flydsl_wave_padded_k,
            "flydsl_alpha_bpermute": args.flydsl_alpha_bpermute,
            "flydsl_lds_padding": args.flydsl_lds_padding,
            "flydsl_rescale_once": args.flydsl_rescale_once,
            "flydsl_fixed_softmax_ref": args.flydsl_fixed_softmax_ref,
            "flydsl_transpose_v": args.flydsl_transpose_v,
            "flydsl_register_p": args.flydsl_register_p,
            "flydsl_cache_q_step": args.flydsl_cache_q_step,
            "flydsl_reuse_q_across_n": args.flydsl_reuse_q_across_n,
            "flydsl_broadcast_indices": args.flydsl_broadcast_indices,
            "flydsl_no_sentinel": args.flydsl_no_sentinel,
            "flydsl_full_tile_fastpath": args.flydsl_full_tile_fastpath,
            "flydsl_p_lane_layout": args.flydsl_p_lane_layout,
            "flydsl_stage_q": args.flydsl_stage_q,
            "flydsl_cache_all_q": args.flydsl_cache_all_q,
            "flydsl_permute_k_scales": args.flydsl_permute_k_scales,
            "flydsl_pairwise_pv": args.flydsl_pairwise_pv,
            "flydsl_post_misched": args.flydsl_post_misched,
            "flydsl_machine_sink": args.flydsl_machine_sink,
            "flydsl_setprio": args.flydsl_setprio,
            "flydsl_dynamic_full_prefix": args.flydsl_dynamic_full_prefix,
            "config": config.name,
            "config_values": kwargs,
            "opus": opus_stats,
            "candidate": candidate_stats,
            "speedup": speedup,
            "delta_pct": (speedup - 1.0) * 100.0,
            "max_abs": diff.max().item(),
            "max_abs_lo256": diff[..., :256].max().item(),
            "max_abs_hi256": diff[..., 256:].max().item(),
            "max_abs_by_32": [
                diff[..., start : start + 32].max().item()
                for start in range(0, 512, 32)
            ],
            "max_abs_by_64": [
                diff[..., start : start + 64].max().item()
                for start in range(0, 512, 64)
            ],
            "worst_tokens": [
                {
                    "token": int(token),
                    "prefix_len": prefix_counts[int(token)],
                    "extend_len": extend_counts[int(token)],
                    "max_abs": float(value),
                }
                for value, token in zip(
                    worst_values.cpu().tolist(), worst_indices.cpu().tolist()
                )
            ],
            "mean_abs": diff.mean().item(),
            "cosine": cosine,
            "opus_nonfinite": int((~torch.isfinite(opus_out)).sum().item()),
            "candidate_nonfinite": int((~torch.isfinite(candidate_out)).sum().item()),
        }
        records.append(record)
        print(
            f"{config.name:23} OPUS={opus_stats['median_us']:9.2f} us "
            f"{args.backend}={candidate_stats['median_us']:9.2f} us "
            f"delta={record['delta_pct']:+7.2f}% cos={cosine:.8f} "
            f"max={record['max_abs']:.6g} "
            f"nonfinite={record['opus_nonfinite']}/{record['candidate_nonfinite']}",
            flush=True,
        )

    payload = {
        "shape": {
            "tokens": args.tokens,
            "heads": args.heads,
            "prefix_len": args.prefix_len,
            "extend_len": args.extend_len,
            "scenario": args.scenario,
            "input_std": args.input_std,
            "prefix_sentinel_period": args.prefix_sentinel_period,
            "assume_full_tiles": args.assume_full_tiles,
            "profile_target": args.profile_target,
            "backend": args.backend,
            "prefix_nnz": int(prefix_indices.numel()),
            "extend_nnz": int(extend_indices.numel()),
        },
        "device": torch.cuda.get_device_name(),
        "records": records,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
