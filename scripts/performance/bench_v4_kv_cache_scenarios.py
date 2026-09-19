#!/usr/bin/env python3
"""Small production-shaped BF16-vs-FP8 V4 decode benchmark.

``prefill length`` is the committed context before decode; prefill itself is
not timed.  Each decode request verifies ``verify_width`` query rows in one
target forward, so ``T = concurrency * verify_width``.  The defaults implement
the requested 8K--32K context, C16--C1024, four-row verification matrix.

For each scenario the benchmark measures the two attention classes in the
61-layer target model:

* 31 HCA layers: 128 SWA rows + all visible ratio-128 compressed rows.
* 30 CSA layers: 128 SWA rows + up to 1024 ratio-4 top-k rows.

The four verification rows of one request share a KV index set, as they do in
production. Different requests use disjoint cache rows. CSA indices are
permuted to represent top-k gather rather than a contiguous scan. A 64 MiB
buffer is touched before each timed attention call to prevent repeated samples
from turning a per-layer cold-cache workload into an artificial L2-hot one.

The indexer remains FP4 in both variants and is not timed.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from atom.model_ops.v4_kernels.paged_decode import sparse_attn_v4_paged_decode
from atom.model_ops.v4_kernels.v4_quant import quantize_bf16_to_v4_2buff_triton
from scripts.performance.bench_v4_kv_cache_dtype import (
    HEAD_DIM,
    LOCAL_HEADS,
    ROPE_HEAD_DIM,
    SOFTMAX_SCALE,
    Comparison,
    _comparison,
    _parse_int_list,
    _time_interleaved,
    benchmark_qk,
)

HCA_LAYERS = 31
CSA_LAYERS = 30
HCA_RATIO = 128
CSA_RATIO = 4
INDEX_TOPK = 1024
WINDOW = 128


@dataclass
class Scenario:
    prefill_len: int
    concurrency: int
    verify_width: int
    tokens: int
    hca_kv_len: int
    csa_kv_len: int


@dataclass
class ScenarioResult:
    scenario: Scenario
    qk: Comparison
    hca: Comparison
    csa: Comparison
    bf16_61_layer_subtotal_ms: float
    fp8_61_layer_subtotal_ms: float
    bf16_saved_per_step_ms: float
    bf16_saved_per_request_verify_position_us: float


def make_scenario(prefill_len: int, concurrency: int, verify_width: int) -> Scenario:
    # The last verification row has absolute position
    # prefill_len + verify_width - 1, and visibility is (pos + 1) // ratio.
    last_pos_plus_one = prefill_len + verify_width
    hca_visible = last_pos_plus_one // HCA_RATIO
    csa_visible = min(last_pos_plus_one // CSA_RATIO, INDEX_TOPK)
    return Scenario(
        prefill_len=prefill_len,
        concurrency=concurrency,
        verify_width=verify_width,
        tokens=concurrency * verify_width,
        hca_kv_len=WINDOW + hca_visible,
        csa_kv_len=WINDOW + csa_visible,
    )


def _make_request_shared_indices(
    concurrency: int,
    verify_width: int,
    kv_len: int,
    *,
    pattern: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    per_request = torch.arange(
        concurrency * kv_len, dtype=torch.int32, device=device
    ).view(concurrency, kv_len)
    if pattern == "csa":
        # One deterministic permutation is sufficient to destroy sequential
        # access while keeping setup cheap at C1024. Requests still address
        # disjoint physical rows because their base offsets differ.
        permutation = torch.randperm(kv_len, dtype=torch.int64, device=device)
        per_request = per_request.index_select(1, permutation)
    elif pattern != "hca":
        raise ValueError(f"unknown pattern {pattern!r}")

    request_ids = torch.arange(concurrency, device=device).repeat_interleave(
        verify_width
    )
    indices = per_request.index_select(0, request_ids).reshape(-1).contiguous()
    tokens = concurrency * verify_width
    kv_indptr = torch.arange(tokens + 1, dtype=torch.int32, device=device) * kv_len
    qo_indptr = torch.arange(tokens + 1, dtype=torch.int32, device=device)
    return indices, kv_indptr, qo_indptr


def benchmark_scenario_attention(
    scenario: Scenario,
    *,
    kv_len: int,
    pattern: str,
    warmup: int,
    iterations: int,
    l2_flush_mib: int,
    device: torch.device,
    seed: int,
) -> Comparison:
    tokens = scenario.tokens
    torch.manual_seed(seed + scenario.prefill_len + scenario.concurrency + kv_len)
    q = torch.randn(
        (tokens, LOCAL_HEADS, HEAD_DIM), dtype=torch.bfloat16, device=device
    )

    # The cache holds one physical set per request. Its speculative verification
    # rows share that set via duplicated CSR indices.
    pages = scenario.concurrency * kv_len
    kv_bf16 = torch.randn((pages, HEAD_DIM), dtype=torch.bfloat16, device=device)
    kv_indices, kv_indptr, qo_indptr = _make_request_shared_indices(
        scenario.concurrency,
        scenario.verify_width,
        kv_len,
        pattern=pattern,
        device=device,
    )
    attn_sink = torch.randn((LOCAL_HEADS,), dtype=torch.float32, device=device)

    kv_fp8, kv_fp8_rope = quantize_bf16_to_v4_2buff_triton(
        kv_bf16.view(pages, 1, HEAD_DIM)
    )
    kv_fp8 = kv_fp8.view(pages, HEAD_DIM)
    kv_fp8_rope = kv_fp8_rope.view(pages, ROPE_HEAD_DIM)
    q_fp8, q_fp8_rope = quantize_bf16_to_v4_2buff_triton(q)

    l2_flush = None
    if l2_flush_mib > 0:
        flush_elements = l2_flush_mib * 1024 * 1024 // 4
        l2_flush = torch.zeros(flush_elements, dtype=torch.float32, device=device)

    def evict_l2() -> None:
        # Enqueued before the start event by _time_interleaved, so this warms
        # clocks and evicts prior KV lines without entering the measured span.
        assert l2_flush is not None
        l2_flush.add_(1.0)

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

    torch.cuda.synchronize()
    bf16, fp8 = _time_interleaved(
        run_bf16,
        run_fp8,
        warmup=warmup,
        iterations=iterations,
        before_each=evict_l2 if l2_flush is not None else None,
    )
    return _comparison(f"{pattern}_paged_decode", tokens, kv_len, bf16, fp8)


def _scenario_result(
    scenario: Scenario,
    qk: Comparison,
    hca: Comparison,
    csa: Comparison,
) -> ScenarioResult:
    bf16_us = (
        61 * qk.bf16.median_us
        + HCA_LAYERS * hca.bf16.median_us
        + CSA_LAYERS * csa.bf16.median_us
    )
    fp8_us = (
        61 * qk.fp8.median_us
        + HCA_LAYERS * hca.fp8.median_us
        + CSA_LAYERS * csa.fp8.median_us
    )
    saved_us = fp8_us - bf16_us
    return ScenarioResult(
        scenario=scenario,
        qk=qk,
        hca=hca,
        csa=csa,
        bf16_61_layer_subtotal_ms=bf16_us / 1000.0,
        fp8_61_layer_subtotal_ms=fp8_us / 1000.0,
        bf16_saved_per_step_ms=saved_us / 1000.0,
        bf16_saved_per_request_verify_position_us=saved_us / scenario.verify_width,
    )


def _print_case_list(scenarios: Sequence[Scenario]) -> None:
    print("case list:", flush=True)
    print(
        f"{'prefill':>8} {'C':>6} {'verify':>7} {'T':>7} {'HCA K':>7} {'CSA K':>7}",
        flush=True,
    )
    for scenario in scenarios:
        print(
            f"{scenario.prefill_len:>8} {scenario.concurrency:>6} "
            f"{scenario.verify_width:>7} {scenario.tokens:>7} "
            f"{scenario.hca_kv_len:>7} {scenario.csa_kv_len:>7}",
            flush=True,
        )


def _print_result(result: ScenarioResult) -> None:
    s = result.scenario
    print(
        f"L={s.prefill_len:>5} C={s.concurrency:>4} T={s.tokens:>4} | "
        f"QK {result.qk.bf16.median_us:7.1f}/{result.qk.fp8.median_us:7.1f} us | "
        f"HCA(K={s.hca_kv_len}) "
        f"{result.hca.bf16.median_us:7.1f}/{result.hca.fp8.median_us:7.1f} us | "
        f"CSA(K={s.csa_kv_len}) "
        f"{result.csa.bf16.median_us:7.1f}/{result.csa.fp8.median_us:7.1f} us | "
        f"61L subtotal {result.bf16_61_layer_subtotal_ms:6.2f}/"
        f"{result.fp8_61_layer_subtotal_ms:6.2f} ms | "
        f"BF16 saves {result.bf16_saved_per_step_ms:+6.2f} ms",
        flush=True,
    )


def _markdown(results: Sequence[ScenarioResult]) -> str:
    lines = [
        "| Prefill | C | Verify | T | HCA K | CSA K | QK BF16/FP8 us | HCA BF16/FP8 us | CSA BF16/FP8 us | 61L BF16/FP8 ms | BF16 saved/step ms |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        s = result.scenario
        lines.append(
            f"| {s.prefill_len} | {s.concurrency} | {s.verify_width} | {s.tokens} | "
            f"{s.hca_kv_len} | {s.csa_kv_len} | "
            f"{result.qk.bf16.median_us:.1f}/{result.qk.fp8.median_us:.1f} | "
            f"{result.hca.bf16.median_us:.1f}/{result.hca.fp8.median_us:.1f} | "
            f"{result.csa.bf16.median_us:.1f}/{result.csa.fp8.median_us:.1f} | "
            f"{result.bf16_61_layer_subtotal_ms:.2f}/"
            f"{result.fp8_61_layer_subtotal_ms:.2f} | "
            f"{result.bf16_saved_per_step_ms:+.2f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prefill-lengths",
        type=_parse_int_list,
        default=[8192, 16384, 32768],
    )
    parser.add_argument(
        "--concurrencies", type=_parse_int_list, default=[16, 64, 256, 1024]
    )
    parser.add_argument("--verify-width", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--l2-flush-mib", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args()

    if args.verify_width <= 0:
        parser.error("--verify-width must be positive")
    if args.warmup < 0 or args.iterations <= 0 or args.l2_flush_mib < 0:
        parser.error("warmup/l2-flush-mib must be >=0; iterations must be >0")
    if not torch.cuda.is_available():
        raise RuntimeError("this benchmark requires a ROCm GPU")

    device = torch.device("cuda", torch.cuda.current_device())
    scenarios = [
        make_scenario(prefill_len, concurrency, args.verify_width)
        for prefill_len in args.prefill_lengths
        for concurrency in args.concurrencies
    ]
    _print_case_list(scenarios)

    # QK depends on T, not context length. Measure it once per concurrency and
    # reuse it in the three context scenarios.
    qk_by_concurrency: dict[int, Comparison] = {}
    for concurrency in args.concurrencies:
        tokens = concurrency * args.verify_width
        print(f"benchmark QK: C={concurrency}, T={tokens}", flush=True)
        qk_by_concurrency[concurrency] = benchmark_qk(
            tokens,
            warmup=args.warmup,
            iterations=args.iterations,
            device=device,
            seed=args.seed,
        )

    results: list[ScenarioResult] = []
    for scenario in scenarios:
        print(
            f"benchmark attention: L={scenario.prefill_len}, "
            f"C={scenario.concurrency}, T={scenario.tokens}",
            flush=True,
        )
        hca = benchmark_scenario_attention(
            scenario,
            kv_len=scenario.hca_kv_len,
            pattern="hca",
            warmup=args.warmup,
            iterations=args.iterations,
            l2_flush_mib=args.l2_flush_mib,
            device=device,
            seed=args.seed,
        )
        csa = benchmark_scenario_attention(
            scenario,
            kv_len=scenario.csa_kv_len,
            pattern="csa",
            warmup=args.warmup,
            iterations=args.iterations,
            l2_flush_mib=args.l2_flush_mib,
            device=device,
            seed=args.seed,
        )
        result = _scenario_result(
            scenario, qk_by_concurrency[scenario.concurrency], hca, csa
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
            "warmup": args.warmup,
            "iterations": args.iterations,
            "l2_flush_mib": args.l2_flush_mib,
            "prefill_is_context_not_timed": True,
            "verify_width_semantics": (
                "total query rows per request; verify_width=4 corresponds "
                "to four-row speculative verification including the current token"
            ),
            "indexer_dtype": "fp4 (not timed)",
            "layer_mix": {"hca": HCA_LAYERS, "csa": CSA_LAYERS},
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
