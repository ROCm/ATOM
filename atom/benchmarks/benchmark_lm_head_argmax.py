# SPDX-License-Identifier: MIT
"""Microbenchmark fused and eager LM-head argmax packing.

Run from the repository root in an ATOM GPU environment:

    python -m atom.benchmarks.benchmark_lm_head_argmax

The eager baseline is the exact operation sequence replaced by
``lm_head_argmax_pack``: max, global-index addition, casts, and stack.
Input creation and correctness checks are outside the timed regions.
"""

import argparse
import csv
from collections.abc import Callable
from pathlib import Path

import torch
import triton
from triton.testing import do_bench

from atom.model_ops.lm_head_argmax import lm_head_argmax_pack

_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def _positive_int_list(value: str) -> list[int]:
    try:
        values = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("all values must be positive integers")
    return values


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _unfused_lm_head_argmax_pack(
    logits: torch.Tensor, vocab_start_idx: int
) -> torch.Tensor:
    local_max_val, local_idx = logits.max(dim=-1)
    global_idx = local_idx + vocab_start_idx
    return torch.stack([local_max_val.float(), global_idx.float()], dim=-1)


def _median_ms(
    operation: Callable[[], torch.Tensor], warmup_ms: float, benchmark_ms: float
) -> float:
    result = do_bench(
        operation,
        warmup=warmup_ms,
        rep=benchmark_ms,
        quantiles=[0.5],
    )
    if isinstance(result, (list, tuple)):
        result = result[0]
    return float(result)


def _print_header() -> None:
    print(
        f"{'rows':>6} {'local_vocab':>12} {'fused_us':>12} "
        f"{'unfused_us':>12} {'speedup':>10} {'saved':>9}"
    )
    print("-" * 67)


def _print_result(result: dict[str, int | float | str]) -> None:
    print(
        f"{result['rows']:>6} {result['local_vocab']:>12} "
        f"{result['fused_us']:>12.2f} {result['unfused_us']:>12.2f} "
        f"{result['speedup']:>9.2f}x {result['saved_percent']:>8.1f}%"
    )


@torch.inference_mode()
def run_benchmark(args: argparse.Namespace) -> list[dict[str, int | float | str]]:
    dtype = _DTYPES[args.dtype]
    results: list[dict[str, int | float | str]] = []
    case_index = 0

    _print_header()
    for local_vocab in args.vocab_sizes:
        for rows in args.rows:
            generator = torch.Generator(device="cuda")
            generator.manual_seed(args.seed + rows + local_vocab)
            logits = torch.randn(
                (rows, local_vocab),
                dtype=dtype,
                device="cuda",
                generator=generator,
            )

            expected = _unfused_lm_head_argmax_pack(logits, args.vocab_start_idx)
            actual = lm_head_argmax_pack(logits, args.vocab_start_idx)
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            torch.cuda.synchronize()
            del actual, expected

            fused = lambda: lm_head_argmax_pack(logits, args.vocab_start_idx)
            unfused = lambda: _unfused_lm_head_argmax_pack(
                logits, args.vocab_start_idx
            )

            # Alternate measurement order to reduce systematic thermal bias.
            if case_index % 2 == 0:
                fused_ms = _median_ms(fused, args.warmup_ms, args.benchmark_ms)
                unfused_ms = _median_ms(unfused, args.warmup_ms, args.benchmark_ms)
            else:
                unfused_ms = _median_ms(unfused, args.warmup_ms, args.benchmark_ms)
                fused_ms = _median_ms(fused, args.warmup_ms, args.benchmark_ms)
            case_index += 1

            speedup = unfused_ms / fused_ms
            result: dict[str, int | float | str] = {
                "dtype": args.dtype,
                "rows": rows,
                "local_vocab": local_vocab,
                "fused_us": fused_ms * 1000,
                "unfused_us": unfused_ms * 1000,
                "speedup": speedup,
                "saved_percent": (1 - fused_ms / unfused_ms) * 100,
            }
            results.append(result)
            _print_result(result)

    return results


def _write_csv(path: Path, results: list[dict[str, int | float | str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved results to {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare fused Triton and eager PyTorch LM-head argmax packing."
    )
    parser.add_argument(
        "--rows",
        type=_positive_int_list,
        default=[1, 4, 8, 15, 16, 64],
        help="comma-separated row counts (default: 1,4,8,15,16,64)",
    )
    parser.add_argument(
        "--vocab-sizes",
        type=_positive_int_list,
        default=[19360, 38720, 77440, 154880],
        help=(
            "comma-separated local vocab sizes; defaults cover GLM TP8/4/2/1 "
            "(default: 19360,38720,77440,154880)"
        ),
    )
    parser.add_argument(
        "--dtype",
        choices=sorted(_DTYPES),
        default="bfloat16",
    )
    parser.add_argument("--vocab-start-idx", type=int, default=32000)
    parser.add_argument(
        "--warmup-ms",
        type=_positive_float,
        default=50.0,
        help="warmup duration per implementation (default: 50)",
    )
    parser.add_argument(
        "--benchmark-ms",
        type=_positive_float,
        default=200.0,
        help="measurement duration per implementation (default: 200)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--csv",
        type=Path,
        help="optional output CSV path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("This benchmark requires a CUDA/ROCm GPU.")

    backend = f"ROCm {torch.version.hip}" if torch.version.hip else "CUDA"
    print(
        f"device={torch.cuda.get_device_name()} backend={backend} "
        f"torch={torch.__version__} triton={triton.__version__} dtype={args.dtype}\n"
    )
    results = run_benchmark(args)
    if args.csv:
        _write_csv(args.csv, results)


if __name__ == "__main__":
    main()
