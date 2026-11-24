#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Generate DSV4 BF16 GEMM untuned shapes using aiter get_padded_m bucketing.

Profiles:
  moe            MoE forward (N,K) with M in [1, 1024]
  flash2node     2-node flash forward gaps (lm_head, wq_b, 7168-K, ...)
  training-bwd  dgrad+wgrad for every (N,K) already in the tuned CSV
"""

import argparse
import csv
from pathlib import Path

LUMEN = Path(__file__).resolve().parents[2]
GEMM_TUNE_DIR = LUMEN / "examples/dsv4/.gemm_tune"
DEFAULT_OUT = GEMM_TUNE_DIR / "dsv4_bf16_moe_bucket_untuned.csv"
DEFAULT_BWD_OUT = GEMM_TUNE_DIR / "dsv4_bf16_bwd_untuned.csv"
DEFAULT_EXISTING = LUMEN / "examples/dsv4/configs/dsv4_bf16_tuned_gemm_mi308x.csv"

MOE_PAIRS = ((4096, 2048), (4096, 4096))
# 2-node flash (TP4/PP4/EP4, seq=4096): lm_head + attention/indexer + 7168-K sparse paths.
FLASH2NODE_NK_PAIRS = (
    (32320, 4096),  # lm_head / output layer (vocab 129280 / TP4)
    (8192, 1024),  # wq_b, indexer linear_wq_b (64*128 / TP4=8192)
    (4096, 256),  # wo_b row-parallel static
    (4096, 1024),  # wo_a column-parallel static
    (384, 7168),
    (512, 7168),
    (1024, 7168),
    (2048, 7168),
)
STATIC_SHAPES = (
    (4096, 1024, 4096),
    (4096, 4096, 1024),
    (4096, 512, 4096),
    (4096, 4096, 256),
    (4096, 8192, 1024),
    (4096, 64, 4096),
)
HEADER = ["M", "N", "K", "bias", "dtype", "outdtype", "scaleAB", "bpreshuffle"]
ROW_DEFAULTS = {
    "bias": "False",
    "dtype": "torch.bfloat16",
    "outdtype": "torch.bfloat16",
    "scaleAB": "False",
    "bpreshuffle": "False",
}

# get_padded_m representatives for M in [1, 1024] on gfx942 (N=4096,K=4096).
REPRESENTATIVE_MS_SMALL = (
    1,
    2,
    4,
    8,
    16,
    32,
    48,
    64,
    80,
    96,
    112,
    128,
    144,
    160,
    176,
    192,
    208,
    224,
    240,
    256,
    288,
    320,
    352,
    384,
    416,
    448,
    480,
    512,
    544,
    576,
    608,
    640,
    672,
    704,
    736,
    768,
    800,
    832,
    864,
    896,
    928,
    960,
    992,
    1024,
)
# get_padded_m representatives for M in [1025, 4096] on gfx942.
REPRESENTATIVE_MS_LARGE = (
    1088,
    1152,
    1216,
    1280,
    1344,
    1408,
    1472,
    1536,
    1600,
    1664,
    1728,
    1792,
    1856,
    1920,
    1984,
    2048,
    2112,
    2176,
    2240,
    2304,
    2368,
    2432,
    2496,
    2560,
    2624,
    2688,
    2752,
    2816,
    2880,
    2944,
    3008,
    3072,
    3136,
    3200,
    3264,
    3328,
    3392,
    3456,
    3520,
    3584,
    3648,
    3712,
    3776,
    3840,
    3904,
    3968,
    4032,
    4096,
)


def _parse_pairs(text):
    if not text:
        return MOE_PAIRS
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if len(parts) % 2:
        raise ValueError("--pairs requires even count: N1,K1,N2,K2,...")
    out = []
    for i in range(0, len(parts), 2):
        out.append((int(parts[i]), int(parts[i + 1])))
    return tuple(out)


def representative_ms(min_m=1, max_m=1024):
    if min_m <= 1024:
        ms_small = [
            m for m in REPRESENTATIVE_MS_SMALL if min_m <= m <= min(max_m, 1024)
        ]
    else:
        ms_small = []
    if max_m > 1024:
        lo = max(min_m, 1025)
        ms_large = [m for m in REPRESENTATIVE_MS_LARGE if lo <= m <= max_m]
    else:
        ms_large = []
    return sorted(set(ms_small + ms_large))


def build_shapes(pairs, min_m=1, max_m=1024, include_static=False):
    reps = representative_ms(min_m, max_m)
    shapes = [(m, n, k) for n, k in pairs for m in reps]
    if include_static:
        shapes.extend(STATIC_SHAPES)
    return sorted(set(shapes))


def expand_backward(forward_pairs, reps):
    """Shapes used by quantized_linear BF16 backward (same TN gemm_a16w16).

    Forward is Y = X @ W.T with W ``(N, K)`` and tokens ``M``.
      dgrad: gemm(dY, W.T)  -> (M, N, K) = (tokens, K_fwd, N_fwd)
      wgrad: gemm(dY.T, X.T) -> (M, N, K) = (N_fwd, K_fwd, tokens)

    Lookup only pads M, not K, so wgrad still sweeps token counts on K.
    """
    shapes = []
    for n_fwd, k_fwd in forward_pairs:
        for tokens in reps:
            shapes.append((tokens, k_fwd, n_fwd))
            shapes.append((n_fwd, k_fwd, tokens))
    return shapes


def _load_existing_nk(path, target_cu=None):
    if not path.is_file():
        return []
    seen = set()
    with path.open() as f:
        for r in csv.DictReader(f):
            if r.get("gfx", "gfx942") != "gfx942":
                continue
            if target_cu is not None and str(r.get("cu_num")) != str(target_cu):
                continue
            seen.add((int(r["N"]), int(r["K"])))
    return sorted(seen)


def _load_existing(path, target_cu=None):
    if not path.is_file():
        return set()
    with path.open() as f:
        rows = csv.DictReader(f)
        out = set()
        for r in rows:
            if r.get("gfx", "gfx942") != "gfx942":
                continue
            if target_cu is not None and str(r.get("cu_num")) != str(target_cu):
                continue
            out.add((int(r["M"]), int(r["N"]), int(r["K"])))
        return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--existing", type=Path, default=DEFAULT_EXISTING)
    parser.add_argument(
        "--profile",
        choices=("moe", "flash2node", "training-bwd"),
        default="moe",
        help="moe / flash2node: forward (N,K); training-bwd: dgrad+wgrad for NK already in --existing",
    )
    parser.add_argument(
        "--pairs",
        default="",
        help="comma-separated N,K pairs only (e.g. 4096,2048). Overrides --profile pairs",
    )
    parser.add_argument(
        "--from-existing-nk",
        action="store_true",
        help="use unique (N,K) from --existing as forward pairs (unioned with --profile/--pairs)",
    )
    parser.add_argument(
        "--include-backward",
        action="store_true",
        help="also emit dgrad (M,K,N) and wgrad (N,K,M) for each forward pair",
    )
    parser.add_argument(
        "--backward-only",
        action="store_true",
        help="only emit dgrad/wgrad (implies --include-backward; skip forward buckets)",
    )
    parser.add_argument(
        "--include-existing",
        action="store_true",
        help="skip shapes already present in --existing CSV (optionally scoped by --target-cu)",
    )
    parser.add_argument(
        "--target-cu",
        type=int,
        default=None,
        help="with --include-existing, only skip shapes tuned for this cu_num (80 or 304)",
    )
    parser.add_argument("--min-m", type=int, default=None)
    parser.add_argument("--max-m", type=int, default=None)
    parser.add_argument(
        "--include-static",
        action="store_true",
        help="append finetune static miss shapes (4096x1024x4096, etc.)",
    )
    args = parser.parse_args()

    if args.backward_only:
        args.include_backward = True
    if args.profile == "training-bwd":
        args.backward_only = True
        args.include_backward = True
        args.from_existing_nk = True
        if args.output == DEFAULT_OUT:
            args.output = DEFAULT_BWD_OUT

    if args.pairs:
        pairs = list(_parse_pairs(args.pairs))
    elif args.profile == "flash2node":
        pairs = list(FLASH2NODE_NK_PAIRS)
    elif args.profile == "training-bwd":
        pairs = []
    else:
        pairs = list(MOE_PAIRS)

    if args.from_existing_nk:
        # NK inventory is global; --target-cu only affects --include-existing skips.
        pairs = sorted(
            set(pairs) | set(_load_existing_nk(args.existing, target_cu=None))
        )

    min_m = 1 if args.min_m is None else args.min_m
    max_m = 4096 if args.max_m is None else args.max_m
    if (
        args.profile == "moe"
        and args.min_m is None
        and args.max_m is None
        and not args.backward_only
    ):
        min_m, max_m = 1, 1024

    reps = representative_ms(min_m, max_m)
    shapes = []
    if not args.backward_only:
        shapes.extend(
            build_shapes(
                pairs,
                min_m=min_m,
                max_m=max_m,
                include_static=args.include_static,
            )
        )
    elif args.include_static:
        shapes.extend(STATIC_SHAPES)
    if args.include_backward:
        if not pairs:
            raise SystemExit(
                "no forward (N,K) pairs — pass --pairs, --profile, or --from-existing-nk"
            )
        shapes.extend(expand_backward(pairs, reps))
    shapes = sorted(set(shapes))

    if args.include_existing:
        have = _load_existing(args.existing, target_cu=args.target_cu)
        shapes = [s for s in shapes if s not in have]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HEADER)
        w.writeheader()
        for m, n, k in shapes:
            row = {"M": m, "N": n, "K": k, **ROW_DEFAULTS}
            w.writerow(row)

    print(f"wrote {args.output} ({len(shapes)} shapes)")
    print(f"  forward (N,K) sources: {len(pairs)}  token buckets: {len(reps)}")
    if args.include_backward:
        remain = set(shapes)
        print("  missing dgrad/wgrad vs tuned CSV:")
        for n_fwd, k_fwd in pairs:
            d = sum(1 for t in reps if (t, k_fwd, n_fwd) in remain)
            w = sum(1 for t in reps if (n_fwd, k_fwd, t) in remain)
            if d or w:
                print(
                    f"    fwd N={n_fwd:6d} K={k_fwd:6d}: dgrad {d:3d}/{len(reps)}  wgrad {w:3d}/{len(reps)}"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
