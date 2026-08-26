#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Is a strided view of a full weight GEMM-equivalent to a contiguous shard?

Context — asymmetric rapidserve (prefill TP=N, decode TP=1). Decode needs FULL
attention matrices; prefill holds Column/RowParallel shards. Rather than keeping
both (shard for prefill + a gathered full copy for decode), each prefill rank
could allocate the FULL weight once, use a VIEW of its own slice for compute,
and let decode alias the whole tensor. One allocation serves both processes and
no all-gather is needed at bootstrap.

That works trivially for ColumnParallelLinear: weights are stored
[output_size, input_size] row-major (linear.py:438-448) and column-parallel
divides `output_size`, so rank r's slice `full.narrow(0, ...)` is CONTIGUOUS.

RowParallelLinear divides `input_size` (linear.py:429-430), so rank r's slice
`full.narrow(1, ...)` is STRIDED — row stride stays `input_size`, not
`input_size/tp`. Today the loader materializes that slice as its own contiguous
tensor (linear.py:1921). The question this script answers:

    does the GEMM produce a bit-identical result from the strided view?

If yes, the full-allocation trick applies uniformly and no gather is ever
needed. If no (or the kernel rejects strided input outright), row-parallel
layers need the fallback and column-parallel ones can still use the trick.

Shapes default to DeepSeek-V4-Pro's `wo_b`
(RowParallelLinear(n_groups*o_lora_rank=16384, dim=7168), deepseek_v4.py:2348).

Usage:
    python tools/check_row_parallel_strided_shard.py
    python tools/check_row_parallel_strided_shard.py --tp 8 --tokens 4096
    python tools/check_row_parallel_strided_shard.py --paths bf16 blockscale
"""

import argparse
import sys
import traceback

import torch


def _banner(msg):
    print(f"\n{'=' * 78}\n{msg}\n{'=' * 78}")


def build_layouts(out_features, in_features, tp, rank, dtype):
    """Return (full, contiguous_shard, strided_view) for one rank's slice.

    contiguous_shard is what the loader produces today; strided_view is what the
    proposed full-allocation scheme would hand the kernel. They hold identical
    VALUES and differ only in memory layout, so any output difference is a
    layout-handling bug in the kernel, not arithmetic.
    """
    assert in_features % tp == 0, f"in_features {in_features} % tp {tp} != 0"
    shard = in_features // tp
    start = rank * shard

    full = torch.randn(out_features, in_features, device="cuda", dtype=torch.float32)
    full = full.to(dtype)

    strided_view = full.narrow(1, start, shard)
    contiguous_shard = strided_view.contiguous()

    assert not strided_view.is_contiguous(), (
        "strided_view came out contiguous — the premise of this test is gone "
        f"(tp={tp}, in_features={in_features}); check the shape arguments."
    )
    assert torch.equal(strided_view, contiguous_shard), "layouts differ in VALUE"
    return full, contiguous_shard, strided_view


def compare(name, run, w_contig, w_strided):
    """Run `run(weight)` on both layouts and report bitwise agreement."""
    try:
        y_ref = run(w_contig)
    except Exception:
        print(f"  {name:<34} SKIP  (contiguous path unavailable)")
        traceback.print_exc(limit=1)
        return None
    try:
        y_view = run(w_strided)
    except Exception as exc:
        print(f"  {name:<34} REJECTED  kernel refused strided input")
        print(f"      {type(exc).__name__}: {exc}")
        return False

    if y_ref.shape != y_view.shape:
        print(f"  {name:<34} SHAPE MISMATCH {y_ref.shape} vs {y_view.shape}")
        return False
    if torch.equal(y_ref, y_view):
        print(f"  {name:<34} BITWISE IDENTICAL")
        return True

    d = (y_ref.float() - y_view.float()).abs()
    denom = y_ref.float().abs().clamp_min(1e-6)
    print(
        f"  {name:<34} DIFFERS  max_abs={d.max().item():.3e} "
        f"max_rel={(d / denom).max().item():.3e} "
        f"mismatched={int((y_ref != y_view).sum())}/{y_ref.numel()}"
    )
    return False


def main():
    ap = argparse.ArgumentParser()
    # V4-Pro wo_b: RowParallelLinear(n_groups * o_lora_rank, dim) = (16384, 7168)
    ap.add_argument("--in-features", type=int, default=16384)
    ap.add_argument("--out-features", type=int, default=7168)
    ap.add_argument("--tp", type=int, default=8)
    ap.add_argument("--rank", type=int, default=3, help="which shard to slice")
    ap.add_argument("--tokens", type=int, default=512)
    ap.add_argument(
        "--paths",
        nargs="*",
        default=None,
        help="subset of: bf16 per_token blockscale blockscale_preshuffle",
    )
    args = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("no GPU visible — run this inside the ATOM container")

    shard_in = args.in_features // args.tp
    _banner(
        f"RowParallelLinear strided-view equivalence\n"
        f"full weight [{args.out_features}, {args.in_features}] "
        f"tp={args.tp} rank={args.rank}\n"
        f"shard       [{args.out_features}, {shard_in}]  "
        f"(contiguous vs stride=({args.in_features}, 1))\n"
        f"activations [{args.tokens}, {shard_in}]"
    )

    want = set(args.paths) if args.paths else None

    def selected(name):
        return want is None or name in want

    results = {}

    # ---- bf16, no quantization: tgemm.mm (linear.py:867) --------------------
    if selected("bf16"):
        print("\n[bf16] tgemm.mm — LinearBase.forward QuantType.No")
        from aiter.tuned_gemm import tgemm

        _, w_c, w_s = build_layouts(
            args.out_features, args.in_features, args.tp, args.rank, torch.bfloat16
        )
        x = torch.randn(args.tokens, shard_in, device="cuda", dtype=torch.bfloat16)
        results["bf16"] = compare(
            "tgemm.mm", lambda w: tgemm.mm(x, w, None, otype=torch.bfloat16), w_c, w_s
        )

    # ---- FP8 per-token: gemm_a8w8_bpreshuffle (linear.py:934) ---------------
    # NOTE: the *_bpreshuffle kernels expect a physically reordered weight. A
    # strided view into a preshuffled full tensor is NOT the preshuffled shard,
    # so this one is expected to fail; it is included to confirm that.
    if selected("per_token"):
        print("\n[per_token] gemm_a8w8_bpreshuffle — linear.py:934")
        from aiter import dtypes, gemm_a8w8_bpreshuffle

        _, w_c, w_s = build_layouts(
            args.out_features, args.in_features, args.tp, args.rank, dtypes.fp8
        )
        x = torch.randn(args.tokens, shard_in, device="cuda").to(dtypes.fp8)
        x_scale = torch.rand(args.tokens, 1, device="cuda", dtype=torch.float32)
        w_scale = torch.rand(args.out_features, 1, device="cuda", dtype=torch.float32)
        results["per_token"] = compare(
            "gemm_a8w8_bpreshuffle",
            lambda w: gemm_a8w8_bpreshuffle(
                x, w, x_scale, w_scale, dtype=torch.bfloat16
            ),
            w_c,
            w_s,
        )

    # ---- FP8 block scale: gemm_a8w8_blockscale (linear.py:963) --------------
    # The scale tiles along the sharded dim too, so it is sliced in lockstep.
    if selected("blockscale"):
        print("\n[blockscale] gemm_a8w8_blockscale — linear.py:963")
        from aiter import dtypes, gemm_a8w8_blockscale

        _, w_c, w_s = build_layouts(
            args.out_features, args.in_features, args.tp, args.rank, dtypes.fp8
        )
        blk = 128
        if shard_in % blk or args.out_features % blk:
            print(
                f"  SKIP: shard_in={shard_in} / out={args.out_features} "
                f"not {blk}-aligned"
            )
            results["blockscale"] = None
        else:
            x = torch.randn(args.tokens, shard_in, device="cuda").to(dtypes.fp8)
            x_scale = torch.rand(
                args.tokens, shard_in // blk, device="cuda", dtype=torch.float32
            )
            w_scale = torch.rand(
                args.out_features // blk, shard_in // blk, device="cuda",
                dtype=torch.float32,
            )
            results["blockscale"] = compare(
                "gemm_a8w8_blockscale",
                lambda w: gemm_a8w8_blockscale(
                    x, w, x_scale, w_scale, dtype=torch.bfloat16
                ),
                w_c,
                w_s,
            )

    # ---- FP8 block scale, preshuffled weight (linear.py:945) ---------------
    if selected("blockscale_preshuffle"):
        print(
            "\n[blockscale_preshuffle] gemm_a8w8_blockscale_bpreshuffle "
            "— linear.py:945"
        )
        from aiter import dtypes, gemm_a8w8_blockscale_bpreshuffle

        _, w_c, w_s = build_layouts(
            args.out_features, args.in_features, args.tp, args.rank, dtypes.fp8
        )
        blk = 128
        if shard_in % blk or args.out_features % blk:
            print("  SKIP: not 128-aligned")
            results["blockscale_preshuffle"] = None
        else:
            x = torch.randn(args.tokens, shard_in, device="cuda").to(dtypes.fp8)
            x_scale = torch.rand(
                args.tokens, shard_in // blk, device="cuda", dtype=torch.float32
            )
            w_scale = torch.rand(
                args.out_features // blk, shard_in // blk, device="cuda",
                dtype=torch.float32,
            )
            results["blockscale_preshuffle"] = compare(
                "gemm_a8w8_blockscale_bpreshuffle",
                lambda w: gemm_a8w8_blockscale_bpreshuffle(
                    x, w, x_scale, w_scale, dtype=torch.bfloat16
                ),
                w_c,
                w_s,
            )

    # ---- verdict -----------------------------------------------------------
    _banner("VERDICT")
    ran = {k: v for k, v in results.items() if v is not None}
    if not ran:
        print("nothing ran — check --paths and that aiter is importable")
        return 2
    for name, ok in ran.items():
        print(f"  {name:<24} {'safe' if ok else 'NOT safe'}")
    if all(ran.values()):
        print(
            "\nEvery tested path is layout-agnostic: RowParallelLinear can use a\n"
            "strided view into a full weight, so the full-allocation scheme\n"
            "applies uniformly and no gather-and-export fallback is needed."
        )
        return 0
    bad = [k for k, v in ran.items() if not v]
    print(
        f"\nStrided views are NOT safe for: {', '.join(bad)}.\n"
        "RowParallelLinear needs its shard kept as a standalone contiguous\n"
        "tensor on those paths, with a gathered full copy exported separately\n"
        "for decode. Column-parallel layers (wq_b, wo_a) are unaffected — their\n"
        "slice is contiguous by construction and can still use the trick."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
