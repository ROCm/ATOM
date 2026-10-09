# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Speedup benchmark for BLASST block skipping in Triton `unified_attention`.

Times the same kernel at `block_skip_threshold=0` (dense) against threshold > 0
(BLASST) on the paged prefill layout, and reports speedup, output drift, and the
achieved sparsity.

Two input modes:

  --mode random  Random Q/K/V. No model required.
  --mode ruler   Real per-layer activations captured from a long-context prompt.
                 Requires `transformers` and a model on disk.

Random scores rarely produce whole skippable tiles, so the two modes are not
interchangeable: compare results at matched `tile_elide`, not at matched
threshold.

Achieved sparsity comes from the KERNEL'S OWN counter (`skip_counter`).

Scope is the 2D kernel. Block skipping only exists there: the wrapper
force-disables it on the 3D, decode and sliding-window paths. Cases that do not
reach the 2D kernel are reported as skipped rather than silently timed at 1.00x.

Run:
    python -m atom.benchmarks.blasst.bench_unified_attention_blasst
    python -m atom.benchmarks.blasst.bench_unified_attention_blasst \\
        --mode ruler --input-file <prompts.jsonl> --model <model> --ruler-mode full
"""

import argparse
import csv
import json
import logging
import math
import os
import sys
import time

import torch

logger = logging.getLogger("atom")


def _make_parser(**kw):
    """ATOM's FlexibleArgumentParser if importable, else plain argparse.

    Imported lazily: `atom/__init__.py` loads the inference engine, which this
    benchmark does not need and which is not always available when the file is
    run directly.
    """
    try:
        from atom.utils.arg_parser import FlexibleArgumentParser

        return FlexibleArgumentParser(**kw)
    except Exception:  # noqa: BLE001
        return argparse.ArgumentParser(**kw)


LOG2E = 1.4426950408889634

# 1e-9 is not a useful sparsity setting; it isolates the fixed cost of the skip
# check itself (a max, a compare, a where per tile) from any benefit.
DEFAULT_THRESHOLDS = [1e-9, 0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0]

# A representative calibrated operating point, included so the sweep brackets a
# realistic value rather than only the extremes.
CALIBRATED_QWEN3_8B = 0.02878

# Overridable so a long sweep can trade timing precision for wall clock. The
# 13x50x36 headline sweep uses 3/10, matching the AITER-side protocol; at 5/20
# the same sweep is ~10 hours instead of ~2.5.
WARMUP = int(os.environ.get("BLASST_BENCH_WARMUP", 5))
REPEAT = int(os.environ.get("BLASST_BENCH_REPEAT", 20))


# --------------------------------------------------------------------------
# timing
# --------------------------------------------------------------------------
def benchmark_fn(fn, warmup=WARMUP, repeat=REPEAT):
    """Median-of-repeats wall time in ms. Median, not mean: an occasional
    scheduler hiccup should not decide a speedup claim."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)
    times.sort()
    return times[len(times) // 2]


# --------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------
def make_paged_inputs(seqlen, num_q_heads, num_kv_heads, head_dim, block_size, seed=0):
    """Random full-prefill inputs in ATOM's paged layout (query_len == kv_len)."""
    torch.manual_seed(seed)
    num_blocks = (seqlen + block_size - 1) // block_size

    query = torch.randn(
        seqlen, num_q_heads, head_dim, dtype=torch.bfloat16, device="cuda"
    )
    key_cache = torch.randn(
        num_blocks, block_size, num_kv_heads, head_dim,
        dtype=torch.bfloat16, device="cuda",
    )
    value_cache = torch.randn_like(key_cache)
    # Identity block table: block i holds positions [i*block_size, ...). Keeps
    # the sparsity replay and the kernel reading the same K/V order.
    block_tables = torch.arange(
        num_blocks, dtype=torch.int32, device="cuda"
    ).unsqueeze(0)

    return _pack(query, key_cache, value_cache, block_tables, seqlen, head_dim)


def paged_inputs_from_capture(cap, block_size):
    """Reshape captured (B, H, S, D) activations into the paged layout.

    K/V are zero-padded up to a whole number of blocks. The padding is never
    read: `seqused_k` stops the kernel at the true length.
    """
    q = cap["Q"][0].transpose(0, 1).contiguous()  # (S, Hq, D)
    k = cap["K"][0].transpose(0, 1).contiguous()  # (S, Hkv, D)
    v = cap["V"][0].transpose(0, 1).contiguous()
    seqlen, _, head_dim = q.shape
    num_kv_heads = k.shape[1]

    num_blocks = (seqlen + block_size - 1) // block_size
    pad = num_blocks * block_size - seqlen
    if pad:
        zeros = torch.zeros(pad, num_kv_heads, head_dim, dtype=k.dtype, device=k.device)
        k = torch.cat([k, zeros])
        v = torch.cat([v, zeros])
    key_cache = k.view(num_blocks, block_size, num_kv_heads, head_dim).contiguous()
    value_cache = v.view(num_blocks, block_size, num_kv_heads, head_dim).contiguous()
    block_tables = torch.arange(
        num_blocks, dtype=torch.int32, device="cuda"
    ).unsqueeze(0)

    return _pack(q, key_cache, value_cache, block_tables, seqlen, head_dim)


def _pack(query, key_cache, value_cache, block_tables, seqlen, head_dim):
    return dict(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        block_tables=block_tables,
        output=torch.empty_like(query),
        cu_seqlens_q=torch.tensor([0, seqlen], dtype=torch.int32, device="cuda"),
        seqused_k=torch.tensor([seqlen], dtype=torch.int32, device="cuda"),
        max_seqlen_q=seqlen,
        max_seqlen_k=seqlen,
        scale=head_dim**-0.5,
        seqlen=seqlen,
    )


def run_attention(inp, threshold, skip_counter=None):
    """Call the kernel directly rather than through ATOM's op wrapper.

    ATOM's `prefill_attention_triton` reaches this same kernel, but going
    through it would fold ATOM's dispatch and metadata setup into the timing.
    Calling the kernel directly keeps the measurement on the kernel itself.
    """
    from aiter.ops.triton.attention.unified_attention import unified_attention

    unified_attention(
        q=inp["query"],
        k=inp["key_cache"],
        v=inp["value_cache"],
        out=inp["output"],
        cu_seqlens_q=inp["cu_seqlens_q"],
        seqused_k=inp["seqused_k"],
        max_seqlen_q=inp["max_seqlen_q"],
        max_seqlen_k=inp["max_seqlen_k"],
        softmax_scale=inp["scale"],
        causal=True,
        window_size=(-1, -1),
        block_table=inp["block_tables"],
        softcap=0,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        block_skip_threshold=float(threshold),
        skip_counter=skip_counter,
    )
    return inp["output"]


def measure_elision(inp, threshold):
    """Achieved tile elision, from the KERNEL'S OWN counter.

    Returns (visited, elided). An UNTIMED extra launch, deliberately: the two
    atomics must not land in the measurement they explain.

    `visited == 0` also proves the 2D kernel did not run -- skipping lives only
    there, and the wrapper force-disables it on the 3D, decode and
    sliding-window paths, which never touch the counter.
    """
    buf = torch.zeros(2, dtype=torch.int64, device=inp["query"].device)
    run_attention(inp, threshold, skip_counter=buf)
    torch.cuda.synchronize()
    visited, elided = (int(x) for x in buf.cpu())
    return visited, elided


# --------------------------------------------------------------------------
# sparsity
# --------------------------------------------------------------------------
def rel_diff(a, b):
    """Mean relative difference between two outputs.

    `b` is the same kernel at threshold=0, so this is the size of the
    approximation, not an error against ground truth. Correctness is covered by
    the kernel's tests in AITER.
    """
    return (
        (a.float() - b.float()).abs().mean() / b.float().abs().mean().clamp_min(1e-6)
    ).item()


# Shape -> tiles visited at a probe threshold. Keyed on everything use_2d_kernel()
# looks at, so a hit means the routing decision is provably identical.
_PROBE_CACHE: dict = {}


def sweep(label, inp, nq, nkv, head_dim, block_size, thresholds, skip_sparsity, rows):
    print(f"\n=== {label} ===")
    # Confirm the 2D kernel actually runs before measuring anything. Block
    # skipping exists only there; on the 3D, decode and sliding-window paths the
    # wrapper force-disables it and the case would report a flat 1.00x while
    # looking perfectly healthy. The counter is the probe: a positive threshold
    # plus a buffer makes the 2D kernel count EVERY tile it inspects, including
    # ones it does not elide, so a zero means it never ran.
    #
    # CACHED BY SHAPE. Routing depends on the shape, not the data -- every layer
    # of every prompt at one sequence length routes identically.
    probe_key = (inp["max_seqlen_q"], inp["max_seqlen_k"], nq, nkv,
                 head_dim, block_size)
    if probe_key not in _PROBE_CACHE:
        _PROBE_CACHE[probe_key] = measure_elision(inp, 1.0)[0]
    probe_visited = _PROBE_CACHE[probe_key]
    if probe_visited == 0:
        print("    SKIPPED: this shape does not reach the 2D kernel, so block "
              "skipping is disabled and the numbers would be meaningless.")
        return

    dense_ms = benchmark_fn(lambda: run_attention(inp, 0.0))
    dense_out = run_attention(inp, 0.0).clone()
    print(f"    dense: {dense_ms:.3f} ms   <- speedup and rel_diff are both against this")
    print(
        f"    {'threshold':>11} {'ms':>9} {'speedup':>8} {'rel_diff':>9} "
        f"{'tile_elide':>11}"
    )

    for thr in thresholds:
        try:
            ms = benchmark_fn(lambda t=thr: run_attention(inp, t))
            out = run_attention(inp, thr).clone()
        except Exception as exc:  # noqa: BLE001
            print(f"    {thr:>11g} FAILED {type(exc).__name__}: {exc}")
            continue
        err = rel_diff(out, dense_out)
        finite = bool(torch.isfinite(out).all())
        if skip_sparsity:
            # Not measured, not zero. Printing nan% here reads like a numerical
            # failure rather than "measurement was switched off".
            tile_s = float("nan")
            tile_str = f"{'-':>10}"
        else:
            visited, elided = measure_elision(inp, thr)
            tile_s = (elided / visited) if visited else float("nan")
            tile_str = f"{tile_s:>10.1%}" if visited else f"{'n/a':>10}"
        note = "" if finite else "  <-- NON-FINITE"
        print(
            f"    {thr:>11g} {ms:>9.3f} {dense_ms / ms:>7.3f}x {err:>9.4f} "
            f"{tile_str}{note}"
        )
        rows.append(
            dict(
                case=label, threshold=thr, dense_ms=round(dense_ms, 4),
                ms=round(ms, 4), speedup=round(dense_ms / ms, 4),
                rel_diff=round(err, 6),
                tile_elide=round(tile_s, 4), finite=finite,
                kernel="2D",
            )
        )


def banner(args):
    import triton

    # Non-default Triton codegen knobs change this kernel's speedup, so echo any
    # that are set. Printed only when present.
    codegen = {
        k: v for k, v in os.environ.items()
        if k.startswith("TRITON_") and k.endswith(("_ACROSS_IF", "_SWIZZLE"))
    }
    # The chain-dot toggle changes codegen (warpsPerCTA [2,2] vs [4,1]), so a run
    # is not interpretable without it. Same for the sparsity columns, which are
    # only populated when the replay runs.
    print("=" * 78)
    print("BLASST block skipping in Triton unified_attention")
    print("=" * 78)
    print(f"  device            {torch.cuda.get_device_name(0)}")
    print(f"  torch / triton    {torch.__version__} / {triton.__version__}")
    print(f"  baseline          same kernel at threshold=0 (Triton dense)")
    print(f"  mode              {args.mode}"
          f"{'  (random scores; understates BLASST)' if args.mode == 'random' else ''}")
    print(f"  causal            True (unified_attention supports causal only)")
    print(f"  dtype             bf16      block_size {args.block_size}")
    for k, v in sorted(codegen.items()):
        print(f"  codegen           {k}={v}")
    print(f"  elision counter   {'off (--skip-sparsity)' if args.skip_sparsity else 'on (kernel counter, 2D only)'}")
    print(f"  reference lambda  {CALIBRATED_QWEN3_8B}")
    print("=" * 78)


def main(argv=None):
    parser = _make_parser(
        description="BLASST unified_attention benchmark (random and RULER inputs)"
    )
    parser.add_argument("--mode", choices=["random", "ruler"], default="random")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument(
        "--seqlens", type=str, default="16384,32768,65536",
        help="comma-separated, random mode only",
    )
    parser.add_argument(
        "--shapes", type=str, default="64x4",
        help="comma-separated NQxNKV head configs, random mode only",
    )
    parser.add_argument("--thresholds", type=str, default="")
    parser.add_argument("--csv", type=str, default="")
    parser.add_argument("--skip-sparsity", action="store_true",
                        help="skip the sparsity replay")
    # ruler mode
    parser.add_argument("--input-file", type=str, default="")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument(
        "--ruler-mode", choices=["quick", "full"], default="quick",
        help="quick = layers 0,7,18,35 (~4 min). full = every layer (~40 min). "
             "Both print a per-layer table AND an aggregate.")
    parser.add_argument(
        "--layers", type=str, default="",
        help="explicit comma-separated layer indices; overrides --ruler-mode")
    parser.add_argument(
        "--quick", action="store_true",
        help="deprecated alias for --ruler-mode quick")
    parser.add_argument("--num-prompts", type=int, default=2)
    args = parser.parse_args(argv)

    if not torch.cuda.is_available():
        print("ERROR: GPU required.")
        return 1

    thresholds = (
        [float(x) for x in args.thresholds.split(",")]
        if args.thresholds
        else sorted(set(DEFAULT_THRESHOLDS + [CALIBRATED_QWEN3_8B]))
    )
    banner(args)
    rows = []

    if args.mode == "random":
        shapes = [tuple(int(v) for v in s.split("x")) for s in args.shapes.split(",")]
        for nq, nkv in shapes:
            for seqlen in [int(s) for s in args.seqlens.split(",")]:
                inp = make_paged_inputs(
                    seqlen, nq, nkv, args.head_dim, args.block_size
                )
                sweep(
                    f"random  {nq}q/{nkv}kv  seqlen={seqlen}",
                    inp, nq, nkv, args.head_dim, args.block_size,
                    thresholds, args.skip_sparsity, rows,
                )
                del inp
                torch.cuda.empty_cache()
    else:
        if not args.input_file:
            print("ERROR: --mode ruler requires --input-file")
            return 1
        # Same reasoning as _make_parser: the package path executes
        # atom/__init__.py and its engine import chain. capture_qkv sits next to
        # this file and needs none of that, so fall back to importing it directly.
        try:
            from atom.benchmarks.blasst.capture_qkv import capture_model_qkv
        except Exception:  # noqa: BLE001
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from capture_qkv import capture_model_qkv

        # --layers wins; otherwise the mode decides. `None` means every layer.
        if args.layers.strip():
            want = None if args.layers.strip() == "all" else {
                int(x) for x in args.layers.split(",")}
        elif args.quick or args.ruler_mode == "quick":
            want = {0, 7, 18, 35}
        else:
            want = None
        with open(args.input_file) as fh:
            prompts = [json.loads(line)["input"] for _, line in
                       zip(range(args.num_prompts), fh)]
        which = "all" if want is None else sorted(want)
        print(f"\n{len(prompts)} RULER prompt(s), replaying layers {which}")

        for pi, text in enumerate(prompts):
            cap = capture_model_qkv(args.model, text, layers=want)
            for layer in sorted(cap):
                inp = paged_inputs_from_capture(cap[layer], args.block_size)
                nq = cap[layer]["Q"].shape[1]
                nkv = cap[layer]["K"].shape[1]
                sweep(
                    f"ruler p{pi} layer{layer}  {nq}q/{nkv}kv  "
                    f"seqlen={inp['seqlen']}",
                    inp, nq, nkv, inp["query"].shape[-1], args.block_size,
                    thresholds, args.skip_sparsity, rows,
                )
                del inp
                cap[layer] = None      # ~400 MB/layer; 36 layers would pile up
                torch.cuda.empty_cache()
            del cap
            torch.cuda.empty_cache()

    # Per-case speedups vary widely -- diffuse layers lose, focused ones win --
    # so summarise as total dense time over total BLASST time, which is what a
    # whole model would see.
    if rows:
        print("\n" + "=" * 78)
        print("OVERALL  (sum dense / sum BLASST, across the cases measured)")
        print("=" * 78)
        by = {}
        for r in rows:
            by.setdefault((r.get("kernel") or "?", r["threshold"]),
                          [0.0, 0.0, 0])
            e = by[(r.get("kernel") or "?", r["threshold"])]
            e[0] += r["dense_ms"]; e[1] += r["ms"]; e[2] += 1
        for kern in sorted({k for k, _ in by}):
            n = max(v[2] for (k, _), v in by.items() if k == kern)
            print(f"  kernel {kern}  ({n} cases)")
            print(f"    {'threshold':>11} {'dense ms':>10} {'blasst ms':>10} {'speedup':>8}")
            for (k, thr), (d, b, _) in sorted(by.items()):
                if k != kern:
                    continue
                print(f"    {thr:>11g} {d:>10.1f} {b:>10.1f} {d / b:>7.3f}x")
            print()

    if args.csv and rows:
        with open(args.csv, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nwrote {len(rows)} rows to {args.csv}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
