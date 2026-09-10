# SPDX-License-Identifier: MIT
"""Perf harness for the ATOM_USE_TRITON_MOE_DECODE path.

``tests/test_mxfp4_triton_moe_decode.py`` checks that path only for
*correctness* -- it mocks the kernel and carries no timing. This driver stands
the same path up with real MXFP4 weights and real kernels on gfx1250, then times
``Mxfp4MoEMethod.apply()`` in the *decode* phase (``is_prefill=False``), which
dispatches to the Triton/gluon fused-SiLU GUGU experts over a zero-copy view of
the FlyDSL weights (see ``_triton_views_of_flydsl_weights``).

Not a pytest module (name is ``bench_*`` so pytest does not collect it). Run on
a gfx1250 box:

    python tests/bench_mxfp4_triton_moe_decode.py \
        --model-dim 7168 --inter-dim 3072 --experts 96 --topk 6 \
        --act silu --data-format a4w4 --no-bias --tokens 512
"""

import argparse
import os
from types import SimpleNamespace

os.environ.setdefault("AITER_LOG_LEVEL", "WARNING")
# fused_moe_triton guards its kernel imports (moe_gemm_a4w4, mxfp4_quant, ...)
# behind these flags. The method itself is built with object.__new__ below, but
# the guarded module-level import still reads them, so set them before any atom
# import.
os.environ.setdefault("ATOM_USE_TRITON_MOE", "1")
os.environ.setdefault("ATOM_USE_TRITON_MOE_DECODE", "1")
os.environ.setdefault("ATOM_MOE_GU_ITLV", "1")

import torch  # noqa: E402

import atom.model_ops.moe as moe_mod  # noqa: E402
from atom.model_ops.moe import (  # noqa: E402
    ActivationType,
    MoEActivationQuant,
    Mxfp4MoEMethod,
)

# a4w4 => MXFP4 activations, a16w4 => bf16 activations. a8w4 (FP8) is not
# implemented for SiLU in the triton MoE kernel, but is accepted so the flag is
# not silently dropped; the kernel raises if you ask for it with --act silu.
_ACT_QUANT_BY_FORMAT = {
    "a4w4": MoEActivationQuant.FP4,
    "a16w4": MoEActivationQuant.BF16,
    "a8w4": MoEActivationQuant.FP8,
}


def _build_method(num_experts, hidden, inter, act_quant):
    """A Mxfp4MoEMethod carrying only what the decode path in apply() and
    _process_weight_layout_after_loading read -- built with object.__new__ so no
    real create_weights()/FusedMoEConfig is needed (mirrors the unit test)."""
    m = object.__new__(Mxfp4MoEMethod)
    m.use_triton = True
    m.use_triton_ep = False
    m.use_triton_decode = True
    m.is_gfx1250 = True
    m.is_guinterleave = True
    m.num_experts = num_experts
    m.hidden_size = hidden
    m.intermediate_size = inter
    m.hidden_pad = 0
    m.intermediate_pad = 0
    m.act_quant = act_quant
    m.quant_type = "mxfp4"
    m.fused_experts = None
    return m


def _make_layer(num_experts, hidden, inter, use_bias, device, seed=0):
    torch.manual_seed(seed)

    def u8(*shape):
        return torch.nn.Parameter(
            torch.randint(0, 255, shape, dtype=torch.uint8, device=device),
            requires_grad=False,
        )

    if use_bias:
        # The triton kernel asserts w{1,2}_bias.dtype == float32 (the cast that
        # process_weights_after_loading would do).
        w13_bias = torch.nn.Parameter(
            torch.randn(num_experts, 2 * inter, dtype=torch.float32, device=device),
            requires_grad=False,
        )
        w2_bias = torch.nn.Parameter(
            torch.randn(num_experts, hidden, dtype=torch.float32, device=device),
            requires_grad=False,
        )
    else:
        w13_bias = None
        w2_bias = None

    return SimpleNamespace(
        activation=ActivationType.Silu,
        num_fused_shared_experts=0,
        routed_scaling_factor=1.0,
        swiglu_limit=0.0,
        w13_bias=w13_bias,
        w2_bias=w2_bias,
        w13_input_scale=None,
        w2_input_scale=None,
        w13_swizzle_layout=None,
        w2_swizzle_layout=None,
        # create_weights() raw MXFP4 layout, pre-FlyDSL-shuffle.
        w13_weight=u8(num_experts, 2 * inter, hidden // 2),
        w2_weight=u8(num_experts, hidden, inter // 2),
        w13_weight_scale=u8(num_experts, 2 * inter, hidden // 32),
        w2_weight_scale=u8(num_experts, hidden, inter // 32),
    )


def _device_us(evt):
    """Self device (GPU) time for a profiler event, in us, across torch versions."""
    for attr in (
        "self_device_time_total",
        "self_cuda_time_total",
        "self_hip_time_total",
    ):
        v = getattr(evt, attr, None)
        if v:
            return float(v)
    return 0.0


def _profile_kernels(args, tokens, step):
    """Run the eager decode step under the profiler and dump per-kernel GPU time.

    Eager (not graph replay) so kineto/ROCTracer attributes every kernel by
    name; the per-kernel *device* durations are the same ones the graph replays,
    so they sum to the graph number -- only the launch gaps between them differ.
    """
    from torch.profiler import ProfilerActivity, profile

    activities = [ProfilerActivity.CPU]
    if hasattr(ProfilerActivity, "CUDA"):
        activities.append(ProfilerActivity.CUDA)

    for _ in range(5):
        step()
    torch.cuda.synchronize()

    with profile(activities=activities) as prof:
        for _ in range(args.profile_iters):
            step()
        torch.cuda.synchronize()

    rows = [
        (evt.key, evt.count, _device_us(evt))
        for evt in prof.key_averages()
        if _device_us(evt) > 0
    ]
    rows.sort(key=lambda r: -r[2])
    total = sum(r[2] for r in rows) or 1.0
    n = args.profile_iters

    w = 72
    print(f"\n  ── per-kernel GPU time, tokens={tokens} "
          f"(eager, {n} iters) ─────────────────────")
    print(f"  {'kernel':<{w}} {'calls/it':>8} {'us/it':>9} {'%':>6}")
    print("  " + "-" * (w + 26))
    for key, count, dev in rows[:35]:
        name = key if len(key) <= w else key[: w - 3] + "..."
        print(f"  {name:<{w}} {count / n:>8.1f} {dev / n:>9.2f} {100 * dev / total:>5.1f}")
    print("  " + "-" * (w + 26))
    print(f"  {'TOTAL device time / iter':<{w}} {'':>8} {total / n:>9.2f} {100.0:>5.1f}")

    trace_path = f"decode_trace_tokens{tokens}.json"
    prof.export_chrome_trace(trace_path)
    print(f"  chrome trace -> {trace_path}")


def _run_one(args, tokens):
    device = "cuda"
    act_quant = _ACT_QUANT_BY_FORMAT[args.data_format]

    method = _build_method(args.experts, args.model_dim, args.inter_dim, act_quant)
    layer = _make_layer(
        args.experts, args.model_dim, args.inter_dim, not args.no_bias, device
    )

    # FlyDSL weight prep (branch C): shuffles the weights/scales in place into
    # the one layout the decode view reads. This is the prep the unit test's
    # test_decode_prep_takes_the_flydsl_branch exercises.
    method._process_weight_layout_after_loading(layer)

    # Decode phase: every rank decoding, over unified tokens.
    moe_mod.get_forward_context = lambda: SimpleNamespace(
        context=SimpleNamespace(is_prefill=False, running_tokens_are_unified=True)
    )

    x = torch.randn(tokens, args.model_dim, dtype=torch.bfloat16, device=device)
    router_logits = torch.randn(
        tokens, args.experts, dtype=torch.float32, device=device
    )

    def step():
        return method.apply(
            layer=layer,
            x=x,
            router_logits=router_logits,
            top_k=args.topk,
            renormalize=True,
            global_num_experts=args.experts,
            activation=ActivationType.Silu,
        )

    # Same statistic as op_tests/test_flydsl_grouped_gemm_gfx1250.py: aiter's
    # run_perftest profiles num_iters calls and returns the average per-iter sum
    # of every kernel's self_device_time_total (warm iter dropped, IQR-filtered).
    # That is GPU device time, so it excludes launch latency without needing a
    # CUDA graph -- testGraph=False matches the FlyDSL bench's `bench` scenario.
    from aiter.test_common import run_perftest

    out, us = run_perftest(
        step,
        num_warmup=args.warmup,
        num_iters=args.iters,
        testGraph=args.test_graph,
    )
    stats = {"us": float(us)}

    if args.profile:
        _profile_kernels(args, tokens, step)

    return stats, out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dim", type=int, default=7168, help="hidden size")
    parser.add_argument("--inter-dim", type=int, default=3072, help="expert intermediate size")
    parser.add_argument("--experts", type=int, default=96)
    parser.add_argument("--topk", type=int, default=6)
    parser.add_argument("--act", choices=("silu",), default="silu",
                        help="only silu takes the fused GUGU decode path")
    parser.add_argument("--data-format", choices=tuple(_ACT_QUANT_BY_FORMAT),
                        default="a4w4")
    parser.add_argument("--tokens", type=int, nargs="+", default=[512], metavar="N",
                        help="one or more decode token counts; timed once per value")
    parser.add_argument("--no-bias", action="store_true")
    # FlyDSL bench defaults (op_tests/test_flydsl_grouped_gemm_gfx1250.py).
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=101)
    parser.add_argument("--test-graph", action="store_true",
                        help="run_perftest testGraph=True (extra hipgraph timing "
                        "pass); off matches the FlyDSL bench scenario")
    # Accepted for command-line compatibility with the aiter FlyDSL bench; the
    # decode path JITs the triton/gluon kernels and has no FlyDSL AOT cache.
    parser.add_argument("--no-check-aot-cache", action="store_true",
                        help="ignored (no FlyDSL AOT cache on the triton decode path)")
    parser.add_argument("--profile", action="store_true",
                        help="dump a per-kernel GPU-time breakdown (torch profiler) "
                        "and a chrome trace per token count")
    parser.add_argument("--profile-iters", type=int, default=10,
                        help="decode steps to profile per token count")
    args = parser.parse_args()

    from aiter.jit.utils.chip_info import get_gfx

    if get_gfx() != "gfx1250":
        raise SystemExit(f"requires gfx1250, got {get_gfx()}")

    print(
        f"[bench decode] {args.data_format} {args.act} "
        f"experts={args.experts} topk={args.topk} "
        f"model_dim={args.model_dim} inter_dim={args.inter_dim} "
        f"bias={not args.no_bias}",
        flush=True,
    )
    rows = []
    for tok in args.tokens:
        stats, out = _run_one(args, tok)
        rows.append((tok, stats))
        # Same statistic and phrasing as the FlyDSL bench: average per-iter GPU
        # device time from run_perftest.
        print(
            f"[bench {args.data_format} {args.act}] apply() decode "
            f"device us/iter = {stats['us']:.2f}  tokens={tok} "
            f"(out {tuple(out.shape)} {out.dtype}, ||out||={out.float().norm():.3e})",
            flush=True,
        )

    print("\n| tokens | device_us_per_iter |")
    print("|-------:|-------------------:|")
    for tok, s in rows:
        print(f"| {tok:6d} | {s['us']:18.2f} |")


if __name__ == "__main__":
    main()
