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

    out = step()
    torch.cuda.synchronize()

    # The production decode path replays a CUDA graph, so eager per-call time is
    # dominated by kernel-launch latency (a ~2.5ms floor at these expert counts)
    # and is NOT representative. Capture the step into a graph and time replays;
    # that is the apples-to-apples number vs the FlyDSL bench's graph=True.
    graph = None
    if not args.eager:
        try:
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(3):
                    step()
            torch.cuda.current_stream().wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out = step()
        except Exception as exc:  # noqa: BLE001
            print(f"  [graph capture failed, falling back to eager: {exc}]", flush=True)
            graph = None

    run = graph.replay if graph is not None else step
    mode = "graph" if graph is not None else "eager"

    for _ in range(args.warmup):
        run()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(args.iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(args.iters)]
    for i in range(args.iters):
        starts[i].record()
        run()
        ends[i].record()
    torch.cuda.synchronize()

    per_us = sorted(s.elapsed_time(e) * 1e3 for s, e in zip(starts, ends))
    stats = {
        "mode": mode,
        "min": per_us[0],
        "median": per_us[len(per_us) // 2],
        "mean": sum(per_us) / len(per_us),
    }
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
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    # Accepted for command-line compatibility with the aiter FlyDSL bench; the
    # decode path JITs the triton/gluon kernels and has no FlyDSL AOT cache.
    parser.add_argument("--no-check-aot-cache", action="store_true",
                        help="ignored (no FlyDSL AOT cache on the triton decode path)")
    parser.add_argument("--eager", action="store_true",
                        help="skip CUDA-graph capture; time eager apply() (launch-"
                        "latency bound, not representative of production decode)")
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
        print(
            f"  tokens={tok:<6d} [{stats['mode']}] apply() decode us: "
            f"min={stats['min']:8.2f} median={stats['median']:8.2f} "
            f"mean={stats['mean']:8.2f}  "
            f"(out {tuple(out.shape)} {out.dtype}, ||out||={out.float().norm():.3e})",
            flush=True,
        )

    print("\n| tokens | mode | min_us | median_us | mean_us |")
    print("|-------:|:-----|-------:|----------:|--------:|")
    for tok, s in rows:
        print(
            f"| {tok:6d} | {s['mode']} | {s['min']:6.2f} | "
            f"{s['median']:9.2f} | {s['mean']:7.2f} |"
        )


if __name__ == "__main__":
    main()
