# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Standalone trace runner for Lumen benchmarks.

Produces Chrome/Perfetto-compatible ``.json`` trace files in
``benchmarks/traces/``.  Open traces at ``chrome://tracing`` or
https://ui.perfetto.dev.

Usage::

    python -m benchmarks.run_traces                          # all traces
    python -m benchmarks.run_traces --only kernel            # one group
    python -m benchmarks.run_traces --run trace_scaling_modes # one function
    python -m benchmarks.run_traces --run trace_scaling_modes trace_fused_qk_rope
    python -m benchmarks.run_traces --list                   # show all names
"""

from __future__ import annotations

import argparse
import os

import torch
from torch import nn

from benchmarks.bench_utils import (
    TRACE_DIR,
    format_bytes,
    require_aiter,
    require_cuda,
    trace_fn,
)

# ---------------------------------------------------------------------------
# Shared dimensions (Llama 3.1 8B, matching the benchmark files)
# ---------------------------------------------------------------------------
B, S = 2, 2048
H, D = 32, 128
HIDDEN = H * D  # 4096
FFN_HIDDEN = 14336
ROTARY_DIM = D

M = B * S  # tokens
K = HIDDEN
N = FFN_HIDDEN
TP_SIZE = 2


def trace_scaling_modes() -> list[str]:
    """Trace quantized_linear under each FP8 scaling mode."""
    from atomrl.lumen.ops.quantize.linear import quantized_linear
    from atomrl.lumen.quantize.config import _get_float8_e4m3

    fp8_dtype = _get_float8_e4m3()
    x = torch.randn(M, HIDDEN, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(FFN_HIDDEN, HIDDEN, device="cuda", dtype=torch.bfloat16)

    paths = []
    for mode in [
        "none",
        "delayed",
        "dynamic",
        "per_token",
        "blockwise",
        "blockwise2d",
        "mxfp8",
    ]:
        p = trace_fn(
            lambda m=mode: quantized_linear(x, w, scaling_type=m, fp8_dtype=fp8_dtype),
            os.path.join(TRACE_DIR, f"kernel_scaling_{mode}.json"),
            label=f"quantized_linear scaling={mode}",
        )
        paths.append(p)
    return paths


def trace_attention_backends() -> list[str]:
    """Trace attention with aiter_csrc vs aiter_triton backends."""
    from atomrl.lumen.ops.attention import attention

    q = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)

    paths = []
    for backend in ["aiter_csrc", "aiter_triton"]:
        p = trace_fn(
            lambda be=backend: attention(q, k, v, causal=True, backend_type=be),
            os.path.join(TRACE_DIR, f"kernel_attn_{backend.split('_')[-1]}.json"),
            label=f"attention causal ({backend})",
        )
        paths.append(p)
    return paths


# ===================================================================
# 3. Wgrad-delay traces
# ===================================================================


def trace_deferred_wgrad_overlap() -> list[str]:
    """Trace wgrad overlapped with simulated comm (sleep) vs sequential."""
    from atomrl.lumen.modules.parallel_linear import _DeferredWgrad

    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    w = nn.Parameter(torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02)
    w.main_grad = torch.zeros(N, K, device="cuda", dtype=torch.bfloat16)
    grad_out = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
    dwg = _DeferredWgrad()
    comm_stream = torch.cuda.Stream()

    # Calibrate sleep cycles to roughly match wgrad GEMM latency
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(5):
        _ = grad_out.T @ x
    torch.cuda.synchronize()
    start.record()
    _ = grad_out.T @ x
    end.record()
    torch.cuda.synchronize()
    dw_ns = int(start.elapsed_time(end) * 1e6)
    sleep_cycles = max(dw_ns * 2, 100_000)

    def _sequential():
        w.main_grad.zero_()
        dwg.defer(w, lambda: grad_out.T @ x)
        dwg.execute()
        torch.cuda.synchronize()
        torch.cuda._sleep(sleep_cycles)
        torch.cuda.synchronize()

    def _overlapped():
        w.main_grad.zero_()
        dwg.defer(w, lambda: grad_out.T @ x)
        with torch.cuda.stream(comm_stream):
            torch.cuda._sleep(sleep_cycles)
        dwg.execute()
        torch.cuda.current_stream().wait_stream(comm_stream)

    paths = [
        trace_fn(
            _sequential,
            os.path.join(TRACE_DIR, "wgrad_comm_sequential.json"),
            label="wgrad+comm sequential",
        ),
        trace_fn(
            _overlapped,
            os.path.join(TRACE_DIR, "wgrad_comm_overlapped.json"),
            label="wgrad||comm overlapped",
        ),
    ]
    return paths


def trace_multi_layer_pipeline() -> list[str]:
    """Trace 4-layer deferred wgrad pipeline vs eager."""
    from atomrl.lumen.modules.parallel_linear import _DeferredWgrad

    n_layers = 4
    weights = [
        nn.Parameter(torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02)
        for _ in range(n_layers)
    ]
    for w_i in weights:
        w_i.main_grad = torch.zeros_like(w_i.data)

    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    grad_outs = [
        torch.randn(M, N, device="cuda", dtype=torch.bfloat16) for _ in range(n_layers)
    ]
    dwg = _DeferredWgrad()
    wgrad_stream = torch.cuda.Stream()

    def _eager():
        for i in range(n_layers):
            weights[i].main_grad.zero_()
            _ = grad_outs[i] @ weights[i]
            weights[i].main_grad.add_(grad_outs[i].T @ x)

    def _deferred():
        for i in range(n_layers):
            weights[i].main_grad.zero_()
            if dwg.has_pending:
                with torch.cuda.stream(wgrad_stream):
                    dwg.execute()
            _ = grad_outs[i] @ weights[i]
            dwg.defer(weights[i], lambda ii=i: grad_outs[ii].T @ x)
        with torch.cuda.stream(wgrad_stream):
            dwg.execute()
        wgrad_stream.synchronize()

    paths = [
        trace_fn(
            _eager,
            os.path.join(TRACE_DIR, "wgrad_4layer_eager.json"),
            label="4-layer eager",
        ),
        trace_fn(
            _deferred,
            os.path.join(TRACE_DIR, "wgrad_4layer_deferred.json"),
            label="4-layer deferred",
        ),
    ]
    return paths


# ===================================================================
# 4. RoPE fusion traces
# ===================================================================


def _make_cos_sin(seqlen: int, rotary_dim: int):
    positions = torch.arange(seqlen, device="cuda", dtype=torch.float32)
    dim_half = rotary_dim // 2
    freqs = 1.0 / (
        10000.0
        ** (torch.arange(0, dim_half, device="cuda", dtype=torch.float32) / dim_half)
    )
    angles = torch.outer(positions, freqs)
    return angles.cos().to(torch.bfloat16), angles.sin().to(torch.bfloat16)


def _pytorch_rope_neox(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    d = x.shape[-1]
    d2 = d // 2
    x1, x2 = x[..., :d2], x[..., d2:]
    cos_exp = cos[: x.shape[-2], :].unsqueeze(0).unsqueeze(0)
    sin_exp = sin[: x.shape[-2], :].unsqueeze(0).unsqueeze(0)
    rotated = torch.cat([-x2, x1], dim=-1)
    return (
        x * cos_exp.repeat(1, 1, 1, 2)[..., :d]
        + rotated * sin_exp.repeat(1, 1, 1, 2)[..., :d]
    )


def trace_rope_fused_vs_pytorch() -> list[str]:
    """Trace Lumen fused RoPE vs decomposed PyTorch RoPE."""
    from atomrl.lumen.ops.rope import apply_rotary_pos_emb

    seqlen = 2048
    x = torch.randn(B, H, seqlen, D, device="cuda", dtype=torch.bfloat16)
    cos, sin = _make_cos_sin(seqlen, ROTARY_DIM)

    paths = [
        trace_fn(
            lambda: apply_rotary_pos_emb(x, cos, sin),
            os.path.join(TRACE_DIR, "rope_fused_s2048.json"),
            label="Lumen fused RoPE S=2048",
        ),
        trace_fn(
            lambda: _pytorch_rope_neox(x, cos, sin),
            os.path.join(TRACE_DIR, "rope_pytorch_s2048.json"),
            label="PyTorch decomposed RoPE S=2048",
        ),
    ]
    return paths


def trace_fused_qk_rope() -> list[str]:
    """Trace fused_rope Q+K in one dispatch."""
    from atomrl.lumen.ops.rope import fused_rope

    seqlen = 2048
    q = torch.randn(B, H, seqlen, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, H, seqlen, D, device="cuda", dtype=torch.bfloat16)
    cos, sin = _make_cos_sin(seqlen, ROTARY_DIM)

    return [
        trace_fn(
            lambda: fused_rope(q, k, cos, sin),
            os.path.join(TRACE_DIR, "rope_fused_qk.json"),
            label="fused_rope Q+K S=2048",
        )
    ]


# ===================================================================
# 5. FP8 param traces
# ===================================================================


def trace_fp8_quant_dequant() -> list[str]:
    """Trace quantize_param_to_fp8 and dequantize_param_from_fp8."""
    from atomrl.lumen.quantize.fp8_params import (
        dequantize_param_from_fp8,
        quantize_param_to_fp8,
    )

    weight = torch.randn(FFN_HIDDEN, HIDDEN, device="cuda", dtype=torch.bfloat16)
    fp8_w, scale = quantize_param_to_fp8(weight)

    return [
        trace_fn(
            lambda: quantize_param_to_fp8(weight),
            os.path.join(TRACE_DIR, "fp8_quant.json"),
            label="quantize_param_to_fp8 (FFN gate)",
        ),
        trace_fn(
            lambda: dequantize_param_from_fp8(fp8_w, scale),
            os.path.join(TRACE_DIR, "fp8_dequant.json"),
            label="dequantize_param_from_fp8 (FFN gate)",
        ),
    ]


def trace_fp8_forward() -> list[str]:
    """Trace forward pass with BF16 params vs FP8 params + dequant hooks."""
    from atomrl.lumen.quantize.fp8_params import FP8ParamManager

    x = torch.randn(2, 2048, HIDDEN, device="cuda", dtype=torch.bfloat16)

    model_bf16 = nn.Sequential(
        nn.Linear(HIDDEN, FFN_HIDDEN, bias=False),
        nn.SiLU(),
        nn.Linear(FFN_HIDDEN, HIDDEN, bias=False),
    ).to(device="cuda", dtype=torch.bfloat16)

    model_fp8 = nn.Sequential(
        nn.Linear(HIDDEN, FFN_HIDDEN, bias=False),
        nn.SiLU(),
        nn.Linear(FFN_HIDDEN, HIDDEN, bias=False),
    ).to(device="cuda", dtype=torch.bfloat16)
    mgr = FP8ParamManager()
    mgr.quantize_params(model_fp8)
    mgr.register_dequant_hooks(model_fp8)

    return [
        trace_fn(
            lambda: model_bf16(x),
            os.path.join(TRACE_DIR, "fp8_forward_bf16.json"),
            label="Linear forward (BF16 params)",
        ),
        trace_fn(
            lambda: model_fp8(x),
            os.path.join(TRACE_DIR, "fp8_forward_fp8.json"),
            label="Linear forward (FP8 params + dequant)",
        ),
    ]


# ===================================================================
# Registry and main
# ===================================================================

TRACE_GROUPS = {
    "kernel": [trace_scaling_modes, trace_attention_backends],
    "wgrad": [trace_deferred_wgrad_overlap, trace_multi_layer_pipeline],
    "rope": [trace_rope_fused_vs_pytorch, trace_fused_qk_rope],
    "fp8": [trace_fp8_quant_dequant, trace_fp8_forward],
}

ALL_TRACES = {}
_TRACE_TO_GROUP = {}
for _grp, _fns in TRACE_GROUPS.items():
    for _fn in _fns:
        ALL_TRACES[_fn.__name__] = _fn
        _TRACE_TO_GROUP[_fn.__name__] = _grp

_AITER_GROUPS = {"kernel", "rope"}


def _needs_aiter(fn_name: str) -> bool:
    return _TRACE_TO_GROUP.get(fn_name, "") in _AITER_GROUPS


def main():
    all_trace_names = list(ALL_TRACES.keys())

    parser = argparse.ArgumentParser(
        description="Generate trace files for Lumen benchmarks",
        epilog="Available trace functions: " + ", ".join(all_trace_names),
    )
    parser.add_argument(
        "--only",
        choices=list(TRACE_GROUPS.keys()),
        default=None,
        help="Run only traces for a specific benchmark group",
    )
    parser.add_argument(
        "--run",
        choices=all_trace_names,
        nargs="+",
        default=None,
        help="Run one or more specific trace functions by name",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available trace functions and exit",
    )
    args = parser.parse_args()

    if args.list:
        for grp, fns in TRACE_GROUPS.items():
            print(f"\n  {grp}:")
            for fn in fns:
                print(f"    {fn.__name__}")
        print()
        return

    if args.only and args.run:
        parser.error("--only and --run are mutually exclusive")

    require_cuda()
    os.makedirs(TRACE_DIR, exist_ok=True)

    all_paths: list[str] = []

    if args.run:
        for fn_name in args.run:
            if _needs_aiter(fn_name):
                require_aiter()
            fn = ALL_TRACES[fn_name]
            grp = _TRACE_TO_GROUP[fn_name]
            print(f"\n{'=' * 72}")
            print(f"  Trace: {fn_name}  (group: {grp})")
            print("=" * 72)
            paths = fn()
            all_paths.extend(paths)
    else:
        groups = {args.only: TRACE_GROUPS[args.only]} if args.only else TRACE_GROUPS
        for group_name, trace_fns in groups.items():
            if group_name in _AITER_GROUPS:
                require_aiter()

            print(f"\n{'=' * 72}")
            print(f"  Trace group: {group_name}")
            print("=" * 72)

            for fn in trace_fns:
                paths = fn()
                all_paths.extend(paths)

    # Summary
    print(f"\n{'=' * 72}")
    print(f"  Trace Summary: {len(all_paths)} file(s) in {TRACE_DIR}")
    print("=" * 72)
    for p in all_paths:
        size = os.path.getsize(p) if os.path.exists(p) else 0
        print(f"  {os.path.basename(p):>40s}  {format_bytes(size)}")
    print("\n  View at: chrome://tracing  or  https://ui.perfetto.dev")
    print()


if __name__ == "__main__":
    main()
