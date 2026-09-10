# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Reproduce K3's small-query cached-chunk attention on one gfx950 GPU.

Correctness UT:
    HIP_VISIBLE_DEVICES=0 python -m pytest -q tests/test_mla_cached_chunk_fmha.py

Attention-only benchmark and kernel traces:
    HIP_VISIBLE_DEVICES=0 python tests/test_mla_cached_chunk_fmha.py \
        --output my_script/results/cached-chunk-fmha-tok70

The traced batch has 70 new queries and a 338,688-token cached prefix:
20 full 16,384-token chunks and one 11,008-token tail. Every chunk runs
noncausal varlen attention with LSE, Hq=Hkv=12, Dq=Dk=192, Dv=128.
Inputs are synthetic and shared by both backends; this is a shape reproduction,
not a replay of model activations. Quantization, gather and LSE merge are
outside the timed region. No performance threshold is asserted by pytest.
"""

import argparse
import json
import os
import statistics
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

import pytest
import torch

QUERY_TOKENS = 70
HEADS = 12
QK_DIM = 192
V_DIM = 128
KV_LENGTHS = (16384, 11008)
SOFTMAX_SCALE = QK_DIM**-0.5


def require_gfx950():
    if not torch.cuda.is_available() or not torch.version.hip:
        pytest.skip("requires a ROCm gfx950 GPU")
    if not torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx950"):
        pytest.skip("requires gfx950 OPUS and FlyDSL FP8 attention")
    if int(os.environ.get("AITER_DISABLE_FMHA_OPUS", "0")):
        raise RuntimeError("unset AITER_DISABLE_FMHA_OPUS to benchmark OPUS")


def make_case(kv_tokens):
    from aiter import flash_attn_varlen_func
    from aiter.ops.flydsl import flydsl_flash_attn_fp8_func
    from aiter.ops.quant import per_tensor_quant_hip

    generator = torch.Generator(device="cuda").manual_seed(42)
    tensors = tuple(
        torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=generator)
        for shape in (
            (QUERY_TOKENS, HEADS, QK_DIM),
            (kv_tokens, HEADS, QK_DIM),
            (kv_tokens, HEADS, V_DIM),
        )
    )
    quantized, scales = [], []
    for x in tensors:
        # Complete vectors for the HIP reference quantizer; done before timing.
        x8, scale = per_tensor_quant_hip(
            x.view(-1, 128), quant_dtype=torch.float8_e4m3fn
        )
        quantized.append(x8.view(x.shape))
        scales.append(scale.reshape(1))
    cu_q = torch.tensor([0, QUERY_TOKENS], device="cuda", dtype=torch.int32)
    cu_k = torch.tensor([0, kv_tokens], device="cuda", dtype=torch.int32)
    calls = {
        "bf16_opus": partial(
            flash_attn_varlen_func,
            *tensors,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            max_seqlen_q=QUERY_TOKENS,
            max_seqlen_k=kv_tokens,
            min_seqlen_q=QUERY_TOKENS,
            dropout_p=0.0,
            softmax_scale=SOFTMAX_SCALE,
            causal=False,
            return_lse=True,
        ),
        "fp8_flydsl": partial(
            flydsl_flash_attn_fp8_func,
            *quantized,
            cu_seqlens_q=cu_q,
            cu_seqlens_kv=cu_k,
            max_seqlen_q=QUERY_TOKENS,
            max_seqlen_kv=kv_tokens,
            cross_seqlen=True,
            causal=False,
            return_lse=True,
            q_descale=scales[0],
            k_descale=scales[1],
            v_descale=scales[2],
            # Keep production defaults for split-K and tile selection.
            stream=None,
        ),
    }
    return calls, tensors, quantized, scales


def reference(q, k, v):
    q, k, v = (x.float().transpose(0, 1) for x in (q, k, v))
    scores = torch.matmul(q, k.transpose(1, 2)) * SOFTMAX_SCALE
    lse = torch.logsumexp(scores, dim=-1)
    out = torch.matmul(torch.softmax(scores, dim=-1), v).transpose(0, 1)
    return out, lse


def relative_rmse(actual, expected):
    return (
        (actual.float() - expected.float()).square().mean().sqrt()
        / expected.float().square().mean().sqrt().clamp_min(1e-12)
    ).item()


@torch.no_grad()
def validate(calls, tensors, quantized, scales):
    refs = {
        "bf16_opus": reference(*tensors),
        "fp8_flydsl": reference(*(x.float() * s for x, s in zip(quantized, scales))),
    }
    outputs, errors = {}, {}
    for name, fn in calls.items():
        out, lse = fn()
        expected, expected_lse = refs[name]
        assert out.shape == (QUERY_TOKENS, HEADS, V_DIM)
        assert lse.shape == (HEADS, QUERY_TOKENS)
        assert out.dtype == torch.bfloat16 and lse.dtype == torch.float32
        assert torch.isfinite(out).all() and torch.isfinite(lse).all()
        error = relative_rmse(out, expected)
        # FlyDSL casts softmax probabilities to E4M3 before P@V, even when the
        # reference starts from the same quantized Q/K/V. Allow that internal
        # FP8 rounding; OPUS's BF16 probability/output path is more precise.
        assert error < (0.04 if name == "fp8_flydsl" else 0.005), (name, error)
        torch.testing.assert_close(lse, expected_lse, rtol=0, atol=1e-4)
        errors[name] = {
            "output_relative_rmse_vs_own_fp32_reference": error,
            "lse_max_abs_error": (lse - expected_lse).abs().max().item(),
        }
        outputs[name] = out
    # Also report the effect of FP8 quantization on the same original inputs.
    errors["fp8_vs_bf16_output_relative_rmse"] = relative_rmse(
        outputs["fp8_flydsl"], outputs["bf16_opus"]
    )
    return errors


def trace_call(name, fn, path=None):
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as prof:
        fn()
        torch.cuda.synchronize()
    # Exporting preserves GPU grid/block information for comparison with the
    # server capture. Only actual device kernels count toward this assertion.
    if path is None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            return export_and_check(prof, name, Path(tmp) / "trace.json")
    return export_and_check(prof, name, path)


def export_and_check(prof, name, path):
    prof.export_chrome_trace(str(path))
    kernels = [
        e
        for e in json.loads(path.read_text())["traceEvents"]
        if e.get("cat") == "kernel"
    ]
    expected = (
        "gqa_d192_v128_kernel" if name == "bf16_opus" else "flash_attn_dualwave_swp_fp8"
    )
    assert len(kernels) == 1, [(e["name"], e.get("dur")) for e in kernels]
    e = kernels[0]
    assert expected in e["name"], e["name"]
    assert e["args"]["grid"] == [12, 1, 1], e["args"]
    assert e["args"]["block"] == [512, 1, 1], e["args"]
    return {
        "name": e["name"],
        "duration_us": e["dur"],
        "grid": e["args"]["grid"],
        "block": e["args"]["block"],
    }


@pytest.mark.parametrize("kv_tokens", KV_LENGTHS)
@torch.no_grad()
def test_cached_chunk_fmha(kv_tokens, monkeypatch):
    require_gfx950()
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    calls, tensors, quantized, scales = make_case(kv_tokens)
    validate(calls, tensors, quantized, scales)
    for name, fn in calls.items():
        fn()
        torch.cuda.synchronize()
        trace_call(name, fn)


@torch.no_grad()
def benchmark(kv_tokens, output, rounds, iterations):
    calls, tensors, quantized, scales = make_case(kv_tokens)
    errors = validate(calls, tensors, quantized, scales)
    graphs, graph_outputs = {}, {}
    for name, fn in calls.items():
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_outputs[name] = fn()
        graphs[name] = graph
        for _ in range(5):
            graph.replay()
    torch.cuda.synchronize()

    # Alternate A/B order each round. Batched graph replay removes Python
    # preparation/launch starvation; events time only GPU execution and replay
    # gaps. Inputs are fixed, warm buffers, not a model-wide cache simulation.
    samples = {name: [] for name in calls}
    names = list(calls)
    for r in range(rounds):
        for name in names[:: 1 if r % 2 == 0 else -1]:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iterations):
                graphs[name].replay()
            end.record()
            end.synchronize()
            samples[name].append(start.elapsed_time(end) * 1000 / iterations)

    row = {
        "query_tokens": QUERY_TOKENS,
        "kv_tokens": kv_tokens,
        "correctness": errors,
        "timing": {},
    }
    for name, fn in calls.items():
        trace = trace_call(name, fn, output / f"{name}-q70-kv{kv_tokens}.trace.json")
        ds = samples[name]
        row["timing"][name] = {
            "median_us": statistics.median(ds),
            "min_us": min(ds),
            "max_us": max(ds),
            "rounds_us": ds,
            "kernel_trace": trace,
        }
    a, b = (row["timing"][name]["median_us"] for name in names)
    row.update(speedup=a / b, time_reduction_pct=100 * (1 - b / a))
    return row


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--output", type=Path, default=Path("my_script/results/cached-chunk-fmha-tok70")
    )
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()
    if args.rounds < 1 or args.iterations < 1:
        parser.error("rounds and iterations must be positive")
    require_gfx950()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    rows = []
    for kv_tokens in KV_LENGTHS:
        row = benchmark(kv_tokens, args.output, args.rounds, args.iterations)
        rows.append(row)
        print(json.dumps(row, indent=2), flush=True)
    props = torch.cuda.get_device_properties(0)
    report = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "device": props.name,
        "arch": props.gcnArchName,
        "hip_visible_devices": os.environ.get("HIP_VISIBLE_DEVICES"),
        "torch_version": torch.__version__,
        "hip_version": torch.version.hip,
        "heads": HEADS,
        "qk_dim": QK_DIM,
        "v_dim": V_DIM,
        "softmax_scale": SOFTMAX_SCALE,
        "causal": False,
        "return_lse": True,
        "rounds": args.rounds,
        "iterations_per_round": args.iterations,
        "timing_method": "HIP events around warmed CUDA-graph replays; alternating A/B order",
        "inputs": "seed-42 synthetic BF16; FP8 quantized once before timing; fixed warm buffers",
        "excluded": ["QKV quantization", "gather", "LSE merge", "CPU wrapper overhead"],
        "cases": rows,
    }
    (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Results: {args.output / 'results.json'}", flush=True)


if __name__ == "__main__":
    main()
