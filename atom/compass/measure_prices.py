# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Time the GEMMs a V4-Pro TP8 + DSpark step issues, on one GPU.

The clock is a CUDA graph of several calls, replayed. That is the method
that does not time the host: a back-to-back Python loop sits on the launch
floor. Kernels are AITER's, the ones the model dispatches:

* ``fp8_block128`` — ``gemm_a8w8_blockscale`` (128-element scales)
* ``fp4_block32`` — ``gemm_a4w4`` (packed e2m1, 32-element scales)
* ``bf16`` — ``gemm_a16w16_opus``
* ``fp32`` — ``torch.mm`` (Markov and mHC)

Attention, collectives and the compressor kernels are not in this table.
A shape that fails to launch is omitted; it is not filled from a peak.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from atom.compass.dsv4_tp8_dspark import (
    DSparkShape,
    V4ProShape,
    gemm_shapes,
)
from atom.compass.price_list import Price, PriceList

# Token counts a TP8 DSpark step actually asks for.
# Verify uses batch * (K+1), the draft uses batch * K, prefill uses the
# prompt length, and the draft's main projection uses the batch.
DEFAULT_TOKEN_COUNTS = (
    1,
    4,
    7,
    8,
    16,
    28,
    32,
    56,
    64,
    112,
    128,
    224,
    256,
    448,
    512,
    896,
    1024,
    2048,
)

_KERNEL = {
    "fp8_block128": "aiter.gemm_a8w8_blockscale",
    "fp4_block32": "aiter.gemm_a4w4",
    "bf16": "aiter.gemm_a16w16_opus",
    "fp32": "torch.mm",
}


def _torch_bf16_mm(m: int, k: int, n: int):
    import torch

    activation = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(k, n, device="cuda", dtype=torch.bfloat16)
    out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)

    def launch():
        torch.mm(activation, weight, out=out)

    return launch


def _replay_seconds(launch, *, repeats: int, iters: int) -> float:
    """Seconds per ``launch()``, from a CUDA graph of ``repeats`` calls.

    The graph gets its own stream. BF16 opus GEMMs keep a split-K workspace
    per stream and refuse to allocate it during capture, so the workspace is
    created and warmed on that stream before ``capture_begin``.
    """
    import torch
    from aiter import opus_gemm_workspace_init

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        opus_gemm_workspace_init()
        for _ in range(2):
            launch()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        graph.capture_begin()
        for _ in range(repeats):
            launch()
        graph.capture_end()
    torch.cuda.current_stream().wait_stream(stream)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / (iters * repeats) / 1e3


def _launch(dtype: str, m: int, k: int, n: int):
    """A zero-argument closure that runs one GEMM on the current device."""
    import torch
    from aiter import dtypes, gemm_a16w16_opus, gemm_a4w4, gemm_a8w8_blockscale

    if dtype == "fp8_block128":
        if k % 128:
            raise ValueError(f"fp8 block scale needs K multiple of 128, got {k}")
        activation = torch.randint(0, 255, (m, k), device="cuda", dtype=torch.uint8).view(
            dtypes.fp8
        )
        weight = torch.randint(0, 255, (n, k), device="cuda", dtype=torch.uint8).view(
            dtypes.fp8
        )
        act_scale = torch.ones(m, k // 128, device="cuda")
        weight_scale = torch.ones(n, k // 128, device="cuda")

        def launch():
            gemm_a8w8_blockscale(activation, weight, act_scale, weight_scale)

        return launch
    if dtype == "fp4_block32":
        # Same layout as ``atom.model_ops.linear.gemm_a4w4_quant``: per-1x32
        # scales, weight pre-shuffled. The unshuffled default kernel faults on
        # the expert down-projection (K=384).
        from aiter import QuantType, dtypes, gemm_a4w4, get_hip_quant

        quant = get_hip_quant(QuantType.per_1x32)
        activation = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
        activation_q, act_scale = quant(
            activation, quant_dtype=dtypes.fp4x2, shuffle=m >= 32
        )
        weight_q, weight_scale = quant(
            weight, quant_dtype=dtypes.fp4x2, shuffle=True
        )

        def launch():
            gemm_a4w4(activation_q, weight_q, act_scale, weight_scale)

        return launch
    if dtype == "bf16":
        activation = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)

        def launch():
            gemm_a16w16_opus(activation, weight)

        return launch
    if dtype == "fp32":
        activation = torch.randn(m, k, device="cuda", dtype=torch.float32)
        weight = torch.randn(k, n, device="cuda", dtype=torch.float32)

        def launch():
            torch.mm(activation, weight)

        return launch
    raise ValueError(f"unknown price dtype {dtype}")


def measure(
    token_counts: tuple[int, ...] = DEFAULT_TOKEN_COUNTS,
    *,
    tp: int = 8,
    repeats: int = 8,
    iters: int = 3,
    device_index: int = 0,
) -> PriceList:
    """Price every distinct GEMM the TP8 DSpark step uses, at each token count."""
    import torch

    torch.cuda.set_device(device_index)
    props = torch.cuda.get_device_properties(device_index)
    shapes = gemm_shapes(V4ProShape(), DSparkShape(), tp)
    entries: dict[tuple[str, int, int, int], Price] = {}
    failures: list[str] = []
    for tokens in token_counts:
        for dtype, k, n in shapes:
            key = (dtype, tokens, k, n)
            if key in entries:
                continue
            try:
                launch = _launch(dtype, tokens, k, n)
                kernel = _KERNEL[dtype]
                # Large fp32 GEMMs (the Markov head) dominate the campaign.
                local_repeats = 2 if dtype == "fp32" and tokens * k * n > 2e9 else repeats
                seconds = _replay_seconds(launch, repeats=local_repeats, iters=iters)
            except Exception as exc:  # noqa: BLE001 — a missed kernel is a miss
                message = str(exc)
                if dtype == "bf16" and "not found in a16w16" in message:
                    # Opus has no tuned kernel for this shape. Time a real
                    # bf16 GEMM instead of leaving a hole, and name that kernel.
                    launch = _torch_bf16_mm(tokens, k, n)
                    kernel = "torch.mm"
                    local_repeats = repeats
                    seconds = _replay_seconds(launch, repeats=local_repeats, iters=iters)
                else:
                    print(f"MISS M={tokens} K={k} N={n} {dtype}: {message}", flush=True)
                    if "illegal memory access" in message:
                        raise RuntimeError(
                            "GPU context is dead after an illegal access; "
                            "not recording the shapes that follow"
                        ) from exc
                    failures.append(f"M={tokens} K={k} N={n} {dtype}: {message}")
                    continue
            entries[key] = Price(
                dtype=dtype,
                m=tokens,
                k=k,
                n=n,
                seconds=seconds,
                kernel=kernel,
            )
            print(
                f"{dtype:14} M={tokens:5} K={k:6} N={n:6}  {seconds * 1e6:10.2f} us",
                flush=True,
            )
    if failures:
        print(f"{len(failures)} shapes missed, {len(entries)} priced", flush=True)
    name = props.name.replace(" ", "_")
    return PriceList(
        entries=entries,
        device=f"{name}:{getattr(props, 'gcnArchName', 'gfx')}",
        method="cuda_graph_replay",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("atom/compass/prices/gfx950_dsv4_tp8_dspark.json"),
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--tokens", type=str, default=",".join(map(str, DEFAULT_TOKEN_COUNTS)))
    args = parser.parse_args()
    counts = tuple(int(part) for part in args.tokens.split(",") if part)
    catalog = measure(counts, device_index=args.device)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    catalog.save(args.out)
    print(f"wrote {len(catalog.entries)} prices to {args.out}")


if __name__ == "__main__":
    main()
