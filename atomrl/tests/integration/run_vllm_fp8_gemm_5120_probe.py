# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Stress the AITER Triton FP8 GEMMs used by vLLM DSV4."""

from __future__ import annotations

import argparse

import torch
from aiter.ops.shuffle import shuffle_weight
from aiter.ops.triton.gemm.basic.gemm_a8w8_blockscale import (
    gemm_a8w8_blockscale_preshuffle,
)
from aiter.utility.dtypes import fp8

BLOCK = 128
SHAPES = (
    # Decode batch and the two untuned TP-local shapes observed in vLLM logs.
    (32, 1536, 4096, 200),
    (32, 4096, 1024, 200),
    # Long-context/prefill token counts around the failing scheduler state.
    (2053, 1536, 4096, 25),
    (2053, 4096, 1024, 25),
    (5120, 1536, 4096, 10),
    (5120, 4096, 1024, 10),
)


def make_inputs(m: int, n: int, k: int) -> tuple[torch.Tensor, ...]:
    x = (torch.rand((m, k), dtype=torch.float32, device="cuda") / 10).to(fp8)
    weight = (torch.rand((n, k), dtype=torch.float32, device="cuda") / 10).to(fp8)
    weight = shuffle_weight(weight, layout=(16, 16))
    scale_k = k // BLOCK
    x_scale = torch.rand((m, scale_k), dtype=torch.float32, device="cuda")
    x_scale = x_scale.transpose(0, 1).contiguous().view(m, scale_k)
    w_scale = torch.rand(
        ((n + BLOCK - 1) // BLOCK, scale_k),
        dtype=torch.float32,
        device="cuda",
    )
    return x, weight.reshape(n // 16, k * 16), x_scale, w_scale


def run(inputs: tuple[torch.Tensor, ...]) -> torch.Tensor:
    output = gemm_a8w8_blockscale_preshuffle(
        *inputs,
        dtype=torch.bfloat16,
    )
    torch.cuda.synchronize()
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iteration-scale", type=float, default=1.0)
    args = parser.parse_args()

    assert torch.cuda.is_available(), "ROCm GPU is required"
    torch.manual_seed(1234)

    for m, n, k, base_iterations in SHAPES:
        iterations = max(1, round(base_iterations * args.iteration_scale))
        inputs = make_inputs(m, n, k)
        reference = run(inputs).clone()
        assert reference.shape == (m, n)
        assert bool(torch.isfinite(reference).all())

        for _ in range(iterations):
            output = run(inputs)
            torch.testing.assert_close(output, reference, rtol=0, atol=0)

        print(
            f"shape=({m},{n},{k}) iterations={iterations} ok",
            flush=True,
        )

    print("PASS all vLLM FP8 GEMM shapes", flush=True)


if __name__ == "__main__":
    main()
