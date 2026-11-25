# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Stress the vLLM DSV4 paged-indexer kernel at the failing 5120 layout."""

from __future__ import annotations

import argparse
import math

import torch
from aiter.ops.triton.attention.pa_mqa_logits import deepgemm_fp8_paged_mqa_logits
from aiter.utility.dtypes import fp8

BATCH_SIZE = 32
NEXT_N = 1
HEADS = 64
HEAD_DIM = 128
BLOCK_SIZE = 256
MAX_MODEL_LEN = 5120
NUM_PHYSICAL_BLOCKS = 6200
SENTINEL = 12345.0

# Reproduces the scheduler lengths dumped immediately after the v11 fault.
CONTEXT_LENS = (
    2053,
    2053,
    2053,
    2053,
    2052,
    2051,
    2051,
    2051,
    2005,
    2005,
    2005,
    2005,
    2005,
    2005,
    2005,
    2005,
    2005,
    2005,
    2005,
    2050,
    2050,
    1595,
    1465,
    1024,
    896,
    768,
    640,
    512,
    448,
    384,
    320,
    285,
)


def make_inputs() -> tuple[torch.Tensor, ...]:
    torch.manual_seed(1234)
    device = "cuda"

    q_bits = torch.randint(
        1,
        64,
        (BATCH_SIZE, NEXT_N, HEADS, HEAD_DIM),
        dtype=torch.uint8,
        device=device,
    )
    q_fp8 = q_bits.view(fp8)

    kv_bits = torch.randint(
        1,
        64,
        (NUM_PHYSICAL_BLOCKS, BLOCK_SIZE, 1, HEAD_DIM + 4),
        dtype=torch.uint8,
        device=device,
    )
    kv_bits[..., HEAD_DIM:] = torch.tensor(
        [0, 0, 128, 63], dtype=torch.uint8, device=device
    )
    kv_cache = kv_bits.view(fp8)

    weights = torch.ones(
        (BATCH_SIZE * NEXT_N, HEADS), dtype=torch.float32, device=device
    )
    context_lens = torch.tensor(CONTEXT_LENS, dtype=torch.int32, device=device)

    max_blocks_per_seq = math.ceil(MAX_MODEL_LEN / BLOCK_SIZE)
    first_physical_block = NUM_PHYSICAL_BLOCKS - BATCH_SIZE * max_blocks_per_seq
    block_tables = torch.arange(
        first_physical_block,
        NUM_PHYSICAL_BLOCKS,
        dtype=torch.int32,
        device=device,
    ).view(BATCH_SIZE, max_blocks_per_seq)
    return q_fp8, kv_cache, weights, context_lens, block_tables


def run_kernel(
    q: torch.Tensor,
    kv: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    output_width: int,
) -> torch.Tensor:
    out = torch.full(
        (BATCH_SIZE * NEXT_N, output_width),
        SENTINEL,
        dtype=torch.float32,
        device=q.device,
    )
    deepgemm_fp8_paged_mqa_logits(
        q,
        kv,
        weights,
        out,
        context_lens,
        block_tables,
        output_width,
        Preshuffle=True,
        KVBlockSize=BLOCK_SIZE,
        ChunkK=256,
        WavePerEU=2,
    )
    torch.cuda.synchronize()
    return out


def verify_output(wide: torch.Tensor, compact: torch.Tensor) -> None:
    for row, context_len in enumerate(CONTEXT_LENS):
        valid = wide[row, :context_len]
        assert not bool(
            (valid == SENTINEL).any()
        ), f"row {row} left valid logits unwritten at context_len={context_len}"
        torch.testing.assert_close(valid, compact[row, :context_len], rtol=0, atol=0)
        tail = wide[row, context_len:]
        tail_is_padding = (tail == SENTINEL) | torch.isneginf(tail)
        assert bool(
            tail_is_padding.all()
        ), f"row {row} produced finite logits beyond context_len={context_len}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=200)
    args = parser.parse_args()

    assert torch.cuda.is_available(), "ROCm GPU is required"
    inputs = make_inputs()
    compact_width = max(CONTEXT_LENS)
    compact = run_kernel(*inputs, compact_width)

    for iteration in range(args.iterations):
        wide = run_kernel(*inputs, MAX_MODEL_LEN)
        verify_output(wide, compact)
        if iteration == 0 or (iteration + 1) % 25 == 0:
            print(f"iteration={iteration + 1}/{args.iterations} ok", flush=True)

    print(
        "PASS "
        f"batch={BATCH_SIZE} heads={HEADS} max_model_len={MAX_MODEL_LEN} "
        f"physical_blocks={NUM_PHYSICAL_BLOCKS} iterations={args.iterations}",
        flush=True,
    )


if __name__ == "__main__":
    main()
