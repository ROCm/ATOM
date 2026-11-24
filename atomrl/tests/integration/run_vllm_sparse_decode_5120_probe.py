# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Stress the vLLM DSV4 sparse-decode kernel at the failing 5120 layout."""

from __future__ import annotations

import argparse
import math

import torch
from vllm.models.deepseek_v4.amd.rocm import (
    compute_global_topk_ragged_indices_and_indptr,
)
from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
    _rocm_sparse_attn_decode_ragged_triton,
)

BATCH_SIZE = 32
LOCAL_HEADS = 16
NOPE_DIM = 448
ROPE_DIM = 64
HEAD_DIM = NOPE_DIM + ROPE_DIM
SWA_BLOCK_SIZE = 256
COMPRESS_RATIO = 4
COMPRESSED_BLOCK_SIZE = SWA_BLOCK_SIZE // COMPRESS_RATIO
MAX_MODEL_LEN = 5120
NUM_PHYSICAL_BLOCKS = 6200
SWA_TOKENS = 128
TOPK_TOKENS = 512
CACHE_BYTES_PER_TOKEN = 576
SCALE_BYTES_PER_TOKEN = 8


def make_cache(block_size: int) -> torch.Tensor:
    cache = torch.zeros(
        (
            NUM_PHYSICAL_BLOCKS,
            block_size,
            CACHE_BYTES_PER_TOKEN + SCALE_BYTES_PER_TOKEN,
        ),
        dtype=torch.uint8,
        device="cuda",
    )
    flat = cache.view(NUM_PHYSICAL_BLOCKS, -1)
    scale_start = block_size * CACHE_BYTES_PER_TOKEN
    flat[:, scale_start:] = 127
    return cache


def make_swa_ragged_indices() -> tuple[torch.Tensor, torch.Tensor]:
    total_rows = NUM_PHYSICAL_BLOCKS * SWA_BLOCK_SIZE
    tokens_per_query = SWA_TOKENS
    first_row = total_rows - BATCH_SIZE * tokens_per_query
    indices = torch.arange(
        first_row,
        total_rows,
        dtype=torch.int32,
        device="cuda",
    )
    indptr = torch.arange(
        0,
        (BATCH_SIZE + 1) * tokens_per_query,
        tokens_per_query,
        dtype=torch.int32,
        device="cuda",
    )
    return indices, indptr


def make_indexer_metadata_inputs() -> tuple[torch.Tensor, ...]:
    compressed_context = MAX_MODEL_LEN // COMPRESS_RATIO
    local_topk = torch.stack(
        [
            torch.randperm(compressed_context, device="cuda", dtype=torch.int32)[
                :TOPK_TOKENS
            ]
            for _ in range(BATCH_SIZE)
        ]
    )
    token_to_req = torch.arange(BATCH_SIZE, dtype=torch.int32, device="cuda")
    block_table_width = math.ceil(MAX_MODEL_LEN / SWA_BLOCK_SIZE)
    first_physical_block = NUM_PHYSICAL_BLOCKS - BATCH_SIZE * block_table_width
    block_table = torch.arange(
        first_physical_block,
        NUM_PHYSICAL_BLOCKS,
        dtype=torch.int32,
        device="cuda",
    ).view(BATCH_SIZE, block_table_width)
    is_valid = torch.ones(BATCH_SIZE, dtype=torch.bool, device="cuda")
    return local_topk, token_to_req, block_table, is_valid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=200)
    args = parser.parse_args()

    assert torch.cuda.is_available(), "ROCm GPU is required"
    assert math.ceil(MAX_MODEL_LEN / SWA_BLOCK_SIZE) < NUM_PHYSICAL_BLOCKS
    torch.manual_seed(1234)

    q = torch.randn(
        BATCH_SIZE,
        LOCAL_HEADS,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device="cuda",
    )
    swa_cache = make_cache(SWA_BLOCK_SIZE)
    extra_cache = make_cache(COMPRESSED_BLOCK_SIZE)
    swa_indices, swa_indptr = make_swa_ragged_indices()
    metadata_inputs = make_indexer_metadata_inputs()

    reference = None
    for iteration in range(args.iterations):
        local_topk, token_to_req, block_table, is_valid = metadata_inputs
        topk_indices, topk_indptr, topk_lens = (
            compute_global_topk_ragged_indices_and_indptr(
                local_topk,
                token_to_req,
                block_table,
                block_size=COMPRESSED_BLOCK_SIZE,
                is_valid_token=is_valid,
            )
        )
        assert bool((topk_lens == TOPK_TOKENS).all())
        assert int(topk_indices.min()) >= 0
        assert int(topk_indices.max()) < NUM_PHYSICAL_BLOCKS * COMPRESSED_BLOCK_SIZE

        output = _rocm_sparse_attn_decode_ragged_triton(
            q=q,
            main_cache=swa_cache,
            main_indices=swa_indices,
            main_indptr=swa_indptr,
            scale=HEAD_DIM**-0.5,
            attn_sink=None,
            nope_head_dim=NOPE_DIM,
            rope_head_dim=ROPE_DIM,
            extra_cache=extra_cache,
            extra_indices=topk_indices,
            extra_indptr=topk_indptr,
        )
        torch.cuda.synchronize()

        assert output.shape == q.shape
        assert bool(
            torch.isfinite(output).all()
        ), "sparse decode produced non-finite output"
        if reference is None:
            reference = output.clone()
        else:
            torch.testing.assert_close(output, reference, rtol=0, atol=0)

        if iteration == 0 or (iteration + 1) % 25 == 0:
            print(f"iteration={iteration + 1}/{args.iterations} ok", flush=True)

    print(
        "PASS "
        f"batch={BATCH_SIZE} local_heads={LOCAL_HEADS} "
        f"max_model_len={MAX_MODEL_LEN} swa={SWA_TOKENS} topk={TOPK_TOKENS} "
        f"compressed_block_size={COMPRESSED_BLOCK_SIZE} "
        f"physical_blocks={NUM_PHYSICAL_BLOCKS} iterations={args.iterations}",
        flush=True,
    )


if __name__ == "__main__":
    main()
