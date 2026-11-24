# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The W4A4 expert host on 256-row blocks (from 4097 tokens): the routing sort writes GEMM1's tile map
(``common/sort_count_all.py`` to ``SORT_COUNT_ALL_MAX_TOKENS``, ``common/sort_tile_map.py`` above), the scale sort runs
without tile-map CTAs, then the persistent GEMM1 + GEMM2 run in one launch (``gemm12_a4w4``) or two.
"""

import functools

import torch

from .prefill_h6144_i768 import GEMM2_N_SPLIT, Mxfp4PrefillScales, gemm2_grid

H, I, BM = 6144, 768, 256  # noqa: E741

# one-launch sort to here: sort + scale sort -1.7 us at 8192, +0.9 at 16384, +9.1 at 32768 vs the two-launch sort
SORT_COUNT_ALL_MAX_TOKENS = 8192


@functools.cache
def _gemm1(num_experts):
    from .gemm1_prefill_h6144_i768 import compile_moe_gemm1_mxfp4_h6144_i768
    return compile_moe_gemm1_mxfp4_h6144_i768(E=num_experts, BLOCK_M=BM)


@functools.cache
def _gemm2(num_experts, topk, n_split):
    from .gemm2_a4w4_bm256_half_rows import compile_moe_gemm2_mxfp4_h6144_i768
    return compile_moe_gemm2_mxfp4_h6144_i768(
        E=num_experts, topk=topk, n_split=n_split, sort_block_m=BM
    )


# one launch (both GEMMs skip a 256-row tile's upper half at <= 128 routes) wins at 4097-8192, two above (README)
GEMM12_TOKENS = (4097, 8192)
# its GEMM2 skips each wave's wholly padded 16-row epilogue groups to here: faster at 4097-7168, +0.4..0.6 us at 8192
GEMM12_EPILOGUE_GROUPS_MAX_TOKENS = 7424


def gemm12_for(n_tokens: int) -> bool:
    return GEMM12_TOKENS[0] <= n_tokens <= GEMM12_TOKENS[1]


@functools.cache
def _gemm12(num_experts, topk, n_split, epilogue_groups=False):
    from .gemm12_a4w4 import compile_moe_gemm12_mxfp4_h6144_i768
    return compile_moe_gemm12_mxfp4_h6144_i768(
        E=num_experts, topk=topk, n_split=n_split, bm=256, epilogue_groups=epilogue_groups
    )


def a4w4_experts(
    x_quant: tuple[torch.Tensor, torch.Tensor],
    w13: torch.Tensor,
    w2: torch.Tensor,
    scales: Mxfp4PrefillScales,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    """The slots ``topk_weights[:, k] * expert_{topk_ids[:, k]}(x)``, bf16 [n, topk, H], for the MXFP4 input ``x_quant``
    = (e2m1 codes [n, H/2], e8m0 [n, H/32], row-major); deterministic (no atomics into outputs)."""
    from .common.launch import (
        _run_compiled,
    )
    from .common.sort import SortBuffers
    from .common.sort_count_all import moe_sort_count_all
    from .common.sort_tile_map import moe_sort_tile_map
    from .common.scale_sort_tile_map import scale_sort_tile_map
    from .common.tile_map import (
        tile_map_grid,
    )

    xq, xs = x_quant
    n = xq.shape[0]
    num_experts = w13.shape[0]
    topk = topk_ids.shape[1]
    dev = xq.device
    stream = torch.cuda.current_stream()
    n_split = GEMM2_N_SPLIT
    b = SortBuffers.allocate(n, num_experts, topk, BM, dev)
    ids32 = topk_ids.to(torch.int32).contiguous()
    w32 = topk_weights.to(torch.float32).contiguous()
    nmb = b.max_sorted // BM
    rows = nmb * BM
    grid1 = tile_map_grid(nmb, I)
    tile_map = torch.empty((grid1 + 1,), dtype=torch.int32, device=dev)
    one_launch = n <= SORT_COUNT_ALL_MAX_TOKENS
    if one_launch:
        moe_sort_count_all(b, ids32, w32, tile_map, grid1, n, E=num_experts, I=I)
    else:
        moe_sort_tile_map(b, ids32, w32, tile_map, grid1, n, E=num_experts)
    a_s = torch.empty(
        ((b.sorted_ids.shape[0] + 31) // 32 * 32, H // 32),
        dtype=torch.uint8,
        device=dev,
    )
    scale_sort_tile_map(
        xs, b.sorted_ids, b.sorted_expert_ids, b.num_valid_ids,
        a_s, tile_map, grid1, n, BM, with_tile_map=False,
    )
    hq = torch.empty((rows, I // 2), dtype=torch.uint8, device=dev)
    hs = torch.empty((rows * (I // 32),), dtype=torch.uint8, device=dev)
    partials = torch.empty((n * topk, H), dtype=torch.bfloat16, device=dev)
    if gemm12_for(n):
        from .experts_a4w4_bm128 import _GEMM12_MAX_BLOCKS, gemm12_ready_counters
        assert nmb <= _GEMM12_MAX_BLOCKS
        _run_compiled(
            _gemm12(num_experts, topk, n_split, n <= GEMM12_EPILOGUE_GROUPS_MAX_TOKENS),
            xq.view(-1),
            w13.view(torch.uint8).view(torch.int64).view(-1),
            hq.view(-1),
            a_s.view(-1),
            scales.w13.view(-1),
            hs,
            b.sorted_ids,
            b.sorted_expert_ids,
            n,
            nmb,
            int(a_s.numel()),
            tile_map,
            grid1,
            w2.view(torch.uint8).view(torch.int64).view(-1),
            partials.view(-1),
            scales.w2.view(-1),
            b.sorted_weights,
            b.num_valid_ids,
            gemm12_ready_counters(dev).data_ptr(),
            gemm2_grid(nmb, n_split),
            stream,
        )
        partials = partials.view(n, topk, H)
        return partials
    _run_compiled(
        _gemm1(num_experts),
        xq.view(-1),
        w13.view(torch.uint8).view(torch.int64).view(-1),
        hq.view(-1),
        a_s.view(-1),
        scales.w13.view(-1),
        hs,
        b.sorted_ids,
        b.sorted_expert_ids,
        n,
        nmb,
        int(a_s.numel()),
        tile_map,
        grid1,
        stream,
    )
    _run_compiled(
        _gemm2(num_experts, topk, n_split),
        hq.view(-1),
        w2.view(torch.uint8).view(torch.int64).view(-1),
        partials.view(-1),
        hs,
        scales.w2.view(-1),
        b.sorted_ids,
        b.sorted_expert_ids,
        b.sorted_weights,
        b.num_valid_ids,
        n,
        nmb,
        gemm2_grid(nmb, n_split),
        stream,
    )
    partials = partials.view(n, topk, H)
    return partials
