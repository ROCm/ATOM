# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Block scores over MiniMax-M3's MXFP4 index cache (``--index-cache-dtype fp4``).

aiter's FlyDSL scorers (``score_flydsl_fp4`` for decode, ``score_prefill_flydsl_fp4``
for prefill) read the cache as the two views of ``mxfp4_pages.page_views`` and write
the same ``[heads, rows, max_block]`` fp32 scores the Triton scorers do, so the
selection and the block-table emit after them are index_topk's own, unchanged.

The query is e4m3 as the fused writer emits it (cast as stored, no per-tensor
scale).

The decode scorer wants a per-step work map, built once per step in the metadata
(outside any graph, `make_sparse_decode_metadata`) into a persistent buffer. Its row
count is the grid, so every call of a step -- and every graph replay -- must pass the
``max_block`` it was built against, the model length's. The scores themselves are
allocated per call, as the Triton scorers' are: the scorer, given no ``out``,
allocates them filled with its -inf sentinel (``alloc_score``).
"""
