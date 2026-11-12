# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""OPUS MQA API adapter for the shared DSA indexer."""


def opus_logits(
    q,
    qs,
    cache,
    cache_scale,
    weights,
    block_tables,
    query_start_loc,
    row_ends,
    *,
    max_qlen,
    max_seq_len,
    page_size=64,
    weight_scale=1.0,
    out=None,
    out_offsets=None,
):
    """The PR 6952 API, imported only for the explicitly selected new layout."""
    from aiter.ops.opus.pa_mqa_logits_mxfp4 import (
        mqa_logits_mxfp4,
        mqa_logits_mxfp4_config,
    )

    cfg = mqa_logits_mxfp4_config(
        q.device,
        heads=q.shape[-2],
        page_size=64,
        pages_per_block=page_size // 64,
        max_rows=q.shape[0],
        max_seqs=query_start_loc.numel() - 1,
        max_qlen=max_qlen,
        max_seq_len=max_seq_len,
    )
    return mqa_logits_mxfp4(
        q,
        qs,
        cache,
        cache_scale,
        weights.contiguous(),
        block_tables,
        query_start_loc,
        row_ends,
        cfg=cfg,
        weight_scale=weight_scale,
        out=out,
        out_offsets=out_offsets,
    )
