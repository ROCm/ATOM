# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Index scoring and top-k: the one entry every DSA indexer scores through.

Paged rows (decode, and FP4 prefill) are each their own item with their own
bound: FP8 reads a row's keys off its own block-table row -- or a request's
`next_n` rows read their request's together -- and FP4 scores a request's
rows together off the request's table, sharing each key load. FP8 prefill
instead gathers its requests' keys end to end and scores them densely
(`ConcatKeys`), which reads each key once for all of a request's rows. Either
way rows run in bands of the plane they were cut to, each band top-k'd as
soon as it is scored, so nothing wider than one band is ever live. The result
is `[rows, topk]` int32 column ids local to the row's request, `-1` past a
row's bound; under DCP each rank scores its share of the rows and the
selections are all-gathered in place.

Under a token graph only the FP4 launch breaks: its plan follows the step's
requests, which a token bucket does not fix. Everything else -- the bands, the
FP8 launch, the top-k -- follows the rows alone and is recorded.
"""

import torch
from aiter import cp_gather_indexer_k_quant_cache, dtypes
from aiter.ops.flydsl import flydsl_pa_mqa_logits_fp4
from aiter.ops.opus.pa_mqa_logits_mxfp4 import pa_mqa_logits_mxfp4
from aiter.ops.topk import top_k_per_row_decode, top_k_per_row_prefill
from aiter.ops.triton.attention.pa_mqa_logits import deepgemm_fp8_paged_mqa_logits
from aiter.ops.triton.fp8_mqa_logits import fp8_mqa_logits

from atom.distributed.dcp_utils import get_dcp_group
from atom.utils import forward_context
from atom.utils.region_graph import eager_region

from .rows import NATURAL, IndexCache, IndexRows


def score_topk(
    query,
    weights: torch.Tensor,
    cache: IndexCache,
    rows: IndexRows,
    *,
    topk: int,
    weight_scale: float = 1.0,
    stable: bool = False,
    on_band=None,
    out: torch.Tensor | None = None,
    keys=None,
) -> torch.Tensor:
    """`[rows.out_rows, topk]` int32 of `rows`' query rows against `cache`,
    each band scored into the plane the rows were cut to (`rows.workspace`).

    FP8: `query` [rows, heads, dim] e4m3 and `weights` [rows, heads] fp32 with
    the query scale folded in. FP4: `query` is `(values, scales)`, each
    [rows, 1, ...], and the kernel applies `weight_scale` to `weights`.

    `on_band(scores, span)`, when given, sees each band's logits before its
    top-k: a scorer that reads more than the top-k off them (a layer picking
    its successors' candidate blocks) does it there. `keys`: FP8 concat rows'
    `(values, scales)` already gathered end to end (DCP's all-gathered
    shards); gathered here from `cache` otherwise."""
    if out is None:
        out = torch.empty(rows.out_rows, topk, dtype=torch.int32, device=weights.device)
    if rows.concat is not None:
        _score_concat(query, weights, cache, rows, topk, stable, out, keys)
    else:
        _score_paged(
            query, weights, cache, rows, topk, weight_scale, stable, on_band, out
        )
    _gather_split(out, rows.split)
    return out


def _gather_split(out, split):
    """Every DCP rank's rows into `out`, in place: the input is this rank's
    own slot inside the output, the form RCCL supports, so nothing is
    allocated or copied. The last slot's tail past the rows is never read.
    A world of one already holds them all."""
    if split.world == 1:
        return
    region = out[: split.stride * split.world]
    torch.distributed.all_gather_into_tensor(
        region,
        region[split.slot : split.slot + split.stride],
        group=get_dcp_group().device_group,
    )


def _score_paged(query, weights, cache, rows, topk, weight_scale, stable, on_band, out):
    """Paged rows, band by band into the plane, each band top-k'd at once."""
    workspace = rows.workspace
    # Without a workspace (rows cut to the logits budget alone), one band's
    # plane for the call, a short band a prefix of it.

    plane = (
        None
        if workspace is not None or not rows.bands
        else torch.empty(
            max(end - first for first, end in rows.bands),
            rows.width,
            dtype=torch.float32,
            device=weights.device,
        )
    )
    group = rows.next_n
    for index, (first, end) in enumerate(rows.bands):
        count, span = end - first, slice(first, end)
        # A band's bounds and tables: its rows', or its requests' when a
        # request's rows share them.
        items = slice(first // group, end // group)
        if rows.packed_offsets is not None:
            scores = workspace.packed_logits()
            offsets = rows.packed_offsets[span]
        else:
            scores = (
                plane[:count]
                if plane is not None
                else workspace.logits(count, rows.width)
            )
            offsets = None
        if cache.fp4:
            values, scales = query
            _fp4_band_logits(
                values[span],
                scales[span],
                weights[span],
                cache,
                scores,
                offsets,
                weight_scale,
                index,
                rows,
            )
        else:
            deepgemm_fp8_paged_mqa_logits(
                query[span].view(
                    count // group, group, weights.shape[1], query.shape[-1]
                ),
                cache.values.unsqueeze(-2),
                weights[span],
                scores,
                rows.row_ends[items],
                rows.row_tables[items],
                rows.width,
                KVBlockSize=cache.rows_per_block,
                Preshuffle=True,
            )
        if on_band is not None:
            on_band(scores, span)
        if offsets is None:
            top_k_per_row_decode(
                scores,
                group,
                rows.row_ends[items],
                out[span],
                count,
                scores.stride(0),
                scores.stride(1),
                k=topk,
                stable=stable,
            )
        else:
            # Packed: a row's logits start at its offset; the top-k is told
            # the plane width the rows stand for, so it picks as that would.
            top_k_per_row_decode(
                scores,
                1,
                rows.row_ends[span],
                out[span],
                count,
                0,
                1,
                k=topk,
                stable=stable,
                row_starts=offsets,
                plane_width=rows.width,
            )


def _gather_concat(query, cache, keys):
    """The step's requests' committed keys, end to end."""
    k_fp8 = torch.empty(
        (keys.total, query.shape[-1]), device=query.device, dtype=dtypes.fp8
    )
    k_scale = torch.empty((keys.total, 1), device=query.device, dtype=torch.float32)
    cp_gather_indexer_k_quant_cache(
        cache.values,
        k_fp8,
        k_scale.view(dtypes.fp8),  # 4-byte scale rows read as fp8 bytes
        keys.block_tables,
        keys.cu_committed,
        preshuffle=True,
    )
    return k_fp8, k_scale


def _score_concat(query, weights, cache, rows, topk, stable, out, gathered):
    """FP8 prefill against the gathered keys. The top-k reports a key's place
    in the gathered run; less its request's start it is the request's own."""
    keys = rows.concat
    k_fp8, k_scale = (
        _gather_concat(query, cache, keys) if gathered is None else gathered
    )
    ends = keys.row_starts + rows.row_ends
    for first, end in rows.bands:
        span = slice(first, end)
        # Allocated by the kernel, a band at a time: a prefill's temporaries
        # are what the memory profile measures, so they are counted already.
        scores = fp8_mqa_logits(
            Q=query[span],
            KV=k_fp8,
            kv_scales=k_scale,
            weights=weights[span],
            cu_starts=keys.row_starts[span],
            cu_ends=ends[span],
            clean_logits=False,
        )
        top_k_per_row_prefill(
            scores,
            keys.row_starts[span],
            ends[span],
            out[span],
            None,
            end - first,
            scores.stride(0),
            scores.stride(1),
            k=topk,
            stable=stable,
        )
    if rows.bands:
        scored = slice(rows.bands[0][0], rows.bands[-1][1])
        picked, base = out[scored], keys.row_starts[scored].unsqueeze(1)
        torch.where(picked < 0, picked, picked - base, out=picked)


def _live_rows():
    """The step a replayed band serves: the live forward's rows."""
    return {"rows": forward_context.get_forward_context().attn_metadata.index_rows}


@eager_region(live=_live_rows)
def _fp4_band_logits(
    values, scales, weights, cache, scores, offsets, weight_scale, band, rows
):
    """FP4 band `band`'s logits into `scores`, launched as the step planned it
    (`IndexRows.fp4_bands`): a token graph's break. `offsets`: packed rows."""
    launch = rows.fp4_bands[band]
    if cache.layout == NATURAL:
        if weights.dtype != torch.bfloat16:
            raise TypeError(f"OPUS FP4 MQA reads BF16 weights, got {weights.dtype}")
        pa_mqa_logits_mxfp4(
            values.squeeze(1),
            scales.squeeze(1),
            cache.values,
            cache.scales,
            launch["block_tables"],
            weights,
            launch["plan"],
            rows.width,
            weight_scale=weight_scale,
            kv_block_size=cache.rows_per_block,
            out=scores,
        )
        return scores
    flydsl_pa_mqa_logits_fp4(
        values,
        scales,
        cache.values,
        cache.scales,
        weights=weights,
        max_seq_len=rows.width,
        weight_scale=weight_scale,
        kv_block_size=cache.rows_per_block,
        out=scores,
        out_offsets=offsets,
        **launch,
    )
    return scores
