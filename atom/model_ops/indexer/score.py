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
row's bound; split across ranks (`RowSplit`: DCP, or TP for a TP-replicated
indexer) each rank scores its share of the rows and the selections are
all-gathered in place -- striped shares left rank-major, slot after slot.

Under a token graph only the FP4 launch breaks: its plan follows the step's
requests, which a token bucket does not fix. Everything else -- the bands, the
FP8 launch, the top-k -- follows the rows alone and is recorded.
"""

import inspect

import torch
from aiter import cp_gather_indexer_k_quant_cache, dtypes
from aiter.dist.parallel_state import get_tp_group
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl import flydsl_fp8_paged_mqa_logits, flydsl_pa_mqa_logits_fp4
from aiter.ops.topk import top_k_per_row_decode, top_k_per_row_prefill
from aiter.ops.triton.attention.pa_mqa_logits import deepgemm_fp8_paged_mqa_logits
from aiter.ops.triton.fp8_mqa_logits import fp8_mqa_logits

from atom.distributed.dcp_utils import get_dcp_group
from atom.model_ops.sparse_indexer_chunk import CONCAT_PLANE_ROW_ALIGN
from atom.utils import forward_context
from atom.utils.region_graph import eager_region

from .layout import MFMA, NATURAL
from .rows import IndexCache, IndexRows
from .workspace import LogitsWorkspace

# Whether aiter's dense FP8 scorer writes into a caller's plane (`logits=`).
# Without it each prefill band's logits are allocated per call.
FP8_MQA_LOGITS_TAKES_OUT = "logits" in inspect.signature(fp8_mqa_logits).parameters

# gfx1250 scores the FP8 plane with FlyDSL's paged MQA logits (8-row pages
# included); it reads the same packed preshuffled pages as the Triton kernel
# and takes the same arguments.
_fp8_paged_mqa_logits = (
    flydsl_fp8_paged_mqa_logits
    if get_gfx() == "gfx1250"
    else deepgemm_fp8_paged_mqa_logits
)


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
    request_local: bool = True,
) -> torch.Tensor:
    """`[rows.out_rows, topk]` int32 of `rows`' query rows against `cache`,
    each band scored into the plane the rows were cut to (`rows.workspace`).
    `request_local=False` leaves FP8 concat rows' ids as places in the
    gathered run, for a consumer that subtracts `rows.concat.row_starts`
    itself (MLA's prefill slot conversion).

    FP8: `query` [rows, heads, dim] e4m3 and `weights` [rows, heads] fp32 with
    the query scale folded in. FP4: `query` is `(values, scales)`, each
    [rows, 1, ...], and the kernel applies `weight_scale` to `weights`.

    `on_band(scores, span)`, when given, sees each band's logits before its
    top-k: a scorer that reads more than the top-k off them (a layer picking
    its successors' candidate blocks) does it there. Concat columns are local
    to the band's key window. `keys`: FP8 concat rows'
    `(values, scales)` already gathered end to end (DCP's all-gathered
    shards); gathered here from `cache` otherwise.

    A striped split (`RowSplit.striped`) scores this rank's rows of `query`
    and `weights` into its slot and returns the gathered slots rank-major:
    slot r holds rows r, r + world, ..., so row i sits at
    `(i % world) * stride + i // world` (`csa_translate_pack` reads it
    there). `out`, `on_band`, `keys` and `request_local=False`, which assume
    rows in order, do not apply."""
    split = rows.split
    if split.stripe:
        if not (out is None and on_band is None and keys is None and request_local):
            raise ValueError(
                "a striped split returns its rows rank-major, not in place"
            )
        query, weights = _stripe_rows(query, weights, cache, split)
    if out is None:
        out = torch.empty(rows.out_rows, topk, dtype=torch.int32, device=weights.device)
    mine = out[split.slot : split.slot + split.stride] if split.stripe else out
    _score_bands(
        query,
        weights,
        cache,
        rows,
        topk,
        weight_scale,
        stable,
        on_band,
        mine,
        keys,
        request_local,
    )
    _gather_split(out, split)
    return out


def _stripe_rows(query, weights, cache, split):
    """This rank's rows of a striped split, every `world`-th from its rank.
    The FP4 kernels assert dense rows, so those are copied into new ones:
    `.contiguous()` hands a single row back as is, the stripe's stride and
    all. FP8 reads them strided."""
    rows = slice(split.rank, None, split.world)
    if cache.fp4:
        values, scales = query
        return (
            (_dense(values[rows]), _dense(scales[rows])),
            _dense(weights[rows]),
        )
    return query[rows], weights[rows]


def _dense(x):
    return x.clone(memory_format=torch.contiguous_format)


def _gather_split(out, split):
    """Every rank's rows into `out`, in place: the input is this rank's own
    slot inside the output, the form RCCL supports, so apart from what RCCL
    sets aside on a group's first all-gather, the gather allocates and
    copies nothing. A slot's tail past its rank's rows is never used
    (`IndexRows.out_rows`). A world of one already holds them all."""
    if split.world == 1:
        return
    region = out[: split.stride * split.world]
    group = get_tp_group() if split.group == "tp" else get_dcp_group()
    torch.distributed.all_gather_into_tensor(
        region,
        region[split.slot : split.slot + split.stride],
        group=group.device_group,
    )


def _score_bands(
    query,
    weights,
    cache,
    rows,
    topk,
    weight_scale,
    stable,
    on_band,
    out,
    gathered=None,
    request_local=True,
):
    """Score and consume one workspace band at a time for every indexer.

    V4.1's paged/packed rows and FP8 concat rows share the buffer lifetime,
    candidate callback and top-k loop. Only key addressing, the kernel launch
    and the mapping of selected columns differ.
    """
    concat = rows.concat
    workspace = rows.workspace
    if concat is not None:
        k_fp8, k_scale = (
            _gather_concat(query, cache, concat) if gathered is None else gathered
        )
    elif workspace is None and rows.bands:
        # Standalone paged calls have no builder-owned workspace. Hold one
        # band's storage for the call and reuse it, including the short tail.
        workspace = LogitsWorkspace(
            max(end - first for first, end in rows.bands) * rows.width,
            weights.device,
        )
    group = rows.next_n
    for index, (first, end) in enumerate(rows.bands):
        count, span = end - first, slice(first, end)
        items = slice(first // group, end // group)
        lo, width, alignment = 0, rows.width, 1
        if concat is not None:
            lo, hi = rows.key_windows[index] if rows.key_windows else (0, concat.total)
            width, alignment = hi - lo, CONCAT_PLANE_ROW_ALIGN
        if rows.packed_offsets is not None:
            scores = workspace.packed_logits()
            offsets = rows.packed_offsets[span]
        else:
            scores = (
                workspace.logits(count, width, row_align=alignment)
                if workspace is not None
                else None
            )
            offsets = None
        if concat is not None:
            starts = concat.row_starts[span]
            if lo:
                # PCP padding rows (start 0, length 0) stay empty.
                starts = (starts - lo).clamp_min_(0)
            stops = starts + rows.row_ends[span]
            scores = fp8_mqa_logits(
                Q=query[span],
                KV=k_fp8[lo : lo + width],
                kv_scales=k_scale[lo : lo + width],
                weights=weights[span],
                cu_starts=starts,
                cu_ends=stops,
                clean_logits=False,
                **({"logits": scores} if scores is not None else {}),
            )
        else:
            _paged_band_logits(
                query,
                weights,
                cache,
                rows,
                index,
                span,
                items,
                scores,
                offsets,
                weight_scale,
            )
        if on_band is not None:
            on_band(scores, span)
        picked = out[span]
        if concat is not None:
            top_k_per_row_prefill(
                scores,
                starts,
                stops,
                picked,
                None,
                count,
                scores.stride(0),
                scores.stride(1),
                k=topk,
                stable=stable,
            )
            # Window columns to request-local or whole-gather columns.
            if request_local:
                torch.where(
                    picked < 0, picked, picked - starts.unsqueeze(1), out=picked
                )
            elif lo:
                torch.where(picked < 0, picked, picked + lo, out=picked)
        elif offsets is None:
            top_k_per_row_decode(
                scores,
                group,
                rows.row_ends[items],
                picked,
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
                picked,
                count,
                0,
                1,
                k=topk,
                stable=stable,
                row_starts=offsets,
                plane_width=rows.width,
            )


def _paged_band_logits(
    query,
    weights,
    cache,
    rows,
    index,
    span,
    items,
    scores,
    offsets,
    weight_scale,
):
    """A paged band's kernel launch, into the shared loop's output view."""
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
        count, group = span.stop - span.start, rows.next_n
        _fp8_paged_mqa_logits(
            query[span].view(count // group, group, weights.shape[1], query.shape[-1]),
            cache.values.unsqueeze(-2),
            weights[span],
            scores,
            rows.row_ends[items],
            rows.row_tables[items],
            rows.width,
            KVBlockSize=cache.rows_per_block,
            Preshuffle=True,
        )


def _gather_concat(query, cache, keys):
    """The step's requests' committed keys, end to end: into `keys.workspace`
    when it has one, else allocated for the step."""
    if keys.workspace is not None:
        k_fp8, k_scale = keys.workspace.keys(keys.total)
    else:
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
    if cache.layout in (NATURAL, MFMA):
        from aiter.ops.opus.pa_mqa_logits_mxfp4 import mqa_logits_mxfp4

        if weights.dtype != torch.bfloat16:
            raise TypeError(f"OPUS FP4 MQA reads BF16 weights, got {weights.dtype}")
        mqa_logits_mxfp4(
            values.squeeze(1),
            scales.squeeze(1),
            cache.values,
            cache.scales,
            weights,
            weight_scale=weight_scale,
            out=scores,
            out_offsets=offsets,
            **launch,
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
