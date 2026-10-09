# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Debug aid (``ATOM_MONO_CHECK``, eager only): each layer's K1, K2a and K2b
outputs against the original modules run on the same inputs, logged on rank 0."""

import logging

import torch
from aiter.ops.batched_gemm_op_a8w8 import batched_gemm_a8w8_mxscale_bpreshuffle
from aiter.ops.inverse_rope_group_quant import inverse_rope_group_quant

from atom.model_ops.blockscale import quantize_fp8
from atom.model_ops.deepseek_v41.mhc_pre_delayed import pre_delayed
from atom.model_ops.deepseek_v41.paged_scoring import quantize_query_rows
from atom.model_ops.deepseek_v41.rope_window import rope_quant_window
from atom.model_ops.v4_kernels.indexer_weights import scale_indexer_weights
from atom.model_ops.v4_kernels.paged_decode import _sparse_attn_v4_paged_decode_triton
from atom.models.deepseek_v41.mono import index_plan as ip
from atom.models.deepseek_v41.mono.kernels import attention as attention_kernel
from atom.models.deepseek_v41.mono.kernels import attn_post as k2a
from atom.models.deepseek_v41.mono.kernels import layer_post as k2
from atom.models.deepseek_v41.mono.kernels import moe as k2b
from atom.models.deepseek_v41.mono.kernels.dims import HEAD_TILE
from atom.mono.runtime.compare import compare

logger = logging.getLogger("atom")


def _diff(name, got, ref):
    return f"{name} {compare(got, ref)}"


def _bad_rows(name, got, ref):
    """The rows (tokens) where ``got`` and ``ref`` differ."""
    bad = (got.float() != ref.float()).flatten(1).any(dim=1)
    return f"{name} rows {bad.nonzero().flatten().tolist()}"


def _mailbox(runner, layout, region, rows):
    """A scratch mailbox region's value words (its pairs' first words), one row a
    token, as int32."""
    off, nbytes = layout[region]
    pairs = runner.scratch[off : off + nbytes].view(torch.int32).view(-1, 2)
    return pairs[:, 0].view(rows, -1)


def _irq_ref(block, attention, rope, cache, step):
    """The original inverse-RoPE MXFP8 quant of ``attention``: (words, codes)."""
    attn = block.attn
    rows = attention.shape[:-2].numel()
    return inverse_rope_group_quant(
        attention.reshape(rows, attn.heads, attn.head_dim),
        cache.rope_positions(step).flatten(),
        rope.cos_cache,
        rope.sin_cache,
        num_groups=attn.groups,
        quant_group_size=32,
    )


def _k2a_stages(runner, block, attention, rope, cache, step):
    """K2a's hand-offs before wo_b against the original's: the inverse-RoPE
    MXFP8 row and its codes, and wo_a's output as wo_b reads it (``quantize_fp8``
    of the bf16 y: FP8 words and group codes)."""
    attn = block.attn
    rows = attention.shape[:-2].numel()
    layout = k2a.scratch_layout(k2a.AttnPostBuild(tokens=rows, tp=runner.tp))
    x_fp8, x_scale = _irq_ref(block, attention, rope, cache, step)
    y = batched_gemm_a8w8_mxscale_bpreshuffle(
        x_fp8,
        attn.wo_a.weight.view(attn.groups, attn.o_rank, -1),
        x_scale,
        attn.wo_a.weight_scale.view(attn.groups, attn.o_rank // 32, -1),
        dtype=attention.dtype,
    )
    y8, y_scale = quantize_fp8(y.reshape(rows, -1))
    xo = _mailbox(runner, layout, "xo", rows)
    return [
        _bad_rows("irq", xo, x_fp8.view(rows, -1).view(torch.int32)),
        _diff("irq", xo, x_fp8.view(rows, -1).view(torch.int32)),
        _diff(
            "irq codes",
            _mailbox(runner, layout, "xos", rows),
            x_scale.reshape(rows, -1).view(torch.uint8).int(),
        ),
        _diff("wo_a", _mailbox(runner, layout, "x8b", rows), y8.view(torch.int32)),
        _diff(
            "wo_a codes",
            _mailbox(runner, layout, "x8bs", rows),
            y_scale.view(torch.uint8).int(),
        ),
    ]


def _index_projections(runner, attn, rope, cache, step):
    """K1's indexer query / scale / head weights against ``Indexer.project`` and
    the scorer's own quant and weight scaling, on K1's own (qr, normed)."""
    indexer = attn.indexer
    query, weights = indexer.project(
        runner.normed, runner.qr, runner.qr_scale, cache, step, rope
    )
    if runner.index_fp4:
        # an FP4 `project` returns the query quantized with its RoPE
        values, scales = query
        mono_values, mono_scales = runner.index_query()
        return [
            _diff("iq", mono_values, values),
            _diff("iq scale", mono_scales, scales),
            _diff("iw", runner.iw, weights.float() * indexer.weights_scale),
        ]
    q_fp8, q_scale = quantize_query_rows(query)
    rows, heads = weights.shape
    scaled = scale_indexer_weights(
        weights.contiguous(), q_scale.view(rows, heads, 1), indexer.weights_scale
    )
    return [
        _diff(
            "iq",
            runner.iq.view(torch.uint8),
            q_fp8.view(torch.uint8).view_as(runner.iq),
        ),
        _diff("iq scale", runner.iq_scale, q_scale.view(rows, heads)),
        _diff("iw", runner.iw, scaled),
    ]


def check_front(runner, layer_id, block, state, rope, cache, step, res_out, gates):
    """The original front (the seam with attn_norm, wqkv_a, q / kv norms, wq_b,
    the window row) on ``state``, against what K1 left."""
    attn = block.attn
    fold = {}
    if state.pending is not None:
        fold = {
            "sublayer_output": state.pending,
            "post_mix": state.post_mix,
            "combination": state.combination,
        }
    residual, normed, pre, post, comb = pre_delayed(
        state.residual,
        state.pre_mix,
        block.hc_attn_fn,
        block.hc_attn_scale,
        block.hc_attn_base,
        block.attn_norm.weight,
        **block.hc_options,
        norm_eps=block.attn_norm.eps,
        **fold,
    )
    q_lora, kv_pre = attn.project_qkv(normed)
    qr, qr_scale, kv_normed = attn.qk_norm(q_lora, kv_pre)
    query = attn.wq_b(qr, x_scale=qr_scale).unflatten(-1, (attn.heads, attn.head_dim))
    width = runner.q.shape[1]
    qat = torch.empty(width, attn.head_dim, device=query.device, dtype=query.dtype)
    rope_quant_window(
        query.view(width, attn.heads, attn.head_dim),
        kv_normed.view(width, attn.head_dim),
        rope.cos_cache,
        rope.sin_cache,
        step.positions,
        rope_dim=rope.rope_dim,
        qat=qat,
    )
    rows = (
        runner.ring_rel.long()
        + cache.geometry.window(layer_id, cache.num_pages).ring_start
    )
    ring = cache.pool[rows.clamp_min(0)]
    parts = [
        _diff("residual", res_out, residual),
        _diff("pre", gates[0], pre),
        _diff("post", gates[1], post),
        _diff("comb", gates[2], comb),
        _diff("q", runner.q, query),
        _diff("window", ring, qat),
    ]
    if attn.indexer is not None:
        parts.append(_diff("normed", runner.normed, normed))
        parts.append(_diff("qr", runner.qr.view(torch.uint8), qr.view(torch.uint8)))
        parts += _index_projections(runner, attn, rope, cache, step)
    if runner.rank == 0:
        logger.info("V4.1 mono check layer %d: %s", layer_id, "; ".join(parts))


def _key_set_rows(runner, spec, keys, cache):
    """The tokens whose pool rows the mono attention reads (``attention.
    _key_row``: its selection's rows, then its window's) differ as a set from
    the original index build's."""
    prefix, indptr = keys
    prefix, indptr = prefix.tolist(), indptr.tolist()
    meta = runner.key_meta.tolist()
    window = cache.geometry.window(spec.layer_id, cache.num_pages)
    owner = spec.topk_owner if spec.ratio else None
    bad = []
    for t, (first, count, _, base) in enumerate(meta):
        rows = []
        if owner is not None:
            n_sel = ip.selected(int(runner.bounds[owner][t]))
            rows = runner.irow[owner][t, :n_sel].tolist()
        rows += [
            base + window.ring_start + (first + k) % window.ring_slots
            for k in range(count)
        ]
        if sorted(rows) != sorted(prefix[indptr[t] : indptr[t + 1]]):
            bad.append(t)
    return f"key rows {bad}"


def _attention_ref(runner, block, keys, cache):
    """The original sparse decode over the mono attention's keys, ``ORDER_ROWS``
    rows and a 16-head tile at a time: its reduce order follows the rows and
    heads of a call (``kernels.attention``), and the mono attention keeps that
    call's at every width and TP size. A short last call repeats its last row,
    and a ragged tile's dead heads its last live head, as the mono attention's
    do."""
    prefix, indptr = keys
    t = attention_kernel.ORDER_ROWS
    q = runner.q.flatten(0, 1)
    rows, heads = q.shape[:2]
    tiles = runner.dims.head_tiles.count
    padded = torch.arange(tiles * HEAD_TILE, device=q.device).clamp(max=heads - 1)
    q, sink = q[:, padded], block.attn.attn_sink[padded]
    bounds = indptr.tolist()
    out = []
    for i in range(0, rows, t):
        n = min(t, rows - i)
        # the call's rows, the last one repeated up to t
        picks = [i + min(r, n - 1) for r in range(t)]
        ids = torch.cat([prefix[bounds[r] : bounds[r + 1]] for r in picks])
        lens = [bounds[r + 1] - bounds[r] for r in picks]
        ptr = torch.tensor([0] + lens, device=indptr.device).cumsum(0)
        call = [
            _sparse_attn_v4_paged_decode_triton(
                q[picks, h : h + HEAD_TILE].contiguous(),
                cache.pool,
                ids,
                ptr.to(indptr.dtype),
                sink[h : h + HEAD_TILE].contiguous(),
                block.attn.softmax_scale,
                block_h=HEAD_TILE,
                kv_splits=attention_kernel.SPLITS,
            )
            for h in range(0, tiles * HEAD_TILE, HEAD_TILE)
        ]
        out.append(torch.cat(call, dim=1)[:n])
    return torch.cat(out)[:, :heads].reshape(runner.q.shape)


def check_post(runner, block, keys, residual, gates, rope, cache, step):
    """The original sparse decode over the same keys, output projections (their
    all-reduce included), FFN seam and ffn_norm, against what K2a left. Collective:
    every rank runs it, twice: the reference against itself tells a mono
    mismatch from the reference's own run-to-run change."""
    attention = _attention_ref(runner, block, keys, cache)
    out = block.attn._project_out(attention, rope, cache.rope_positions(step))
    again = block.attn._project_out(attention, rope, cache.rope_positions(step))
    pre, post, comb = gates
    res, normed, pre_f, post_f, comb_f = pre_delayed(
        residual,
        pre,
        block.hc_ffn_fn,
        block.hc_ffn_scale,
        block.hc_ffn_base,
        block.ffn_norm.weight,
        **block.hc_options,
        norm_eps=block.ffn_norm.eps,
        sublayer_output=out,
        post_mix=post,
        combination=comb,
    )
    g = runner.gates_ffn
    parts = [_key_set_rows(runner, block.attn.spec, keys, cache)]
    parts += _k2a_stages(runner, block, attention, rope, cache, step) + [
        _diff("ref self", again, out),
        _diff("residual", runner.res_ffn, res),
        _diff("pre", g[0], pre_f),
        _diff("post", g[1], post_f),
        _diff("comb", g[2], comb_f),
        _diff("normed", runner.ffn_normed, normed),
    ]
    if runner.rank == 0:
        logger.info("V4.1 mono check post %s: %s", block.layer_name, "; ".join(parts))


def check_moe(runner, block):
    """The original MoE (its all-reduce included) on the same input, against
    K2b's output. Collective: every rank runs it, twice (see ``check_post``)."""
    ref = block.ffn(runner.ffn_normed)
    again = block.ffn(runner.ffn_normed)
    # K2b's router logits: LOGIT's (value, tag) pairs, each logit's f32 K part
    # sums, added in part order and rounded to bf16 as ``route`` does
    x = runner.ffn_normed.reshape(-1, runner.ffn_normed.shape[-1])
    # K2b's regions where either launch shape puts them (``layer_post.moe_base``)
    layout = k2.scratch_layout(runner._post_key(x.shape[0]))
    off, nbytes = layout["logit"]
    pairs = runner.scratch[off : off + nbytes].view(torch.float32).view(-1, 2)
    parts = pairs[:, 0].view(x.shape[0], -1, k2b.ROUTER_PARTS)
    logits = torch.zeros_like(parts[..., 0])
    for p in range(k2b.ROUTER_PARTS):
        logits = logits + parts[..., p]
    logits = logits.to(torch.bfloat16)
    # the union U of every token's routed experts: what the step's ug and down read
    off, nbytes = layout["route"]
    route = runner.scratch[off : off + nbytes].view(torch.int32).view(-1, 2)[:, 0]
    ids = route.view(k2b.ROUTE_COPIES, x.shape[0], 2, runner.topk)[0, :, 0]
    parts = [
        _diff("ref self", again, ref),
        _diff("logits", logits, block.ffn.router_logits(x)),
        _diff("ffn out", runner.ffn_out, ref),
        f"|U| {ids.unique().numel()}",
    ]
    if runner.rank == 0:
        logger.info("V4.1 mono check moe %s: %s", block.layer_name, "; ".join(parts))


def check_selection(runner, spec, step):
    """K2a's selection (and a candidate producer's blocks) against the original
    scorer's, run on the same inputs before K2a."""
    layer = spec.layer_id
    parts = [_diff("selected", runner.isel[layer], step.selected[layer][0])]
    if spec.produces_candidates:
        parts.append(_diff("candidates", runner.icand, step.candidates[layer]))
    if runner.rank == 0:
        logger.info("V4.1 mono check index %d: %s", layer, "; ".join(parts))
