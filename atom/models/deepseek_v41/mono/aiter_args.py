# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""ATOM's V4.1 decode tensors as the arguments of AITER's two decode-layer ops
(``aiter/ops/deepseek_v41_decode.md``, v1): ``deepseek_v41_decode_attn_pre``
(K1) and ``deepseek_v41_decode_layer_post`` (K2).

Nothing is copied. A weight is the parameter the loaded block holds (the layout
``weights.bind_layer`` already checked); an activation is a row view of the
runner's buffers, which keep ATOM's leading batch dim of 1 and lose it here.
The per-step integers are small torch expressions over what
``step_meta.write_step_meta`` and the indexer leave, so a graph capture
records them:

- ``kv_slots`` (K1's window row, a layer): ``ring_rel`` + the layer's ring
  start, -1 kept for a pad row (the contract's absolute cache rows);
- ``window_meta`` (K2's window keys, a layer): ``key_meta``'s
  ``(first, count, -, slot row base)`` with the ring start folded into the
  base, the reserved word 0;
- ``topk_rows`` / ``topk_count`` (K2's selected keys): the scorer's
  ``step.selected`` columns as pool rows, through the block table, the way the
  original decode's ``indices._indices`` maps them, valid ones first.

Imports torch only, so it is testable on meta tensors without AITER.
"""

import torch

# DeepSeek-V4.1-Flash, as the contract sheet names them (section 1)
HC = 4
HIDDEN = 5120
Q_RANK = 1280
HEAD_DIM = 512
ROPE_HALF = 32
Q_HEADS = 64
O_GROUPS = 8
O_RANK = 1024
EXPERTS = 384
MOE_INTER = 2304
MOE_PAD = 128
INDEX_HEADS = 32
INDEX_DIM = 128
BLOCK = 32
TOPK = 512
KMETA = 4

F32, BF16, I32, I64 = torch.float32, torch.bfloat16, torch.int32, torch.int64
FP8 = torch.float8_e4m3fn
SCALES = (torch.float8_e8m0fnu, torch.uint8)
FP4 = (getattr(torch, "float4_e2m1fn_x2", torch.uint8), torch.uint8)

ATTN_PRE = "deepseek_v41_decode_attn_pre"
LAYER_POST = "deepseek_v41_decode_layer_post"


def _ceil(n: int, m: int) -> int:
    return -(-n // m) * m


def _rows2d(scale: torch.Tensor) -> torch.Tensor:
    """An MXFP4 expert scale as the contract's 2-D ``[E * N, K / 32 padded]``
    (aiter's ``shuffle_scale`` order; a 3-D ``[E, N, K']`` is the same bytes)."""
    return scale if scale.dim() == 2 else scale.reshape(-1, scale.shape[-1])


# ---------------------------------------------------------------------------
# weights: once a layer
# ---------------------------------------------------------------------------


def attn_pre_weights(weights, feeds_index: bool) -> dict:
    """K1's weight arguments of a bound layer (``weights.LayerWeights``); with
    ``feeds_index`` (a FULL / REINDEX layer) the indexer projections as well."""
    out = {
        "hc_fn": weights["hc_attn_fn"],
        "hc_scale": weights["hc_attn_scale"],
        "hc_base": weights["hc_attn_base"],
        "attn_norm_weight": weights["attn_norm.weight"],
        "wqkv_a": weights["attn.wqkv_a.weight"],
        "wqkv_a_scale": weights["attn.wqkv_a.weight_scale"],
        "q_norm_weight": weights["attn.q_norm.weight"],
        "kv_norm_weight": weights["attn.kv_norm.weight"],
        "wq_b": weights["attn.wq_b.weight"],
        "wq_b_scale": weights["attn.wq_b.weight_scale"],
    }
    if feeds_index:
        out.update(
            index_wq_b=weights["attn.indexer.wq_b.weight"],
            index_wq_b_scale=weights["attn.indexer.wq_b.weight_scale"],
            index_weights_proj=weights["attn.indexer.weights_proj.weight"],
        )
    return out


def layer_post_weights(weights) -> dict:
    """K2's weight arguments of a bound layer."""
    return {
        "attn_sink": weights["attn.attn_sink"],
        "wo_a": weights["attn.wo_a.weight"],
        "wo_a_scale": weights["attn.wo_a.weight_scale"],
        "wo_b": weights["attn.wo_b.weight"],
        "wo_b_scale": weights["attn.wo_b.weight_scale"],
        "hc_fn": weights["hc_ffn_fn"],
        "hc_scale": weights["hc_ffn_scale"],
        "hc_base": weights["hc_ffn_base"],
        "ffn_norm_weight": weights["ffn_norm.weight"],
        "gate_weight": weights["ffn.gate.weight"],
        "gate_bias": weights["ffn.gate.e_score_correction_bias"],
        "w13": weights["ffn.experts.w13_weight"],
        "w13_scale": _rows2d(weights["ffn.experts.w13_weight_scale"]),
        "w2": weights["ffn.experts.w2_weight"],
        "w2_scale": _rows2d(weights["ffn.experts.w2_weight_scale"]),
        "shared_w13": weights["ffn.shared_experts.gate_up_proj.weight"],
        "shared_w13_scale": weights["ffn.shared_experts.gate_up_proj.weight_scale"],
        "shared_w2": weights["ffn.shared_experts.w2.weight"],
        "shared_w2_scale": weights["ffn.shared_experts.w2.weight_scale"],
    }


def _dense(name, n, k):
    return {
        name: ((n, k), (FP8,)),
        f"{name}_scale": ((n // BLOCK, k // BLOCK), SCALES),
    }


def weight_shapes(tp: int, feeds_index: bool) -> tuple[dict, dict]:
    """(K1's, K2's) weight ``name -> (shape, dtypes)`` at TP ``tp``: the
    contract sheet's (sections 2, 4, 5)."""
    heads, groups = Q_HEADS // tp, O_GROUPS // tp
    inter = _ceil(MOE_INTER // tp, MOE_PAD)
    shared = MOE_INTER // tp
    mix = HC * (HC + 2)
    seam = {
        "hc_fn": ((mix, HC * HIDDEN), (F32,)),
        "hc_scale": ((3,), (F32,)),
        "hc_base": ((mix,), (F32,)),
    }
    pre = {
        **seam,
        "attn_norm_weight": ((HIDDEN,), (BF16,)),
        **_dense("wqkv_a", Q_RANK + HEAD_DIM, HIDDEN),
        "q_norm_weight": ((Q_RANK,), (BF16,)),
        "kv_norm_weight": ((HEAD_DIM,), (BF16,)),
        **_dense("wq_b", heads * HEAD_DIM, Q_RANK),
    }
    if feeds_index:
        pre.update(_dense("index_wq_b", INDEX_HEADS * INDEX_DIM, Q_RANK))
        pre["index_weights_proj"] = ((INDEX_HEADS, HIDDEN), (BF16,))
    post = {
        **seam,
        "attn_sink": ((heads,), (F32,)),
        **_dense("wo_a", groups * O_RANK, (heads // groups) * HEAD_DIM),
        **_dense("wo_b", HIDDEN, groups * O_RANK),
        "ffn_norm_weight": ((HIDDEN,), (BF16,)),
        "gate_weight": ((EXPERTS, HIDDEN), (BF16,)),
        "gate_bias": ((EXPERTS,), (F32,)),
        "w13": ((EXPERTS, 2 * inter, HIDDEN // 2), FP4),
        "w13_scale": ((EXPERTS * 2 * inter, _ceil(HIDDEN // BLOCK, 8)), SCALES),
        "w2": ((EXPERTS, HIDDEN, inter // 2), FP4),
        "w2_scale": ((EXPERTS * HIDDEN, _ceil(inter // BLOCK, 8)), SCALES),
        **_dense("shared_w13", 2 * shared, HIDDEN),
        **_dense("shared_w2", HIDDEN, shared),
    }
    return pre, post


def shape_errors(args: dict, expected: dict) -> list[str]:
    """Each of ``expected``'s arguments whose tensor in ``args`` is missing, of
    another shape or dtype, or not contiguous."""
    errors = []
    for name, (shape, dtypes) in expected.items():
        t = args.get(name)
        if t is None:
            errors.append(f"{name} missing")
        elif tuple(t.shape) != shape or t.dtype not in dtypes:
            errors.append(
                f"{name} is {tuple(t.shape)} {t.dtype}, the contract's {shape} "
                f"{'/'.join(map(str, dtypes))}"
            )
        elif not t.is_contiguous():
            errors.append(f"{name} not contiguous")
    return errors


# ---------------------------------------------------------------------------
# a call: every step
# ---------------------------------------------------------------------------


def _rope_table(table: torch.Tensor) -> torch.Tensor:
    return table if table.dim() == 2 else table.reshape(-1, table.shape[-1])


def attn_pre_kwargs(
    static: dict,
    *,
    layer_index: int,
    residual,
    pre_mix,
    fold: tuple | None,
    residual_out,
    rope,
    positions,
    kv_cache,
    kv_slots,
    gates_out: tuple,
    q_out,
    index_out: dict | None,
    aux_out,
    group,
    stream,
) -> dict:
    """Every argument of one ``attn_pre`` call, by its contract name.

    ``residual`` / ``pre_mix``: the seam's state (``SinglePassHCState``, ATOM's
    ``[1, S, ...]``). ``fold``: the owed ``(sublayer_output, post_mix,
    combination)`` or None (layer 0, an Engram layer); ``residual_out`` is
    written only with it. ``gates_out``: ``(pre, post, comb)``. ``index_out``
    (FULL / REINDEX): ``normed``, ``qr``, ``qr_scale``, ``iq``, ``iq_scale``,
    ``iw`` buffers. ATOM's row tensors (the state, the gates, ``q_out``,
    ``normed`` / ``qr`` / ``qr_scale``) come with its leading batch dim of 1;
    ``positions``, ``kv_slots``, ``aux_out`` and the index query buffers as
    the contract shapes them.
    """
    pre, post, comb = gates_out
    kw = {
        **{k: v for k, v in static.items() if not k.startswith("index_")},
        "residual": _s(residual),
        "pre_mix": _s(pre_mix),
        "cos_cache": _rope_table(rope.cos_cache),
        "sin_cache": _rope_table(rope.sin_cache),
        "positions": positions,
        "kv_cache": kv_cache,
        "kv_slots": kv_slots,
        "pre_out": _s(pre),
        "post_out": _s(post),
        "comb_out": _s(comb),
        "q_out": _s(q_out),
        "layer_index": layer_index,
        "aux_out": aux_out,
        "group": group,
        "stream": stream,
    }
    if fold is not None:
        pending, post_mix, combination = fold
        kw.update(
            sublayer_output=_s(pending),
            post_mix=_s(post_mix),
            comb_mix=_s(combination),
            residual_out=_s(residual_out),
        )
    if index_out is not None:
        kw.update(
            normed_out=_s(index_out["normed"]),
            q_latent_out=_s(index_out["qr"]),
            q_latent_scale_out=_s(index_out["qr_scale"]),
            index_wq_b=static["index_wq_b"],
            index_wq_b_scale=static["index_wq_b_scale"],
            index_weights_proj=static["index_weights_proj"],
            index_q_out=index_out["iq"],
            index_q_scale_out=index_out["iq_scale"],
            index_weights_out=index_out["iw"],
        )
    return kw


def layer_post_kwargs(
    static: dict,
    *,
    layer_index: int,
    q,
    kv_cache,
    topk_rows,
    topk_count,
    window_meta,
    positions,
    rope,
    residual,
    gates_in: tuple,
    residual_out,
    gates_out: tuple,
    ffn_out,
    softmax_scale: float,
    ring_slots: int,
    group,
    stream,
) -> dict:
    """Every argument of one ``layer_post`` call, by its contract name.
    ``gates_in``: the attention seam's ``(pre, post, comb)`` (K1's outputs),
    ``gates_out`` the FFN seam's."""
    pre, post, comb = gates_in
    pre_o, post_o, comb_o = gates_out
    return {
        **static,
        "q": _s(q),
        "kv_cache": kv_cache,
        "topk_rows": topk_rows,
        "topk_count": topk_count,
        "window_meta": window_meta,
        "positions": positions,
        "cos_cache": _rope_table(rope.cos_cache),
        "sin_cache": _rope_table(rope.sin_cache),
        "residual": _s(residual),
        "pre_mix": _s(pre),
        "post_mix": _s(post),
        "comb_mix": _s(comb),
        "residual_out": _s(residual_out),
        "pre_out": _s(pre_o),
        "post_out": _s(post_o),
        "comb_out": _s(comb_o),
        "ffn_out": _s(ffn_out),
        "softmax_scale": float(softmax_scale),
        "ring_slots": int(ring_slots),
        "layer_index": layer_index,
        "group": group,
        "stream": stream,
    }


def _s(t):
    """An ATOM row tensor ``[1, S, ...]`` as the contract's ``[S, ...]`` (a
    view; contiguous when ``t`` is)."""
    if t.shape[0] != 1:
        raise ValueError(f"expected ATOM's [1, S, ...] rows, got {tuple(t.shape)}")
    return t[0]


# ---------------------------------------------------------------------------
# per-step integers
# ---------------------------------------------------------------------------


def layer_kv_slots(ring_rel: torch.Tensor, ring_starts: torch.Tensor) -> torch.Tensor:
    """``[L, S]`` int32: each layer's cache row of each token's window row
    (``ring_rel`` + the layer's ring start), -1 where ``ring_rel`` is -1."""
    rel = ring_rel.to(I64).unsqueeze(0)
    rows = torch.where(rel >= 0, rel + ring_starts.to(I64).unsqueeze(1), -1)
    return rows.to(I32)


def layer_window_meta(
    key_meta: torch.Tensor, ring_starts: torch.Tensor
) -> torch.Tensor:
    """``[L, S, 4]`` int32: ``(first, count, 0, base)`` a layer and token, the
    base ``key_meta``'s slot row base plus the layer's ring start."""
    layers = ring_starts.shape[0]
    meta = key_meta.to(I64).unsqueeze(0).repeat(layers, 1, 1)
    meta[..., 2] = 0
    meta[..., 3] += ring_starts.to(I64).unsqueeze(1)
    return meta.to(I32)


def selection_rows(
    selected: torch.Tensor,
    batch_ids: torch.Tensor,
    block_tables: torch.Tensor,
    rows_per_page: int,
    page_rows: int,
    main_offset: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(topk_rows [S, K] int32, topk_count [S] int32)`` of a selection.

    ``selected`` ``[S, K]``: the scorer's compressed-key columns (-1 = none).
    A column's pool row is its request's PAGE (``block_tables[batch, col //
    rows_per_page]``) times ``page_rows``, plus the owner's ``main_offset`` and
    ``col % rows_per_page``, as ``indices._indices`` addresses it. Valid
    entries are packed first in their order (that kernel's ``cumsum`` rank); a
    pad row (batch id -1) has none."""
    sel = selected.to(I64)
    s, k = sel.shape
    live = batch_ids.narrow(0, 0, s) >= 0
    valid = (sel >= 0) & live.unsqueeze(1)
    cols = sel.clamp_min(0)
    batch = batch_ids.narrow(0, 0, s).to(I64).clamp_min(0).unsqueeze(1)
    pages = block_tables[batch, cols // rows_per_page].to(I64)
    rows = pages * page_rows + main_offset + cols % rows_per_page
    rank = valid.to(I64).cumsum(1) - 1
    dest = torch.where(valid, rank, k)
    packed = torch.zeros(s, k + 1, dtype=I64, device=sel.device)
    packed.scatter_(1, dest, torch.where(valid, rows, 0))
    return packed[:, :k].to(I32).contiguous(), valid.sum(1).to(I32)


def no_selection(rows: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """A window-only layer's ``(topk_rows [S, 1], topk_count [S])``: no keys."""
    return (
        torch.zeros(rows, 1, dtype=I32, device=device),
        torch.zeros(rows, dtype=I32, device=device),
    )
