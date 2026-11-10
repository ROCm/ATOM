# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# Modifications: namespace/dependency extraction for ATOM, 2026-10-13.

"""Build-time device layouts, tile geometry and task counts for V4 Mono."""

from atom.models.deepseek_v4.mono.config import (
    HEAD_DIM,
    HIDDEN,
    INTER,
    MAX_LAYERS_PER_STEP,
    N_EXPERTS,
    O_GROUPS,
    O_LORA,
    Q_LORA,
    TOP_K,
    WINDOW,
    MoeMode,
    moe_format,
)

BLOCKS = 256
LAYER_SLOTS = MAX_LAYERS_PER_STEP
THREADS = 512
WAVES = THREADS // 64
QKV_A_TILE = 16
Q_B_TILE = 16
UV_TILE = 64  # output dims merged per uv task, at least: see uv_tile
ROW_TILE = 32  # rows per o_a / o_b / attention peer-reduce tile
ROUTER_TILE = 8  # experts per router task (a part of a 16-row MFMA group)
UG_TILE = 16  # intermediates per up/gate task (16 gate rows + 16 up rows)
UG8 = 8  # intermediates one up/gate task actually owns
SPLIT_KEYS = 64
HC_CPW = 2  # hc_pre 64-K chunks per wave; sets the K split across tasks
MIN_I32 = -(1 << 31)  # flips the sign bit: signed-ordered <-> unsigned-ordered


# task counts per stage
QKV_A_ROWS = Q_LORA + HEAD_DIM
N_QKV_A = QKV_A_ROWS // QKV_A_TILE
N_ROW_TILES = HIDDEN // ROW_TILE


def dn_tile(S: int, hidden: int = HIDDEN) -> int:
    """Hidden rows per expert-down / FFN peer-reduce task: 32 at S = 1, else about one
    task per CTA, rounded up to a multiple of 16 so ``emit_dn``'s tile never straddles
    the ``DN_R * 16`` rows ``reduce_rows`` produces (V4-Pro's 28 would drop rows)."""
    return 32 if S == 1 else max(16, -(-(hidden // BLOCKS) // 16) * 16)


def qkv_a_groups(n_tiles: int) -> int:
    """16-row groups per qkv_a task: two (four waves each splitting K) when the tiles
    outnumber the grid, so the input is not staged twice."""
    return 2 if n_tiles > BLOCKS and n_tiles % 2 == 0 else 1


def q_b_groups(n_tiles: int) -> int:
    """16-row groups per q_b task, by qkv_a_groups' rule."""
    return qkv_a_groups(n_tiles)


def uv_tile(S: int, heads: int, head_dim: int) -> int:
    """Output dims merged per uv task: UV_TILE, widened until the tasks fit one round
    of the grid (every task redoes its head's split weights)."""
    t = UV_TILE
    while S * heads * head_dim // t > BLOCKS and t * 2 <= head_dim:
        t *= 2
    return t


def o_a_spt(S: int, o_groups: int, o_lora: int) -> int:
    """Samples per o_a task: two (in two MFMA B columns) once the tasks outgrow the grid."""
    return 2 if S % 2 == 0 and S * o_groups * o_lora // ROW_TILE > BLOCKS else 1


def ffn_hcc(S: int, hc_mult: int, n_experts: int = N_EXPERTS) -> bool:
    """Whether the FFN contracts its hc_mult streams in a stage of its own (hcc_f,
    publishing ``ain``) rather than in every router task. Keep this handoff for
    workloads exceeding one grid round even when a non-divisible sample count
    prevents multi-column router grouping; inline contraction spills at q7.
    """
    return hc_mult > 1 and S * (n_experts // ROUTER_TILE) > BLOCKS


def router_spt(S: int, n_experts: int = N_EXPERTS) -> int:
    """Samples per router task, retaining whole groups of live samples.

    Prefer enough columns to fit the grid in one round, but never create a
    partial group: router inputs and score publication have no padded rows.
    Non-divisible verify widths use more task rounds instead.
    """
    n_router = n_experts // ROUTER_TILE
    spt = 1
    while S * n_router > BLOCKS * spt and spt < ROUTER_TILE:
        spt *= 2
    while S % spt:
        spt //= 2
    if S > WAVES:
        # Four hidden-wide normalized columns keep too many inputs live next
        # to router weight prefetches. More two-column tasks preserve every
        # row's K reduction and publication while avoiding scratch at S16.
        spt = min(spt, 2)
    return spt


def down_route_rounds(samples: int) -> int:
    """Wave-uniform rounds needed to select routes for every live sample."""
    return -(-samples // WAVES)


def down_route_samples(samples: int, tokens_per_seq: int, bf16_routes: bool) -> int:
    """Rows sharing a routed-down K partition, not a launch/admission limit.

    The wide q6 partition changes a request's K split relative to its qualified
    narrow execution. Keep whole q6 requests separate internally so per-route
    BF16 rounding sees the same full-K sum; other schedules are unchanged.
    """
    if bf16_routes and samples > WAVES and tokens_per_seq == 6:
        return tokens_per_seq
    return samples


def route_key_capacity(samples: int, top_k: int = TOP_K) -> int:
    """Shared key storage serves attention tiles and complete MoE route sets."""
    return max(SPLIT_KEYS, samples * (1 + top_k))


N_UG_PER_SLOT = INTER // UG_TILE


# CPol scope: SC1 = device (past the per-XCD caches), SC0|SC1 = system (peers over XGMI)
POLL_MAX = 12  # mailbox specs polled per batch
# opt-in poll bound (poll_timeout_us): a poll waiting this long gives up and flags `hang`
POLL_TIMEOUT_US = 10_000_000
TL_COLS = (
    5  # timeline stamps per task: start, hint seen, inputs staged, compute done, end
)


def _align(n, a=256):
    return (n + a - 1) // a * a


def hc_shape(hc_mult: int, hidden: int):
    """(tasks per side, K per task, rows, values per task) for hc_pre: K = hc * hidden is
    split across tasks, each also publishing a partial sum of squares as an extra row.
    """
    if hc_mult <= 1:
        return 0, 0, 0, 0
    rows = ((2 + hc_mult) * hc_mult + 15) // 16 * 16
    k_total = hc_mult * hidden
    n_tasks = k_total // (64 * WAVES * HC_CPW)
    return n_tasks, k_total // n_tasks, rows, rows + 1


def layout(
    S: int,
    heads: int,
    npes: int,
    window: int = WINDOW,
    moe_mode: MoeMode | str = MoeMode.A8W4,
    hidden: int = HIDDEN,
    q_lora: int = Q_LORA,
    head_dim: int = HEAD_DIM,
    o_groups: int = O_GROUPS,
    o_lora: int = O_LORA,
    hc_mult: int = 1,
    compress_ratio: int = 0,
    n_keys: int | None = None,
    c_coff: int = 1,
    index_head_dim: int = 0,
    index_heads: int = 0,
    # unused here: all three builders take the same dims dict
    index_topk: int = 0,
    max_seq: int = 0,
    kv_fp8: bool = False,
    shared_bf16_block128: bool = False,
    routed_down_bf16: bool = False,
    compressor_projection_bf16: bool = False,
    attention_fp8_bf16: bool = False,
    qkv_norm_handoff: bool = False,
    indexer_hadamard: bool = True,
    indexer_mfma_scales: bool = False,
    n_experts: int = N_EXPERTS,
    top_k: int = TOP_K,
    inter: int = INTER,
):
    """Byte offsets of the per-rank scratch and of the symmetric buffer.

    Every mailbox holds ``(value, tag)`` int32 pairs (8 bytes per element)."""
    fmt = moe_format(moe_mode)
    quant_group = fmt.activation_group
    xq_blocks = 0 if quant_group is None else hidden // quant_group
    n_split = (window if n_keys is None else n_keys) // SPLIT_KEYS
    hc_tasks, _, _hc_rows, hc_vals = hc_shape(hc_mult, hidden)
    hc_coef = 2 * hc_mult + hc_mult * hc_mult  # pre | post | comb
    pr = 8
    items = [
        ("hc_d", S * 2 * max(hc_tasks, 1) * max(hc_vals, 1) * pr),
        ("hc_c", S * 2 * max(hc_coef, 1) * pr),
        # hc_pre's output: the hc_mult streams contracted by `pre` to one
        ("xin", S * hidden * pr if hc_mult > 1 else pr),
        # Ordinary fused mHC norm squares FP32 collapse, but consumes its
        # BF16 numerator. Explicit tagged plane, shared by the same variant.
        *([("xin_raw", S * hidden * pr)] if attention_fp8_bf16 and hc_mult > 1 else []),
        # Optional packed-BF16 attention input.  A small producer stage writes
        # the normalized input once; qkv projection CTAs consume this stable
        # handoff instead of carrying RMSNorm values beside weight vectors.
        *([("qkv_x", S * hidden // 2 * pr)] if qkv_norm_handoff else []),
        # The FP8 q_a/kv projection consumes the E8M0/128 round trip while the
        # compressors' BF16 projection consumes qkv_x before that quantization.
        *([("qkv_xq", S * hidden // 2 * pr)] if qkv_norm_handoff else []),
        (
            "ain",
            S * hidden * pr if ffn_hcc(S, hc_mult, n_experts) else pr,
        ),  # the FFN side's, see ffn_hcc
        *(
            [("ain_raw", S * hidden * pr)]
            if attention_fp8_bf16 and ffn_hcc(S, hc_mult, n_experts)
            else []
        ),  # preserve the fused mHC RMS denominator across the FFN handoff
        ("q_a", S * q_lora * pr),
        ("kv_a", S * head_dim * pr),  # the single shared KV row, pre-norm
        # the compressor's kv / gate from the fused qkv_a GEMV (c_coff-wide: overlap)
        ("c_kv", S * c_coff * head_dim * pr if compress_ratio else pr),
        ("c_gate", S * c_coff * head_dim * pr if compress_ratio else pr),
        ("i_kv", S * c_coff * index_head_dim * pr if index_head_dim else pr),
        ("i_gate", S * c_coff * index_head_dim * pr if index_head_dim else pr),
        # rows this launch just wrote (cnew, kvnew, i_cnew): a CTA cannot rely on
        # seeing its own global store, so readers take them from these mailboxes
        ("cnew", S * head_dim * pr if compress_ratio else pr),
        ("kvnew", S * head_dim * pr),  # this launch's KV ring rows (bf16 values)
        ("q_raw", S * heads * head_dim * pr),  # q_b output, before the per-head RMS
        # the indexer's query, raw then rotated / Hadamard / FP4
        ("i_q_raw", S * index_heads * index_head_dim * pr if index_head_dim else pr),
        ("i_q", S * index_heads * index_head_dim * pr if index_head_dim else pr),
        ("i_wp", S * index_heads * pr if index_head_dim else pr),
        ("i_cnew", S * index_head_dim * pr if index_head_dim else pr),
        # one score per compressed entry; entries not yet written score NEG
        ("i_score", S * max(n_compressed(max_seq, compress_ratio), 1) * pr),
        # the indexer's picks, gathered instead of the caller's compressed indices
        ("i_sel", S * max((n_keys or window) - window, 1) * pr),
        # the top-k parts' bins, one set per radix digit
        (
            "tk_hist",
            S * 4 * n_topk_parts(max_seq, compress_ratio, index_head_dim) * 256 * pr
            or pr,
        ),
        ("q", S * heads * head_dim * pr),  # full per-head query: rope is inside it
        ("sp_acc", S * n_split * heads * head_dim * pr),
        ("sp_m", S * n_split * heads * pr),
        ("sp_l", S * n_split * heads * pr),
        ("o", S * heads * head_dim * pr),  # merged, de-rotated attention output
        ("o_lora", S * o_groups * o_lora * pr),
        (
            "a",
            S * max(hc_mult, 1) * hidden * pr,
        ),  # post-attention residual stream (bf16)
        ("scores", S * n_experts * pr),
        ("xq", S * hidden // (4 if quant_group is not None else 2) * pr),
        ("xqs", S * xq_blocks * pr),
        # Independent shared input, BEFORE routed MXFP8/32 quantization.
        # Packed BF16 pairs; the shared consumer quantizes each 128 group.
        *([("sx", S * hidden // 2 * pr)] if shared_bf16_block128 else []),
        ("sel", S * (1 + top_k) * pr),
        ("prob", S * (1 + top_k) * pr),
        ("mid", S * (1 + top_k) * inter * pr),
        # Plain-f32 observer plane, excluded from tagged-pair scrub. Routed
        # input first, then real shared UG and down ACTUALLY staged operands.
        # Real CSA appends its normalized BF16 index-weights input/output for
        # fixed-operand attribution; these stores are not performance-ready.
        (
            "xqd",
            S
            * (
                hidden
                + (hidden + inter if shared_bf16_block128 else 0)
                + (hidden + index_heads if attention_fp8_bf16 and index_head_dim else 0)
            )
            * 4,
        ),
    ]
    off, scratch = 0, {}
    for name, size in items:
        scratch[name] = off
        off += _align(size)
    scratch["_bytes"] = off
    part = npes * S * hidden * pr
    sym = {"attn": 0, "ffn": part, "_bytes": 2 * part}
    return scratch, sym


SCORE_TILE = 512  # one candidate per thread
IH_TASK = WAVES  # index heads per i_q / i_wp task (i_q: one per wave)


def n_compressed(max_seq: int, compress_ratio: int) -> int:
    return max_seq // compress_ratio if compress_ratio else 0


def n_index(
    max_seq: int, compress_ratio: int, index_head_dim: int, index_topk: int
) -> int:
    """Compressed slots the attention can gather: the indexer's pick, capped."""
    if not index_head_dim:
        return 0
    return min(index_topk, n_compressed(max_seq, compress_ratio))


def n_topk_parts(max_seq: int, compress_ratio: int, index_head_dim: int) -> int:
    """CTAs the indexer's top-k splits its candidates over; the parts agree on each
    radix digit by summing one another's bins. Capped, since that exchange grows
    quadratically with the part count."""
    if not index_head_dim:
        return 0
    return min(16, max(1, n_compressed(max_seq, compress_ratio) // 4096))


def n_score_tiles(max_seq: int, compress_ratio: int, index_head_dim: int) -> int:
    """Tiles of compressed entries the indexer scores, sized for the whole cache;
    entries not yet written score NEG."""
    if not index_head_dim:
        return 0
    n = n_compressed(max_seq, compress_ratio)
    return (n + SCORE_TILE - 1) // SCORE_TILE


def qkv_a_rows(
    q_lora: int,
    head_dim: int,
    compress_ratio: int,
    c_coff: int,
    index_head_dim: int = 0,
    index_topk_refresh: bool = True,
) -> int:
    """Rows of the fused qkv_a GEMV: q_a, kv, the compressor's c_coff-wide pair and
    the indexer compressor's pair (reference.qkv_a_tail())."""
    if not compress_ratio:
        return q_lora + head_dim
    index_width = index_head_dim if index_topk_refresh else 0
    tail = 2 * c_coff * (head_dim + index_width)
    return q_lora + head_dim + tail


def stage_tasks(
    S: int,
    heads: int,
    window: int = WINDOW,
    hidden: int = HIDDEN,
    q_lora: int = Q_LORA,
    head_dim: int = HEAD_DIM,
    o_groups: int = O_GROUPS,
    o_lora: int = O_LORA,
    top_k: int = TOP_K,
    inter: int = INTER,
    hc_mult: int = 1,
    compress_ratio: int = 0,
    n_keys: int | None = None,
    c_coff: int = 1,
    index_head_dim: int = 0,
    index_heads: int = 0,
    index_topk: int = 0,
    max_seq: int = 0,
    kv_fp8: bool = False,
    shared_bf16_block128: bool = False,
    routed_down_bf16: bool = False,
    compressor_projection_bf16: bool = False,
    attention_fp8_bf16: bool = False,
    qkv_norm_handoff: bool = False,
    indexer_hadamard: bool = True,
    indexer_mfma_scales: bool = False,
    index_topk_refresh: bool = True,
    n_experts: int = N_EXPERTS,
):
    """[(stage name, task count)] in execution order."""
    n_qkv_a = (q_lora + head_dim) // QKV_A_TILE
    n_qkv_c = (
        qkv_a_rows(
            q_lora,
            head_dim,
            compress_ratio,
            c_coff,
            index_head_dim,
            index_topk_refresh,
        )
        - q_lora
        - head_dim
    ) // QKV_A_TILE
    hc_tasks, _, _, _ = hc_shape(hc_mult, hidden)
    n_iqb = index_heads * index_head_dim // Q_B_TILE
    return [
        ("hcd_a", S * hc_tasks),
        ("hcc_a", (hidden // ROW_TILE) if hc_mult > 1 else 0),
        ("qkv_norm", S if qkv_norm_handoff else 0),
        ("qkv_a", n_qkv_a // qkv_a_groups(n_qkv_a)),
        # the compressors' projections, BF16 as ATOM
        ("qkv_c", n_qkv_c // qkv_a_groups(n_qkv_c)),
        ("cache", 1),
        ("cmp", S if compress_ratio else 0),
        ("i_cmp", S if index_head_dim and index_topk_refresh else 0),
        (
            "q_b",
            heads * head_dim // Q_B_TILE // q_b_groups(heads * head_dim // Q_B_TILE),
        ),
        ("q_norm", S * heads),
        (
            "i_q_b",
            n_iqb // q_b_groups(n_iqb) if index_head_dim and index_topk_refresh else 0,
        ),
        (
            "i_q",
            S * index_heads // IH_TASK if index_head_dim and index_topk_refresh else 0,
        ),
        (
            "i_wp",
            S * index_heads // IH_TASK if index_head_dim and index_topk_refresh else 0,
        ),
        (
            "i_score",
            (
                S * n_score_tiles(max_seq, compress_ratio, index_head_dim)
                if index_topk_refresh
                else 0
            ),
        ),
        (
            "i_topk",
            (
                S * n_topk_parts(max_seq, compress_ratio, index_head_dim)
                if index_topk_refresh
                else 0
            ),
        ),
        ("split", S * ((window if n_keys is None else n_keys) // SPLIT_KEYS)),
        ("uv", S * (heads * head_dim // uv_tile(S, heads, head_dim))),
        ("o_a", S * o_groups * o_lora // ROW_TILE // o_a_spt(S, o_groups, o_lora)),
        ("o_b", hidden // ROW_TILE),
        ("hcd_f", S * hc_tasks),
        ("hcc_f", (hidden // ROW_TILE) if ffn_hcc(S, hc_mult, n_experts) else 0),
        ("router", S * (n_experts // ROUTER_TILE) // router_spt(S, n_experts)),
        # one tile per (routed slot, 8 intermediates); the first INTER / UG8 also carry the shared expert
        ("ug", S * top_k * (inter // UG8)),
        ("down", hidden // dn_tile(S, hidden)),
    ]
