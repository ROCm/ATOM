# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Argument table, scratch and peer-buffer layouts of the Qwen3.5 mono layer kernels.

A layer kernel takes one int64 table (``args``) of the pointers that do not change
between steps (weights, scratch, caches, the activation buffers), in the order
below; the per-step metadata (positions, state slots, block table, ...) are direct
kernel arguments, so a captured graph records them like any other kernel's. Scratch is rank-local f32, reused by every layer; the
peer buffer holds the LL all-reduce slots.
"""

from atom.models.qwen3_5_mono.config import (
    ATT_D,
    ATT_QH,
    ATT_QKVG,
    BLOCKS,
    GDN_BA,
    GDN_QKV,
    GDN_Z,
    HIDDEN,
    INTER,
    MAX_TOKENS,
    MTP_MAX_TOKENS,
    TOP_K,
    TP,
)

# ---- argument table (int64 entries)
COMMON = [
    "h_in", "res_in", "h_out", "res_out", "ln1_w", "ln2_w",
    "m1_w", "sh_w2", "w13", "w13_s", "w2", "w2_s",
    "scratch", "mbox", "peers", "stamps",
    "nx_first", "nx_rows", "nx_out", "nx_m1",
]  # fmt: skip
GDN_ARGS = COMMON + [
    "g1_w", "conv_w", "dt_bias", "a_log", "gnorm_w", "out_proj",
    "conv_state", "ssm_state",
]  # fmt: skip
ATT_ARGS = COMMON + [
    "qkvg", "q_norm_w", "k_norm_w", "o_proj", "k_cache", "v_cache",
    "cos", "sin",
]  # fmt: skip


NARGS = 28  # kernel arguments for the table (each kind's, zero-padded)
assert len(GDN_ARGS) <= NARGS and len(ATT_ARGS) <= NARGS


def arg_index(kind: str) -> dict:
    names = GDN_ARGS if kind == "gdn" else ATT_ARGS
    return {n: i for i, n in enumerate(names)}


# ---- timing builds: clock stamps a CTA a layer ([layer][BLOCKS][STAMPS] int64)
STAMPS = 32  # 0..15 the stages; 16.. sub-stage probes (N1: inputs in, first barrier; A2; 24..: G3)
GDN_STAMPS = ["start", "N1", "G1 in_proj+conv", "barrier", "G2 recurrence", "barrier",
              "G3 norm+out_proj", "AR attn", "barrier (all-gather)", "N2", "M1 router+shared",
              "barrier", "M2 top-k + M3 experts", "barrier", "M4 down", "AR moe"]  # fmt: skip
ATT_STAMPS = ["start", "N1", "A1 qkvg", "barrier", "A2 q/k + attention", "barrier", "A3 combine",
              "barrier + A4 load", "o_proj", "AR attn", "barrier (all-gather)", "N2 + M1",
              "barrier", "M2 + M3", "barrier + M4", "AR moe"]  # fmt: skip

# ---- L2 prefetch of the next layer's weights (nx_*): its first projection (g1_w or
# qkvg, nx_rows rows; nx_rows = 0: none), its out_proj / o_proj and its m1_w. CTA b
# prefetches the rows CTA b will read (rows b mod BLOCKS, its 8 out rows, its M1
# task's rows), so the lines land in the L2 of the XCD that reads them.

# ---- packed weights (runner copies): one buffer a stage, so a wave picks its row by
# offset (an early load cannot depend on which of two buffers a branch picks)
#   g1_w [GDN_QKVZ + GDN_BA, HIDDEN]: in_proj_qkvz | in_proj_ba
#   m1_w [ROUTER_ROWS + 2 SH_INTER, HIDDEN]: router (256 + shared gate) | shared gate | up
M1_SH = 257  # first shared gate row in m1_w; shared up at M1_SH + SH_INTER

# ---- bf16 projections on WMMA (v_wmma_f32_16x16x32_bf16): a weight is the A operand,
# the T tokens the B operand's columns. The runner packs a [R, K] weight as
# [R / 16][K / 32][32 lanes][16 bf16]: lane l holds row 16 tile + l % 16, k = 32 s +
# 16 (l // 16) + 0..15 (rows zero-padded to 16), so a wave's fragment for a K step
# is 1 KB contiguous; one CTA a 16-row tile, its waves split K.
G1_TILES = (3072 + 16) // 16  # in_proj_qkvz | in_proj_ba: 193
A1_TILES = 2560 // 16  # qkvg: 160
OUT_TILES = (
    2048 // 16
)  # out_proj / o_proj: 128 (all-reduce 0: CTA b < 128 owns dims 16b..)
M1_RT_TILES = (
    257 + 15
) // 16  # router (+ shared gate logit, row 256), padded to 272: 17
M1_TILES = M1_RT_TILES + 16  # then shared tiles: gate rows 8s..8s+7 | up rows 8s..8s+7

# ---- launch counters: every CTA arrives once, at the end of a launch; a launch's
# index (the hand-off tags) is the sum of both kinds' counters / BLOCKS at its start
ARRIVALS = 1

# ---- scratch (f32 offsets)
LOGIT_STRIDE = 260
ACT_SLOTS = TOP_K + 1  # 8 routed, then the shared expert
ATT_CHUNKS = BLOCKS  # split-context partials a token: one a CTA
ACTQ_PAIRS = 36

# hand-off regions are sized for the largest step (MTP: MTP_MAX_TOKENS tokens); the
# attention partials for either kernel family (one a (token, chunk, head), or one a
# (CTA, B column) under MTP)
TOK = max(MAX_TOKENS, MTP_MAX_TOKENS)
ATT_PARTS = max(MAX_TOKENS * ATT_CHUNKS * ATT_QH, BLOCKS * 16)

_SCRATCH = [  # f32, plain (bulk data published behind an LL flag)
    ("aml", ATT_PARTS * 2),  # partial (max, sum) a chunk
    ("aop", ATT_PARTS * ATT_D),  # partial outputs
    (
        "xq",
        TOK * HIDDEN // 4,
    ),  # MTP: N2's x as MXFP8 (M3's B, a token's 2 KB), raw dwords
    ("xs", TOK * HIDDEN // 128),  # its scales, a dword a 128-wide K step
    (
        "actd",
        TOK * TOP_K * INTER // 4,
    ),  # MTP: routed activations, MXFP8 (a slot's 128 B)
    ("acts", TOK * TOP_K * 4),  # their scales, a dword a 32-block (low byte)
    ("shact", TOK * INTER),  # MTP: the shared expert's activations, f32
    (
        "agp",
        TOK * HIDDEN // 2,
    ),  # MTP: N2 output (all-reduce 0 + residual, RMSNorm), packed bf16 rows: M1 B
    (
        "xn1",
        TOK * HIDDEN // 2,
    ),  # MTP: the next layer's N1 output (all-reduce 1 + residual, its RMSNorm), packed bf16 rows
    ("gqkvp", TOK * GDN_QKV // 2),  # MTP: conv outputs q | k | v, bf16
    ("gzp", TOK * GDN_Z // 2),  # MTP: z, bf16
    ("gbap", TOK * GDN_BA),  # MTP: b, a (pairs a v head), f32
    ("gop", TOK * GDN_Z // 2),  # MTP: the recurrence output, bf16
    ("qkvgp", TOK * ATT_QKVG // 2),  # MTP: attention's gate | q | k | v, bf16
    ("aop2", TOK * ATT_QH * ATT_D // 2),  # MTP: the gated attention output, bf16
]

_MBOX = [  # LL pairs {value, tag}: offsets in pairs
    ("logits", TOK * LOGIT_STRIDE),
    ("act", TOK * ACT_SLOTS * INTER),
    ("ag", TOK * HIDDEN),  # the attention all-reduce, all-gathered
    ("gqkv", TOK * GDN_QKV),  # conv outputs q | k | v
    ("gz", TOK * GDN_Z),
    ("gba", TOK * GDN_BA),
    ("go", TOK * GDN_Z),  # recurrence output
    ("qkvg", TOK * ATT_QKVG),
    ("ao", TOK * ATT_QH * ATT_D),  # gated attention output
    ("aflag", TOK * ATT_CHUNKS),  # a chunk's partial is published
    ("xqf", TOK),  # MTP: a token's xq is published
    ("sel", TOK * (TOP_K + 1) * 2),
    ("m3done", BLOCKS),
    ("agdone", BLOCKS),
    ("g1done", BLOCKS),  # MTP: a CTA's G1 / conv stores have landed
    ("g2done", BLOCKS),
    ("a1done", BLOCKS),  # MTP: a CTA's A1 (qkvg) stores have landed
    (
        "a3done",
        BLOCKS,
    ),  # MTP: a CTA's A3 (gated output) stores have landed  # MTP: a CTA's G2 stores (and its G1 ones) have landed  # MTP: an all-reduce-0 owner CTA's agp rows have landed  # MTP: a CTA's M3 stores (and its M1 shared activations) have landed  # MTP: a token's top-8 (expert, weight) lines + the shared gate
    (
        "actq",
        TOK * TOP_K * ACTQ_PAIRS,
    ),  # MTP: routed activations, MXFP8 (32 dword pairs + 4 scales a slot)
]


def _offsets(regions):
    out, off = {}, 0
    for name, n in regions:
        out[name] = off
        off += (n + 63) // 64 * 64  # aligned regions (16-byte lines stay aligned)
    return out, off


SCRATCH, SCRATCH_F32 = _offsets(_SCRATCH)
MBOX, MBOX_PAIRS = _offsets(_MBOX)

# ---- peer buffer: LL pairs {two bf16, tag} [PEER_SLOTS][TP src][TOK][HIDDEN / 2]
PEER_SLOTS = 2  # all-reduce 0, all-reduce 1
PEER_BYTES = PEER_SLOTS * TP * TOK * (HIDDEN // 2) * 8


def pair_index(slot, src, t, dpair):
    """The pair holding dims (2 dpair, 2 dpair + 1) of token t from rank src."""
    return ((slot * TP + src) * TOK + t) * (HIDDEN // 2) + dpair
