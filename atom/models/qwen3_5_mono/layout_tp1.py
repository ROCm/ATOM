# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Argument table, tile counts and scratch / mailbox layouts of the Qwen3.5 TP1 mono
layer kernels (kernels_tp1.py).

As layout.py: one int64 table (``args``) of the pointers that do not change between
steps, the per-step metadata as direct kernel arguments. No peers: the hand-offs that
TP4 all-reduces across ranks gather inside the GPU, through the tagged mailbox.
"""

from atom.models.qwen3_5_mono.config_tp1 import (
    ATT_D,
    ATT_O,
    ATT_QKVG,
    BLOCKS,
    GDN_BA,
    GDN_KH,
    GDN_QKV,
    GDN_QKVZ,
    GDN_VH,
    GDN_Z,
    HIDDEN,
    INTER,
    SH_INTER,
    TOK,
    TOP_K,
)

# ---- argument table (int64 entries)
COMMON = [
    "h_in", "res_in", "h_out", "res_out", "ln1_w", "ln2_w",
    "m1_w", "sh_w2", "w13", "w13_s", "w2", "w2_s",
    "scratch", "mbox", "stamps",
    "nx_first", "nx_rows", "nx_out", "nx_m1",
]  # fmt: skip
GDN_ARGS = COMMON + [
    "g1_w", "conv_w", "dt_bias", "a_log", "gnorm_w", "out_proj",
    "conv_state", "ckpt", "rec_k", "rec_u", "rec_g",
]  # fmt: skip
ATT_ARGS = COMMON + [
    "qkvg", "q_norm_w", "k_norm_w", "o_proj", "k_cache", "v_cache",
    "cos", "sin",
]  # fmt: skip

NARGS = 32  # kernel arguments for the table (each kind's, zero-padded)
assert len(GDN_ARGS) <= NARGS and len(ATT_ARGS) <= NARGS


def arg_index(kind: str) -> dict:
    names = GDN_ARGS if kind == "gdn" else ATT_ARGS
    return {n: i for i, n in enumerate(names)}


# ---- timing builds: clock stamps a CTA a layer ([layer][BLOCKS][STAMPS] int64)
STAMPS = 32
GDN_STAMPS = ["start", "N1", "G1 in_proj+conv", "barrier", "G2 replay+recurrence", "barrier + norm",
              "barrier + out_proj", "gather 0", "barrier", "N2", "M1 router+shared",
              "barrier", "M2 top-k + M3 experts", "barrier", "M4 down", "gather 1"]  # fmt: skip
ATT_STAMPS = ["start", "N1", "A1 qkvg", "barrier", "A2 q/k + attention", "-", "barrier + A3 combine",
              "-", "barrier + o_proj", "gather 0", "barrier", "N2 + M1",
              "barrier", "M2 + M3", "barrier + M4", "gather 1"]  # fmt: skip

# ---- bf16 projections on WMMA (layout.py): a [R, K] weight packed [R / 16][K / 32]
# [32 lanes][16 bf16], one 16-row tile a CTA at a time, its waves split K.
G1_TILES = (GDN_QKVZ + GDN_BA) // 16  # in_proj_qkvz | in_proj_ba: 772
G1_QKV_TILES = GDN_QKV // 16  # 512: the conv channels (tiles b and b + 256 of CTA b)
G1_Z_END = GDN_QKVZ // 16  # 768: z tiles 512.. 767, then 4 b|a tiles
A1_TILES = ATT_QKVG // 16  # qkvg: 576
OUT_TILES = HIDDEN // 16  # out_proj / o_proj / w2: 128 (CTA b < 128 owns dims 16 b ..)
OUT_K = GDN_Z  # out_proj / o_proj input width: 4096 (== ATT_O)
M1_RT_TILES = (
    257 + 15
) // 16  # router (+ shared gate logit, row 256), padded to 272: 17
M1_SH_TILES = 2 * SH_INTER // 16  # shared gate rows 8s.. 8s+7 | up rows 8s.. 8s+7: 64
M1_TILES = M1_RT_TILES + M1_SH_TILES  # 81
assert OUT_K == ATT_O

# ---- launch counters (layout.py)
ARRIVALS = 1

# ---- scratch (f32 offsets)
LOGIT_STRIDE = 260
ACT_SLOTS = TOP_K + 1
ATT_COLS = 48  # an attention CTA's B columns (8 q heads x up to 6 tokens, padded to 16)

_SCRATCH = [  # f32, plain (bulk data published behind a flag)
    ("aml", BLOCKS * ATT_COLS * 2),  # attention partial (max, sum) a (CTA, column)
    ("aop", BLOCKS * ATT_COLS * ATT_D),  # attention partial outputs
    ("xq", TOK * HIDDEN // 4),  # N2's x as MXFP8 (M3's B), raw dwords
    ("xs", TOK * HIDDEN // 128),  # its scales, a dword a 128-wide K step
    ("actd", TOK * TOP_K * INTER // 4),  # routed activations, MXFP8 (a slot's 512 B)
    ("acts", TOK * TOP_K * INTER // 32),  # their scales, a dword a 32-block (low byte)
    ("shact", TOK * SH_INTER),  # the shared expert's activations, f32
    ("agp", TOK * HIDDEN // 2),  # N2 output, packed bf16 rows: M1 B
    ("xn1", TOK * HIDDEN // 2),  # the next layer's N1 output, packed bf16 rows
    ("gqkvp", TOK * GDN_QKV // 2),  # conv outputs q | k | v, bf16
    ("gzp", TOK * GDN_Z // 2),  # z, bf16
    ("gbap", TOK * GDN_BA),  # b, a (pairs a v head), f32
    (
        "gop",
        TOK * GDN_Z // 2,
    ),  # the recurrence output, bf16; then gated-normed in place
    ("rks", TOK * GDN_KH * 128 // 2),  # each token's normalized k (bf16): its k records
    ("rgs", TOK * GDN_VH),  # each (token, v head)'s gate: its g records
    ("qkvgp", TOK * ATT_QKVG // 2),  # attention's gate | q | k | v, bf16
    ("aop2", TOK * ATT_O // 2),  # the gated attention output, bf16
]

_MBOX = [  # LL pairs {value, tag}: offsets in pairs
    ("logits", TOK * LOGIT_STRIDE),
    ("ar0", TOK * HIDDEN // 2),  # gather 0: two bf16 a pair (out_proj / o_proj)
    ("ar1", TOK * HIDDEN // 2),  # gather 1 (M4)
    ("aflag", BLOCKS),  # an attention CTA's partials are published
    ("xqf", TOK),  # a token's N2 outputs are published
    ("sel", TOK * (TOP_K + 1) * 2),
    ("m3done", BLOCKS),  # a CTA's M3 (and M1 shared) stores have landed
    ("g1done", BLOCKS),  # a CTA's G1 / conv stores have landed
    ("g2done", BLOCKS),  # a CTA's G2 stores have landed
    ("gndone", BLOCKS),  # a CTA's gated RMSNorm stores (out_proj's input) have landed
    ("a1done", BLOCKS),  # a CTA's A1 (qkvg) stores have landed
    ("a3done", BLOCKS),  # a CTA's A3 (gated output) stores have landed
]


def _offsets(regions):
    out, off = {}, 0
    for name, n in regions:
        out[name] = off
        off += (n + 63) // 64 * 64  # aligned regions (16-byte lines stay aligned)
    return out, off


SCRATCH, SCRATCH_F32 = _offsets(_SCRATCH)
MBOX, MBOX_PAIRS = _offsets(_MBOX)
_ = (ATT_QKVG, GDN_QKVZ, GDN_VH)
