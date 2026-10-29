# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Shapes the Qwen3.5-35B-A3B TP1 mono kernels (kernels_tp1.py) are built for: the
whole model on one gfx1250 GPU, MTP verify steps on ReplaySSM state.

Every value is checked against the loaded model when the runner binds; a model that
does not match never takes the mono path.
"""

# execution model: one CTA a CU, all co-resident, THREADS threads (wave32)
BLOCKS = 256
THREADS = 512
WAVE = 32
WAVES = THREADS // WAVE

HIDDEN = 2048
EPS = 1e-6

# tokens travel in groups of 16 (the columns of one WMMA B tile); at most two groups
TG = 16
MAX_GROUPS = 2
TOK = TG * MAX_GROUPS

# GDN (linear attention)
GDN_KH = 16  # k heads
GDN_VH = 32  # v heads
GDN_D = 128  # k and v head dim
GDN_QKV = (2 * GDN_KH + GDN_VH) * GDN_D  # conv channels: q | k | v = 8192
GDN_Z = GDN_VH * GDN_D  # 4096
GDN_QKVZ = GDN_QKV + GDN_Z  # in_proj_qkvz rows = 12288
GDN_BA = 2 * GDN_VH  # in_proj_ba rows: b 32 | a 32
CONV_K = 4  # conv taps; the window keeps CONV_K - 1 + the drafts

# full attention: gate | q | k | v rows of qkv_proj
ATT_QH = 16
ATT_KVH = 2
ATT_GQA = ATT_QH // ATT_KVH  # 8 q heads a kv head
ATT_D = 256
ATT_ROT = 64  # rotary dims (NeoX halves of 32)
ATT_QKVG = 2 * ATT_QH * ATT_D + 2 * ATT_KVH * ATT_D  # 9216
ATT_G_OFF = 0
ATT_Q_OFF = ATT_QH * ATT_D  # 4096
ATT_K_OFF = 2 * ATT_QH * ATT_D  # 8192
ATT_V_OFF = ATT_K_OFF + ATT_KVH * ATT_D  # 8704
ATT_O = ATT_QH * ATT_D  # o_proj input width: 4096
ATT_BLOCK = 16  # KV cache block (tokens)
ATT_SCALE = ATT_D**-0.5

# MoE
N_EXPERTS = 256
TOP_K = 8
INTER = 512  # an expert's intermediate (gate / up rows each)
SH_INTER = 512  # the shared expert's
ROUTER_ROWS = N_EXPERTS + 1  # + the shared-expert gate row

# MXFP4 expert weights in ATOM's gfx1250 FlyDSL MoE layout (bytes an expert)
W13_ROWS = 2 * INTER  # gate / up interleaved: 1024 rows
W13_EXPERT_BYTES = W13_ROWS * (HIDDEN // 2)  # 1 MiB
W13S_EXPERT_BYTES = (W13_ROWS // 32) * 2048  # e8m0 scales: 64 KiB
W2_EXPERT_BYTES = HIDDEN * (INTER // 2)  # 512 KiB
W2S_EXPERT_BYTES = (HIDDEN // 32) * (INTER // 128) * 128  # 32 KiB
