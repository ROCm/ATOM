# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Shapes the Qwen3.5-35B-A3B mono kernels are built for: one TP4 rank on gfx1250.

Every value is checked against the loaded model in ``weights.py``; a model that does
not match never takes the mono path.
"""

# execution model: one CTA a CU, all co-resident, THREADS threads (wave32)
BLOCKS = 256
THREADS = 512
WAVE = 32
WAVES = THREADS // WAVE
GRID_WAVES = BLOCKS * WAVES

TP = 4
MAX_TOKENS = 4  # decode tokens a mono step serves (one token a request)
# MTP verify steps (kernels_mtp.py): SPEC tokens a request (the anchor + the drafts),
# up to MTP_MAX_REQS requests, in groups of 16 tokens (one WMMA B tile of columns)
SPEC = 4
MTP_MAX_REQS = 16
MTP_MAX_TOKENS = SPEC * MTP_MAX_REQS

HIDDEN = 2048
EPS = 1e-6
SLICE = HIDDEN // BLOCKS  # hidden dims CTA b owns in the all-reduces: [8b, 8b + 8)

# GDN (linear attention), per rank
GDN_KH = 16 // TP  # k heads
GDN_VH = 32 // TP  # v heads
GDN_D = 128  # k and v head dim
GDN_QKV = (2 * GDN_KH + GDN_VH) * GDN_D  # conv channels: q | k | v = 2048
GDN_Z = GDN_VH * GDN_D  # 1024
GDN_QKVZ = GDN_QKV + GDN_Z  # in_proj_qkvz rows = 3072
GDN_BA = 2 * GDN_VH  # in_proj_ba rows: b 8 | a 8
CONV_K = 4  # conv taps; the state keeps CONV_K - 1

# full attention, per rank
ATT_QH = 16 // TP  # q heads
ATT_D = 256
ATT_ROT = 64  # rotary dims (NeoX halves of 32)
ATT_QKVG = 2 * ATT_QH * ATT_D + 2 * ATT_D  # gate 1024 | q 1024 | k 256 | v 256 = 2560
ATT_BLOCK = 16  # KV cache block (tokens)
ATT_SCALE = ATT_D**-0.5

# MoE, per rank (experts sharded along the intermediate dim)
N_EXPERTS = 256
TOP_K = 8
INTER = 512 // TP  # 128 real rows of each expert's gate / up on a rank
INTER_PAD = 256  # the FlyDSL layout pads the rank's intermediate to 256
SH_INTER = 512 // TP  # shared expert intermediate on a rank
ROUTER_ROWS = N_EXPERTS + 1  # + the shared-expert gate row
