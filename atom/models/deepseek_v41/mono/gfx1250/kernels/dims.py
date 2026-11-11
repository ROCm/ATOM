# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The step kernel's model and deployment dimensions, defined once (every kernel module takes its sizes from here).

DeepSeek-V4.1-Flash, TP4, DSpark with 5 draft tokens, at most 8 requests a step. Per-rank widths are the model's over
TP; the step's row counts are capacities (a build's actual rows, 6 a request, are its TR).
"""

# --- model --------------------------------------------------------------------------------------------------------
HID = 5120  # hidden size
Q_RANK = 1280  # q latent (q_lora_rank)
KV_DIM = 512  # compressed kv / head dim
Q_HEADS = 64  # attention heads
MOE_INTER = 2304  # routed / shared expert intermediate size
EXPERTS = 384  # routed experts
TOPK = 6  # experts a token
VOCAB = 129280
LAYERS = 40  # target layers
HC = 4  # mHC residual streams
INDEX_HEADS, INDEX_DIM = 32, 128
RMS_EPS = 1e-20  # rms_norm_eps

# --- the DSpark draft -----------------------------------------------------------------------------------------------
DRAFT_LAYERS = 3
DRAFT_EXPERTS, DRAFT_TOPK = 128, 3
DSPARK_W = 5  # draft tokens a request (a draft block's rows)

# --- deployment -----------------------------------------------------------------------------------------------------
TP = 4
HEADS = Q_HEADS // TP  # 16 attention heads a rank
MOE_I = MOE_INTER // TP  # 576 expert intermediate columns a rank
MOE_IPAD = 640  # MOE_I as the production MoE stores it, padded (the kernel reads its experts in place)
VR = VOCAB // TP  # 32320 vocab rows a rank
MAX_REQ = 8  # requests a step
ROWS_PER_REQ = DSPARK_W + 1  # target rows a request: the anchor + its drafts
T_MAX = MAX_REQ * ROWS_PER_REQ  # 48 target rows (buffer / layout capacity)

# --- the launch -----------------------------------------------------------------------------------------------------
CTAS, THREADS, WAVES = 256, 256, 8
CL = 16  # CTAs a hardware cluster

assert Q_HEADS % TP == 0 and MOE_INTER % TP == 0 and VOCAB % TP == 0
