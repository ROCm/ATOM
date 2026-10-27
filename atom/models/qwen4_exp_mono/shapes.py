# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Fixed model dimensions and ordinary decode build capabilities."""

# Every request contributes one token. Persistent and batched schedules differ.
PERSISTENT_ROWS = (1,)
BATCH_ROWS = (2, 4)
DECODE_ROWS = (*PERSISTENT_ROWS, *BATCH_ROWS)
SUPPORTED_TP = (1, 8)
ALIGN = 16
LAYERS = 48
HIDDEN = 2560
HC = 4
HC_RANK = 320
EXPERTS = 512
TOPK = 10
INTERMEDIATE = 640
SHARED_EXPERTS = 1
ROUTER_OUTPUTS = EXPERTS + SHARED_EXPERTS
FP8_BLOCK = 128

# QSA specialization used by every fourth layer.  Keep these beside the GDN
# dimensions: they are part of the deployment/build key, not values a kernel
# may discover from runtime metadata.
QSA_HEADS = 24
QSA_KV_HEADS = 2
QSA_HEAD_DIM = 256
QSA_Q_SIZE = QSA_HEADS * QSA_HEAD_DIM
QSA_KV_SIZE = QSA_KV_HEADS * QSA_HEAD_DIM
QSA_QKV_SIZE = 2 * QSA_Q_SIZE + 2 * QSA_KV_SIZE
QSA_INDEX_HEADS = 4
QSA_INDEX_KV_HEADS = 1
QSA_INDEX_HEAD_DIM = 128
QSA_INDEX_Q_SIZE = QSA_INDEX_HEADS * QSA_INDEX_HEAD_DIM
QSA_INDEX_KV_SIZE = QSA_INDEX_KV_HEADS * QSA_INDEX_HEAD_DIM
QSA_INDEX_QK_SIZE = QSA_INDEX_Q_SIZE + QSA_INDEX_KV_SIZE
QSA_INDEX_BUDGET = 2048
QSA_COMPRESS_RATIO = 4
QSA_DIRECT_CONTEXT = 256
QSA_ROTARY_DIM = 64
QSA_MROPE_SECTION = (11, 11, 10)
QSA_MROPE_INTERLEAVED = True

# Gated DeltaNet specialization used by the 36 linear-attention layers.  The
# production FP8 checkpoint deliberately leaves every GDN projection in BF16;
# these widths therefore describe the packed BF16 projection that mono reads,
# not a generic Qwen configuration.
GDN_KEY_HEADS = 16
GDN_VALUE_HEADS = 48
GDN_HEAD_DIM = 128
GDN_CONV_WIDTH = 4
GDN_KEY_DIM = GDN_KEY_HEADS * GDN_HEAD_DIM
GDN_VALUE_DIM = GDN_VALUE_HEADS * GDN_HEAD_DIM
GDN_CONV_DIM = 2 * GDN_KEY_DIM + GDN_VALUE_DIM
GDN_PROJ_DIM = GDN_CONV_DIM + GDN_VALUE_DIM + 2 * GDN_VALUE_HEADS
EXPECTED_LAYER_TYPES = tuple(
    "full_attention" if (layer + 1) % 4 == 0 else "linear_attention"
    for layer in range(LAYERS)
)
