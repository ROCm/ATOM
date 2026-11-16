# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The step kernel's device-wide counters and its fixed LDS regions (byte offsets).

Counters live in the ``cnt`` buffer (zeroed every step by the host): one block of LAYER_BYTES per layer (target layers,
then draft layers), then the tail block. Every handoff / group counter has one named slot here; the CPU test checks that
no two slots overlap.
"""

from __future__ import annotations

# --- per-layer counters: cnt + layer * LAYER_BYTES + slot -----------------------------------------------------------
LAYER_BYTES = 512
ATTN_A = 0  # stage A (wqkv GEMV) done
INDEX_QUANT = 4  # indexer layers: iquant / compressor rows done
KV_C = 8  # kv / C (+ iq / iw / compressor GEMM) done
ATTN_D = 12  # sparse attention done
ATTN_E = 16  # wo_a done
MOE_DONE = 20  # MoE (down) done -> combine + TP2
FRONT = 24  # FFN front done -> MoE
LAYER_END = 28  # the attention front of the next layer done
MOE_UG = 40  # MoE: UG done -> down (moe.moe)
SHARED_DOWN = 44  # MoE: the dedicated CTAs' shared-expert units (narrow batches)
SCORE = 48  # indexer score done
SELECT0 = 52  # selection pass p done: SELECT0 + 4 p (p < 2 IDX_LEVELS)
SELECT_PASSES = 6  # room for 6 passes (52 .. 72)
X8_PUB = 92  # FFN front's x8 / xs out (the early shared-expert UG waits on it)
WOA_QUARTERS = 96  # wo_a done, by y8 column quarter: 4 x 4 B (wo_b CTA b waits for quarter b % 4 only)
TP1_GROUPS = 128  # wo_b + TP1 column groups: 16 x 4 B
TP2_GROUPS = 192  # combine + TP2 column groups: 16 x 4 B
GROUPS = 16
MIX_POST = 424  # (v2) the owners' mix partials done (sinkhorn2 waits for all 256)
MIX_L1 = 432  # (v2) the mix's level-1 sums done (NL1 CTAs)
SINK_DONE = 428  # (v2) the FFN mix's Sinkhorn done (the attention front's pre / post / comb)
FFN_POST = 420  # (v2) the TP1 column owners' FFN mHC post done (ffn_post_owner; ffn_c2 waits for all 256)
DN_BLK = 256  # (v2) routed + shared down units done, by 128-col hidden block: 40 x 4 B (the combine of a CTA's cols waits for its blocks)

# --- tail counters: cnt + (NL + NDR) * LAYER_BYTES + slot -------------------------------------------------------------
HEAD = 0  # target head start
HEAD_DONE = 16  # target head done -> sampling
SAMPLE = 64  # sampling / verification counters (sample.sample_once_p: 64 .. 159)
VERIFY_DONE = 160
DRAFT_HEAD_DONE = 176
MARKOV = 192  # Markov chain counters (markov.markov_once: 16 B a block)
DRAFT_STATE = 400  # draft entry: block state done
DRAFT_FRONT = 416  # draft entry: attention front done
TARGET_STATE = 432  # target entry: block state done
TARGET_FRONT = 448  # target entry: attention front done

# --- fixed LDS regions --------------------------------------------------------------------------------------------
TP_PAYLOAD = 0  # [TR][20] f32 TP reduction payload, kept across the FFN front
FRONT_LDS = 4096  # the fronts' / MoE's LDS (after the TP payload)
RKEEP = 294912  # the residual streams kept between the fronts

# --- layer-table rows -------------------------------------------------------------------------------------------------
TARGET_ENTRY_ROW = (
    44  # the target entry's attention front (after 40 + 3 layers and the draft entry)
)


def layer_slots() -> dict:
    """Every per-layer slot: name -> (offset, bytes)."""
    return {
        "ATTN_A": (ATTN_A, 4),
        "INDEX_QUANT": (INDEX_QUANT, 4),
        "KV_C": (KV_C, 4),
        "ATTN_D": (ATTN_D, 4),
        "ATTN_E": (ATTN_E, 4),
        "MOE_DONE": (MOE_DONE, 4),
        "FRONT": (FRONT, 4),
        "LAYER_END": (LAYER_END, 4),
        "MOE_UG": (MOE_UG, 4),
        "SHARED_DOWN": (SHARED_DOWN, 4),
        "SCORE": (SCORE, 4),
        "SELECT": (SELECT0, 4 * SELECT_PASSES),
        "X8_PUB": (X8_PUB, 4),
        "WOA_QUARTERS": (WOA_QUARTERS, 16),
        "TP1_GROUPS": (TP1_GROUPS, 4 * GROUPS),
        "TP2_GROUPS": (TP2_GROUPS, 4 * GROUPS),
        "DN_BLK": (DN_BLK, 160),
        "FFN_POST": (FFN_POST, 4),
        "MIX_POST": (MIX_POST, 4),
        "SINK_DONE": (SINK_DONE, 4),
        "MIX_L1": (MIX_L1, 4),
    }
