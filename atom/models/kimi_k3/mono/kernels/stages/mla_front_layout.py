# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Pure-Python geometry of the gfx1250 MLA front (``mla_front_gfx1250``): which
CTA projects which weight rows and which waves normalize which rows, what the
LDS and the launch's buffers hold. No FlyDSL, so every coordinate is checked on
a CPU-only host.

The front projects a full-attention layer's normed input through its two FP8
projections that read it, ``fused_qkv_a_proj`` (``[q_a 1536 | kv_a 512 |
rope 64]``, 2112 rows) and this rank's ``g_proj`` rows, as one split-K TDM
GEMV (``gemv_tdm_gfx1250.project_split``): a cluster of four CTAs owns a
group of four 16-row tasks (64 rows, inside one 128-row scale block) and
splits K four ways. The 33 groups of the first weight come first, then the
gate's, so cluster ``c`` < 24 projects q_a, 24..31 kv_a, 32 the rope rows and
the rest the gate.
"""

from __future__ import annotations

from dataclasses import dataclass

from atom.models.kimi_k3.mono.config import GFX1250_BLOCKS, HIDDEN, MAX_ROWS
from atom.models.kimi_k3.mono.kernels.stages.gemv_layout import (
    ROWS,
    TDM_CLUSTER,
    TDM_SPLIT_GROUP,
    tdm_split_lds_words,
    tdm_split_scratch_bytes,
    tdm_split_slabs,
)
from atom.mono.plan.architecture import GFX1250

Q_LORA = 1536
KV_LORA = 512
ROPE = 64
# fused_qkv_a_proj's logical rows; ATOM pads its FP8 weight to the next 128
QKV_A_ROWS = Q_LORA + KV_LORA + ROPE
QKV_A_PADDED = -(-QKV_A_ROWS // 128) * 128
# the unnormalized q_a | kv_a rows the norm waves read back
QKV_C = Q_LORA + KV_LORA
GROUP_ROWS = TDM_SPLIT_GROUP * ROWS
Q_CLUSTERS = Q_LORA // GROUP_ROWS
KV_CLUSTERS = KV_LORA // GROUP_ROWS
A_CLUSTERS = QKV_A_ROWS // GROUP_ROWS
ROPE_CLUSTER = Q_CLUSTERS + KV_CLUSTERS
assert A_CLUSTERS * GROUP_ROWS == QKV_A_ROWS and ROPE == GROUP_ROWS
# A ``qkv_c`` word (two BF16) no producer has stored this launch: both halves
# an all-ones NaN. The producers store that pair (a NaN input row's) as two
# quiet NaNs instead, so it never reads as unwritten.
UNWRITTEN = 0xFFFFFFFF
QUIET_NAN_PAIR = 0x7FC07FC0
THREADS = 256
WAVES = THREADS // GFX1250.wave_size
# AITER fused_qk_rmsnorm at q_lora 1536 (its widest): 256 threads of eight
# consecutive elements, the q rows on threads [0, 192), the kv rows on
# [0, 64). A norm wave lane holds eight such threads' runs, 64 elements.
NORM_THREADS = 256
NORM_RUN = 8
LANE_ELEMENTS = NORM_THREADS * NORM_RUN // GFX1250.wave_size
Q_LANES = Q_LORA // LANE_ELEMENTS
KV_LANES = KV_LORA // LANE_ELEMENTS
# q_b_proj's activation quant: 128-element groups, E8M0 scales
QUANT_GROUP = 128
Q_GROUPS = Q_LORA // QUANT_GROUP
assert QUANT_GROUP == 2 * LANE_ELEMENTS
# aiter's kAmaxFloor and FP8 E4M3's 1 / 448, as their f32 bits
AMAX_FLOOR_BITS = 0x2EDBE6FF
INV_FP8_MAX_BITS = 0x3B124925
# s_sleep units (64 clocks) the gate CTAs hold their copies back by default:
# measured at 8 and 24 rows with the weights cold, 16 and 32 both faster
# than none, 32 the faster at 24 rows
GATE_DELAY = 32


def gate_clusters(gate_rows: int) -> int:
    if gate_rows <= 0 or gate_rows % GROUP_ROWS:
        raise ValueError(f"g_proj rows {gate_rows} are not 64-row groups")
    return gate_rows // GROUP_ROWS


def gemv_ctas(gate_rows: int) -> int:
    """One CTA a task: the first weight's 33 groups, then the gate's."""
    return (A_CLUSTERS + gate_clusters(gate_rows)) * TDM_CLUSTER


def spare_ctas(gate_rows: int) -> int:
    """The whole clusters of CUs the GEMV leaves free: the norms' own CTAs by
    default, so a norm wave polls from the start instead of past a gate
    CTA's GEMV (28 at TP8)."""
    return (GFX1250_BLOCKS - gemv_ctas(gate_rows)) // TDM_CLUSTER * TDM_CLUSTER


def grid(gate_rows: int, norm_ctas: int = 0) -> int:
    """The GEMV's CTAs, then ``norm_ctas`` of the norms' own (whole
    clusters), if any."""
    if norm_ctas < 0 or norm_ctas % TDM_CLUSTER:
        raise ValueError(f"{norm_ctas} norm CTAs: whole clusters only")
    return gemv_ctas(gate_rows) + norm_ctas


def cluster_rows(cluster: int, gate_rows: int) -> tuple[str, int]:
    """``(kind, first row in its weight)`` of cluster ``cluster``'s group:
    kind ``q``, ``kv`` and ``rope`` read fused_qkv_a_proj, ``gate`` g_proj."""
    if not 0 <= cluster < gemv_ctas(gate_rows) // TDM_CLUSTER:
        raise ValueError(f"cluster {cluster} outside the grid")
    if cluster >= A_CLUSTERS:
        return "gate", (cluster - A_CLUSTERS) * GROUP_ROWS
    row = cluster * GROUP_ROWS
    if cluster < Q_CLUSTERS:
        return "q", row
    if cluster < ROPE_CLUSTER:
        return "kv", row
    return "rope", row


def norm_jobs(rows: int) -> int:
    """A wave a row: the q rows, then the kv rows."""
    return 2 * rows


def norm_placement(gate_rows: int, norm_ctas: int = 0) -> tuple[int, int]:
    """``(first CTA, CTAs)`` the norm waves run on: the norms' own CTAs past
    the GEMV's, or with none the gate CTAs, which publish nothing the norms
    wait for."""
    if norm_ctas:
        return gemv_ctas(gate_rows), norm_ctas
    return A_CLUSTERS * TDM_CLUSTER, gate_clusters(gate_rows) * TDM_CLUSTER


def norm_job(cta: int, wave: int, gate_rows: int, norm_ctas: int = 0) -> int | None:
    """The norm job of a CTA's wave, or None: job ``j`` on norm CTA
    ``j % CTAs``, wave ``j // CTAs`` (``norm_placement``)."""
    first, count = norm_placement(gate_rows, norm_ctas)
    if not first <= cta < first + count:
        return None
    return cta - first + count * wave


@dataclass(frozen=True)
class FrontPlan:
    """The split-K GEMV's weight staging at ``rows`` rows: ``resident``
    (every slab of the CTA's K quarter copied up front) or ``stages`` slab
    buffers; ``words`` its LDS."""

    rows: int
    resident: bool
    stages: int
    words: int

    @property
    def tiles(self) -> int:
        return -(-self.rows // ROWS)


def lds_budget_words() -> int:
    return GFX1250.max_workgroup_lds_bytes // 4


def front_plan(rows: int) -> FrontPlan:
    if not 1 <= rows <= MAX_ROWS:
        raise ValueError(f"the MLA front takes 1..{MAX_ROWS} rows, got {rows}")
    budget = lds_budget_words()
    slabs = tdm_split_slabs(HIDDEN)
    whole = tdm_split_lds_words(rows, HIDDEN, "fp8", slabs, groups=1)
    if whole <= budget:
        return FrontPlan(rows, True, slabs, whole)
    for stages in (3, 2):
        words = tdm_split_lds_words(rows, HIDDEN, "fp8", stages, groups=1)
        if words <= budget:
            return FrontPlan(rows, False, stages, words)
    raise ValueError(f"the MLA front's LDS does not fit at {rows} rows")


def partial_bytes(rows: int, gate_rows: int) -> int:
    """The split-K exchange's f32 partials (``project_split``'s scratch)."""
    return tdm_split_scratch_bytes(gemv_ctas(gate_rows), rows)


def qkv_words(rows: int = MAX_ROWS) -> int:
    """``qkv_c``'s i32 words for ``rows`` rows: one buffer serves every width
    up to it (a launch reads and restores only its own rows)."""
    return rows * QKV_C // 2
