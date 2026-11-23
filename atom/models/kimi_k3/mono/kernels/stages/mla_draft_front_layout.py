# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Pure-Python geometry of the gfx1250 DSpark draft MLA front
(``mla_draft_front_gfx1250``): which CTA projects which ``fused_qkv_a_proj``
rows over which K, which waves normalize which rows, what the launch's
buffers hold. No FlyDSL, so every coordinate is checked on a CPU-only host.

The draft's input norm hands ``fused_qkv_a_proj`` an FP8 row with E8M0
per-1x128 codes, and production's FlyDSL ``gemm_a8w8_mxfp8_128`` sums the
scaled WMMAs of a K split in order, its ``split_k`` partials summed in order
by the split-K reduce. The front keeps both: a cluster of four CTAs owns a
64-row group (four 16-row tasks); at ``ksplit`` 4 CTA ``r`` chains every task
over K quarter ``r`` and owns task ``r``'s sum over the quarters, at
``ksplit`` 1 CTA ``r`` chains task ``r`` over all of K. A CTA's weights are
four 28 KB runs of the preshuffled weight either way, a wave copying half of
one. The 33 groups are q_a (clusters 0..23), kv_a (24..31) and the rope rows
(32); the norm CTAs follow.
"""

from __future__ import annotations

from dataclasses import dataclass

from atom.models.kimi_k3.mono.config import HIDDEN, MAX_ROWS
from atom.models.kimi_k3.mono.kernels.stages import mla_front_layout as front
from atom.models.kimi_k3.mono.kernels.stages.gemv_layout import ROWS, TDM_CLUSTER
from atom.mono.plan.architecture import GFX1250

Q_LORA = front.Q_LORA
KV_LORA = front.KV_LORA
ROPE = front.ROPE
QKV_A_ROWS = front.QKV_A_ROWS
QKV_A_PADDED = front.QKV_A_PADDED
QKV_C = front.QKV_C
GROUP_ROWS = front.GROUP_ROWS
Q_CLUSTERS = front.Q_CLUSTERS
A_CLUSTERS = front.A_CLUSTERS
ROPE_CLUSTER = front.ROPE_CLUSTER
UNWRITTEN = front.UNWRITTEN
QUIET_NAN_PAIR = front.QUIET_NAN_PAIR
THREADS = 256
WAVES = THREADS // GFX1250.wave_size
LANES = GFX1250.wave_size
TASKS = GROUP_ROWS // ROWS
# a scale group of K (one E8M0 code, one scaled WMMA)
K_GROUP = 128
GROUPS = HIDDEN // K_GROUP
# a CTA's weights: four runs of 14 groups, each copied by two waves
RUNS = 4
RUN_GROUPS = GROUPS // RUNS
PIECE_GROUPS = RUN_GROUPS // 2
GROUP_BYTES = K_GROUP * ROWS
# the chains walk K a piece (seven groups) at a time: a weight copy's, and
# the token rows staged in LDS, double-buffered
PIECE_BYTES = PIECE_GROUPS * K_GROUP
PIECE_UNITS = PIECE_BYTES // 16
KSPLITS = (1, 4)
# AITER fused_qk_rmsnorm_group_quant at q_lora 1536 (128 threads of 16
# elements, the q rows on threads [0, 96), the kv rows on [0, 32)): a norm
# wave lane holds thread ``lane + 32 r`` for run r, three runs of a q row,
# one of a kv row
NORM_THREAD_ELEMENTS = 16
NORM_WAVE_ELEMENTS = LANES * NORM_THREAD_ELEMENTS
Q_RUNS = Q_LORA // NORM_WAVE_ELEMENTS
KV_RUNS = KV_LORA // NORM_WAVE_ELEMENTS
NORM_BLOCK_WAVES = 4
# q_b_proj's activation quant: 128-element groups (8 AITER threads), fp32
# scales amax * f32(1 / 448), the amax floored at 1e-10
QUANT_GROUP = 128
Q_GROUPS = Q_LORA // QUANT_GROUP
GROUP_LANES = QUANT_GROUP // NORM_THREAD_ELEMENTS
AMAX_FLOOR = 1e-10
INV_FP8_MAX_BITS = front.INV_FP8_MAX_BITS
FP8_MAX = 448.0
assert Q_RUNS == 3 and KV_RUNS == 1 and RUN_GROUPS == 14 and PIECE_GROUPS == 7


def gemv_ctas() -> int:
    return A_CLUSTERS * TDM_CLUSTER


def norm_jobs(rows: int) -> int:
    """A wave a row: the q rows, then the kv rows."""
    return 2 * rows


def norm_ctas(rows: int) -> int:
    """Whole clusters of norm CTAs, a wave a job."""
    ctas = -(-norm_jobs(rows) // WAVES)
    return -(-ctas // TDM_CLUSTER) * TDM_CLUSTER


def grid(rows: int) -> int:
    return gemv_ctas() + norm_ctas(rows)


def norm_job(cta: int, wave: int, rows: int) -> int | None:
    """Job ``j`` runs on norm CTA ``j % CTAs``, wave ``j // CTAs``."""
    first, count = gemv_ctas(), norm_ctas(rows)
    if not first <= cta < first + count:
        return None
    job = cta - first + count * wave
    return job if job < norm_jobs(rows) else None


def cluster_rows(cluster: int) -> tuple[str, int]:
    """``(kind, first row)`` of cluster ``cluster``'s 64-row group."""
    if not 0 <= cluster < A_CLUSTERS:
        raise ValueError(f"cluster {cluster} outside the GEMV")
    row = cluster * GROUP_ROWS
    if cluster < Q_CLUSTERS:
        return "q", row
    if cluster < ROPE_CLUSTER:
        return "kv", row
    return "rope", row


def run_source(ksplit: int, cluster: int, rank: int, run: int) -> tuple[int, int]:
    """``(task, first K group)`` of run ``run`` of CTA ``rank``'s weights:
    task ``run`` over quarter ``rank`` at ksplit 4, task ``rank`` over quarter
    ``run`` at ksplit 1."""
    if ksplit not in KSPLITS:
        raise ValueError(f"split-K {ksplit}: the front mirrors 1 or 4")
    task, quarter = (run, rank) if ksplit == 4 else (rank, run)
    return task, quarter * RUN_GROUPS


def run_offset(ksplit: int, cluster: int, rank: int, run: int) -> int:
    """Byte offset of the run in the preshuffled weight: a 16-row task's
    K groups lie one after another, 2 KB each."""
    task, group = run_source(ksplit, cluster, rank, run)
    return (cluster * GROUP_ROWS + task * ROWS) * HIDDEN + group * GROUP_BYTES


def wave_piece(wave: int) -> tuple[int, int]:
    """``(run, half)`` wave ``wave`` copies."""
    if not 0 <= wave < WAVES:
        raise ValueError(f"wave {wave}")
    return wave % RUNS, wave // RUNS


def chains(ksplit: int, rows: int) -> list[tuple[int, int]]:
    """A CTA's ``(task slot, token tile)`` chains, wave ``w`` running chains
    ``w, w + 8, ..``: four tasks a tile over the CTA's K quarter at ksplit 4,
    one task a tile over all K at ksplit 1."""
    tiles = -(-rows // ROWS)
    if ksplit == 4:
        return [(t, m) for m in range(tiles) for t in range(TASKS)]
    return [(0, m) for m in range(tiles)]


def chain_groups(ksplit: int) -> int:
    return RUN_GROUPS if ksplit == 4 else GROUPS


@dataclass(frozen=True)
class DraftFrontPlan:
    rows: int
    ksplit: int

    def __post_init__(self):
        if not 1 <= self.rows <= MAX_ROWS:
            raise ValueError(
                f"the draft front takes 1..{MAX_ROWS} rows, got {self.rows}"
            )
        if self.ksplit not in KSPLITS:
            raise ValueError(f"split-K {self.ksplit}: the front mirrors 1 or 4")

    @property
    def tiles(self) -> int:
        return -(-self.rows // ROWS)

    @property
    def chain_count(self) -> int:
        return len(chains(self.ksplit, self.rows))


def pieces(ksplit: int) -> int:
    """K pieces of a chain: its quarter's two at ksplit 4, all of K's eight
    at ksplit 1."""
    return chain_groups(ksplit) // PIECE_GROUPS


def weight_words() -> int:
    return RUNS * RUN_GROUPS * GROUP_BYTES // 4


def token_words(tiles: int) -> int:
    """One token buffer: a piece of every tile's 16 rows in the WMMA operand
    layout."""
    return tiles * PIECE_GROUPS * GROUP_BYTES // 4


CODE_WORDS = PIECE_GROUPS * MAX_ROWS


def token_offset(buffer: int, tiles: int) -> int:
    return weight_words() + buffer * token_words(tiles)


def code_offset(buffer: int, tiles: int) -> int:
    return weight_words() + 2 * token_words(tiles) + buffer * CODE_WORDS


def lds_words(rows: int) -> int:
    """A CTA's LDS: its weight runs, two token buffers and two of their
    E8M0 codes (a word each)."""
    tiles = -(-rows // ROWS)
    return code_offset(2, tiles)


def partial_bytes() -> int:
    """The split-K exchange: every GEMV CTA's three foreign tasks' f32 sums
    at up to MAX_ROWS rows (a v8 a lane a tile)."""
    tiles = -(-MAX_ROWS // ROWS)
    return gemv_ctas() * TASKS * tiles * LANES * 8 * 4


def qkv_words(rows: int = MAX_ROWS) -> int:
    return front.qkv_words(rows)


def qkv_a_split_k(kernel_name: str | None) -> int | None:
    """The split_k of a FlyDSL mxfp8_128 kernelName (``..._sk4_..``), or None
    where it names none."""
    if not kernel_name:
        return None
    for part in str(kernel_name).split("_"):
        if part.startswith("sk") and part[2:].isdigit():
            return int(part[2:])
    return None
