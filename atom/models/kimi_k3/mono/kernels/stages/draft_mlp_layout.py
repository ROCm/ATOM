# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Pure-Python geometry of the gfx1250 DSpark draft layer's MLP and
projection kernels (``draft_proj_ar_gfx1250``, ``draft_gate_up_gfx1250``):
which CTA projects which columns, what a CTA's LDS holds, and every word a
launch pushes to or polls from a rank's receive bank. No FlyDSL, so every
coordinate -- and every peer address -- is checked on a CPU-only host.

A draft layer's row-parallel projections (o_proj, the MLP's down_proj) take a
BF16 activation ``[rows, k]``, quantize it per 1x128 group to FP8 with an E8M0
code, multiply by the rank's FP8 shard ``[HIDDEN, k]`` (128x128 E8M0 block
scales, AITER's 16x16 preshuffle) and all-reduce the BF16 partials. CTA ``b``
owns output columns ``32 b .. 32 b + 32`` (two 16-row tasks), holds its tasks'
weights and every token row's FP8 bytes in LDS, and pushes its partials as
``k2_oproj_ar_gfx1250``'s one-shot lines: the bank layout below is that
kernel's, word for word.
"""

from __future__ import annotations

from atom.models.kimi_k3.mono.config import HIDDEN, MAX_ROWS
from atom.models.kimi_k3.mono.kernels.stages import mla_draft_front_layout as front
from atom.models.kimi_k3.mono.kernels.stages.gemv_layout import ROWS
from atom.mono.plan.architecture import GFX1250

THREADS = 256
LANES = GFX1250.wave_size
WAVES = THREADS // LANES
# a scale group of K: one E8M0 code, one scaled 16x16x128 WMMA
K_GROUP = 128
# a 16-row task's K group in AITER's preshuffle, as it lies in the weight
GROUP_BYTES = K_GROUP * ROWS
GROUP_WORDS = GROUP_BYTES // 4
TASKS = 2
CTA_COLUMNS = TASKS * ROWS
CTAS = HIDDEN // CTA_COLUMNS
CTA_PAIRS = CTA_COLUMNS // 2
# a CTA's weights arrive in this many TDM copies a task
WEIGHT_PARTS = 2
# the activation quant: a lane holds 16 elements, eight lanes a group
QUANT_LANE_ELEMENTS = 16
QUANT_LANES = K_GROUP // QUANT_LANE_ELEMENTS
AMAX_FLOOR_BITS = 0x2EDBE6FF  # f32(1e-10), aiter::kAmaxFloor
INV_FP8_MAX_BITS = 0x3B124925  # f32(1 / 448)
MAX_CODE = 254  # mxfp8_cap_scale: 2^127
BF16_INF = 0x7F80

# k2_oproj_ar_gfx1250's one-shot line: three tokens' BF16 pair of one column
# pair, then the launch's epoch
LINE_TOKENS = 3
LINE_WORDS = LINE_TOKENS + 1
LINE_BYTES = 4 * LINE_WORDS
# a bounded (tool / test) build's give-up mark, in word 0 of its flags
GAVE_UP = 0x7AB00000
# k2_allreduce_gfx1250's epoch counters (its MAX_CTAS)
EPOCH_COUNTERS = 256


def groups(k: int) -> int:
    return k // K_GROUP


def tiles(rows: int) -> int:
    return -(-rows // ROWS)


def triples(rows: int) -> int:
    return -(-rows // LINE_TOKENS)


def line_rounds(rows: int) -> int:
    """A CTA's ``[triple][pair]`` lines a thread: one up to 48 rows."""
    return -(-triples(rows) * CTA_PAIRS // THREADS)


def quant_items(rows: int, k: int, phases: int = 1) -> int:
    """A K phase's 16-element runs of the activation: ``[group][token][run]``."""
    return rows * (groups(k) // phases) * QUANT_LANES


def quant_rounds(rows: int, k: int, phases: int = 1) -> int:
    return -(-quant_items(rows, k, phases) // THREADS)


def weight_words(k: int) -> int:
    return TASKS * groups(k) * GROUP_WORDS


def token_words(rows: int, k: int) -> int:
    """Every token tile's 16 rows of every K group in the WMMA operand
    layout (``[tile][group][4 chunks][2 halves][16 rows][16 B]``)."""
    return tiles(rows) * groups(k) * GROUP_WORDS


def code_words(rows: int, k: int) -> int:
    """The tokens' E8M0 codes, a word each: ``[group][rows]``."""
    return groups(k) * rows


def stage_words(rows: int) -> int:
    """The CTA's BF16 pairs ``[token][pair]``, cut into lines."""
    return triples(rows) * LINE_TOKENS * CTA_PAIRS


def scratch_words(rows: int, k: int, phases: int) -> int:
    """The BF16 activation of one K phase as the TDM lands it:
    ``[rows][k / phases]``."""
    return rows * (k // phases) // 2


def token_offset(k: int) -> int:
    return weight_words(k)


def code_offset(rows: int, k: int) -> int:
    return token_offset(k) + token_words(rows, k)


def stage_offset(rows: int, k: int) -> int:
    return code_offset(rows, k) + code_words(rows, k)


def scratch_offset(rows: int, k: int) -> int:
    return stage_offset(rows, k) + stage_words(rows)


def lds_words(rows: int, k: int, phases: int) -> int:
    return scratch_offset(rows, k) + scratch_words(rows, k, phases)


# K phases the activation is copied to LDS and quantized in: all of K where
# the LDS holds it beside the FP8 tokens and the weights, else its halves or
# quarters in turn
TOKEN_PHASES = (1, 2, 4)


def token_phases(rows: int, k: int) -> int | None:
    """The fewest K phases whose LDS fits (None: none does)."""
    limit = GFX1250.max_workgroup_lds_bytes
    for phases in TOKEN_PHASES:
        if groups(k) % phases == 0 and 4 * lds_words(rows, k, phases) <= limit:
            return phases
    return None


def token_rows_a_wave(rows: int) -> int:
    """Rows of the activation a wave's TDM copy moves."""
    return -(-rows // WAVES)


def refusal(rows: int, k: int, tp: int) -> str | None:
    """Why no kernel is built at this shape (None: one is)."""
    if not 1 <= rows <= MAX_ROWS:
        return f"{rows} rows"
    if k <= 0 or k % (WEIGHT_PARTS * K_GROUP):
        return f"K {k}: not whole pairs of 128-wide groups"
    if groups(k) > 32:
        return f"K {k}: more than 32 K groups a chain"
    if tp < 1 or tp > 8:
        return f"TP{tp}"
    if token_phases(rows, k) is None:
        return f"no LDS for {rows} rows at K {k}"
    return None


def task_weight_byte(cta: int, task: int, k: int) -> int:
    """First byte of a CTA's task in the preshuffled weight: 16 rows' K
    groups lie one after another, 2 KB each."""
    return (cta * CTA_COLUMNS + task * ROWS) * k


def weight_block(cta: int) -> int:
    """The 128-row block of the weight's E8M0 scales a CTA's rows lie in."""
    return cta * CTA_COLUMNS // 128


# ---- the receive banks: k2_oproj_ar_gfx1250's one-shot layout ----


def bank_words(tp: int, rows: int) -> int:
    """A bank: every rank's partials ``[source rank][token triple][pair]``,
    a line each."""
    return LINE_WORDS * triples(rows) * tp * (HIDDEN // 2)


def peer_bytes(tp: int, rows: int) -> int:
    """One rank's two banks (epoch parity)."""
    return 2 * bank_words(tp, rows) * 4


def line_live(rows: int, item):
    """Whether a thread's ``item`` (its index, plus 256 a round) is one of
    the CTA's lines: only those are pushed and polled."""
    return item < triples(rows) * CTA_PAIRS


def line_word(source: int, rows: int, bank, cta, item):
    """First word, in a rank's bank region, of the line holding rank
    ``source``'s partial of CTA ``cta``'s line ``item``: the word rank
    ``source`` pushes it to on every rank, and the word every rank polls it
    at. ``bank`` is the epoch's bank offset in words (0 or
    ``bank_words``); ``cta`` and ``item`` may be device values."""
    triple = item // CTA_PAIRS
    pair = item % CTA_PAIRS
    return bank + LINE_WORDS * (
        (source * triples(rows) + triple) * (HIDDEN // 2) + cta * CTA_PAIRS + pair
    )


# ---- gate_up_proj and SiLU * mul (``draft_gate_up_gfx1250``) ----
#
# The MLP's gate_up_proj takes the post-attention norm's FP8 row (E8M0
# per-1x128 codes, transposed: the draft front's input format) through the
# rank's ``[2 inter, HIDDEN]`` FP8 shard, gate rows first. A cluster of four
# CTAs owns 32 intermediate columns -- four 16-row tasks: two gate tasks,
# then the two up tasks of the same columns -- and splits K as the draft
# front does at ``ksplit`` 4: CTA ``r`` chains every task over K quarter
# ``r`` (its weights four 28 KB runs, its token quarter staged in two
# seven-group pieces). Production's GEMM sums K in ``split_k`` splits, each
# from zero, added in split order by its reduce: at 4 a CTA's quarter is one
# split, at 8 each of its two pieces is one. The CTAs trade their splits'
# f32 sums through ``partials`` past one cluster barrier; CTA 0 and CTA 1
# then each own 16 columns: gate and up summed in split order, rounded to
# BF16, SiLU(gate) * up.

GATE_UP_GROUPS = HIDDEN // K_GROUP
PIECE_GROUPS = front.PIECE_GROUPS
QUARTER_GROUPS = front.RUN_GROUPS
GATE_UP_CLUSTER = 4
# a cluster's task slots: gate columns [0, 16), gate [16, 32), up [0, 16),
# up [16, 32) of its 32 columns
GATE_UP_SLOTS = 4
GATE_UP_OWNERS = 2
GATE_UP_COLUMNS = GATE_UP_OWNERS * ROWS
# production's split_k values the kernel mirrors
GATE_UP_SPLITS = (4, 8)
MAX_TILES = -(-MAX_ROWS // ROWS)
SILU_NEG_LOG2E_HI_BITS = 0xBFB8AA3B
SILU_NEG_LOG2E_LO_BITS = 0xB2A5705F
SILU_EXP_ZERO_BITS = 0x42CE8ED0  # past it exp(-x) is 0
SILU_EXP_INF_BITS = 0xC2B17218  # below it exp(-x) is +inf
assert GATE_UP_SLOTS == front.TASKS == front.RUNS
assert QUARTER_GROUPS * GATE_UP_CLUSTER == GATE_UP_GROUPS == front.GROUPS
assert QUARTER_GROUPS == 2 * PIECE_GROUPS


def gate_up_clusters(inter: int) -> int:
    return inter // GATE_UP_COLUMNS


def gate_up_ctas(inter: int) -> int:
    return gate_up_clusters(inter) * GATE_UP_CLUSTER


def gate_up_slot_row(cluster, slot, inter: int):
    """First weight row of a cluster's task slot: gate rows, then up rows."""
    return (slot // GATE_UP_OWNERS) * inter + (
        cluster * GATE_UP_OWNERS + slot % GATE_UP_OWNERS
    ) * ROWS


def gate_up_column(cluster, owner):
    """First intermediate column CTA ``owner`` (0, 1) of a cluster owns."""
    return (cluster * GATE_UP_OWNERS + owner) * ROWS


def gate_up_lds_words(rows: int) -> int:
    """A CTA's LDS: the draft front's (four weight runs, two token pieces
    and their codes)."""
    return front.lds_words(rows)


def gate_up_chains(rows: int) -> int:
    """A CTA's (task slot, token tile) chains, wave ``w`` running chains
    ``w, w + 8``."""
    return GATE_UP_SLOTS * tiles(rows)


def gate_up_piece_split(split_k: int, rank, piece: int):
    """The split a CTA's K piece closes (its f32 sum is traded), or None
    where the split runs on into the CTA's next piece."""
    if split_k == 8:
        return rank * 2 + piece
    return rank if piece == 1 else None


def gate_up_partial_index(cluster, slot, split, tile, lane, split_k: int, rows: int):
    """First f32 of a lane's eight sums of (cluster, task slot, split, token
    tile) in ``partials``."""
    return (
        (((cluster * GATE_UP_SLOTS + slot) * split_k + split) * tiles(rows) + tile)
        * LANES
        + lane
    ) * 8


def gate_up_partial_bytes(inter: int) -> int:
    """The split exchange at up to MAX_ROWS rows and eight splits."""
    return gate_up_clusters(inter) * GATE_UP_SLOTS * 8 * MAX_TILES * LANES * 8 * 4


def gate_up_refusal(rows: int, inter: int, split_k: int) -> str | None:
    """Why no gate_up kernel is built at this shape (None: one is)."""
    limit = GFX1250.max_workgroup_lds_bytes
    if not 1 <= rows <= MAX_ROWS:
        return f"{rows} rows"
    if inter <= 0 or inter % K_GROUP:
        return f"an intermediate shard of {inter}"
    if split_k not in GATE_UP_SPLITS:
        return f"split-K {split_k}: the kernel mirrors {GATE_UP_SPLITS}"
    if gate_up_ctas(inter) > 256:
        return f"an intermediate shard of {inter}: more CTAs than CUs"
    if 4 * gate_up_lds_words(rows) > limit:
        return f"{4 * gate_up_lds_words(rows)} B of LDS at {rows} rows"
    return None
