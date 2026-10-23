"""Mailbox regions, LDS arena and stage placement of the layer kernel.

Scratch is one device buffer of (u32 payload, u32 tag) pairs, split into named
regions. Every region is written once per launch with that launch's tag, so
nothing is ever cleared. Offsets here are in bytes from the scratch base.

Stage ``X`` runs task ``t`` on CTA ``(BASE[X] + t) % GRID``. Every CTA walks the
stages of its layer kind in ``ORDERS[kind]`` and every dependency points
forward in it, so a CTA never waits on a task that it, or a CTA stuck behind
it, has yet to run.
"""

import functools
from types import SimpleNamespace

from atom.models.qwen4_exp.mono.kernels.config import DEFAULT, GRID, THREADS

C = DEFAULT
WAVES = THREADS // 64

# ------------------------------------------------------------- stages

_HC_IN, _OUT_MLP = ("a1", "a2"), ("c1", "m1", "m2", "m3", "m4", "m5")
ORDERS = {
    "delta": _HC_IN + ("b1", "b3") + _OUT_MLP,
    "qsa": _HC_IN + ("q1", "q2", "q3", "q4", "q5", "q6", "q7") + _OUT_MLP,
}
# Layer 1 (0-based): the PLE prelude (P1 projections, P2 gate/conv/residual)
# runs first and feeds the attention HC.
ORDERS["delta_ple"] = ("p1", "p2") + ORDERS["delta"]
# Selected keys (list positions) at most: 512 blocks of 4, then the incomplete
# block's up to 3 tokens. Q5 scores them in tiles of 32 positions.
QSA_LIST = 4 * (C.index_budget // C.index_compress) + 3  # 2051
SCORE_TILE = 32
SCORE_TILES = -(-QSA_LIST // SCORE_TILE)  # 65
SS_ROW = SCORE_TILE * SCORE_TILES  # 2080 positions per head row of ``ss``
TASKS = {
    "a1": C.hc_lowrank // 32
    + C.hc_count,  # 10 down groups of 32 low-rank columns + 4 injection rows
    "a2": C.hidden // 32,  # 80 up/read tiles of 32 columns x 4 streams
    "b1": (C.delta_qkv + C.delta_value) // 64,  # 256 QKVZ tiles of N64
    "b3": C.delta_v_heads * 4,  # 192 (value head, 32-row split)
    "c1": C.hidden // 16,  # 160 out_proj tiles of N16
    "m1": C.hc_lowrank // 32 + C.hc_count,  # as a1, on the MLP HC
    "m2": C.hidden // 32,  # as a2
    "m3": C.experts // 8 + 1,  # router rows 8 t .. 8 t + 7, then the shared expert gate
    "m4": (C.top_k + 1) * (C.moe_inter // 64),  # 110: (rank or shared, 64 columns)
    "m5": C.hidden // 16,  # 160: down tiles of N16, combine, HC write
    "q1": C.qsa_proj // 64,  # 218 tiles of N64 of the fused QSA projection
    "q2": C.q_heads + C.kv_heads,  # 26: norm + RoPE of one q head, then of each k head
    "q3": GRID,  # index blocks 8 t + w + 2048 i (wave w of task t, round i)
    "q4": 9,  # 8 chunk tasks of 8192 blocks, then the top 512 blocks in score order
    "q5": C.kv_heads * SCORE_TILES,  # 130: (score tile t // 2, kv head t % 2)
    "q6": C.q_heads,  # 24: the softmax of one head
    "q7": C.kv_heads * 3 * 8,  # 48: (kv head, 4 q heads, 32-dim tile)
    "p1": C.hidden // 16,  # 160: key n-blocks 4 t .. 4 t + 3 and value n-block t
    "p2": C.hidden // 32,  # 80: columns 32 t .. 32 t + 31 of every stream
}
# Pairs of a row's ``isc`` region: Q3 and Q4 keep the index scores in the
# scratch's second half, from ``isc``'s offset on (``stages.qsa_fast``).
QSA_BLOCKS = 32768
# M1/M2 sit on CTAs without a C1 task, which are idle once their B3 task ends.
# M3 follows on CTAs whose C1 task ended early; M4's shared tasks (100-109)
# and M5 share CTAs 0-9 and 156-159, in that order. Q4 sits on CTAs with no
# Q5-Q7 task (those end at 73 and start at 128). P2 sits on CTAs without a
# P1 task, so its loads go out at the kernel start. A B3 task on an A2 CTA
# starts its B1 only after the A2 task: A2 on CTAs 176-255 shares 16 with B3
# (64 from 128: 0.3 µs a layer more).
BASE = {
    "a1": 0, "a2": 176, "b1": 0, "b3": 0, "c1": 192,
    "m1": 96, "m2": 110, "m3": 190, "m4": 156, "m5": 0,
    "q1": 0, "q2": 218, "q3": 0, "q4": 74, "q5": 128, "q6": 2, "q7": 26,
    "p1": 0, "p2": 160,
}  # fmt: skip

for _name, _n in TASKS.items():
    assert _n <= GRID, f"{_name}: one task per CTA at most"
# The stages a batched build may run in rounds (``common.task_rounds``).
ROUNDS = ("q5", "q7", "p2", "m4")

# ------------------------------------------------------------- mailboxes

PAIR = 8
# name -> pairs. A1 -> A2: fp8 words and exponents of the low-rank row's
# MXFP8 pack. A2 -> B1/B3/C1: the attention HC's mixed row (fp8 words of
# its MXFP8 pack, one E8M0 exponent per 32 values, bf16 words) and the 4
# injection weights. B1 -> B3: bf16 words of qkvz. B3 siblings: the f32 core rows of each split.
# B3 -> C1: fp8 words and exponents of the gated output pack. C1 -> M1/M2:
# the MLP HC's streams (bf16 words) and their square sums per 16-column tile
# and stream (index ``160 b + t``). The MLP HC's own regions mirror the
# attention HC's. M3 -> M4/M5: the router logits (bf16 words) and the shared
# expert's gate ``bf16(sigmoid(.))`` (f32). M4 -> M5: the down projections'
# MXFP8 activations, one row of ``DN_ROW`` pairs per rank (row 10: the shared
# expert), fp8 words then exponents. Q1 -> Q2: bf16 words of the fused QSA
# projection (index q and k, then per head q and gate, then k, then v).
# Q2 -> Q5: bf16 words of the normalized, rotated q heads, then k heads.
# Q3 -> Q4: the index score of every block (f32, at ``isc``, see QSA_BLOCKS).
# Q4 -> Q5-Q7: the selected blocks in score order, then a hint. Q5 -> Q6:
# per head a row of scores (f32, ``SS_ROW`` positions), then one hint per
# task. Q6 -> Q7: per head the probabilities (bf16 words), then a hint. Q7
# -> C1: the gated output's pack, in ``d8``/``de`` as B3's. P1 -> P2: bf16
# words of PLE's key (4 streams) and value. P2 -> A1/A2/C1: the PLE output
# ``h'`` (the attention HC's streams and C1's residual) and its 16-column
# square sums, laid out as ``w`` and ``whs``. A1 -> A2 (``inv``) and M1 -> M2
# (``minv``), from ``hc.INV_ROWS`` rows: the inverse RMS of the 4 streams
# (f32), from each row's task 0.
REGIONS = {
    "lr8": C.hc_lowrank // 4,
    "lre": C.hc_lowrank // 32,
    "x8": C.hidden // 4,
    "xe": C.hidden // 32,
    "xbf": C.hidden // 2,
    "inj": C.hc_count,
    "qkvz": (C.delta_qkv + C.delta_value) // 2,
    "core": TASKS["b3"] * 32,
    "d8": C.delta_value // 4,
    "de": C.delta_value // 32,
    "w": C.hc_width // 2,
    "whs": C.hc_count * TASKS["c1"],
    "mlr8": C.hc_lowrank // 4,
    "mlre": C.hc_lowrank // 32,
    "mx8": C.hidden // 4,
    "mxe": C.hidden // 32,
    "mxbf": C.hidden // 2,
    "minj": C.hc_count,
    "lg": C.experts // 2,
    "sg": 1,
    "dn": (C.top_k + 1) * (C.moe_inter // 4 + C.moe_inter // 32),
    "qp": C.qsa_proj // 2,
    "qn": (C.q_heads + C.kv_heads) * C.head_dim // 2,
    "isc": QSA_BLOCKS,
    "sel": C.index_budget // C.index_compress,
    "selh": 1,
    "ss": C.q_heads * SS_ROW,
    "ssh": TASKS["q5"],
    "pp": C.q_heads * SS_ROW // 2,
    "pph": C.q_heads,
    "pk": C.hc_width // 2,
    "pv": C.hidden // 2,
    "ph": C.hc_width // 2,
    "phs": C.hc_count * TASKS["c1"],
    "inv": C.hc_count,
    "minv": C.hc_count,
}
DN_ROW = C.moe_inter // 4 + C.moe_inter // 32  # 180 pairs: 160 fp8 words, 20 exponents

SC = {}
_off = 0
for _name, _pairs in REGIONS.items():
    SC[_name] = _off
    _off += (_pairs * PAIR + 255) // 256 * 256
SCRATCH_BYTES = _off
# Batched builds (``rows`` requests per launch): request row ``r`` of every
# region sits ``r * SCRATCH_BYTES`` past row 0, i.e. ``r * ROW_PAIRS`` pairs
# on. Row 0 is the unbatched layout. The scratch buffer holds ``MAX_ROWS`` rows.
MAX_ROWS = 4
ROW_PAIRS = SCRATCH_BYTES // PAIR


def scratch_bytes(rows):
    """The scratch buffer of a ``rows``-row build: every row's regions (for
    ``MAX_ROWS`` rows at least), then as many bytes again, where Q3 and Q4
    keep the index scores (``stages.qsa_fast``)."""
    return 2 * max(rows, MAX_ROWS) * SCRATCH_BYTES


# ------------------------------------------------------------- LDS arena
# int32 words; f32 views share the same offsets. Regions are 16-byte aligned.

# A1 injection: lane ``L`` of a row's dot product owns values ``320 L ..
# 320 L + 319``; its 160 words sit at ``INJ_SEG * L``, padded against bank
# conflicts.
INJ_LANES = 32
INJ_SEG = C.hc_width // 2 // INJ_LANES + 4

# M3 router (and B3's b and a rows): row ``r`` of the task's 8 weight rows
# keeps K part ``p`` (a 16-part split, ``ROUTER_PARTS``) as bf16 words
# at ``r * RW_ROW + p * RW_PART``; the mixed row's part ``p`` sits as f32 at
# ``p * RX_PART``. Both are padded against bank conflicts.
ROUTER_PARTS = tuple(
    (192 * p, 192) if p < 8 else (128 * p + 512, 128) for p in range(16)
)  # (first k, length) of each part
RW_PART = 100
RW_ROW = len(ROUTER_PARTS) * RW_PART
RX_PART = 196

_ATTN = (
    (
        "x8",
        C.delta_value // 4,
    ),  # B1: 640 fp8 words of mixed; C1: 1536 of the gated output
    ("xe", C.delta_value // 32),  # exponents for x8
    ("red", WAVES * 16),  # per-wave MFMA rows
    ("qk", 2 * C.delta_k_dim),  # B3: q, k of one key head (f32)
    ("vz", 2 * 32),  # B3: v, z of one 32-row split (f32)
    ("core", 32),  # B3: core rows
    ("sums", 4 * WAVES),  # B3: b and a
    ("cores", C.delta_v_dim),  # B3: the head's core rows from all four splits
    (
        "baw",
        2 * RW_ROW,
    ),  # B3: the b and a weight rows in the router's part layout (bf16)
    ("bax", len(ROUTER_PARTS) * RX_PART),  # B3: the bf16 mixed row by part (f32)
    ("hsum", C.hc_width // 32),  # A1/A2: square sums of the streams' 32-chunks
    ("inv", 4),  # A1/A2: inverse RMS per stream
    ("n8", C.hc_width // 4),  # A1 down: fp8 words of the norm pack
    ("ne", C.hc_width // 32),  # exponents for n8
    ("nbf", INJ_LANES * INJ_SEG),  # A1 injection: bf16 words of the normalized streams
    ("injw", INJ_LANES * INJ_SEG),  # A1 injection: bf16 words of the task's weight row
    ("l8", 3 * 32),  # A2: low-rank fp8 words, zero past 320 values (3 k_iters)
    ("le", 3 * 4),  # exponents for l8, 127 past the 10 real blocks
    ("prod", 4 * 32),  # A2: sigma(up) * norm per stream
)

_MOE = (
    ("rw", 8 * RW_ROW),  # M3: the task's 8 router weight rows
    ("rx", len(ROUTER_PARTS) * RX_PART),  # M3: the mixed row (f32)
    ("lgl", 8),  # M3: the 8 logits
    (
        "top",
        96,
    ),  # M4/M5: the routing (moe.TOP_*), and M5's residual words and injection weights
    ("gx8", C.hidden // 4),  # M4: fp8 words of the mixed row
    ("gxe", C.hidden // 32),  # exponents for gx8
    ("gred", WAVES * 16),  # M4: per-wave MFMA rows (per slice, ``m4_slices``)
    ("dn", (C.top_k + 1) * DN_ROW),  # M5: the polled down activations
    (
        "ys",
        (C.top_k + 1) * 16,
    ),  # M5: each rank's 16 down outputs, then the shared expert's
)

_QSA = (
    (
        "iqn",
        C.index_heads * C.index_dim,
    ),  # Q3: the normalized, rotated index query (f32)
    (
        "q4_keys",
        5120,
    ),  # Q4: radix histograms, wave counts, candidates (qsa_fast.LDS_WORDS)
)

# Q5: key row ``r`` of the tile at ``r * K_STRIDE`` (bf16 words, padded against
# bank conflicts). Q7: probabilities of its 4 heads, ``SS_ROW`` apart (f32).
K_STRIDE = C.head_dim // 2 + 4
_ATTN_QSA = (
    (
        "q5_q",
        (C.q_heads // C.kv_heads + 1) * C.head_dim // 2,
    ),  # Q5: 12 q heads, then the k head
    ("q5_k", SCORE_TILE * K_STRIDE),  # Q5: the tile's key rows
    ("q5_sel", SCORE_TILE // 4),  # Q5: the tile's blocks
    ("q6_e", 9 * 256),  # Q6: e_j, zero from SS_ROW on (9 terms per thread of 256)
    ("q6_red", 4 * WAVES),  # Q6: wave maxima, then wave sums
    ("q7_sel", C.index_budget // C.index_compress),  # Q7: the selected blocks
    ("q7_p", 4 * SS_ROW),  # Q7: the probabilities
    ("q7_cur", 16),  # Q7: the current token's value tile (bf16 words)
    ("q7_part", WAVES * 128),  # Q7: the waves' partial sums
)

_PLE = (
    ("p_x8", C.hidden // 4),  # P1: fp8 words of the embedding row's MXFP8 pack
    ("p_xe", C.hidden // 32),  # exponents for p_x8
    ("p_sig", C.hc_count),  # P2: the 4 gates
    ("p_inv", C.hc_count),  # P2: inverse RMS of the gated streams
)


# Entries of a batched build that hold one copy per request row, ``stride``
# words apart (``for_rows(rows).stride``).
_PER_ROW = {
    "x8", "xe", "red", "qk", "vz", "core", "sums", "cores", "bax",
    "hsum", "inv", "n8", "ne", "l8", "le", "prod",
    "rx", "lgl", "top", "gx8", "gxe", "gred", "dn", "ys",
    "p_x8", "p_xe",
}  # fmt: skip


# Batched builds whose ``_ATTN`` exceeds the LDS split it in two: the HC
# stages' (A1/A2/M1/M2) and the DeltaNet and projection stages' (B1/B3/C1/Q1),
# both starting with ``red`` (shared, as one entry).
_ATTN_D = dict(_ATTN)
_HC_TABLE = tuple(
    (n, _ATTN_D[n])
    for n in ("red", "hsum", "inv", "n8", "ne", "nbf", "injw", "l8", "le", "prod")
)
_DELTA_TABLE = tuple(
    (n, _ATTN_D[n])
    for n in ("red", "x8", "xe", "qk", "vz", "core", "sums", "cores", "baw", "bax")
)
assert sorted(_HC_TABLE + _DELTA_TABLE[1:]) == sorted(_ATTN)
# The stages of each half: a split build runs a CTA barrier ahead of a stage of
# one half that follows a stage of the other.
SPLIT_GROUPS = {"a1": 0, "a2": 0, "m1": 0, "m2": 0, "b1": 1, "b3": 1, "c1": 1, "q1": 1}


def _place(lds, stride, table, rows=1):
    w = 0
    for name, words in table:
        assert lds.get(name, w) == w, name
        lds[name] = w
        stride[name] = (words + 3) // 4 * 4
        if rows > 1 and name in _PER_ROW:
            # Rows 16 words (mod 64) apart: the rows' copies of a word sit in
            # distinct LDS banks, so reading all rows at once has no conflict.
            stride[name] += (16 - stride[name]) % 64
        w += stride[name] * (rows if name in _PER_ROW else 1)
    return w


# Seq builds (``for_rows(rows, seq=True)``): Q5 and Q7 hold the current key
# head and value tile of every row, rows ``q5_q`` / ``q7_cur`` words apart
# after the one-row entry's.
SEQ_GROW = {"q5_q": C.head_dim // 2, "q7_cur": 16}


def _arena(rows=1, split=False, m4_slices=1, seq=False):
    """(LDS offsets, row strides, words) of a build serving ``rows`` requests."""
    lds, stride = {}, {}
    attn = (_HC_TABLE, _DELTA_TABLE) if split else (_ATTN,)
    moe = tuple((n, w * m4_slices if n == "gred" else w) for n, w in _MOE)
    attn_qsa = _ATTN_QSA
    if seq:
        attn_qsa = tuple((n, w + (rows - 1) * SEQ_GROW.get(n, 0)) for n, w in _ATTN_QSA)
    tables = (*attn, moe, _QSA, attn_qsa, _PLE)
    return lds, stride, max(_place(lds, stride, t, rows) for t in tables)


# The attention stages (or, split, each of their halves), the MoE stages
# (M3-M5), Q3-Q4, Q5-Q7 and PLE overlay each other: each of those writes LDS
# only after a CTA barrier.
LDS, STRIDE, LDS_WORDS = _arena()
LDS_LIMIT = 40960


@functools.cache
def for_rows(rows, seq=False):
    """The layout of a build serving ``rows`` requests: ``tasks``, ``base``
    (as :data:`TASKS`, :data:`BASE`), ``lds``, ``stride`` (row ``r`` of a
    per-row LDS entry sits ``r * stride[name]`` words past ``lds[name]``) and
    ``lds_words``, ``split`` (the ``_ATTN`` split, :data:`SPLIT_GROUPS`),
    ``m4_slices`` (64-column slices per M4 task) and ``m3_task_rows``
    (request rows per M3 task). One request is exactly the
    module-level layout. ``seq``: the rows are consecutive tokens of one
    request (an MTP verify build), which only grows two LDS entries
    (:data:`SEQ_GROW`)."""
    assert 1 <= rows <= MAX_ROWS, rows
    assert not seq or rows > 1, "a seq build has 2 rows at least"
    if rows == 1:
        return SimpleNamespace(
            rows=1, tasks=TASKS, base=BASE, lds=LDS, stride=STRIDE, lds_words=LDS_WORDS,
            split=False, m4_slices=1, seq=False, m3_task_rows=1,
        )  # fmt: skip
    tasks, base = dict(TASKS), dict(BASE)
    # A1/M1: every request row's one-request tasks. M2 then M3 follow M1's
    # CTAs (sharing C1's, which end first), so neither waits on an M1 task.
    for name in ("a1", "m1"):
        tasks[name] = rows * TASKS[name]
    base["m2"] = base["m1"] + tasks["m1"]
    base["m3"] = (base["m2"] + tasks["m2"]) % GRID
    # M3: each router block's rows in parts, as few rows a task as fit the
    # grid (one at 2 and 3 rows, two at 4: 130, 195, 130 tasks).
    fit = (
        d
        for d in range(1, rows + 1)
        if rows % d == 0 and rows // d * TASKS["m3"] <= GRID
    )
    m3_task_rows = next(fit)
    tasks["m3"] = rows // m3_task_rows * TASKS["m3"]
    # M4: every row's routed tasks, then the shared expert's (all rows at once);
    # two slices per task where one per task would not fit the grid.
    m4_slices = 1 if (rows * C.top_k + 1) * (C.moe_inter // 64) <= GRID else 2
    tasks["m4"] = (rows * C.top_k + 1) * (C.moe_inter // 64 // m4_slices)
    # QSA: per request row, Q3's blocks over GRID / rows tasks, Q4-Q7 the
    # one-request tasks.
    tasks["q3"] = rows * (GRID // rows)
    for name in ("q4", "q5", "q6", "q7"):
        tasks[name] = rows * TASKS[name]
    # P2: per request row, the one-request tasks.
    tasks["p2"] = rows * TASKS["p2"]
    for name, n in tasks.items():
        assert n <= GRID or name in ROUNDS, f"{name}: one task per CTA at most"
    assert base["q4"] + tasks["q4"] <= GRID  # one CTA per row's selection
    lds, stride, words = _arena(rows, m4_slices=m4_slices, seq=seq)
    split = words > LDS_LIMIT
    if split:
        lds, stride, words = _arena(rows, split=True, m4_slices=m4_slices, seq=seq)
    assert words <= LDS_LIMIT, words
    return SimpleNamespace(
        rows=rows, tasks=tasks, base=base, lds=lds, stride=stride, lds_words=words, split=split,
        m4_slices=m4_slices, seq=seq, m3_task_rows=m3_task_rows,
    )  # fmt: skip


def b3_tiles(kh, vh, s):
    """The 6 B1 tiles a B3 task reads: q and k of key head ``kh`` (2 each), and
    the v and z tiles holding rows ``32 s .. 32 s + 31`` of value head ``vh``."""
    q = 2 * kh
    k = C.delta_key // 64 + 2 * kh
    v = 2 * C.delta_key // 64 + 2 * vh + s // 2
    z = C.delta_qkv // 64 + 2 * vh + s // 2
    return q, q + 1, k, k + 1, v, z
