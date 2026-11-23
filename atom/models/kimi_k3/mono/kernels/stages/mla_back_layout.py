# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Pure-Python geometry of the gfx1250 MLA back half (``mla_back_gfx1250``):
which wave projects which ``W_UV`` rows, which wave merges which output row,
the lines a row travels as and the launch's banks. No FlyDSL, so every
coordinate is checked on a CPU-only host.

The back half takes a full-attention layer from the decode attention's
partials to the attention output: ``o [rows, heads, 512]`` BF16 and
``lse [rows, heads]`` f32 over this rank's KV shard for the DCP group's
``heads = dcp * local_heads`` query heads; ``W_UV [heads, 128, 512]`` FP8 with
one scale (project before merge); the merge of each rank's own
``local_heads`` over the group's shards by their LSE; the output gate; and
o_proj (``[HIDDEN, local_heads * 128]``) with its TP all-reduce.

A ``(token, head)`` row of 128 BF16 travels as 32 lines of 16 B -- four BF16,
the row's LSE, the launch's epoch -- so lane ``j`` of a merging wave polls
line ``j`` of every shard and needs nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass

from atom.models.kimi_k3.mono.config import HIDDEN, MAX_ROWS
from atom.models.kimi_k3.mono.kernels.stages.gemv_layout import ROWS
from atom.mono.plan.architecture import GFX1250

V_DIM = 128
KV_LORA = 512
THREADS = 256
LANES = GFX1250.wave_size
WAVES = THREADS // LANES
GRID = 256
# o_proj's CTAs: 32 output columns each, as ``k2_oproj_ar_gfx1250``
OPROJ_COLUMNS = 2 * ROWS
OPROJ_CTAS = HIDDEN // OPROJ_COLUMNS
# W_UV: a head's 128 rows as eight 16-row tasks, K as 16 chained K32 steps
N_BLOCKS = V_DIM // ROWS
K_STEPS = KV_LORA // 32
# a line: four BF16 (two words), the row's LSE, the epoch
LINE_VALUES = 4
LINE_WORDS = 4
ROW_LINES = V_DIM // LINE_VALUES
assert ROW_LINES == LANES
# a CTA's waves [0, WUV_WAVES) project and push (a wave's loads issued after
# its remote stores wait out their round trip); the others merge
WUV_WAVES = 3
MERGE_WAVES = WAVES - WUV_WAVES
WUV_SLOTS = GRID * WUV_WAVES
MERGE_SLOTS = GRID * MERGE_WAVES
# the core's flags a thread of an o_proj CTA polls, four a load
FLAG_VECTOR = 4
# the o_proj all-reduce's lines (``k2_oproj_ar_gfx1250``): three tokens' BF16
# pair and the epoch
AR_LINE_TOKENS = 3


@dataclass(frozen=True)
class BackPlan:
    """The back half of one layer shape at ``rows`` token rows: ``tp`` ranks
    all-reduce o_proj, ``dcp`` ranks merge (1: no merge, the rank's own
    heads), ``local_heads`` query heads a rank."""

    rows: int
    tp: int
    dcp: int
    local_heads: int

    def __post_init__(self):
        if not 1 <= self.rows <= MAX_ROWS:
            raise ValueError(f"the MLA back takes 1..{MAX_ROWS} rows, got {self.rows}")
        if self.tp < 1 or self.dcp < 1 or self.local_heads < 1:
            raise ValueError(f"TP{self.tp} DCP{self.dcp} {self.local_heads} heads")
        if HIDDEN % (self.tp * OPROJ_COLUMNS):
            raise ValueError(f"a rank's columns must be whole CTAs at TP{self.tp}")
        if self.heads * N_BLOCKS > WUV_SLOTS:
            raise ValueError(
                f"{self.heads} heads: W_UV holds {WUV_SLOTS // N_BLOCKS} at most"
            )
        if self.merge_rows > THREADS * FLAG_VECTOR * 2:
            raise ValueError(f"{self.merge_rows} merged rows a rank: too many flags")

    @property
    def heads(self) -> int:
        """The DCP group's query heads: W_UV's batch."""
        return self.dcp * self.local_heads

    @property
    def proj(self) -> int:
        """o_proj's input width on a rank."""
        return self.local_heads * V_DIM

    @property
    def tiles(self) -> int:
        return -(-self.rows // ROWS)

    @property
    def merge_rows(self) -> int:
        """The ``(token, local head)`` rows a rank merges."""
        return self.rows * self.local_heads

    @property
    def merge_rounds(self) -> int:
        """Rows a merging wave takes, in turn."""
        return -(-self.merge_rows // MERGE_SLOTS)

    @property
    def flag_vectors(self) -> int:
        """Four-flag loads a thread of an o_proj CTA polls."""
        return -(-self.merge_rows // (THREADS * FLAG_VECTOR))

    @property
    def flag_words(self) -> int:
        """``flags``: a word a merged row, padded to whole loads."""
        return self.flag_vectors * THREADS * FLAG_VECTOR

    @property
    def merge_bank_words(self) -> int:
        """A merge bank: ``[source rank][merged row][line]``."""
        return self.dcp * self.merge_rows * ROW_LINES * LINE_WORDS

    @property
    def ar_triples(self) -> int:
        """Token triples: an all-reduce line carries three tokens."""
        return -(-self.rows // AR_LINE_TOKENS)

    @property
    def ar_rounds(self) -> int:
        """All-reduce lines a thread of an o_proj CTA pushes and polls
        (``k2_oproj_ar_gfx1250.line_rounds``): one up to 48 rows."""
        return -(-self.ar_triples * (OPROJ_COLUMNS // 2) // THREADS)

    @property
    def ar_bank_words(self) -> int:
        """The one-shot o_proj all-reduce's bank
        (``k2_oproj_ar_gfx1250.oneshot_bank_words``)."""
        return LINE_WORDS * self.ar_triples * self.tp * (HIDDEN // 2)

    @property
    def bank_words(self) -> int:
        """A bank: the merge's lines, then the all-reduce's."""
        return self.merge_bank_words + self.ar_bank_words

    @property
    def peer_bytes(self) -> int:
        """A rank's receive memory: two banks (epoch parity)."""
        return 2 * self.bank_words * 4

    def line_word(self, source: int, row: int, line: int) -> int:
        """The word of ``line`` of merged row ``row`` from rank ``source``
        inside a merge bank."""
        if not (
            0 <= source < self.dcp
            and 0 <= row < self.merge_rows
            and 0 <= line < ROW_LINES
        ):
            raise ValueError(f"line {line} of row {row} from rank {source}")
        return ((source * self.merge_rows + row) * ROW_LINES + line) * LINE_WORDS


# --- where a line lives. The kernel computes every address it stores to or
# polls with these (on device integers), the tests with the same functions on
# host integers: a line's byte offset from a rank's bank base is
# ``4 * (bank_base(plan, epoch) + word)``, 16 B long.


def bank_base(plan: BackPlan, epoch):
    """The word offset of launch ``epoch``'s bank in a rank's receive
    memory: the banks alternate, so a line is rewritten two launches on."""
    return (epoch & 1) * plan.bank_words


def uv_line_word(plan: BackPlan, source, token, local_head, block, lane_group, half):
    """The word, inside a bank of the head's owner, of the line a W_UV lane
    of rank ``source`` pushes: values ``block * 16 + lane_group * 8 + half *
    4 ..+ 4`` of ``(token, local_head)``."""
    row = token * plan.local_heads + local_head
    line = block * (ROWS // LINE_VALUES) + lane_group * 2 + half
    return ((source * plan.merge_rows + row) * ROW_LINES + line) * LINE_WORDS


def merge_line_word(plan: BackPlan, source, row, lane):
    """The word, inside this rank's bank, of line ``lane`` of merged row
    ``row`` from rank ``source``: what lane ``lane`` of the merging wave
    polls."""
    return ((source * plan.merge_rows + row) * ROW_LINES + lane) * LINE_WORDS


def ar_line_word(plan: BackPlan, source, triple, column_pair):
    """The word, inside a bank (past the merge's lines), of rank ``source``'s
    all-reduce line of token triple ``triple`` and output column pair
    ``column_pair``: pushed to every rank, polled by the pair's CTA."""
    return plan.merge_bank_words + LINE_WORDS * (
        (source * plan.ar_triples + triple) * (HIDDEN // 2) + column_pair
    )


def uv_weight_word(head, n, step, half, lane_group):
    """The word of ``W_UV`` (``[heads, 128, 512]`` FP8) a lane reads eight
    weights (two words) of row ``n`` of ``head`` at: K ``32 step + 16 half +
    8 lane_group``."""
    return (head * V_DIM + n) * (KV_LORA // 4) + (
        32 * step + 16 * half + lane_group * 8
    ) // 4


def uv_token_word(plan: BackPlan, token, head, step, half, lane_group):
    """The word of ``o`` (``[rows, heads, 512]`` BF16) a lane reads eight
    values (four words) of ``(token, head)`` at: the same K."""
    return (token * plan.heads + head) * (KV_LORA // 2) + (
        32 * step + 16 * half + lane_group * 8
    ) // 2


def core_word(plan: BackPlan, row, lane):
    """The word of ``core`` (``[rows, proj]`` BF16) lane ``lane`` of merged
    row ``row``'s wave stores its four gated values at (two words)."""
    token = row // plan.local_heads
    local_head = row % plan.local_heads
    return (token * plan.proj + local_head * V_DIM) // 2 + lane * (LINE_VALUES // 2)


def wuv_job(plan: BackPlan, cta: int, wave: int) -> tuple[int, int] | None:
    """``(head, 16-row block)`` of ``W_UV`` wave ``wave`` of CTA ``cta``
    projects (every token tile of it), or None."""
    if not (0 <= cta < GRID and 0 <= wave < WAVES):
        raise ValueError(f"wave {wave} of CTA {cta}")
    slot = cta * WUV_WAVES + wave
    if wave >= WUV_WAVES or slot >= plan.heads * N_BLOCKS:
        return None
    return divmod(slot, N_BLOCKS)


def merge_row(plan: BackPlan, cta: int, wave: int, round_index: int) -> int | None:
    """The merged row (``token * local_heads + local head``) wave ``wave`` of
    CTA ``cta`` takes in round ``round_index``, or None."""
    if not (0 <= cta < GRID and 0 <= wave < WAVES):
        raise ValueError(f"wave {wave} of CTA {cta}")
    if wave < WUV_WAVES or not 0 <= round_index < plan.merge_rounds:
        return None
    row = cta * MERGE_WAVES + wave - WUV_WAVES + round_index * MERGE_SLOTS
    return row if row < plan.merge_rows else None


def pushed_lines(plan: BackPlan) -> int:
    """16 B lines a rank pushes for the merge (its own heads' stay local)."""
    return plan.rows * (plan.heads - plan.local_heads) * ROW_LINES
