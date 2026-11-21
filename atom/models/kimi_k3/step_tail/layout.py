# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The step tail's sizes, its LDS regions and every address it pushes to or
polls on a peer: plain integers, shared by the kernels, the runner and the CPU
tests (no FlyDSL, no GPU).

One deployment: TP8, hidden 7168, a 163840-id vocabulary (20480 ids a rank),
steps of ``requests x (drafts + 1)`` rows up to 64.
"""

from dataclasses import dataclass

TP = 8
HIDDEN = 7168
VOCAB = 163840
VR = VOCAB // TP  # vocab ids a rank
MAX_ROWS = 64  # target rows a step
MAX_QLEN = 8  # rows a request: the drafts and the bonus row
MAX_REQ = 16  # requests a step
MAX_DRAFTS = MAX_QLEN - 1
MARKOV_RANK_MAX = 512

CTAS, THREADS, WAVES, LANES = 256, 160, 5, 32  # every wave holds a vocab tile
TILE = 16  # vocab ids of a wave's WMMA tile
TILES = VR // TILE  # 1280
TILE_WAVES = -(-TILES // CTAS)  # waves of a CTA holding a tile: 5
K_STEP = 32  # hidden columns of a WMMA
STEPS = HIDDEN // K_STEP  # 224
PHASE_STEPS = 8  # K steps a staged phase holds (the CTA's weights and x)
PHASES = STEPS // PHASE_STEPS  # 28
DEPTH = 3  # phases in flight (TDM copies issued ahead of the one multiplied)
NBUF = DEPTH + 1  # LDS phase buffers
LDS_PAD = 8  # BF16 elements the TDM pads a staged row with
LDS_LIMIT = 327680

LINE = 16  # bytes of a peer line: three words and the launch's epoch
# peer region of a rank, two banks each (a line stays until launch epoch + 2):
# the argmax lines [bank][source][row], then the Markov lines
# [bank][position][source][request]
SAMPLE_BANK = TP * MAX_ROWS * LINE
SAMPLE_OFF = 0
MARKOV_BANK = MAX_DRAFTS * TP * MAX_REQ * LINE
MARKOV_OFF = SAMPLE_OFF + 2 * SAMPLE_BANK
PEER_BYTES = 1 << 16
assert MARKOV_OFF + 2 * MARKOV_BANK <= PEER_BYTES
assert TILE_WAVES == WAVES and TILES == CTAS * TILE_WAVES
assert PHASE_STEPS * K_STEP in (256, 512)  # a width the TDM pads rows of
assert STEPS == PHASES * PHASE_STEPS

# this rank's scratch: every CTA's best of each row (i64), then the counters
REC_BYTES = CTAS * MAX_ROWS * 8
# counters (i32): the launches' epoch, the head hand-off's count, timeouts
CNT_EPOCH, CNT_HEAD, CNT_TIMEOUT, CNT_SITE, CNT_MARKOV = 0, 1, 2, 3, 8
CNT_WORDS = CNT_MARKOV + MAX_DRAFTS + 1
# the pointer table the kernels read (8 B each)
POINTERS = (
    "w",  # this rank's LM head rows [VR, HIDDEN] bf16
    "rec",  # REC_BYTES
    "cnt",  # CNT_WORDS i32
    "peers",  # every rank's peer region, i64 a rank
    "outtok",  # [requests, qlen] i32
    "verdict",  # [3, requests] i32
    "tap",  # a check build's logits [rows, VR] bf16
    "dlog",  # the draft's logits [draft rows, VR] bf16
    "w1",  # Markov W1 [VOCAB, rank] bf16
    "w2",  # this rank's Markov W2 rows [VR, rank] bf16
    "drafts",  # [requests, drafts] i32
    "mbest",  # a Markov round's x_k, CTA 0 to the rank's other CTAs [MAX_REQ] i32
    "stamps",  # a timeline build's clock stamps [requests][8] i64
)


def sample_line(bank: int, source: int, row: int) -> int:
    """Byte offset in a rank's peer region of ``source``'s argmax line of
    ``row`` in ``bank``."""
    return SAMPLE_OFF + ((bank * TP + source) * MAX_ROWS + row) * LINE


def markov_line(bank: int, position: int, source: int, request: int) -> int:
    """... of ``source``'s Markov line of ``request`` at block ``position``."""
    return (
        MARKOV_OFF
        + (((bank * MAX_DRAFTS + position) * TP + source) * MAX_REQ + request) * LINE
    )


@dataclass(frozen=True)
class Step:
    """A served width: ``rows`` target rows of ``qlen`` a request."""

    rows: int
    qlen: int

    @property
    def requests(self) -> int:
        return self.rows // self.qlen

    @property
    def drafts(self) -> int:
        return self.qlen - 1

    @property
    def draft_rows(self) -> int:
        return self.requests * self.drafts


def step_refusal(rows: int, qlen: int) -> str:
    """Why the tail has no build for a step of ``rows`` rows, ``qlen`` a
    request, or ''."""
    if not 2 <= qlen <= MAX_QLEN:
        return f"{qlen} rows a request (2..{MAX_QLEN})"
    if rows <= 0 or rows % qlen:
        return f"{rows} rows are not whole requests of {qlen}"
    if rows > MAX_ROWS or rows // qlen > MAX_REQ:
        return f"{rows} rows / {rows // qlen} requests past {MAX_ROWS} / {MAX_REQ}"
    return ""


def row_groups(rows: int) -> int:
    """16-row WMMA groups of ``rows`` rows."""
    return -(-rows // 16)


def x_stride() -> int:
    """Bytes of a staged row in LDS (a weight row or an x row): a phase's
    columns and a pad."""
    return (PHASE_STEPS * K_STEP + LDS_PAD) * 2


def w_buffer() -> int:
    """Bytes of one staged phase of a CTA's weight rows."""
    return TILE_WAVES * TILE * x_stride()


def x_buffer(rows: int) -> int:
    """Bytes of one staged x phase."""
    return row_groups(rows) * 16 * x_stride()


def phase_buffer(rows: int) -> int:
    """Bytes of one staged phase, the weights then x (NBUF are live)."""
    return w_buffer() + x_buffer(rows)


def x_rows_per_wave(rows: int) -> int:
    """x rows a wave's TDM copy of a phase brings (the last waves fewer)."""
    return -(-rows // WAVES)


# LDS scratch of the reductions: a lane's best a row group [groups][THREADS] i64
# (the head's epilogue), then a wave's lanes' bests of a request's rows
# [WAVES][MAX_QLEN][LANES] i64 (the verify CTA)
RED_BYTES = WAVES * MAX_QLEN * LANES * 8
assert RED_BYTES >= 4 * THREADS * 8


@dataclass(frozen=True)
class Lds:
    """LDS byte offsets of a head launch at ``rows`` rows."""

    red: int  # the reductions' scratch (RED_BYTES)
    wbest: int  # the verify CTA's [wave][row of the request] bests, i64
    peer: int  # ... its [row of the request][source] bests, i64
    ids: int  # ... its rows' ids, i32
    size: int


def head_lds(rows: int) -> Lds:
    red = NBUF * phase_buffer(rows)
    wbest = red + RED_BYTES
    peer = wbest + WAVES * MAX_QLEN * 8
    ids = peer + MAX_QLEN * TP * 8
    size = ids + MAX_QLEN * 4
    if size > LDS_LIMIT:
        raise ValueError(f"{rows} rows need {size} B of LDS (> {LDS_LIMIT})")
    return Lds(red, wbest, peer, ids, size)


def split_steps(kind: str, split_k: int, block_k: int) -> tuple[int, ...]:
    """The K steps (of 32 columns) at which production's LM head GEMM starts a
    new partial sum: none for one pass over K. ``kind`` 'opus': AITER's split-K
    workspace kernel, ``split_k`` slices of whole ``block_k`` tiles, the last
    one short; 'flydsl': equal slices of K."""
    if split_k <= 1:
        return ()
    if kind == "opus":
        tiles = -(-HIDDEN // block_k)
        while split_k > 1 and (split_k - 1) * -(-tiles // split_k) >= tiles:
            split_k -= 1  # the launcher's clamp: no empty trailing slice
        per = -(-tiles // split_k) * block_k
    elif kind == "flydsl":
        if HIDDEN % split_k:
            raise ValueError(f"split_k {split_k} does not divide {HIDDEN}")
        per = HIDDEN // split_k
    else:
        raise ValueError(f"unknown LM head kernel kind {kind!r}")
    if per % K_STEP:
        raise ValueError(f"a {per}-column K slice is not whole WMMA steps")
    return tuple(z * per // K_STEP for z in range(1, split_k))


def peer_addresses(rows: int, qlen: int, markov: bool = True):
    """Every (kind, source rank, destination rank, byte offset) a launch of
    this width pushes, and every (kind, polling rank, byte offset) it polls,
    over both banks: what the address audit holds inside ``PEER_BYTES``."""
    step = Step(rows, qlen)
    pushes, polls = [], []
    for bank in (0, 1):
        for source in range(TP):
            for row in range(rows):
                at = sample_line(bank, source, row)
                polls.extend(("sample", rank, at) for rank in range(TP))
                pushes.extend(
                    ("sample", source, dest, at) for dest in range(TP) if dest != source
                )
            if not markov:
                continue
            for position in range(step.drafts):
                for request in range(step.requests):
                    at = markov_line(bank, position, source, request)
                    polls.extend(("markov", rank, at) for rank in range(TP))
                    pushes.extend(
                        ("markov", source, dest, at)
                        for dest in range(TP)
                        if dest != source
                    )
    return pushes, polls
