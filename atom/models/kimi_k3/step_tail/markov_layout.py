# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The draft launch's sizes and LDS regions (the draft head, then the Markov
block sampling): plain integers beside ``layout`` (whose peer lines, counters
and pointer table it shares), in a file of its own so the head + verify
kernel's build key does not move with it."""

from dataclasses import dataclass

from atom.models.kimi_k3.step_tail import layout as L

RANKS = (256, 512)  # Markov ranks the kernel is built for (a TDM tile width)
# counters beside layout's (i32 words of ``cnt``): the draft launches' epoch;
# round k's hand-off count is layout.CNT_MARKOV + k
CNT_MEPOCH = 4
# ... and the token of the last round whose x_k CTA 0 published to the rank's
# other CTAs: epoch * 8 + round + 1
CNT_MFLAG = 5
assert L.CNT_SITE < CNT_MEPOCH < CNT_MFLAG < L.CNT_MARKOV and L.MAX_DRAFTS < 8


def draft_refusal(requests: int, drafts: int, rank: int) -> str:
    """Why there is no draft launch for ``requests`` blocks of ``drafts``
    tokens at Markov rank ``rank``, or ''."""
    if not 1 <= drafts <= L.MAX_DRAFTS:
        return f"{drafts} drafts a request (1..{L.MAX_DRAFTS})"
    if not 1 <= requests <= L.MAX_REQ or requests * drafts > L.MAX_ROWS:
        return f"{requests} requests of {drafts} drafts past {L.MAX_REQ} / {L.MAX_ROWS} rows"
    if rank not in RANKS:
        return f"Markov rank {rank} (built for {RANKS})"
    return ""


def pitch(rank: int) -> int:
    """Bytes of a staged W1 / W2 row in LDS."""
    return (rank + L.LDS_PAD) * 2


@dataclass(frozen=True)
class MarkovLds:
    """LDS byte offsets of the Markov rounds: below the head's phase buffers'
    end (free once the head is done); the draft logits sit in the head
    layout's ``red`` region, written by the head's epilogue."""

    w2: int  # the CTA's 80 W2 rows (TDM, once)
    w1: int  # W1[x] of every request (a round's B operand rows)
    x: int  # x_{k-1} a request, i32
    best: int  # a lane's best of its request, i64 [THREADS]
    lane: int  # CTA 0: a wave's lanes' bests [WAVES][MAX_REQ][LANES] i64
    wave: int  # CTA 0: [WAVES][MAX_REQ] i64
    peer: int  # the round's [request][source] bests, i64
    size: int


def markov_lds(rank: int) -> MarkovLds:
    w1 = L.TILE_WAVES * L.TILE * pitch(rank)
    x = w1 + L.MAX_REQ * pitch(rank)
    best = x + L.MAX_REQ * 4
    lane = best + L.THREADS * 8
    wave = lane + L.WAVES * L.MAX_REQ * L.LANES * 8
    peer = wave + L.WAVES * L.MAX_REQ * 8
    size = peer + L.MAX_REQ * L.TP * 8
    return MarkovLds(0, w1, x, best, lane, wave, peer, size)


# the draft logits [WAVES][MAX_ROWS] rows of 16 bf16 fit the head's scratch,
# and the rounds' regions end below it at every width
assert L.WAVES * L.MAX_ROWS * 32 <= L.RED_BYTES
assert max(markov_lds(r).size for r in RANKS) <= L.NBUF * L.phase_buffer(1)
