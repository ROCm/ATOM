# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""The peer buffer: every region one rank's kernels store into another's, laid
out once for the kernels, the runner's allocation and the debug report."""

from atom.models.deepseek_v41.mono import index_plan as ip
from atom.models.deepseek_v41.mono.kernels.dims import (
    HC,
    HIDDEN,
    KV_DIM,
    Q_RANK,
    ROUTE_COPIES,
    TOPK,
    Dims,
)
from atom.mono.plan.execution import BLOCKS
from atom.mono.plan.layout import pair_layout
from atom.mono.runtime.step_begin import ZERO_ALIGN

SLICE_COLS = 32  # ``attn_pre.COLS``
# ``col_gather``'s planes a row: the residual's HC streams, then up to 4 taps
CG_PLANES = HC + 4
SLICE_PART = 4 * (4 + 2) + 1  # ``attn_pre.PART``: the mixes, then the squares


def peer_layout(tokens: int, d: Dims) -> dict[str, tuple[int, int]]:
    """Region -> (byte offset, bytes) for a step of ``tokens`` MoE rows, 8 B a
    (word, tag) pair. ATTN and FFN: the attention and FFN partials, [source]
    [token][hidden / 2] (``Dims.attn_sources``, ``Dims.ranks``). Then a
    token-parallel step's gathers, [token][..] each: XG the normed rows (K2's
    MoE input); QG / QGS the q latent's MXFP8 words / codes and KG the window
    row (K1); NBG the normed rows (the compressor's) and IWG the head weights
    (an indexer layer's K1); IRDY / ICRDY: a row's selection / candidate
    blocks are in every rank's ``plain_layout`` (put by its owner's K2a). RT:
    under DP attention every rank's routes (``moe.stage_route``'s copies, each
    rank its own step's tokens). MTOP / MRANK: the draft head's task and rank
    bests of a block position, HXR its rows of x landed (``draft_head``). CG:
    a sharded residual's columns gathered (``col_gather``). EPF / ENG: an
    Engram layer's projection flags and gate sums (``engram``). CX: the
    draft's context projection's columns gathered (``draft_ctx``)."""
    row = tokens * HIDDEN // 2
    slices = HIDDEN // SLICE_COLS
    return pair_layout(
        [
            ("attn", d.attn_sources * row),
            # a sharded seam (``attn_pre.shard_seam``): every rank's slices'
            # LIN and its sums of squares, gathered; the mixes' gate block
            # sums, seam A's and seam F's apart
            ("ling", row),
            ("lssqg", tokens * slices),
            ("pmixg", slices * tokens * SLICE_PART),
            ("pmixgf", slices * tokens * SLICE_PART),
            ("ffn", d.ranks * row),
            ("xg", row),
            ("qg", tokens * Q_RANK // 4),
            ("qgs", tokens * Q_RANK // 32),
            ("kg", tokens * KV_DIM // 2),
            ("nbg", row),
            ("iwg", tokens * ip.HEADS),
            ("irdy", tokens),
            ("icrdy", tokens),
            # a split row's answers in its ``plain_layout`` IXSEL, [token]
            # [source rank] (``index_score.stage_imerge``)
            ("ixrdy", tokens * d.tp),
            ("rt", ROUTE_COPIES * tokens * 2 * TOPK),
            # the draft head's (key, id) bests (``draft_head``): its tasks'
            # on this rank [row][task], every rank's [source rank][row]
            ("mtop", tokens * BLOCKS * 2),
            ("mrank", d.ranks * tokens * 2),
            # a sharded residual's columns gathered (``col_gather``): [row]
            # [plane][hidden / 2]
            ("cg", tokens * CG_PLANES * HIDDEN // 2),
            # the draft head's rows of x landed (``draft_head.tail_row``), a
            # flag a row
            ("hxr", tokens),
            # an Engram layer's (``engram``): its projection CTAs done, a flag
            # a CTA; every rank's gate sums, [row][stream][source rank][3]
            ("epf", BLOCKS),
            ("eng", tokens * HC * d.ranks * 3),
            # the draft's context projection (``draft_ctx``): every group
            # rank's columns of a row, bf16 pairs [row][hidden / 2]
            ("cx", row),
        ]
    )


PEER_REGIONS = tuple(peer_layout(1, Dims(1)))


def peer_offsets(tokens: int, d: Dims) -> dict[str, int]:
    return {region: off for region, (off, _) in peer_layout(tokens, d).items()}


def peer_bytes(tokens: int, d: Dims) -> int:
    """The buffer's bytes: past its last region, in whole ``ZERO_ALIGN``
    units (the step begin zeroes it in those)."""
    end = max(off + n for off, n in peer_layout(tokens, d).values())
    return -(-end // ZERO_ALIGN) * ZERO_ALIGN


def plain_layout(tokens: int, selecting: int, tp: int) -> dict[str, tuple[int, int]]:
    """Region -> (byte offset, bytes) in the peer buffer's never-zeroed part
    (``PeerBuffer.plain``), int32 words: IROW the ``selecting`` layers' pool
    rows, [layer][token][TOPK], and ICAND the candidate blocks, [token]
    [TOPK_BLOCKS]. Each row's are written whole by the K2a that selects it
    (``index_score._st_out``), before the step reads them. IXSEL: a split
    row's (value bits, column) answers, [token][source rank][TOPK][2]
    (``index_score._put_answer``), read past their IXRDY flags."""
    out, off = {}, 0
    for name, words in (
        ("irow", selecting * tokens * ip.TOPK),
        ("icand", tokens * ip.TOPK_BLOCKS),
        ("ixsel", tokens * tp * ip.TOPK * 2),
    ):
        out[name] = (off, words * 4)
        off += words * 4
    return out
