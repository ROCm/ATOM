# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The peer buffer: every region one rank's kernels store into another's, laid
out once for the kernels, the runner's allocation and the debug report."""

from atom.models.deepseek_v41.mono import index_plan as ip
from atom.models.deepseek_v41.mono.kernels.dims import HIDDEN, KV_DIM, Q_RANK
from atom.mono.plan.layout import pair_layout


def peer_layout(tokens: int, tp: int) -> dict[str, tuple[int, int]]:
    """Region -> (byte offset, bytes) for a ``tokens``-row step, 8 B a (word,
    tag) pair. ATTN and FFN: the attention and FFN partials, [source rank]
    [token][hidden / 2]. Then a token-parallel step's gathers, [token][..]
    each: XG the normed rows (K2's MoE input); QG / QGS the q latent's MXFP8
    words / codes and KG the window row (K1); NBG the normed rows (the
    compressor's) and IWG the head weights (an indexer layer's K1)."""
    row = tokens * HIDDEN // 2
    return pair_layout(
        [
            ("attn", tp * row),
            ("ffn", tp * row),
            ("xg", row),
            ("qg", tokens * Q_RANK // 4),
            ("qgs", tokens * Q_RANK // 32),
            ("kg", tokens * KV_DIM // 2),
            ("nbg", row),
            ("iwg", tokens * ip.HEADS),
        ]
    )


PEER_REGIONS = tuple(peer_layout(1, 1))


def peer_offsets(tokens: int, tp: int) -> dict[str, int]:
    return {region: off for region, (off, _) in peer_layout(tokens, tp).items()}


def peer_bytes(tokens: int, tp: int) -> int:
    return max(off + n for off, n in peer_layout(tokens, tp).values())
