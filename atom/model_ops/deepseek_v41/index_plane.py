# SPDX-License-Identifier: MIT
"""The index plane's two formats, as every reader and the writer see them.

FP8: one plane, a block of `index_block_rows` preshuffled E4M3 rows followed by
their fp32 scales (`fp8_indexer_block_fields`).

FP4: the official arithmetic (E2M1, one E8M0 per 32 dims, `quantize_fp4`'s
group-32 grid), values and scales in two planes, a page per candidate block.
gfx1250's OPUS scorer reads natural [page, row, byte/group] order. gfx950's
FlyDSL scorer reads the page-8 layout below. Key `t` of a page, its K tile `kt`, its
32-dim chunk `c` and that chunk's byte `b` (two E2M1 values) sit at

    values  [kt][t % 4][c][t // 4][b]      (16 bytes a chunk)
    scales  [kt][c][t]

so the four keys an MFMA lane reads (`4 l + nt`) are adjacent, in one page,
and their four E8M0 bytes are one dword (`pa_mqa_logits_fp4_rowgroup`).
"""

from typing import NamedTuple

import torch

FP4_FLYDSL = "preshuffle"
FP4_OPUS = "natural"


def fp4_index_layout_for_arch(arch):
    return FP4_OPUS if arch == "gfx1250" else FP4_FLYDSL


FP4_KEYS_A_LANE = 4  # the keys of a lane that sit adjacent in a page
FP4_CHUNK_DIMS = 32  # dims an E8M0 scale covers: a chunk of 16 bytes
FP4_TILE_DIMS = 128  # dims an MFMA K tile covers: four chunks


class IndexUnits(NamedTuple):
    """One owner's index plane as `[tiles, tile rows, bytes]` views, which is
    what a block id addresses and what a scorer is handed. ``scales`` is the
    FP4 E8M0 plane; None for FP8, whose scales sit inside ``values``' blocks.
    """

    values: torch.Tensor
    scales: torch.Tensor | None = None
    # Both FP4 layouts have the same shape and byte count. Keep the layout
    # with the plane so the writer, query quantizer and scorer agree.
    layout: str = FP4_FLYDSL

    @property
    def fp4(self):
        return self.scales is not None
