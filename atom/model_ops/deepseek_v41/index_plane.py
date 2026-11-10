# SPDX-License-Identifier: MIT
"""The index plane's two formats, as every reader and the writer see them.

FP8: one plane, a block of `index_block_rows` preshuffled E4M3 rows followed by
their fp32 scales (`fp8_indexer_block_fields`).

FP4: the official arithmetic (E2M1, one E8M0 per 32 dims, `quantize_fp4`'s
group-32 grid), values and scales in two planes, a page per candidate block.
gfx1250's OPUS scorer reads natural [page, row, byte/group] order. gfx950's
OPUS scorer reads 64-row groups: values [group, chunk, row, 16 bytes],
scales [group, chunk % 2, row % 32, chunk // 2, row // 32]. Its logical
8-row page p is rows (p % 8) * 8 through +7 of group p // 8. gfx950
FlyDSL reads the page-8 layout below. Key `t` of a page, its K tile `kt`, its
32-dim chunk `c` and that chunk's byte `b` (two E2M1 values) sit at

    values  [kt][t % 4][c][t // 4][b]      (16 bytes a chunk)
    scales  [kt][c][t]

so the four keys an MFMA lane reads (`4 l + nt`) are adjacent, in one page,
and their four E8M0 bytes are one dword (`pa_mqa_logits_fp4_rowgroup`).
"""

from typing import NamedTuple

import torch

from atom.model_ops.attentions.pool_layout.v4_pool_fields import (
    FP4_GFX950_OPUS,
    FP4_GFX950_PRESHUFFLE,
    FP4_GFX1250_NATURAL,
    fp4_indexer_layout_for_arch,
)
from atom.model_ops.indexer.layout import MFMA as FP4_MFMA
from atom.model_ops.indexer.layout import NATURAL as FP4_OPUS
from atom.model_ops.indexer.layout import PRESHUFFLE as FP4_FLYDSL


def fp4_index_layout_for_arch(arch, *, backend):
    """Use the shared layout policy with V4.1's index-plane names."""
    return {
        FP4_GFX950_OPUS: FP4_MFMA,
        FP4_GFX950_PRESHUFFLE: FP4_FLYDSL,
        FP4_GFX1250_NATURAL: FP4_OPUS,
    }[fp4_indexer_layout_for_arch(arch, backend=backend)]


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
    # FP4 layouts have the same logical shape and byte count. MFMA stores
    # eight logical pages together as one 64-row group; a whole PAGE still
    # contains complete groups, so PAGE copies preserve the byte layout.
    # Keep the layout with the plane so its writer and readers agree.
    layout: str = FP4_FLYDSL

    @property
    def fp4(self):
        return self.scales is not None
