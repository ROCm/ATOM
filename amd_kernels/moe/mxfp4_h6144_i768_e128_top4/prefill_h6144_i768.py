# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""What the BM128 and BM256 expert hosts share (H 6144, I 768 per rank): ``Mxfp4PrefillScales``, the expert scales every
host reads (the weights are the bytes of ``layout``, one copy per layer), and GEMM2's column chunks and grid.
"""

from dataclasses import dataclass

import torch

from .common.epilogue import _NUM_XCDS

H, I = 6144, 768  # noqa: E741

# GEMM2's column chunks per m-tile at every row count: 4 n-tiles of 256 columns each
GEMM2_N_SPLIT = 6


def gemm2_grid(num_m_blocks: int, n_split: int) -> int:
    """Workgroups to launch: every (m-tile, chunk) of the allocation rounded up to the 8 XCDs (the kernel drops the
    fully padded tail tiles at run time)."""
    return (num_m_blocks * n_split + _NUM_XCDS - 1) // _NUM_XCDS * _NUM_XCDS


@dataclass
class Mxfp4PrefillScales:
    """A layer's routed-expert e8m0 scales for the FP4 kernels: W13 gate / up interleaved and aiter-shuffled
    (``i3072.gemm1_prefill_h6144_i3072_persistent.interleave_w13_scales``), W2 aiter-shuffled (``e8m0_shuffle``)."""

    w13: torch.Tensor
    w2: torch.Tensor

    @classmethod
    def from_row_major(
        cls, w13_scale: torch.Tensor, w2_scale: torch.Tensor
    ) -> "Mxfp4PrefillScales":
        """From the checkpoint's row-major e8m0 ``[E, 2I, H/32]`` (gate rows, then up rows) and ``[E, H, I/32]``."""
        from aiter.utility.fp4_utils import e8m0_shuffle

        from .i3072.gemm1_prefill_h6144_i3072_persistent import interleave_w13_scales

        e = w13_scale.shape[0]
        w13 = e8m0_shuffle(w13_scale.reshape(e * 2 * I, -1)).view(e, 2 * I, -1)
        w2 = e8m0_shuffle(w2_scale.reshape(e * H, -1)).view(e, H, -1)
        return cls(interleave_w13_scales(w13.view(torch.uint8)), w2.view(torch.uint8))
