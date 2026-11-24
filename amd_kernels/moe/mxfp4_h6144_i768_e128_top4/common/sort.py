# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The routing sort's buffers and bounds (aiter's ``moe_3stage_sort.cuh`` contract), written by ``sort_count_all``,
``sort_scales_all`` and ``sort_tile_map``.

Outputs: ``sorted_ids[row] = token | slot << 24`` (padding rows: ``n_tokens``), ``sorted_weights`` (padding 0),
``sorted_expert_ids`` per block, ``num_valid_ids`` = [padded total, the last expert's first row]. Rows inside an expert
come out in atomic arrival order; consumers do not rely on it.
"""

from dataclasses import dataclass

import torch

SORT_CTAS = 128  # histogram / placement blocks
THREADS = 1024  # >= E and >= block_m


def max_sorted_rows(n_tokens: int, E: int, topk: int, block_m: int) -> int:
    """Aiter's bound: every active expert pads by < block_m rows."""
    active = min(E, n_tokens * topk)
    cumsum_max = n_tokens * topk + active * (block_m - 1)
    return ((cumsum_max + block_m - 1) // block_m) * block_m


@dataclass
class SortBuffers:
    sorted_ids: torch.Tensor
    sorted_expert_ids: torch.Tensor
    num_valid_ids: (
        torch.Tensor
    )  # [2] i32: padded row count, the last expert's first row
    sorted_weights: torch.Tensor
    block_offsets: torch.Tensor  # workspace [E * SORT_CTAS]: per-CTA counts
    max_sorted: int
    block_m: int

    @staticmethod
    def allocate(n_tokens, E, topk, block_m, device):
        ms = max_sorted_rows(n_tokens, E, topk, block_m)
        i32 = torch.int32
        return SortBuffers(
            sorted_ids=torch.empty(ms, dtype=i32, device=device),
            sorted_expert_ids=torch.empty(ms // block_m, dtype=i32, device=device),
            num_valid_ids=torch.empty(
                2, dtype=i32, device=device
            ),  # written by sort_place_pad
            sorted_weights=torch.empty(ms, dtype=torch.float32, device=device),
            block_offsets=torch.empty(E * SORT_CTAS, dtype=i32, device=device),
            max_sorted=ms,
            block_m=block_m,
        )

    def launch_args(self, topk_ids, topk_weights, n_tokens):
        return (
            topk_ids.contiguous().int().view(-1),
            topk_weights.contiguous().float().view(-1),
            self.sorted_ids,
            self.sorted_expert_ids,
            self.num_valid_ids,
            self.sorted_weights,
            self.block_offsets,
            int(n_tokens),
            torch.cuda.current_stream(),
        )
