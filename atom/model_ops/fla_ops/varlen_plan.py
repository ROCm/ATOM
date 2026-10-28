# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""A bucket-shaped launch plan for the varlen chunked GDN prefill kernels.

The chunk kernels are launched over the step's chunks and sequences, counts
the host reads off ``cu_seqlens``, so a recorded graph could not replay them
for another step. With a plan every grid is sized for the token bucket
(`nt64_max`, `nt16_max`, `n_max`, conv programs) and each program reads the
step's real count from a device scalar and exits past it; the chunk index
tables, chunk offsets and per-sequence state slots live at fixed addresses
the step's metadata rewrites, so a replay reads the step being served. The
recurrence gathers its initial state from the state pool by slot and writes
the final state back by slot, in place, instead of an `index_select` /
`index_copy_` whose row count varies with the batch.
"""
from dataclasses import dataclass

import torch


@dataclass
class VarlenPlan:
    chunk_indices_64: torch.Tensor  # [nt64_max, 2] int (sequence, chunk in sequence)
    nt64: torch.Tensor  # [1] int32, the step's 64-chunk count
    nt64_max: int
    chunk_indices_16: torch.Tensor  # [nt16_max, 2]
    nt16: torch.Tensor  # [1] int32
    nt16_max: int
    chunk_offsets: torch.Tensor  # [n_max + 1] int32, per-sequence 64-chunk prefix sums
    num_seqs: torch.Tensor  # [1] int32, the step's sequence count
    n_max: int
    state_idx: torch.Tensor  # [n_max] int32, the slot each sequence's final state goes to
    state_idx_in: torch.Tensor  # [n_max] int32, the slot its initial state comes from
    has_initial_state: torch.Tensor  # [n_max] int32, 0/1
