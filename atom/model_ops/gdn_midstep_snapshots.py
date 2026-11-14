# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Index tables for the unnarrowed interior states of a GDN prefill forward.

The chunk kernel stores the recurrent state at every 64-token boundary in `h`,
narrowed to the activation dtype. A state pool wider than that (Qwen4Exp keeps
fp32) takes each interior midstep checkpoint from a snapshot the kernel stores
unnarrowed instead. Torch only, so the plan is testable without triton.
"""

import torch


def interior_state_snapshots(
    rows: torch.Tensor,
    offs: torch.Tensor,
    interior: list[bool],
    chunk_offsets: torch.Tensor,
    num_chunks_bound: int,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Where the chunk kernel stores each interior target's unnarrowed state.

    Returns ``snap_idx`` (``[num_chunks_bound]``: chunk boundary of the
    flattened batch -> snapshot row, or -1), ``h_rows`` (per target: its
    snapshot row, -1 for an end target) and the number of snapshots.
    ``rows``/``offs`` are the device targets `write_state_checkpoints` takes and
    ``interior`` their host-side flags, so nothing here waits on the device.
    """
    dev = rows.device
    sel = [i for i, f in enumerate(interior) if f]
    snap_rows = [-1] * len(interior)
    for j, i in enumerate(sel):
        snap_rows[i] = j
    snap_idx = torch.full((num_chunks_bound,), -1, dtype=torch.int32, device=dev)
    if sel:
        pick = torch.tensor(sel, dtype=torch.long, device=dev)
        chunk = chunk_offsets[rows[pick].long()] + offs[pick] // chunk_size
        snap_idx[chunk.long()] = torch.arange(len(sel), dtype=torch.int32, device=dev)
    return snap_idx, torch.tensor(snap_rows, dtype=torch.int32, device=dev), len(sel)
