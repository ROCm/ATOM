# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""TP rank 0's sampled token ids on every rank, without an RCCL broadcast.

The decode postprocess syncs a few dozen int32 ids per step so every rank's
drafter and output agree. `GroupCoordinator.broadcast` sends them through
RCCL, whose launch, proxy handshake and kernel cost tens of microseconds for
bytes. aiter's custom all-gather moves them over the IPC peer buffers the
model's own collectives already use.

Why a gather and not a custom all-reduce of the ids encoded as fp32 (zeros
off rank 0): the all-reduce path is gated by AITER_CUSTOM_AR_MIN_SIZE, which
deployments set well above a few hundred bytes so small reduces go to RCCL;
the gather has no lower bound. And it is a memcpy -- aiter views int32 as
float32 only to pick a 4-byte element type -- so it is bit-exact for any
value, where the fp32 encoding is exact only below 2**24.

Correctness: every rank contributes a buffer of the same size (the callers'
broadcasts already require identical shapes), and every rank receives the
same `[tp, n]` concatenation. Reading row 0 gives each rank rank 0's bytes,
which is what `broadcast(src=0)` delivers. Unlike the broadcast, the input is
not overwritten; the callers use the returned tensor.
"""

import torch

from atom.utils import envs

# Custom collectives move whole 16-byte vectors.
_ALIGN_ELEMS = 4


def _custom_gather_usable(group, padded: torch.Tensor) -> bool:
    """Whether this sync may take the custom gather.

    Every term is rank-invariant (process-wide env, the group's ca_comm fixed
    at init, a byte count every rank shares), so all ranks pick the same
    collective; one rank in the gather and another in the broadcast would hang.
    """
    if not envs.ATOM_USE_CUSTOM_ALL_GATHER:
        return False
    ca_comm = getattr(getattr(group, "device_communicator", None), "ca_comm", None)
    if ca_comm is None or ca_comm.disabled:
        return False
    return bool(ca_comm.should_custom_ag(padded))


def broadcast_from_rank0(tokens: torch.Tensor, group) -> torch.Tensor:
    """`group.broadcast(tokens, src=0)`'s values, via the custom all-gather
    with ATOM_TP_TOKEN_SYNC_ALLGATHER. 4- and 8-byte dtypes; others, and any
    group the gather cannot serve, take the broadcast."""
    if group.world_size == 1:
        return tokens
    tokens = tokens.contiguous()
    if not envs.ATOM_TP_TOKEN_SYNC_ALLGATHER or tokens.element_size() not in (4, 8):
        return group.broadcast(tokens, src=0)
    flat = tokens.view(-1).view(torch.int32)
    n = flat.numel()
    padded_n = -(-max(n, 1) // _ALIGN_ELEMS) * _ALIGN_ELEMS
    padded = flat
    if padded_n != n:
        # The tail is never read back, so it is left unset.
        padded = torch.empty(padded_n, dtype=flat.dtype, device=flat.device)
        padded[:n].copy_(flat)
    if not _custom_gather_usable(group, padded):
        return group.broadcast(tokens, src=0)
    gathered = group.all_gather(padded, use_custom=True, dim=0)
    return gathered[:n].view(tokens.dtype).view(tokens.shape)
