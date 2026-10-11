# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Qwen3.8 QSA index-only context parallelism over the existing TP group.

Each rank scores a contiguous slice of the replicated compressed-key cache,
selects its local candidates, and gathers only those candidates. Every rank
then makes the same global selection, so sparse attention keeps its ordinary
TP-owned KV heads. No new process group or KV-cache partition is needed.

The custom gather removes RCCL's launch bubble for small candidate payloads;
RCCL handles larger ones. Transport selection must depend on rank-invariant
inputs because the two choices are different collectives on the same TP group.
"""

import torch
import torch.distributed as dist

from atom.utils import envs

# Graph measurements on MI355X: TP4 custom gather still saves 12-22 us at
# 768 KiB-1 MiB. TP2 has a smaller, less consistent gain above 512 KiB.
_CUSTOM_GATHER_MAX_PAYLOAD_BYTES = {2: 512 * 1024, 4: 1024 * 1024}


def qsa_cp_enabled() -> bool:
    """Whether QSA index-only context partitioning was requested."""
    return envs.ATOM_QWEN38_QSA_CP


def get_qsa_cp_group():
    """The TP group is also the QSA index context-partition group."""
    from aiter.dist.parallel_state import get_tp_group

    return get_tp_group()


def get_qsa_cp_decode_group(*, rows: int, max_seq_len: int, tp_size: int):
    """Return the group when the saved score work pays for candidate exchange.

    This decision is identical on all ranks: rows, the context bucket and TP
    size are graph/metadata properties shared by every rank. The score width
    doubles at powers of two in compressed groups, so the measured 128k and
    256k buckets have different batch-size crossovers. The gates include
    ragged, padded decode rows, not just distinct full-length requests.
    """
    if not qsa_cp_enabled() or tp_size not in (2, 4) or max_seq_len < 1:
        return None
    compressed = (max_seq_len + 3) // 4
    bucket = 1 << (compressed - 1).bit_length()
    if bucket >= 65536:
        minimum_rows = 2 if tp_size == 4 else 16
    elif bucket >= 32768:
        minimum_rows = 24 if tp_size == 4 else 48
    else:
        return None
    if rows < minimum_rows:
        return None
    return get_qsa_cp_group()


def _can_custom_all_gather(payload: torch.Tensor, group) -> bool:
    """Select transport from properties all ranks hold in common.

    A per-rank fallback after a collective fails would leave other ranks
    waiting in a different collective. The payload geometry is shared; the
    global opt-out and group communicator are fixed before graph capture.
    """
    limit = _CUSTOM_GATHER_MAX_PAYLOAD_BYTES.get(group.world_size, 0)
    if payload.numel() * payload.element_size() > limit:
        return False
    if not envs.ATOM_USE_CUSTOM_ALL_GATHER:
        return False
    ca_comm = getattr(getattr(group, "device_communicator", None), "ca_comm", None)
    return bool(
        ca_comm is not None
        and not ca_comm.disabled
        and ca_comm.should_custom_ag(payload)
    )


def exchange_qsa_candidates(payload: torch.Tensor, group) -> torch.Tensor:
    """All-gather [rows, k, 2] packed FP32-score/int32-id candidates.

    Each rank needs every shard's candidates for the same query rows. Unlike
    M3's head-owner routing, an all-to-all would not reduce this payload.
    """
    world = group.world_size
    if _can_custom_all_gather(payload, group):
        return group.custom_all_gather(payload).view(world, *payload.shape)
    received = torch.empty(
        (world, *payload.shape), dtype=payload.dtype, device=payload.device
    )
    dist.all_gather_into_tensor(received, payload, group=group.device_group)
    return received


def warmup_qsa_cp_exchange(device: torch.device) -> None:
    """Establish both possible transports before decode graph capture.

    The model warmup is prefill and never exchanges candidates. The probe
    warms RCCL even when custom gather will handle small real batches: a
    larger batch can cross the byte threshold during a later graph capture.
    """
    if not qsa_cp_enabled():
        return
    group = get_qsa_cp_group()
    world = group.world_size
    probe = torch.zeros((1, 1, 2), dtype=torch.int32, device=device)
    received = torch.empty((world, *probe.shape), dtype=torch.int32, device=device)
    dist.all_gather_into_tensor(received, probe, group=group.device_group)
    if _can_custom_all_gather(probe, group):
        group.custom_all_gather(probe)
    torch.cuda.synchronize(device)
