"""Warm the TP collective used by Qwen QSA's compute-only index partition."""

import torch
import torch.distributed as dist

from atom.utils import envs

_CUSTOM_GATHER_MAX_BYTES = 512 * 1024


def _use_custom_gather(payload: torch.Tensor, group) -> bool:
    """Use only rank-invariant inputs to choose the collective transport."""
    if payload.numel() * payload.element_size() > _CUSTOM_GATHER_MAX_BYTES:
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
    """All-gather [rows, k, 2] packed FP32-score/int32-id candidates."""
    world = group.world_size
    if _use_custom_gather(payload, group):
        return group.custom_all_gather(payload).view(world, *payload.shape)
    received = torch.empty(
        (world, *payload.shape), dtype=payload.dtype, device=payload.device
    )
    dist.all_gather_into_tensor(received, payload, group=group.device_group)
    return received


def warmup_qsa_cp_exchange(device: torch.device) -> None:
    """Establish all-gather peers before a decode CUDA graph is captured."""
    if not envs.ATOM_QWEN38_QSA_CP:
        return
    from aiter.dist.parallel_state import get_tp_group

    group = get_tp_group()
    world = group.world_size
    probe = torch.zeros((1, 1, 2), dtype=torch.int32, device=device)
    received = torch.empty((world, *probe.shape), dtype=torch.int32, device=device)
    dist.all_gather_into_tensor(received, probe, group=group.device_group)
    if _use_custom_gather(probe, group):
        group.custom_all_gather(probe)
    torch.cuda.synchronize(device)
