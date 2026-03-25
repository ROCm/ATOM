"""Reuse vLLM's TP GroupCoordinator and inject aiter's ca_comm to avoid double IPC init.

When ATOM runs as vLLM plugin, both vLLM and aiter would create separate TP groups
with their own ProcessGroups and CustomAllreduce IPC setup. This causes:
- Duplicate gloo/NCCL groups for the same ranks
- Double IPC handle allocation and potential 2x slowdown in reduce kernels

This module creates an aiter-compatible TP group adapter that:
1. Uses vLLM's existing TP ProcessGroups (cpu_group, device_group)
2. Creates only aiter's CudaCommunicator (with ca_comm) attached to those groups
3. Registers as aiter's get_tp_group() so model collectives use single IPC setup
"""

from __future__ import annotations

import logging
from typing import Any

import torch

logger = logging.getLogger("atom")


def _create_aiter_tp_adapter_from_vllm() -> Any:
    """Create aiter-compatible TP adapter using vLLM's TP groups and aiter's ca_comm."""
    import vllm.distributed.parallel_state as vllm_ps

    vllm_tp = vllm_ps.get_tp_group()
    if vllm_tp.world_size == 1:
        return None

    # Import aiter components - must use aiter's CudaCommunicator with aiter's ca_comm
    from aiter.dist.device_communicators.communicator_cuda import CudaCommunicator
    from aiter.dist.parallel_state import _register_group

    # Create aiter CudaCommunicator with vLLM's ProcessGroups (no new groups created)
    device_communicator = CudaCommunicator(
        cpu_group=vllm_tp.cpu_group,
        device=vllm_tp.device,
        device_group=vllm_tp.device_group,
        unique_name="tp",
    )

    if device_communicator.ca_comm is None or device_communicator.ca_comm.disabled:
        logger.warning(
            "ATOM tp_group_reuse: aiter ca_comm not available on vLLM's TP group, "
            "caller will fall back to standard aiter distributed initialization "
            "(e.g., via aiter.init_dist_env(...))"
        )
        return None

    # Inherit from aiter GroupCoordinator - only override __init__ to use existing groups.
    # All reduce/gather/broadcast methods are inherited, no need to reimplement.
    from aiter.dist.parallel_state import GroupCoordinator as AiterGroupCoordinator

    class AiterTPAdapter(AiterGroupCoordinator):
        """Reuse vLLM's TP groups + aiter's ca_comm. Inherits all methods from GroupCoordinator."""

        def __init__(self, vllm_tp: Any, device_comm: Any):
            # Skip GroupCoordinator.__init__ (it creates new ProcessGroups).
            # Set attributes directly to match what parent expects.
            self.unique_name = "tp:0"
            _register_group(self)
            self.rank = vllm_tp.rank
            self.local_rank = vllm_tp.local_rank
            self.ranks = vllm_tp.ranks
            self.world_size = vllm_tp.world_size
            self.rank_in_group = vllm_tp.rank_in_group
            self.cpu_group = vllm_tp.cpu_group
            self.device_group = vllm_tp.device_group
            self.device = vllm_tp.device
            self.use_device_communicator = True
            self.device_communicator = device_comm
            self.mq_broadcaster = None

    adapter = AiterTPAdapter(vllm_tp, device_communicator)
    return adapter


def _setup_ca_comm_signal(adapter: Any, tensor_model_parallel_size: int) -> None:
    """Register signal buffer for custom allreduce (required by aiter)."""
    ca_comm = adapter.device_communicator.ca_comm
    if ca_comm is None:
        return
    signal = torch.zeros(
        tensor_model_parallel_size * 64, dtype=torch.int64, device=adapter.device
    )
    ca_comm.signal = signal
    ca_comm.register_input_buffer(signal)


def init_aiter_tp_from_vllm(tensor_model_parallel_size: int) -> bool:
    """
    Initialize aiter's TP group by reusing vLLM's TP and injecting aiter's ca_comm.

    Also sets _PP from vLLM so get_pp_group() works (required by model_wrapper).

    Returns True if reuse succeeded, False if fallback to init_aiter_dist is needed.
    """
    try:
        import vllm.distributed.parallel_state as vllm_ps

        adapter = _create_aiter_tp_adapter_from_vllm()
        if adapter is None:
            return False

        from aiter.dist import parallel_state as aiter_ps

        aiter_ps._TP = adapter  # type: ignore[attr-defined]
        aiter_ps._PP = vllm_ps.get_pp_group()  # type: ignore[attr-defined]
        aiter_ps._DP = vllm_ps.get_dp_group()  # type: ignore[attr-defined]
        aiter_ps._EP = getattr(
            vllm_ps, "_EP", None
        )  # EP may not exist in all vLLM configs
        _setup_ca_comm_signal(adapter, tensor_model_parallel_size)

        from aiter.dist.parallel_state import set_custom_all_reduce

        set_custom_all_reduce(True)

        logger.info(
            "ATOM plugin: reused vLLM TP group with aiter ca_comm "
            "(single IPC init, no duplicate ProcessGroups)"
        )
        return True
    except Exception as e:
        logger.warning(
            "ATOM tp_group_reuse failed (%s), caller will fall back to standard "
            "aiter distributed initialization (e.g., via aiter.init_dist_env(...))",
            e,
        )
        return False
