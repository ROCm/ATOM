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

import functools
import logging
from contextlib import ExitStack
from typing import Any

import torch

logger = logging.getLogger("atom")


def _create_aiter_device_communicator(vllm_tp: Any, cleanup: ExitStack) -> Any:
    """Create AITER collectives without creating a duplicate PyNccl communicator."""
    from aiter.dist.device_communicators.base_device_communicator import (
        DeviceCommunicatorBase,
    )
    from aiter.dist.device_communicators.communicator_cuda import CudaCommunicator
    from aiter.dist.device_communicators.custom_all_reduce import CustomAllreduce
    from aiter.dist.device_communicators.quick_all_reduce import QuickAllReduce
    from aiter.dist.parallel_state import _ENABLE_CUSTOM_ALL_REDUCE

    pynccl_comm = getattr(vllm_tp.device_communicator, "pynccl_comm", None)
    if pynccl_comm is None:
        raise RuntimeError("vLLM TP group does not expose a PyNccl communicator")

    class ReusedPyNcclCudaCommunicator(CudaCommunicator):
        def __init__(self) -> None:
            self._all2all_manager = None
            self._all2all_manager_created = False
            DeviceCommunicatorBase.__init__(
                self,
                cpu_group=vllm_tp.cpu_group,
                device=vllm_tp.device,
                device_group=vllm_tp.device_group,
                unique_name="tp",
            )

            self.use_custom_allreduce = _ENABLE_CUSTOM_ALL_REDUCE
            self.use_torch_symm_mem = False
            self.pynccl_comm = pynccl_comm
            # PyNccl and ProcessGroups are borrowed, but ca/qr are ours. The
            # parent destroy only drops PyNccl's reference; it never closes it.
            self._owns_comms = True
            self.ca_comm = None
            self.qr_comm = None
            self.symm_mem_comm = None
            cleanup.callback(self.destroy)
            self.ca_comm = (
                CustomAllreduce(group=self.cpu_group, device=self.device)
                if self.use_custom_allreduce and self.world_size > 1
                else None
            )
            self.qr_comm = (
                QuickAllReduce(group=self.cpu_group, device=self.device)
                if self.world_size > 1
                else None
            )

    return ReusedPyNcclCudaCommunicator()


def _create_aiter_tp_adapter_from_vllm(vllm_tp: Any, cleanup: ExitStack) -> Any:
    """Create aiter-compatible TP adapter using vLLM's TP groups and aiter's ca_comm."""
    if vllm_tp.world_size == 1:
        return None

    device_communicator = _create_aiter_device_communicator(vllm_tp, cleanup)

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
            self._owns_cpu_group = False
            self._owns_device_group = False
            self._owns_device_communicator = True
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


def init_aiter_dist_from_vllm(tensor_model_parallel_size: int) -> bool:
    """
    Initialize aiter's distributed groups by reusing vLLM's, and inject aiter's
    ca_comm into the TP group.

    Reuses vLLM's TP/PP/DP groups (and EP when present) so get_tp_group() /
    get_pp_group() / get_dp_group() work without a duplicate IPC init.

    Returns True if reuse succeeded, False if fallback to init_aiter_dist is needed.
    """
    import vllm.distributed.parallel_state as vllm_ps
    from aiter.dist import parallel_state as aiter_ps

    vllm_tp = vllm_ps.get_tp_group()
    existing = getattr(vllm_tp, "_atom_aiter_tp_adapter", None)
    if existing is not None:
        if aiter_ps._TP is existing and not getattr(existing, "_atom_closing", False):
            return True
        raise RuntimeError(
            "Cannot reinitialize an AITER TP adapter after replacement/close"
        )

    # Cleanup is outside the fallback handler: failed cleanup must propagate,
    # not create a second group on top of resources we could not release.
    with ExitStack() as cleanup:
        try:
            adapter = _create_aiter_tp_adapter_from_vllm(vllm_tp, cleanup)
            if adapter is None:
                return False
            groups = {
                "_TP": adapter,
                "_PP": vllm_ps.get_pp_group(),
                "_DP": vllm_ps.get_dp_group(),
                "_EP": getattr(vllm_ps, "_EP", None),
            }
            _setup_ca_comm_signal(adapter, tensor_model_parallel_size)
            aiter_ps.set_custom_all_reduce(True)
        except Exception as e:
            logger.warning(
                "ATOM tp_group_reuse failed (%s), caller will fall back to standard "
                "aiter distributed initialization (e.g., via aiter.init_dist_env(...))",
                e,
                exc_info=True,
            )
            return False

        original_destroy = vllm_tp.destroy
        plugin_closed = owner_closed = False
        owner_result = None

        @functools.wraps(original_destroy)
        def destroy(*args, **kwargs):
            nonlocal plugin_closed, owner_closed, owner_result
            if not plugin_closed:
                adapter._atom_closing = True
                # vLLM reaches TP.destroy after synchronizing the runner and
                # clearing its graphs. Close our IPC buffers while its borrowed
                # communicators/groups still exist. On failure, retain both the
                # globals and owner so close can be retried without double-free.
                adapter.destroy()
                for name, group in groups.items():
                    if getattr(aiter_ps, name, None) is group:
                        setattr(aiter_ps, name, None)
                plugin_closed = True
                logger.info("ATOM plugin: closed reused AITER TP collectives")
            if not owner_closed:
                owner_result = original_destroy(*args, **kwargs)
                owner_closed = True
            return owner_result

        # Publish only after setup succeeds. Never call AITER's global destroy:
        # PP/DP/EP are bare borrowed vLLM groups, not owned adapters.
        aiter_ps._register_group(adapter)
        vllm_tp.destroy = destroy
        vllm_tp._atom_aiter_tp_adapter = adapter
        for name, group in groups.items():
            setattr(aiter_ps, name, group)
        cleanup.pop_all()

    logger.info(
        "ATOM plugin: reused vLLM TP group with aiter ca_comm "
        "(single IPC init, no duplicate ProcessGroups)"
    )
    return True
