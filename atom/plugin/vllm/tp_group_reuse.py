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
_REUSED_TP_ADAPTER: Any = None
_REUSED_TP_SIGNATURE: tuple | None = None


def reset_reused_tp_adapter() -> None:
    """Release ATOM-owned communicators and clear borrowed-group aliases.

    The adapter's destroy method intentionally leaves vLLM-owned ProcessGroups alive.
    """
    global _REUSED_TP_ADAPTER, _REUSED_TP_SIGNATURE
    from aiter.dist import parallel_state as aiter_ps

    adapter = _REUSED_TP_ADAPTER
    if adapter is not None:
        adapter.destroy()
    if getattr(aiter_ps, "_TP", None) is adapter:
        aiter_ps._TP = None
    if getattr(aiter_ps, "_PCP", None) is adapter:
        aiter_ps._PCP = None
    _REUSED_TP_ADAPTER = None
    _REUSED_TP_SIGNATURE = None
    try:
        from atom.plugin.vllm.attention.cp_gather import close_cp_gather_pipeline

        close_cp_gather_pipeline()
    except Exception:
        pass


def _create_aiter_device_comm(vllm_group: Any, comm_unique_name: str) -> Any:
    """Create an aiter CudaCommunicator (with its own ca_comm) bound to vLLM's
    existing ProcessGroups. Returns None if custom all-reduce is unavailable.

    Each CudaCommunicator owns an INDEPENDENT ``CustomAllreduce`` (ca_comm): its
    own IPC pool, rank-data buffer and graph-buffer slot allocator. Creating a
    second one over the same ranks is how CP gets a collective communicator that
    is isolated from TP's.
    """
    from aiter.dist.device_communicators.communicator_cuda import CudaCommunicator

    device_communicator = CudaCommunicator(
        cpu_group=vllm_group.cpu_group,
        device=vllm_group.device,
        device_group=vllm_group.device_group,
        unique_name=comm_unique_name,
    )
    if device_communicator.ca_comm is None or device_communicator.ca_comm.disabled:
        return None
    return device_communicator


def _build_aiter_adapter(
    vllm_group: Any, device_comm: Any, adapter_unique_name: str
) -> Any:
    """Wrap vLLM's ProcessGroups + an aiter device_communicator in an aiter
    GroupCoordinator, WITHOUT creating new ProcessGroups.

    Inherits all reduce/gather/broadcast methods from GroupCoordinator; only
    __init__ is overridden to reuse existing groups. Registers itself under
    ``adapter_unique_name`` so aiter's ``outplace_all_gather`` /
    ``outplace_reduce_scatter`` custom ops (which resolve ``_groups[group_name]``)
    dispatch to THIS group's ca_comm.
    """
    from aiter.dist.parallel_state import GroupCoordinator as AiterGroupCoordinator
    from aiter.dist.parallel_state import _register_group

    class AiterGroupAdapter(AiterGroupCoordinator):
        def __init__(self) -> None:
            # Skip GroupCoordinator.__init__ (it creates new ProcessGroups).
            self.unique_name = adapter_unique_name
            _register_group(self)
            self.rank = vllm_group.rank
            self.local_rank = vllm_group.local_rank
            self.ranks = vllm_group.ranks
            self.world_size = vllm_group.world_size
            self.rank_in_group = vllm_group.rank_in_group
            self.cpu_group = vllm_group.cpu_group
            self.device_group = vllm_group.device_group
            self.device = vllm_group.device
            self.use_device_communicator = True
            self.device_communicator = device_comm
            self.mq_broadcaster = None

        def destroy(self) -> None:
            # cpu_group / device_group are vLLM's -- owned and torn down by vLLM.
            # Only release the aiter device_communicator (ca_comm/IPC) we created;
            # NEVER destroy the borrowed ProcessGroups (a second reuse adapter
            # over the same groups would otherwise double-free them at shutdown).
            dc = getattr(self, "device_communicator", None)
            if dc is not None:
                try:
                    dc.destroy()
                except Exception:  # noqa: BLE001
                    pass
                self.device_communicator = None

    return AiterGroupAdapter()


def _create_aiter_tp_adapter_from_vllm() -> Any:
    """Create aiter-compatible TP adapter using vLLM's TP groups and aiter's ca_comm."""
    import vllm.distributed.parallel_state as vllm_ps

    vllm_tp = vllm_ps.get_tp_group()
    if vllm_tp.world_size == 1:
        return None

    device_communicator = _create_aiter_device_comm(vllm_tp, "tp")
    if device_communicator is None:
        logger.warning(
            "ATOM tp_group_reuse: aiter ca_comm not available on vLLM's TP group, "
            "caller will fall back to standard aiter distributed initialization "
            "(e.g., via aiter.init_dist_env(...))"
        )
        return None

    return _build_aiter_adapter(vllm_tp, device_communicator, "tp:0")


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
    global _REUSED_TP_ADAPTER, _REUSED_TP_SIGNATURE
    cp_enabled = False
    adapter = None
    try:
        import vllm.distributed.parallel_state as vllm_ps

        vllm_tp = vllm_ps.get_tp_group()
        signature = (tuple(vllm_tp.ranks), id(vllm_tp.device_group))
        from aiter.dist import parallel_state as aiter_ps
        from atom.utils import envs

        cp_enabled = (
            envs.ATOM_VLLM_ATTN_CP
            and vllm_tp.world_size > 1
            and getattr(aiter_ps, "_ATOM_PLUGIN_MODE", True)
        )
        if _REUSED_TP_ADAPTER is not None:
            if _REUSED_TP_SIGNATURE != signature:
                raise RuntimeError(
                    "vLLM TP topology changed after ATOM reuse initialization"
                )
            # Idempotent repeated model construction: retain the existing adapter
            # and communicator instead of overwriting globals and leaking ca_comm.
            aiter_ps._TP = _REUSED_TP_ADAPTER  # type: ignore[attr-defined]
            if cp_enabled:
                aiter_ps._PCP = _REUSED_TP_ADAPTER  # type: ignore[attr-defined]
            return True

        adapter = _create_aiter_tp_adapter_from_vllm()
        if adapter is None:
            return False
        _setup_ca_comm_signal(adapter, tensor_model_parallel_size)
        if cp_enabled:
            from atom.plugin.vllm.attention.cp_gather import (
                initialize_cp_gather_pipeline,
            )

            initialize_cp_gather_pipeline()

        # Publish only after every fallible resource initialization succeeds.
        aiter_ps._TP = adapter  # type: ignore[attr-defined]
        aiter_ps._PP = vllm_ps.get_pp_group()  # type: ignore[attr-defined]
        aiter_ps._DP = vllm_ps.get_dp_group()  # type: ignore[attr-defined]
        aiter_ps._EP = getattr(
            vllm_ps, "_EP", None
        )  # EP may not exist in all vLLM configs

        # RFC ROCm/ATOM#196: run the Context-Parallel dimension over the TP ranks.
        # Setting aiter's `_PCP` makes get_pcp_group() /
        # get_prefill_context_model_parallel_{world_size,rank}() and
        # pcp_allgather_rerange() operate over the TP ranks, so the native
        # round-robin split/gather/indexer paths (all keyed off pcp_is_enabled)
        # light up. The reuse-specific model rebuild (full-head attention, MoE
        # all-gather/reduce-scatter, no fused-AR norms) is gated separately on
        # plugin_attn_cp_enabled() at construction time.
        #
        # TP and CP activations intentionally alias the same adapter.  Weight
        # prefetch owns a separate raw process group/side stream in cp_gather;
        # creating another aiter CP ca_comm here duplicates IPC state and violates
        # the reuse-TP-as-CP contract.
        if cp_enabled:
            aiter_ps._PCP = adapter  # type: ignore[attr-defined]
            logger.info(
                "ATOM plugin: aliased the vLLM TP adapter as Context-Parallel "
                "(_PCP) over %d ranks (ATOM_VLLM_ATTN_CP=1)",
                adapter.world_size,
            )

        _REUSED_TP_ADAPTER = adapter
        _REUSED_TP_SIGNATURE = signature

        from aiter.dist.parallel_state import set_custom_all_reduce

        set_custom_all_reduce(True)

        logger.info(
            "ATOM plugin: reused vLLM TP group with aiter ca_comm "
            "(single IPC init, no duplicate ProcessGroups)"
        )
        return True
    except Exception as e:
        # CP-on cannot fall back: its model graph already contains CP collectives
        # and requires the independent weight communicator on every rank.
        if adapter is not None and adapter is not _REUSED_TP_ADAPTER:
            adapter.destroy()
        if cp_enabled:
            raise RuntimeError("ATOM TP/CP reuse initialization failed") from e
        logger.warning(
            "ATOM tp_group_reuse failed (%s), caller will fall back to standard "
            "aiter distributed initialization (e.g., via aiter.init_dist_env(...))",
            e,
        )
        return False
