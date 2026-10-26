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
from typing import Any

import torch

logger = logging.getLogger("atom")

# What this module built and what it wrote into aiter's globals, kept so that
# teardown can close the first and withdraw the second without guessing. Empty
# on every path that does not reach the reuse fast path -- TP=1, a failed
# reuse, the DP+EP fallback where aiter owns its groups and tears them down
# itself -- which is what makes the teardown a structural no-op there rather
# than one guarded by a flag.
_ATOM_OWNED_TP: Any | None = None
_ATOM_INSTALLED_GROUPS: dict[str, Any] = {}


def _create_aiter_device_communicator(vllm_tp: Any) -> Any:
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
            self.symm_mem_comm = None

    return ReusedPyNcclCudaCommunicator()


def _create_aiter_tp_adapter_from_vllm() -> Any:
    """Create aiter-compatible TP adapter using vLLM's TP groups and aiter's ca_comm."""
    import vllm.distributed.parallel_state as vllm_ps

    vllm_tp = vllm_ps.get_tp_group()
    if vllm_tp.world_size == 1:
        return None

    from aiter.dist.parallel_state import _register_group

    device_communicator = _create_aiter_device_communicator(vllm_tp)

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


def shutdown_aiter_dist_from_vllm() -> bool:
    """Close the collectives this module created, while torch can still free them.

    aiter frees a `CustomAllreduce`'s IPC buffers through `torch.ops.aiter`, and
    nothing closed one on this path: the adapter is reachable only from aiter's
    `_TP` global (its group registry holds a weakref), so the first thing to
    release it was interpreter finalization. By then `torch.library` has
    deregistered the operator namespace, and each rank ends a run with

        AttributeError: '_OpNamespace' 'aiter' object has no attribute
        'free_meta_buffer'

    raised inside `__del__`, where nothing can handle it. aiter expects the
    explicit call -- its own `close()` says it frees the buffers
    "deterministically instead of leaving them for GC" -- and closing here
    makes that later `__del__` a no-op, because `close()` guards on the pointer
    it has already zeroed. The traceback stops happening rather than moving.

    Only what this module built is closed. The PyNccl communicator, the process
    groups and the PP/DP/EP coordinators belong to vLLM, which destroys them
    itself a few lines further on; destroying them here would be a double free.
    aiter's own `destroy()` is not usable for this either: `AiterTPAdapter`
    skips `GroupCoordinator.__init__`, so the `_owns_*` fields that `destroy()`
    reads do not exist on it, and `CudaCommunicator.destroy()` gates every
    close on an `_owns_comms` this communicator never sets.

    Deliberately without a barrier. Each rank frees only mappings it opened
    itself, and no collective is issued after this point. A barrier would be a
    collective on the groups being torn down, on the very path that is already
    failing to exit, so one unresponsive rank would hang the rest -- turning a
    destructor warning into a four-way stall. The residual risk is that ranks
    are not in lockstep; it is bounded by nothing reading those mappings again.

    Returns whether there was anything to tear down.
    """
    global _ATOM_OWNED_TP, _ATOM_INSTALLED_GROUPS

    adapter = _ATOM_OWNED_TP
    if adapter is None:
        return False
    installed = _ATOM_INSTALLED_GROUPS
    # Dropped before any of the work below, so that a close which raises cannot
    # leave a second call to find the adapter and free the same pointers twice.
    _ATOM_OWNED_TP = None
    _ATOM_INSTALLED_GROUPS = {}

    try:
        from aiter.dist import parallel_state as aiter_ps
    except ImportError:  # torn down already, or aiter never imported
        aiter_ps = None
    if aiter_ps is not None:
        for name, group in installed.items():
            # Withdraw what this module installed and nothing else: a later
            # owner's group is theirs to clear.
            if getattr(aiter_ps, name, None) is group:
                setattr(aiter_ps, name, None)

    communicator = getattr(adapter, "device_communicator", None)
    for name in ("ca_comm", "qr_comm"):
        comm = getattr(communicator, name, None)
        if comm is None:
            continue
        try:
            comm.close()
        except Exception:
            # Broad on purpose: one comm failing to close must not strand the
            # other, nor the globals still pointing at a half-torn-down group.
            logger.warning(
                "ATOM plugin: closing the aiter %s raised; continuing teardown",
                name,
                exc_info=True,
            )
        # Dropping the last reference lets the closed object be collected now,
        # rather than at finalization where this started.
        setattr(communicator, name, None)

    if communicator is not None:
        # Borrowed from vLLM's TP group: released here, never destroyed.
        communicator.pynccl_comm = None
    adapter.device_communicator = None
    logger.info(
        "ATOM plugin: closed the aiter collectives this worker created "
        "(custom and quick all-reduce)"
    )
    return True


def apply_vllm_worker_shutdown_patch() -> int:
    """Run that teardown from vLLM's worker shutdown. Returns classes covered.

    `Worker.shutdown` is where it has to go. It runs inside the worker process;
    `worker_main`'s `finally` reaches it on a clean exit and on the SystemExit
    a SIGTERM raises; and the executor calls it before `destroy_model_parallel`
    tears down the groups this adapter borrows. vLLM does the same kind of
    thing in the same method for the same reason -- it releases its allocator
    pools there so that teardown "is not deferred to interpreter finalization".
    `atexit` and `weakref.finalize` both run in exactly the regime that causes
    the bug, which is why neither is used.

    The original runs first, and that ordering is load-bearing:
    `model_runner.shutdown()` inside it is what drops the captured graphs, and
    freeing an IPC buffer a live graph still points into would be worse than
    the warning being fixed. The teardown goes in a `finally` so a failing
    vLLM shutdown cannot skip it, and swallows its own errors so it can never
    turn an exit into a crash.

    The count is returned, and logged by the caller, because a patch installed
    on a class nobody instantiates reads exactly like one that works.
    """
    try:
        from vllm.v1.worker.gpu_worker import Worker
    except ImportError:
        return 0

    original = getattr(Worker, "shutdown", None)
    if original is None or getattr(original, "_atom_worker_shutdown_patched", False):
        return 0

    @functools.wraps(original)
    def shutdown(self, *args, **kwargs):
        try:
            return original(self, *args, **kwargs)
        finally:
            try:
                shutdown_aiter_dist_from_vllm()
            except Exception:
                # Broad on purpose: this runs on the way out, and no failure
                # in it may replace the exit the worker was already making.
                logger.warning(
                    "ATOM plugin: aiter teardown raised during worker shutdown",
                    exc_info=True,
                )

    shutdown._atom_worker_shutdown_patched = True
    Worker.shutdown = shutdown
    return 1


def init_aiter_dist_from_vllm(tensor_model_parallel_size: int) -> bool:
    """
    Initialize aiter's distributed groups by reusing vLLM's, and inject aiter's
    ca_comm into the TP group.

    Reuses vLLM's TP/PP/DP groups (and EP when present) so get_tp_group() /
    get_pp_group() / get_dp_group() work without a duplicate IPC init.

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

        # Remember what was built and what was installed, so teardown closes
        # the first and withdraws the second without inferring either. The
        # adapter needs a strong reference here regardless: aiter's group
        # registry keeps only a weakref, so until now `aiter_ps._TP` was the
        # one thing holding it alive.
        global _ATOM_OWNED_TP, _ATOM_INSTALLED_GROUPS
        _ATOM_OWNED_TP = adapter
        _ATOM_INSTALLED_GROUPS = {
            "_TP": adapter,
            "_PP": aiter_ps._PP,
            "_DP": aiter_ps._DP,
            "_EP": aiter_ps._EP,
        }
        # Installed from here, not from `register_platform`: this is the only
        # path that creates anything to close, so every other one -- TP=1, a
        # failed reuse, the DP+EP fallback where aiter owns and destroys its
        # own groups -- is left exactly as it was, with no hook to reason
        # about. It also keeps `vllm.v1.worker.gpu_worker` out of the import
        # that resolves vLLM's platform plugins.
        covered = apply_vllm_worker_shutdown_patch()
        if covered:
            logger.info(
                "ATOM plugin: aiter collective teardown covers %d worker class(es)",
                covered,
            )
        else:
            logger.debug("ATOM plugin: aiter collective teardown already installed")

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
