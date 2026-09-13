from __future__ import annotations

import logging
import threading
from typing import Any, Callable, Optional

logger = logging.getLogger("atom")

_FALLBACK_A2A_BACKEND = "allgather_reducescatter"
_ASSERTION_A2A_BACKEND = "deepep_low_latency"
_A2A_BACKEND_PATCH_FLAG = "_atom_dbo_backend_gate_patched"
_ATOM_BRIDGE_PATCH_FLAG = "_atom_tbo_bridge_patched"

def patch_vllm_dbo_backend() -> None:
    """Relax vLLM's DeepEP-only gate so ATOM plugin mode can run DBO over mori.

    vLLM has a hard-assert that when DBO is enabled, the all2all backend is one
    of the two DeepEP backends (deepep_low_latency or deepep_high_throughput)
    as its MoE prepare/finalize is implemented on the contract for the DeepEP
    backends. In ATOM plugin mode where ATOM's own MoE prepare/finalize is used,
    this assertion is over-broad as DeepEP is not used and it rules out mori.
    (See vllm/config/vllm.py:1375)

    Therefore, here we patch VllmConfig.__post_init__ to bypass the assertion by
    spoofing the all2all backend to "deepep_low_latency" during the construction
    of VllmConfig, which is safe because vLLM's MoE prepare/finalize is not used
    in ATOM's vLLM plugin mode. At the end of the construction, the backend in
    VllmConfig is set to "allgather_reducescatter". Note that when "mori" is
    explicitly selected, VllmConfig still coerces the restored backend to
    "allgather_reducescatter" so that vLLM doesn't build a mori manager that would
    collide with ATOM's mori manager from aiter.
    """
    from vllm.config import VllmConfig

    if getattr(VllmConfig, _A2A_BACKEND_PATCH_FLAG, False):
        return

    _orig_post_init = VllmConfig.__post_init__

    def __post_init__(self):  # noqa: N807 - mirror dataclass hook name
        pc = getattr(self, "parallel_config", None)
        original_backend = getattr(pc, "all2all_backend", None) if pc is not None else None
        spoofed = (
            pc is not None
            and getattr(pc, "use_ubatching", False)
        )
        if spoofed:
            restore_backend = _FALLBACK_A2A_BACKEND
            logger.info(
                "ATOM plugin DBO: bypassing vLLM's DeepEP-only micro-batching "
                "assertion. vLLM all2all_backend '%s' -> '%s'.",
                original_backend,
                restore_backend,
            )
            pc.all2all_backend = _ASSERTION_A2A_BACKEND
        try:
            _orig_post_init(self)
        finally:
            if spoofed:
                pc.all2all_backend = restore_backend

    VllmConfig.__post_init__ = __post_init__
    setattr(VllmConfig, _A2A_BACKEND_PATCH_FLAG, True)
    logger.info("ATOM plugin: patched VllmConfig DBO backend gate")


def _get_vllm_ubatching() -> Optional[Any]:
    try:
        from vllm.v1.worker import ubatching as vub  # type: ignore
        return vub
    except ImportError:
        return None


def patch_atom_dbo_bridge() -> None:
    """Patch ATOM ``tbo_*`` helpers to dispatch to vLLM ``dbo_*``."""
    import atom.utils.tbo.ubatching as ub

    if getattr(ub, _ATOM_BRIDGE_PATCH_FLAG, False):
        return

    vub = _get_vllm_ubatching()
    if vub is None:
        logger.info(
            "ATOM plugin: vllm.v1.worker.ubatching not importable; "
            "skipping tbo_* -> dbo_* bridge install."
        )
        return

    def _in_vllm() -> bool:
        return threading.get_ident() in vub._THREAD_ID_TO_CONTEXT

    def _vllm_ctx():
        cid = vub._THREAD_ID_TO_CONTEXT[threading.get_ident()]
        return vub._CURRENT_CONTEXTS[cid]

    # ============= Patch helpers

    _orig_tbo_active = ub.tbo_active

    def tbo_active() -> bool:
        return _in_vllm() or _orig_tbo_active()

    ub.tbo_active = tbo_active

    _orig_tbo_current_ubatch_id = ub.tbo_current_ubatch_id

    def tbo_current_ubatch_id() -> int:
        if _in_vllm():
            return vub.dbo_current_ubatch_id()
        return _orig_tbo_current_ubatch_id()

    ub.tbo_current_ubatch_id = tbo_current_ubatch_id

    _orig_tbo_yield = ub.tbo_yield

    def tbo_yield() -> None:
        if _in_vllm():
            return vub.dbo_yield()
        return _orig_tbo_yield()

    ub.tbo_yield = tbo_yield

    _orig_tbo_register_recv_hook = ub.tbo_register_recv_hook

    def tbo_register_recv_hook(hook: Callable) -> None:
        if _in_vllm():
            return vub.dbo_register_recv_hook(hook)
        return _orig_tbo_register_recv_hook(hook)

    ub.tbo_register_recv_hook = tbo_register_recv_hook

    _orig_tbo_maybe_run_recv_hook = ub.tbo_maybe_run_recv_hook

    def tbo_maybe_run_recv_hook() -> None:
        if _in_vllm():
            return vub.dbo_maybe_run_recv_hook()
        return _orig_tbo_maybe_run_recv_hook()

    ub.tbo_maybe_run_recv_hook = tbo_maybe_run_recv_hook

    _orig_tbo_get_comm_stream = ub.tbo_get_comm_stream

    def tbo_get_comm_stream():
        if _in_vllm():
            return _vllm_ctx().comm_stream
        return _orig_tbo_get_comm_stream()

    ub.tbo_get_comm_stream = tbo_get_comm_stream

    _orig_tbo_get_compute_stream = ub.tbo_get_compute_stream

    def tbo_get_compute_stream():
        if _in_vllm():
            return _vllm_ctx().compute_stream
        return _orig_tbo_get_compute_stream()

    ub.tbo_get_compute_stream = tbo_get_compute_stream

    _orig_tbo_yield_compute_to_comm = ub.tbo_yield_and_switch_from_compute_to_comm

    def tbo_yield_and_switch_from_compute_to_comm() -> None:
        if _in_vllm():
            return vub.dbo_yield_and_switch_from_compute_to_comm()
        return _orig_tbo_yield_compute_to_comm()

    ub.tbo_yield_and_switch_from_compute_to_comm = (
        tbo_yield_and_switch_from_compute_to_comm
    )

    _orig_tbo_yield_comm_to_compute = ub.tbo_yield_and_switch_from_comm_to_compute

    def tbo_yield_and_switch_from_comm_to_compute() -> None:
        if _in_vllm():
            return vub.dbo_yield_and_switch_from_comm_to_compute()
        return _orig_tbo_yield_comm_to_compute()

    ub.tbo_yield_and_switch_from_comm_to_compute = (
        tbo_yield_and_switch_from_comm_to_compute
    )

    _orig_tbo_switch_to_compute_sync = ub.tbo_switch_to_compute_sync

    def tbo_switch_to_compute_sync() -> None:
        if _in_vllm():
            return vub.dbo_switch_to_compute_sync()
        return _orig_tbo_switch_to_compute_sync()

    ub.tbo_switch_to_compute_sync = tbo_switch_to_compute_sync

    _orig_tbo_switch_to_compute = ub.tbo_switch_to_compute

    def tbo_switch_to_compute() -> None:
        if _in_vllm():
            return vub.dbo_switch_to_compute()
        return _orig_tbo_switch_to_compute()

    ub.tbo_switch_to_compute = tbo_switch_to_compute

    _orig_tbo_switch_to_comm = ub.tbo_switch_to_comm

    def tbo_switch_to_comm() -> None:
        if _in_vllm():
            return vub.dbo_switch_to_comm()
        return _orig_tbo_switch_to_comm()

    ub.tbo_switch_to_comm = tbo_switch_to_comm

    setattr(ub, _ATOM_BRIDGE_PATCH_FLAG, True)
    logger.info(
        "ATOM plugin: patched tbo_* -> dbo_* bridge on atom.utils.tbo.ubatching"
    )
