"""Patch vLLM graph capture to also enter aiter's ca_comm.capture().

When ATOM model runs as vLLM plugin, the model uses aiter's collectives
(tensor_model_parallel_fused_allreduce_rmsnorm etc.) but vLLM's graph_capture
only calls vLLM's ca_comm.capture(). aiter's ca_comm never enters capture mode,
causing _IS_CAPTURING=False -> registered=False -> hipMemcpyAsync on every call.

This patch wraps vLLM's GroupCoordinator.graph_capture to also nest aiter's
ca_comm.capture(), so fused_allreduce_rmsnorm uses registered=True and avoids
the extra hipMemcpyAsync.
"""

import functools
import logging
from contextlib import contextmanager, nullcontext

logger = logging.getLogger("atom")

# Avoid applying patch multiple times
_GRAPH_CAPTURE_PATCH_APPLIED = False


def _get_aiter_ca_capture_context():
    """Lazily get aiter's ca_comm.capture() context, or nullcontext if unavailable."""
    try:
        from aiter.dist.parallel_state import get_tp_group

        aiter_tp = get_tp_group()
    except Exception:
        return nullcontext()

    if aiter_tp is None:
        return nullcontext()

    device_communicator = getattr(aiter_tp, "device_communicator", None)
    if device_communicator is None:
        return nullcontext()

    aiter_ca_comm = getattr(device_communicator, "ca_comm", None)
    if aiter_ca_comm is None or getattr(aiter_ca_comm, "disabled", True):
        return nullcontext()

    capture_method = getattr(aiter_ca_comm, "capture", None)
    if capture_method is None:
        return nullcontext()

    return capture_method()


def _patched_graph_capture(original_graph_capture):
    """Wrap vLLM's graph_capture to also enter aiter's ca_comm.capture()."""

    @functools.wraps(original_graph_capture)
    @contextmanager
    def wrapped(self, graph_capture_context=None):
        aiter_ca_context = _get_aiter_ca_capture_context()
        with aiter_ca_context:
            with original_graph_capture(self, graph_capture_context) as ctx:
                yield ctx

    return wrapped


def apply_graph_capture_patch() -> None:
    """Patch vLLM's GroupCoordinator.graph_capture to nest aiter's ca_comm.capture()."""
    global _GRAPH_CAPTURE_PATCH_APPLIED

    if _GRAPH_CAPTURE_PATCH_APPLIED:
        return

    try:
        import vllm.distributed.parallel_state as parallel_state

        GroupCoordinator = getattr(parallel_state, "GroupCoordinator", None)
        if GroupCoordinator is None:
            logger.debug("ATOM graph_capture patch: GroupCoordinator not found, skip")
            return

        original = getattr(GroupCoordinator, "graph_capture", None)
        if original is None or getattr(original, "_atom_aiter_patched", False):
            return

        GroupCoordinator.graph_capture = _patched_graph_capture(original)
        GroupCoordinator.graph_capture._atom_aiter_patched = True  # type: ignore
        _GRAPH_CAPTURE_PATCH_APPLIED = True
        logger.info(
            "ATOM plugin: patched vLLM graph_capture to nest aiter ca_comm.capture() "
            "(avoids hipMemcpyAsync in fused_allreduce_rmsnorm)"
        )
    except ImportError as e:
        logger.debug("ATOM graph_capture patch: vllm not available (%s), skip", e)
    except Exception as e:
        logger.warning(
            "ATOM graph_capture patch failed: %s. "
            "fused_allreduce_rmsnorm may incur extra hipMemcpyAsync in vLLM plugin mode.",
            e,
        )
