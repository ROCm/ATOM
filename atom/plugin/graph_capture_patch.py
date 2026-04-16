"""Patch a framework's graph_capture to also enter aiter's ca_comm.capture().

When ATOM model runs as a plugin backend (vLLM or SGLang), the model uses aiter's
collectives (tensor_model_parallel_fused_allreduce_rmsnorm etc.)
but the host framework's graph_capture only enters its own ca_comm.capture().
aiter's ca_comm never enters capture mode, causing _IS_CAPTURING=False ->
registered=False -> hipMemcpyAsync on every call.

This module provides a shared helper that patches any framework(vLLM or SGLang)'s
GroupCoordinator.graph_capture to also nest aiter's ca_comm.capture(),
so fused_allreduce_rmsnorm uses registered=True and avoids the extra hipMemcpyAsync.
"""

import functools
import logging
from contextlib import contextmanager, nullcontext

logger = logging.getLogger("atom")


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
    """Wrap a framework's graph_capture to also enter aiter's ca_comm.capture()."""

    @functools.wraps(original_graph_capture)
    @contextmanager
    def wrapped(self, graph_capture_context=None, **kwargs):
        aiter_ca_context = _get_aiter_ca_capture_context()
        with aiter_ca_context:
            with original_graph_capture(self, graph_capture_context, **kwargs) as ctx:
                yield ctx

    return wrapped


def apply_graph_capture_patch(framework_module_path: str) -> bool:
    """Patch a framework's GroupCoordinator.graph_capture to nest aiter's
    ca_comm.capture().

    Args:
        framework_module_path: Dotted import path to the framework's
            parallel_state module containing GroupCoordinator
            (e.g. "vllm.distributed.parallel_state" or
            "sglang.srt.distributed.parallel_state").

    Returns:
        True if the patch was applied, False otherwise.
    """
    import importlib

    try:
        parallel_state = importlib.import_module(framework_module_path)
    except ImportError as e:
        logger.debug(
            "ATOM graph_capture patch: %s not available (%s), skip",
            framework_module_path,
            e,
        )
        return False

    GroupCoordinator = getattr(parallel_state, "GroupCoordinator", None)
    if GroupCoordinator is None:
        logger.debug(
            "ATOM graph_capture patch: GroupCoordinator not found in %s, skip",
            framework_module_path,
        )
        return False

    original = getattr(GroupCoordinator, "graph_capture", None)
    if original is None or getattr(original, "_atom_aiter_patched", False):
        return False

    try:
        GroupCoordinator.graph_capture = _patched_graph_capture(original)
        GroupCoordinator.graph_capture._atom_aiter_patched = True  # type: ignore
        logger.info(
            "ATOM plugin: patched %s.GroupCoordinator.graph_capture to nest "
            "aiter ca_comm.capture() (avoids hipMemcpyAsync in aiter collectives)",
            framework_module_path,
        )
        return True
    except Exception as e:
        logger.warning(
            "ATOM graph_capture patch for %s failed: %s. "
            "aiter collectives may incur extra hipMemcpyAsync in plugin mode.",
            framework_module_path,
            e,
        )
        return False
