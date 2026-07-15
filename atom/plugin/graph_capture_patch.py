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
from contextlib import ExitStack, contextmanager

logger = logging.getLogger("atom")


def _ca_comm_of(group):
    """Return a group's capturable aiter ca_comm, or None if unavailable."""
    if group is None:
        return None
    device_communicator = getattr(group, "device_communicator", None)
    if device_communicator is None:
        return None
    ca_comm = getattr(device_communicator, "ca_comm", None)
    if ca_comm is None or getattr(ca_comm, "disabled", True):
        return None
    if getattr(ca_comm, "capture", None) is None:
        return None
    return ca_comm


def _iter_aiter_capture_ca_comms():
    """Yield the aiter ca_comm(s) that must enter capture mode during graph
    capture: the TP group, plus the dedicated Context-Parallel (_PCP) group when
    it has its OWN ca_comm (reuse-TP-as-CP with a dedicated CP group).

    The model's custom collectives run on aiter's groups (TP all-reduce on TP;
    CP all-gather / reduce-scatter on _PCP). Each ca_comm must be in capture
    mode so its graph buffers register (flush_graph_buffers on exit); otherwise
    its collectives take the non-registered / hipMemcpyAsync path. Deduped by
    identity, so the aliased case (_PCP is TP) enters exactly once.
    """
    getters = []
    try:
        from aiter.dist.parallel_state import get_tp_group

        getters.append(get_tp_group)
    except Exception:
        pass
    try:
        from aiter.dist.parallel_state import get_pcp_group

        getters.append(get_pcp_group)
    except Exception:
        pass

    seen = set()
    for getter in getters:
        try:
            group = getter()
        except Exception:
            # get_pcp_group() asserts when _PCP is unset (non-CP runs): skip.
            continue
        ca_comm = _ca_comm_of(group)
        if ca_comm is not None and id(ca_comm) not in seen:
            seen.add(id(ca_comm))
            yield ca_comm


@contextmanager
def _get_aiter_ca_capture_context():
    """Nest ca_comm.capture() for every aiter group that needs it (TP + CP)."""
    ca_comms = list(_iter_aiter_capture_ca_comms())
    with ExitStack() as stack:
        for ca_comm in ca_comms:
            stack.enter_context(ca_comm.capture())
        yield


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
