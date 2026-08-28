"""Patch SGLang graph capture to also enter aiter's ca_comm.capture().

Delegates to the shared implementation in atom.plugin.graph_capture_patch.
"""

_GRAPH_CAPTURE_PATCH_APPLIED = False


def apply_graph_capture_patch() -> None:
    """Patch SGLang's GroupCoordinator.graph_capture to nest aiter's ca_comm.capture()."""
    global _GRAPH_CAPTURE_PATCH_APPLIED

    if _GRAPH_CAPTURE_PATCH_APPLIED:
        return

    from atom.plugin.graph_capture_patch import apply_graph_capture_patch as _apply

    _GRAPH_CAPTURE_PATCH_APPLIED = _apply("sglang.srt.distributed.parallel_state")
