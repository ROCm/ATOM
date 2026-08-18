"""Narrow compatibility fixes for SGLang's TBO metadata contract."""

import functools
import logging
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

logger = logging.getLogger("atom.plugin.sglang.tbo")

_current_child_attn_backend: ContextVar[Any | None] = ContextVar(
    "atom_sglang_tbo_child_attn_backend",
    default=None,
)


@contextmanager
def bind_sglang_tbo_child_attn_backend(backend):
    """Bind one SGLang child backend to the current ATOM worker thread."""

    token = _current_child_attn_backend.set(backend)
    try:
        yield
    finally:
        _current_child_attn_backend.reset(token)


def _install_thread_local_tbo_attention_backend_patch() -> None:
    """Route SGLang attention calls to the current ATOM TBO child backend.

    SGLang 0.5.15's forward context is a process-global variable because its
    native TBO executor interleaves both children on one Python thread. ATOM
    executes complete child forwards on two worker threads, so changing that
    global context in either worker would race. Use a ContextVar override in
    ``get_attn_backend`` instead. Patch the RadixAttention module too because it
    imports the getter by value.
    """

    from sglang.srt.layers import radix_attention
    from sglang.srt.model_executor import forward_context

    original_attr = "_atom_sglang_original_get_attn_backend"
    if hasattr(forward_context, original_attr):
        return

    original_get_attn_backend = forward_context.get_attn_backend
    setattr(forward_context, original_attr, original_get_attn_backend)

    @functools.wraps(original_get_attn_backend)
    def _get_thread_local_attn_backend():
        child_backend = _current_child_attn_backend.get()
        if child_backend is not None:
            return child_backend
        return original_get_attn_backend()

    forward_context.get_attn_backend = _get_thread_local_attn_backend
    # radix_attention imported get_attn_backend with ``from ... import ...``,
    # so it holds the original function object instead of looking the name up
    # on forward_context at call time. Replace that cached module reference as
    # well; patching forward_context.get_attn_backend alone would not affect it.
    radix_attention.get_attn_backend = _get_thread_local_attn_backend
    logger.info(
        "ATOM SGLang TBO compat: installed thread-local child attention routing"
    )


def _install_cuda_graph_tbo_capture_token_count_patch() -> None:
    """Preserve the real token count on synthetic CUDA Graph capture batches.

    SGLang 0.5.15 passes ``num_tokens`` to its TBO capture preparer but still
    does not copy it to ``ForwardBatch.num_token_non_padded_cpu``. The ATOM
    adapter needs that CPU value to trim child output ranges without reading a
    GPU scalar during graph capture.
    """

    from sglang.srt.batch_overlap.two_batch_overlap import TboCudaGraphRunnerPlugin

    original_attr = "_atom_sglang_original_capture_one_batch_size"
    if hasattr(TboCudaGraphRunnerPlugin, original_attr):
        return

    original_capture_one_batch_size = TboCudaGraphRunnerPlugin.capture_one_batch_size
    setattr(TboCudaGraphRunnerPlugin, original_attr, original_capture_one_batch_size)

    @functools.wraps(original_capture_one_batch_size)
    def _capture_one_batch_size_with_real_token_count(self, batch, num_tokens):
        batch.num_token_non_padded_cpu = num_tokens
        return original_capture_one_batch_size(self, batch, num_tokens)

    TboCudaGraphRunnerPlugin.capture_one_batch_size = (
        _capture_one_batch_size_with_real_token_count
    )
    logger.info(
        "ATOM SGLang TBO compat: installed CUDA graph capture token-count patch"
    )


def install_sglang_tbo_compat_patches() -> None:
    """Install compatibility fixes still required by SGLang 0.5.15.

    The old 0.5.12 ``rids`` padding patch is intentionally absent: 0.5.15's
    TBO splitter now handles request-ID lists shorter than a padded batch.
    """

    _install_thread_local_tbo_attention_backend_patch()
    _install_cuda_graph_tbo_capture_token_count_patch()
