"""Resolve runtime objects owned by the active SGLang attention backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SGLangRuntimeObjects:
    attn_backend: Any
    token_to_kv_pool: Any
    req_to_token_pool: Any


def _get_current_attention_backend():
    try:
        from sglang.srt.model_executor.forward_context import get_attn_backend

        return get_attn_backend()
    except (AssertionError, ImportError):
        return None


def _get_backend_pools(backend):
    token_to_kv_pool = getattr(backend, "token_to_kv_pool", None)
    req_to_token_pool = getattr(backend, "req_to_token_pool", None)
    if token_to_kv_pool is not None and req_to_token_pool is not None:
        return token_to_kv_pool, req_to_token_pool
    full_backend = getattr(backend, "full_attn_backend", None)
    return (
        (
            token_to_kv_pool
            if token_to_kv_pool is not None
            else getattr(full_backend, "token_to_kv_pool", None)
        ),
        (
            req_to_token_pool
            if req_to_token_pool is not None
            else getattr(full_backend, "req_to_token_pool", None)
        ),
    )


def resolve_sglang_runtime(forward_batch=None) -> SGLangRuntimeObjects:
    """Return the current backend and its authoritative pool pair.

    SGLang v0.5.15 selects an attention backend through ``ForwardContext`` for
    every forward and draft substep. A batch may still carry pool attributes
    under older SGLang versions, but those values are only a compatibility
    fallback when no current backend exists.
    """

    backend = _get_current_attention_backend()
    if backend is not None:
        token_to_kv_pool, req_to_token_pool = _get_backend_pools(backend)
    else:
        token_to_kv_pool = getattr(forward_batch, "token_to_kv_pool", None)
        req_to_token_pool = getattr(forward_batch, "req_to_token_pool", None)

    if token_to_kv_pool is None or req_to_token_pool is None:
        owner = type(backend).__name__ if backend is not None else "ForwardBatch"
        raise RuntimeError(
            "SGLang runtime owner does not provide a complete KV pool pair: "
            f"owner={owner}, token_to_kv_pool={token_to_kv_pool is not None}, "
            f"req_to_token_pool={req_to_token_pool is not None}"
        )

    return SGLangRuntimeObjects(
        attn_backend=backend,
        token_to_kv_pool=token_to_kv_pool,
        req_to_token_pool=req_to_token_pool,
    )
