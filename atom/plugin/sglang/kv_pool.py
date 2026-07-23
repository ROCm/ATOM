# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Compatibility helpers for SGLang KV pool lookup."""

from __future__ import annotations

from typing import Any


def _maybe_get_current_attn_backend() -> Any | None:
    try:
        from sglang.srt.model_executor.forward_context import get_attn_backend

        return get_attn_backend()
    except Exception:
        return None


def _maybe_setattr(obj: Any, name: str, value: Any) -> None:
    if obj is None or value is None or getattr(obj, name, None) is not None:
        return
    try:
        setattr(obj, name, value)
    except Exception:
        pass


def maybe_get_sglang_kv_pools(
    forward_batch: Any | None = None,
) -> tuple[Any | None, Any | None]:
    """Resolve SGLang token/request KV pools across ForwardBatch API versions.

    SGLang 0.5.15 keeps these pools on the active attention backend instead of
    exposing them directly on ``ForwardBatch``. Older versions exposed the same
    objects on ``forward_batch``. Return ``None`` entries when a pool is not
    available so callers can decide whether to error or skip optional work.
    """

    token_to_kv_pool = None
    req_to_token_pool = None

    if forward_batch is not None:
        token_to_kv_pool = getattr(forward_batch, "token_to_kv_pool", None)
        req_to_token_pool = getattr(forward_batch, "req_to_token_pool", None)
        if token_to_kv_pool is not None and req_to_token_pool is not None:
            return token_to_kv_pool, req_to_token_pool

    backend = _maybe_get_current_attn_backend()
    if backend is not None:
        if token_to_kv_pool is None:
            token_to_kv_pool = getattr(backend, "token_to_kv_pool", None)
        if req_to_token_pool is None:
            req_to_token_pool = getattr(backend, "req_to_token_pool", None)

    _maybe_setattr(forward_batch, "token_to_kv_pool", token_to_kv_pool)
    _maybe_setattr(forward_batch, "req_to_token_pool", req_to_token_pool)
    return token_to_kv_pool, req_to_token_pool


def get_sglang_token_to_kv_pool(
    forward_batch: Any | None = None,
    *,
    caller: str = "SGLang plugin",
) -> Any:
    token_to_kv_pool, _ = maybe_get_sglang_kv_pools(forward_batch)
    if token_to_kv_pool is None:
        raise RuntimeError(
            f"{caller} requires SGLang token_to_kv_pool, but it could not be "
            "resolved from ForwardBatch or the current attention backend."
        )
    return token_to_kv_pool
