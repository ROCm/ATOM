# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Production R3 is vLLM + hard-assignment expert ids only.

``assert_supported_r3`` fail-closes ATOM rollout and ``replay_mode=distribution``.
Do not import this from ``atomrl.moe`` inside ``atomrl.core.config`` — import
this module directly so config load does not cycle through ``R3Manager``.
"""

from __future__ import annotations

from typing import Any

SUPPORTED_R3_GENERATION_BACKEND = "vllm"
SUPPORTED_R3_REPLAY_MODE = "hard_assignment"


def _nested(obj: Any, *keys: str, default: Any = None) -> Any:
    cur: Any = obj
    for key in keys:
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(key, default)
        else:
            cur = getattr(cur, key, default)
    return cur


def r3_fields_from_config(config: Any) -> tuple[bool, str, str]:
    """Return ``(enabled, replay_mode, generation_backend)`` from a config tree."""
    enabled = bool(_nested(config, "moe", "r3", "enabled", default=False) or False)
    replay_mode = str(
        _nested(
            config,
            "moe",
            "r3",
            "replay_mode",
            default=SUPPORTED_R3_REPLAY_MODE,
        )
        or SUPPORTED_R3_REPLAY_MODE
    )
    generation_backend = str(
        _nested(config, "policy", "generation_backend", default="") or ""
    )
    return enabled, replay_mode, generation_backend


def assert_supported_r3(*, replay_mode: str, generation_backend: str) -> None:
    """Raise unless this enabled R3 job is vLLM hard-assignment.

    Call only when ``moe.r3.enabled`` (or equivalent) is already true.
    """
    backend = (generation_backend or "").strip().lower()
    mode = (replay_mode or "").strip().lower()
    if backend != SUPPORTED_R3_GENERATION_BACKEND:
        raise RuntimeError(
            "moe.r3.enabled=true requires policy.generation_backend=vllm; "
            f"got {generation_backend!r}. ATOM R3 is not supported yet."
        )
    if mode != SUPPORTED_R3_REPLAY_MODE:
        raise RuntimeError(
            "moe.r3.enabled=true requires moe.r3.replay_mode=hard_assignment; "
            f"got {replay_mode!r}. Distribution replay is not supported yet "
            "(vLLM does not return router logits)."
        )
