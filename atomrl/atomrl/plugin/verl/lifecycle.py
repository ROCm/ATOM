# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ZhangDanyang-AMD
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""ATOM rollout-manager lifecycle for verl integration."""

from __future__ import annotations

from typing import Any

from atomrl.plugin._lifecycle import (
    setup_adapter_manager,
    teardown_adapter_manager,
)


def setup_atom_manager(
    rollout_adapter: Any,
    actor_wg: Any,
    model_path: str,
    engine_kwargs: dict[str, Any],
    *,
    max_concurrency: int = 64,
    base_seed: int | None = None,
) -> Any:
    """Create, launch, and inject an ATOM manager into a verl adapter."""
    return setup_adapter_manager(
        rollout_adapter,
        actor_wg,
        model_path,
        engine_kwargs,
        max_concurrency=max_concurrency,
        base_seed=base_seed,
    )


def teardown_atom_manager(rollout_adapter: Any) -> None:
    """Shut down and detach verl's ATOM manager."""
    teardown_adapter_manager(rollout_adapter)
