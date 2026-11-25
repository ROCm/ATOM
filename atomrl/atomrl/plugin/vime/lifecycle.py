# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ZhangDanyang-AMD
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""ATOM rollout-manager lifecycle for vime integration."""

from __future__ import annotations

from typing import Any

from atomrl.plugin._lifecycle import setup_args_manager, teardown_args_manager


def setup_atom_manager(
    args: Any,
    actor_wg: Any,
    *,
    engine_kwargs: dict[str, Any] | None = None,
    max_concurrency: int = 64,
) -> Any:
    """Create, launch, and store an ATOM manager on vime args."""
    return setup_args_manager(
        args,
        actor_wg,
        framework="vime",
        engine_kwargs=engine_kwargs,
        max_concurrency=max_concurrency,
    )


def teardown_atom_manager(args: Any) -> None:
    """Shut down and detach vime's ATOM manager."""
    teardown_args_manager(args)
