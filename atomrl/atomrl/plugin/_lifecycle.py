# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ZhangDanyang-AMD
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Shared ATOM rollout-manager lifecycle helpers for framework plugins."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def _shutdown_best_effort(manager: Any) -> None:
    try:
        manager.shutdown()
    except Exception as error:  # noqa: BLE001 - preserve the initiating failure
        logger.warning("atomrl: partial ATOM manager cleanup failed: %s", error)


def default_engine_kwargs(args: Any, framework: str) -> dict[str, Any]:
    """Build public ATOM engine kwargs from a host framework's arguments."""
    kwargs: dict[str, Any] = {}
    if framework == "slime":
        tp = getattr(args, "sglang_tp_size", None) or getattr(
            args, "tensor_model_parallel_size", 1
        )
        max_seqs = getattr(args, "sglang_server_concurrency", None) or getattr(
            args, "max_num_seqs", 256
        )
    else:
        tp = getattr(args, "tensor_model_parallel_size", 1)
        max_seqs = getattr(args, "max_num_seqs", 256)

    kwargs["tensor_parallel_size"] = int(tp)
    kwargs["max_num_seqs"] = int(max_seqs)
    kwargs["enforce_eager"] = bool(getattr(args, "enforce_eager", True))
    kwargs["enable_chunked_prefill"] = bool(
        getattr(args, "enable_chunked_prefill", True)
    )

    memory = getattr(args, "gpu_memory_utilization", None)
    if memory is not None:
        kwargs["gpu_memory_utilization"] = float(memory)

    max_model_len = int(getattr(args, "rollout_max_prompt_len", 0) or 0) + int(
        getattr(args, "rollout_max_response_len", 0) or 0
    )
    if max_model_len > 0:
        kwargs["max_model_len"] = max_model_len
    return kwargs


def create_atom_manager(
    actor_wg: Any,
    model_path: str,
    engine_kwargs: dict[str, Any],
    *,
    max_concurrency: int = 64,
    base_seed: int | None = None,
) -> Any:
    """Create and launch an ``ATOMReplicaManager`` through its public API."""
    from atomrl.engine.inference.atom_ray_server import ATOMReplicaManager

    manager = ATOMReplicaManager(
        actor_wg=actor_wg,
        model_name=model_path,
        engine_kwargs=engine_kwargs,
        max_concurrency=max_concurrency,
        base_seed=base_seed,
    )
    try:
        manager.create()
    except Exception:
        _shutdown_best_effort(manager)
        raise
    return manager


def setup_args_manager(
    args: Any,
    actor_wg: Any,
    *,
    framework: str,
    engine_kwargs: dict[str, Any] | None = None,
    max_concurrency: int = 64,
) -> Any:
    """Launch ATOM and store the manager on a host framework's args object."""
    model_path = getattr(args, "hf_checkpoint", None) or getattr(args, "model_name", "")
    if not model_path:
        raise ValueError(
            "args must define 'hf_checkpoint' or 'model_name' for the ATOM model"
        )
    manager = create_atom_manager(
        actor_wg,
        model_path,
        engine_kwargs or default_engine_kwargs(args, framework),
        max_concurrency=max_concurrency,
        base_seed=getattr(args, "seed", None),
    )
    args._atomrl_atom_manager = manager
    logger.info(
        "atomrl: launched %d ATOM replicas for %s",
        manager.num_replicas,
        framework,
    )
    return manager


def teardown_args_manager(args: Any) -> None:
    """Shut down and detach a manager stored on a host args object."""
    manager = getattr(args, "_atomrl_atom_manager", None)
    if manager is None:
        return
    manager.shutdown()
    args._atomrl_atom_manager = None


def setup_adapter_manager(
    adapter: Any,
    actor_wg: Any,
    model_path: str,
    engine_kwargs: dict[str, Any],
    *,
    max_concurrency: int = 64,
    base_seed: int | None = None,
) -> Any:
    """Launch ATOM and inject it into a verl rollout adapter."""
    manager = create_atom_manager(
        actor_wg,
        model_path,
        engine_kwargs,
        max_concurrency=max_concurrency,
        base_seed=base_seed,
    )
    try:
        adapter.set_manager(manager)
    except Exception:
        _shutdown_best_effort(manager)
        raise
    return manager


def teardown_adapter_manager(adapter: Any) -> None:
    """Shut down and detach a verl rollout adapter's manager."""
    manager = getattr(adapter, "_manager", None)
    if manager is None:
        return
    manager.shutdown()
    adapter._manager = None
