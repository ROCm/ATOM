# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from atom.model_engine.engine_utility import EngineUtilityHandler
from atom.rollout.memory_manager import MemoryManagerMixin
from atom.rollout.weight_sync import load_weights_via_ipc, load_weights_via_shm
from atom.rollout.weight_updater import WeightUpdaterMixin

__all__ = [
    "AsyncLLMEngine",
    "EngineUtilityHandler",
    "MemoryManagerMixin",
    "RLHFModelRunner",
    "WeightUpdaterMixin",
    "load_weights_via_ipc",
    "load_weights_via_shm",
]


def __getattr__(name):
    # Lazy import to break circular dependency:
    #   atom.model_engine.llm_engine -> engine_core_mgr -> engine_core
    #   -> atom.rollout (this __init__) -> async_engine -> atom.model_engine.llm_engine
    if name == "AsyncLLMEngine":
        from atom.rollout.async_engine import AsyncLLMEngine

        return AsyncLLMEngine
    if name == "RLHFModelRunner":
        from atom.rollout.model_runner_ext import RLHFModelRunner

        return RLHFModelRunner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
