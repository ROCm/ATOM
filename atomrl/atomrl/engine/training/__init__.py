# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Training backends and engine abstractions."""

from __future__ import annotations

from atomrl.engine.training.base_engine import BaseEngine, BaseEngineCtx, EngineRegistry
from atomrl.engine.training.fsdp_backend import FSDP2Backend
from atomrl.engine.training.fsdp_engine import FSDP2Engine, FSDP2EngineWithLMHead
from atomrl.engine.training.megatron_engine import (
    MegatronEngine,
    MegatronEngineWithLMHead,
    MegatronEngineWithValueHead,
)
from atomrl.engine.training.megatron_lumen_dsv4_engine import (
    MegatronLumenDSV4Engine,
    MegatronLumenDSV4EngineWithLMHead,
    MegatronLumenDSV4EngineWithValueHead,
)
from atomrl.engine.training.megatron_native_engine import (
    MegatronNativeEngine,
    MegatronNativeEngineWithLMHead,
    MegatronNativeEngineWithValueHead,
)

__all__ = [
    "BaseEngine",
    "BaseEngineCtx",
    "EngineRegistry",
    "FSDP2Backend",
    "FSDP2Engine",
    "FSDP2EngineWithLMHead",
    "MegatronEngine",
    "MegatronEngineWithLMHead",
    "MegatronEngineWithValueHead",
    "MegatronLumenDSV4Engine",
    "MegatronLumenDSV4EngineWithLMHead",
    "MegatronLumenDSV4EngineWithValueHead",
    "MegatronNativeEngine",
    "MegatronNativeEngineWithLMHead",
    "MegatronNativeEngineWithValueHead",
]
