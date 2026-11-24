# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Shared type definitions for AtomRL."""

from __future__ import annotations

from enum import Enum


class TrainingBackend(str, Enum):
    FSDP2 = "fsdp2"
    MEGATRON_NATIVE = "megatron_native"
    NONE = "none"


class GenerationBackend(str, Enum):
    ATOM = "atom"
    VLLM = "vllm"


class AlgorithmName(str, Enum):
    GRPO = "grpo"
    DAPO = "dapo"
    PPO = "ppo"
    OPD = "opd"
    SPEC_DISTILL = "spec_distill"
