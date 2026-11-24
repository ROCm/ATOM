# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Typed component registries used by runtime assembler."""

from __future__ import annotations

from atomrl.core.registry import Registry

worker_role_registry = Registry("worker_role")
training_backend_registry = Registry("training_backend")
inference_backend_registry = Registry("inference_backend")
trainer_registry = Registry("trainer")
