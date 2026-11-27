# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Factory helpers for resolving worker classes from role keys."""

from __future__ import annotations

from atomrl.architecture.assembly.default_bindings import register_default_bindings
from atomrl.architecture.registry.component_registries import worker_role_registry


def resolve_worker_class(role_key: str):
    """Resolve a worker class by registry role key."""
    register_default_bindings()
    return worker_role_registry.get(role_key)
