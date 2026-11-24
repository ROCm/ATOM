#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Verify AtomRL plugin registration in verl."""

import importlib.metadata

eps = list(importlib.metadata.entry_points(group="verl.plugins"))
print(f"verl.plugins entry-points: {eps}")

import verl

print(f"verl version: {verl.__version__}")

from verl.workers.rollout.base import _ROLLOUT_REGISTRY

print(f"Rollout registry keys: {list(_ROLLOUT_REGISTRY.keys())}")

if ("atom", "async") not in _ROLLOUT_REGISTRY:
    print("WARN: auto-register failed, calling register() manually")
    from atomrl.plugin.verl.register import register

    register()
    print(f"After manual register: {list(_ROLLOUT_REGISTRY.keys())}")

assert ("atom", "async") in _ROLLOUT_REGISTRY, "ATOM rollout not registered"
print("Plugin verification: OK")
