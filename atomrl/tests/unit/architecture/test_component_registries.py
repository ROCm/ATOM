# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

from __future__ import annotations

from atomrl.architecture.registry.component_registries import worker_role_registry


class DummyWorker:
    pass


def test_worker_registry_register_and_get() -> None:
    worker_role_registry.register("unit.dummy", DummyWorker)
    assert worker_role_registry.get("unit.dummy") is DummyWorker
