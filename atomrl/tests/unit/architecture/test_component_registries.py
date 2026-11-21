from __future__ import annotations

from atomrl.architecture.registry.component_registries import worker_role_registry


class DummyWorker:
    pass


def test_worker_registry_register_and_get() -> None:
    worker_role_registry.register("unit.dummy", DummyWorker)
    assert worker_role_registry.get("unit.dummy") is DummyWorker
