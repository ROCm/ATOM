# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Runtime graph construction for trainer/controller composition."""

from __future__ import annotations

from dataclasses import dataclass

from atomrl.architecture.assembly.default_bindings import register_default_bindings
from atomrl.architecture.assembly.policy_validator import validate_backend_policy
from atomrl.architecture.registry.component_registries import (
    inference_backend_registry,
    trainer_registry,
    training_backend_registry,
    worker_role_registry,
)
from atomrl.core.config import AtomRLConfig


@dataclass
class RuntimeGraph:
    actor_worker_cls: type
    rollout_worker_cls: type
    trainer_cls: type
    training_backend_cls: type
    inference_backend_cls: type


class RuntimeAssembler:
    """Build runtime component graph from config and registries."""

    def __init__(self, config: AtomRLConfig):
        self.config = config
        register_default_bindings()

    def build_graph(self) -> dict[str, type]:
        if self.config.assembly.strict_policy:
            validate_backend_policy(self.config.assembly)

        actor_key = self.config.assembly.role_impl_keys.get("actor", "actor.default")
        rollout_key = self.config.assembly.role_impl_keys.get("rollout", "rollout.atom")
        trainer_key = self._resolve_trainer_key()

        graph = RuntimeGraph(
            actor_worker_cls=worker_role_registry.get(actor_key),
            rollout_worker_cls=worker_role_registry.get(rollout_key),
            trainer_cls=trainer_registry.get(trainer_key),
            training_backend_cls=training_backend_registry.get(
                self.config.assembly.training_backend
            ),
            inference_backend_cls=inference_backend_registry.get(
                self.config.assembly.inference_backend
            ),
        )
        return graph.__dict__

    def _resolve_trainer_key(self) -> str:
        algo = self.config.algorithm.name.lower()
        if algo == "opd":
            return "trainer.opd"
        if algo == "spec_distill":
            return "trainer.spec_distill"
        if self.config.async_training.enabled:
            return "trainer.async"
        return "trainer.rl"
