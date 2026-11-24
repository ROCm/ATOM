# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Default component registrations for phase-1 assembler rollout."""

from __future__ import annotations

from atomrl.architecture.registry.component_registries import (
    inference_backend_registry,
    trainer_registry,
    training_backend_registry,
    worker_role_registry,
)


def _ensure(registry, key: str, value) -> None:
    if key not in registry:
        registry.register(key, value)


def register_default_bindings() -> None:
    """Register built-in components for assembler usage.

    Heavy component imports are deferred into this function: ``rl_trainer``
    imports ``atomrl.controller`` which (via ``worker_group_factory``) imports
    this module, so importing trainers/engines at module top creates a circular
    import. Importing them lazily here breaks that cycle.
    """
    from atomrl.engine.inference.atom_engine import AtomEngine
    from atomrl.engine.training.fsdp_backend import FSDP2Backend
    from atomrl.trainer.async_trainer import AsyncRLTrainer
    from atomrl.trainer.opd_trainer import OPDTrainer
    from atomrl.trainer.rl_trainer import RLTrainer
    from atomrl.trainer.spec_distill_trainer import SpecDistillTrainer
    from atomrl.workers.actor_worker import AtomActorWorker
    from atomrl.workers.critic_worker import CriticWorker
    from atomrl.workers.ref_worker import RefPolicyWorker
    from atomrl.workers.reward_worker import RewardWorker
    from atomrl.workers.rollout_worker import AtomRolloutWorker
    from atomrl.workers.teacher_worker import TeacherWorker

    _ensure(worker_role_registry, "actor.default", AtomActorWorker)
    _ensure(worker_role_registry, "critic.default", CriticWorker)
    _ensure(worker_role_registry, "ref.default", RefPolicyWorker)
    _ensure(worker_role_registry, "reward.default", RewardWorker)
    _ensure(worker_role_registry, "rollout.atom", AtomRolloutWorker)
    _ensure(worker_role_registry, "teacher.default", TeacherWorker)

    _ensure(training_backend_registry, "fsdp", FSDP2Backend)
    _ensure(training_backend_registry, "fsdp2", FSDP2Backend)
    _ensure(inference_backend_registry, "atom", AtomEngine)

    _ensure(trainer_registry, "trainer.rl", RLTrainer)
    _ensure(trainer_registry, "trainer.async", AsyncRLTrainer)
    _ensure(trainer_registry, "trainer.opd", OPDTrainer)
    _ensure(trainer_registry, "trainer.spec_distill", SpecDistillTrainer)
