# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Co-located actor + rollout worker sharing local GPUs."""

from __future__ import annotations

import logging
from typing import Any

from atomrl.core.protocol import DataProto
from atomrl.workers.actor_worker import AtomActorWorker
from atomrl.workers.base_worker import BaseWorker
from atomrl.workers.rollout_worker import AtomRolloutWorker

logger = logging.getLogger(__name__)


class HybridWorker(BaseWorker):
    """Hosts training and generation stacks in one Ray actor process.

    Weight sync between the two stacks is a local ``state_dict`` transfer,
    avoiding Ray object store round-trips when policies are co-located.
    """

    def __init__(
        self, rank: int, world_size: int, config: dict[str, Any] | None = None
    ) -> None:
        super().__init__(rank, world_size, config)
        self._actor = AtomActorWorker(rank, world_size, config)
        self._rollout = AtomRolloutWorker(rank, world_size, config)

    def init_model(self) -> None:
        """Construct both actor (train) and ATOM rollout stacks."""
        self._actor.init_model()
        self._rollout.init_model()
        self._log.info("HybridWorker: actor and rollout initialized.")

    def generate(self, batch: DataProto) -> DataProto:
        """Delegate to the rollout worker."""
        return self._rollout.generate(batch)

    def update_policy(self, batch: DataProto) -> dict[str, float]:
        """Delegate to the actor worker."""
        return self._actor.update_policy(batch)

    def sync_weights(self) -> None:
        """Push actor weights into the rollout engine."""
        state = self._actor.get_state_dict()
        self._rollout.update_weights(state)
        self._log.info("HybridWorker.sync_weights: transferred %d tensors.", len(state))

    def prepare_for_generation(self) -> None:
        """Forward KV / cache preparation to rollout."""
        self._rollout.prepare_for_generation()

    def finish_generation(self) -> None:
        """Forward rollout teardown hooks."""
        self._rollout.finish_generation()

    def cleanup(self) -> None:
        self._actor.cleanup()
        self._rollout.cleanup()
        super().cleanup()
