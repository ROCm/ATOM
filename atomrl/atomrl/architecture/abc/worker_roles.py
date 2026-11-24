# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Role-specific worker contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod

from atomrl.core.protocol import DataProto


class ActorWorkerABC(ABC):
    @abstractmethod
    def init_model(self) -> None: ...

    @abstractmethod
    def compute_log_probs(self, batch: DataProto) -> DataProto: ...

    @abstractmethod
    def train_step(self, batch: DataProto) -> dict[str, float]: ...

    @abstractmethod
    def get_state_dict(self) -> dict: ...


class CriticWorkerABC(ABC):
    @abstractmethod
    def init_model(self) -> None: ...

    @abstractmethod
    def compute_values(self, batch: DataProto) -> DataProto: ...

    @abstractmethod
    def train_step(self, batch: DataProto) -> dict[str, float]: ...


class RolloutWorkerABC(ABC):
    @abstractmethod
    def init_model(self) -> None: ...

    @abstractmethod
    def prepare_for_generation(self) -> None: ...

    @abstractmethod
    def generate(self, batch: DataProto) -> DataProto: ...

    @abstractmethod
    def update_weights(self, state_dict: dict) -> None: ...


class RewardWorkerABC(ABC):
    @abstractmethod
    def init_model(self) -> None: ...

    @abstractmethod
    def compute_rewards(self, batch: DataProto) -> DataProto: ...


class TeacherWorkerABC(ABC):
    @abstractmethod
    def init_model(self) -> None: ...

    @abstractmethod
    def compute_teacher_outputs(
        self, batch: DataProto, return_mode: str = "logits"
    ) -> DataProto: ...
