"""Backend contracts for runtime assembly."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class TrainingBackendABC(ABC):
    @abstractmethod
    def build_model(self, model_name: str, cfg: dict[str, Any]) -> Any: ...


class InferenceBackendABC(ABC):
    @abstractmethod
    def init(self) -> None: ...
