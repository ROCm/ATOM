# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2025 AtomRL Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This file contains code adapted from the verl project
# (https://github.com/verl-project/verl).
# Original: verl/workers/engine/base.py

"""Abstract base classes and registry for training engines."""

from __future__ import annotations

import typing
from abc import abstractmethod
from collections.abc import Callable, Generator
from contextlib import AbstractContextManager, nullcontext
from typing import Any

import torch


class BaseEngine:
    """Abstract base class defining the interface for model training engines.

    Engine implementations must subclass BaseEngine and provide concrete
    behaviour for all methods.
    """

    def initialize(self) -> None:
        """Instantiate or load the model, optimizer, and learning-rate scheduler."""
        raise NotImplementedError

    @property
    @abstractmethod
    def is_param_offload_enabled(self) -> bool:
        raise NotImplementedError

    @property
    @abstractmethod
    def is_optimizer_offload_enabled(self) -> bool:
        raise NotImplementedError

    def train_mode(self, **kwargs) -> AbstractContextManager:
        """Context manager that switches the engine into training mode."""
        raise NotImplementedError

    def eval_mode(self, **kwargs) -> AbstractContextManager:
        """Context manager that switches the engine into evaluation mode."""
        raise NotImplementedError

    def optimizer_zero_grad(self) -> None:
        raise NotImplementedError

    def optimizer_step(self) -> float:
        """Perform an optimisation step. Returns grad_norm."""
        raise NotImplementedError

    def lr_scheduler_step(self) -> float:
        """Advance the LR scheduler. Returns current learning rate."""
        raise NotImplementedError

    def forward_backward_batch(
        self,
        data: dict[str, Any],
        loss_function: Callable,
        forward_only: bool = False,
    ) -> Any:
        raise NotImplementedError

    def train_batch(self, data: dict[str, Any], loss_function: Callable) -> Any:
        self.optimizer_zero_grad()
        outputs = self.forward_backward_batch(data, loss_function, forward_only=False)
        grad_norm = self.optimizer_step()
        if self.is_mp_src_rank_with_outputs():
            assert "grad_norm" not in outputs.get("metrics", {})
            outputs.setdefault("metrics", {})["grad_norm"] = grad_norm
        return outputs

    def infer_batch(
        self,
        data: dict[str, Any],
        loss_function: Callable | None = None,
    ) -> Any:
        with torch.no_grad():
            outputs = self.forward_backward_batch(
                data, loss_function, forward_only=True
            )
        return outputs

    def get_per_tensor_param(
        self,
    ) -> tuple[Generator[tuple[str, torch.Tensor], None, None], dict | None]:
        """Yield ``(name, tensor)`` pairs and an optional peft config dict."""
        raise NotImplementedError

    def get_data_parallel_size(self) -> int:
        raise NotImplementedError

    def get_data_parallel_rank(self) -> int:
        raise NotImplementedError

    def get_data_parallel_group(self):
        raise NotImplementedError

    def to(
        self,
        device: str,
        model: bool = True,
        optimizer: bool = True,
        grad: bool = True,
    ) -> None:
        if grad:
            assert model, "Gradient buffers must be moved along with model parameters"

    def save_checkpoint(
        self,
        local_path: str,
        global_step: int = 0,
        max_ckpt_to_keep: int | None = None,
        **kwargs: Any,
    ) -> None:
        raise NotImplementedError

    def load_checkpoint(self, local_path: str, **kwargs: Any) -> None:
        raise NotImplementedError

    def is_mp_src_rank_with_outputs(self) -> bool:
        """Whether the current rank is the model-parallel source that holds outputs."""
        raise NotImplementedError

    def disable_adapter(self) -> AbstractContextManager:
        """Temporarily disable LoRA adapters."""
        return nullcontext()


class BaseEngineCtx:
    """Context manager that handles device offload/reload around mode switches.

    Adapted from verl's ``BaseEngineCtx``.
    """

    def __init__(self, engine: BaseEngine, mode: str, **kwargs: Any) -> None:
        self.engine = engine
        self.mode = mode
        assert self.mode in ("train", "eval")
        self.disable_auto_offload = kwargs.pop("disable_auto_offload", False)

    def _get_device_name(self) -> str:
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _context_switch(self, device: str) -> None:
        if self.disable_auto_offload:
            return
        if device != "cpu" and (
            not self.engine.is_param_offload_enabled
            and not self.engine.is_optimizer_offload_enabled
        ):
            return
        if self.mode == "eval":
            self.engine.to(
                device=device,
                model=self.engine.is_param_offload_enabled,
                optimizer=False,
                grad=False,
            )
        elif self.mode == "train":
            self.engine.to(
                device=device,
                model=self.engine.is_param_offload_enabled,
                optimizer=self.engine.is_optimizer_offload_enabled,
                grad=self.engine.is_param_offload_enabled,
            )

    def __enter__(self):
        self._context_switch(self._get_device_name())
        self.engine.mode = self.mode

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._context_switch("cpu")
        self.engine.mode = None


class EngineRegistry:
    """Registry for mapping ``(model_type, backend)`` to concrete engine classes."""

    _engines: typing.ClassVar[dict[str, dict[str, dict[str, type]]]] = {}

    @classmethod
    def register(
        cls,
        model_type: str,
        backend: list[str] | str,
        device: list[str] | str = "cuda",
    ):
        def decorator(engine_class: type) -> type:
            assert issubclass(engine_class, BaseEngine)
            if model_type not in cls._engines:
                cls._engines[model_type] = {}

            backends = backend if isinstance(backend, list) else [backend]
            devices = device if isinstance(device, list) else [device]
            for b in backends:
                if b not in cls._engines[model_type]:
                    cls._engines[model_type][b] = {}
                for d in devices:
                    cls._engines[model_type][b][d] = engine_class

            return engine_class

        return decorator

    @classmethod
    def get_engine_cls(cls, model_type: str, backend: str) -> type:
        assert model_type in cls._engines, f"Unknown model_type: {model_type}"
        assert backend in cls._engines[model_type], f"Unknown backend: {backend}"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        assert (
            device in cls._engines[model_type][backend]
        ), f"No engine for device={device}, model_type={model_type}, backend={backend}"
        return cls._engines[model_type][backend][device]

    @classmethod
    def new(
        cls, model_type: str, backend: str, *args: Any, **kwargs: Any
    ) -> BaseEngine:
        engine_cls = cls.get_engine_cls(model_type, backend)
        return engine_cls(*args, **kwargs)
