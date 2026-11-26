# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Reward model or rule-based reward worker."""

from __future__ import annotations

import importlib
import logging
from collections.abc import Callable
from typing import Any

import torch
from torch import nn

from atomrl.core.protocol import DataProto
from atomrl.engine.training.fsdp_backend import FSDP2Backend
from atomrl.workers.base_worker import BaseWorker, get_nested_config

logger = logging.getLogger(__name__)


def _default_sequence_reward(batch: DataProto) -> torch.Tensor:
    """Deterministic placeholder reward from token ids (no external deps)."""
    input_ids = batch["input_ids"].float()
    scores = input_ids.sum(dim=-1) / max(1, input_ids.shape[-1])
    return scores


class RewardWorker(BaseWorker):
    """Computes scalar or vector rewards packaged back into :class:`DataProto`."""

    def __init__(
        self, rank: int, world_size: int, config: dict[str, Any] | None = None
    ) -> None:
        super().__init__(rank, world_size, config)
        self._model: nn.Module | None = None
        self._fn: Callable[[DataProto], torch.Tensor] | None = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def init_model(self) -> None:
        """Instantiate a reward model or import a Python reward function."""
        reward_cfg = get_nested_config(self.config, "reward", default={}) or {}
        r_type = str(reward_cfg.get("type", "function")).lower()
        if r_type == "model":
            self._fn = None
            training_stub: dict[str, Any] = {"tiny_lm": {"dim": 128, "n_layers": 1}}
            model_name = str(reward_cfg.get("model_name", "reward-model"))
            self._model = FSDP2Backend.build_model(model_name, training_stub)
            self._model = self._model.to(self._device)
            for p in self._model.parameters():
                p.requires_grad_(False)
            self._model.eval()
            self._log.info("RewardWorker: reward model loaded.")
            return

        self._model = None
        fn_path = str(reward_cfg.get("function", ""))
        if fn_path and "." in fn_path:
            mod_name, attr = fn_path.rsplit(".", 1)
            try:
                mod = importlib.import_module(mod_name)
                self._fn = getattr(mod, attr)
                self._log.info("RewardWorker: registered reward fn %s", fn_path)
                return
            except (ImportError, AttributeError) as exc:
                self._log.warning(
                    "Reward fn %s unavailable (%s); using built-in default.",
                    fn_path,
                    exc,
                )

        self._fn = _default_sequence_reward
        self._log.info("RewardWorker: using default_sequence_reward.")

    def compute_rewards(self, batch: DataProto) -> DataProto:
        """Attach ``rewards`` tensor [batch] or [batch, seq] to a new proto."""
        if "input_ids" not in batch.tensors:
            raise KeyError("batch must contain 'input_ids'")

        if self._model is not None:
            input_ids = batch["input_ids"].to(self._device)
            with torch.no_grad():
                hidden = self._model(input_ids)
                if hidden.dim() == 3:
                    scores = hidden[:, -1].mean(dim=-1)
                else:
                    scores = hidden.mean(dim=tuple(range(1, hidden.dim())))
            rewards = scores.float().cpu()
        elif self._fn is not None:
            rewards = self._fn(batch).detach().cpu()
        else:
            raise RuntimeError("init_model() must be called before compute_rewards().")

        out = DataProto(
            tensors={"rewards": rewards, "input_ids": batch["input_ids"]},
            meta=dict(batch.meta),
        )
        return out

    def cleanup(self) -> None:
        self._model = None
        self._fn = None
        super().cleanup()
