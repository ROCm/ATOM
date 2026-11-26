# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Frozen reference policy for KL and baseline log-prob targets."""

from __future__ import annotations

import logging
from typing import Any

import torch

from atomrl.core.protocol import DataProto
from atomrl.engine.training.fsdp_backend import FSDP2Backend
from atomrl.workers.base_worker import BaseWorker, get_nested_config

logger = logging.getLogger(__name__)


class RefPolicyWorker(BaseWorker):
    """Reference model in eval mode for stable log-probability estimates."""

    def __init__(
        self, rank: int, world_size: int, config: dict[str, Any] | None = None
    ) -> None:
        super().__init__(rank, world_size, config)
        self._model: torch.nn.Module | None = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def init_model(self) -> None:
        """Load a frozen reference policy (same architecture path as actor)."""
        policy = get_nested_config(self.config, "policy", default={}) or {}
        training_cfg = policy.get("training", {}) or {}
        self._model = FSDP2Backend.build_model(
            str(policy.get("model_name", "")), training_cfg
        )
        self._model = FSDP2Backend.apply_lumen_optimizations(
            self._model,
            get_nested_config(self.config, "quantization", "training", default={})
            or {},
        )
        self._model.to(self._device)
        for p in self._model.parameters():
            p.requires_grad_(False)
        self._model.eval()
        self._log.info("RefPolicyWorker: frozen reference ready.")

    def compute_log_probs(self, batch: DataProto) -> DataProto:
        """Return detached reference log-probs."""
        if self._model is None:
            raise RuntimeError(
                "init_model() must be called before compute_log_probs()."
            )
        if "input_ids" not in batch.tensors:
            raise KeyError("batch must contain 'input_ids'")

        input_ids = batch["input_ids"].to(self._device)
        with torch.no_grad():
            logits = self._model(input_ids)
            log_probs = torch.log_softmax(logits[:, :-1], dim=-1)
            targets = input_ids[:, 1:]
            token_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

        return DataProto(
            tensors={
                "ref_log_probs": token_log_probs.cpu(),
                "input_ids": batch["input_ids"],
            },
            meta=dict(batch.meta),
        )

    def cleanup(self) -> None:
        self._model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        super().cleanup()
