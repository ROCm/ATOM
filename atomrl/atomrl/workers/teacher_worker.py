# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Teacher model worker for On-Policy Distillation and Speculative Distillation.

Supports two output modes:

* ``"logits"``: Returns full-vocabulary logits ``[B, T, V]`` for OPD.
* ``"hidden"``: Returns last-layer hidden states ``[B, T, D]`` for Spec Distill.

The teacher model is always frozen (eval mode, no gradients).
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from atomrl.core.protocol import DataProto
from atomrl.engine.training.fsdp_backend import FSDP2Backend
from atomrl.workers.base_worker import BaseWorker, get_nested_config

logger = logging.getLogger(__name__)


class TeacherWorker(BaseWorker):
    """Frozen teacher model that produces logits or hidden states on demand."""

    def __init__(
        self,
        rank: int,
        world_size: int,
        config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(rank, world_size, config)
        self._model: torch.nn.Module | None = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def init_model(self) -> None:
        """Load the teacher model in frozen eval mode."""
        teacher_cfg = (
            get_nested_config(self.config, "algorithm", "teacher", default={}) or {}
        )
        model_name = teacher_cfg.get("model_name", "")
        if not model_name:
            policy = get_nested_config(self.config, "policy", default={}) or {}
            model_name = str(policy.get("model_name", ""))

        training_cfg = (
            get_nested_config(self.config, "policy", "training", default={}) or {}
        )
        self._model = FSDP2Backend.build_model(model_name, training_cfg)
        self._model = FSDP2Backend.apply_lumen_optimizations(
            self._model,
            get_nested_config(self.config, "quantization", "training", default={})
            or {},
        )
        self._model.to(self._device)
        for p in self._model.parameters():
            p.requires_grad_(False)
        self._model.eval()
        self._log.info("TeacherWorker: frozen teacher model ready (%s).", model_name)

    def compute_teacher_outputs(
        self,
        batch: DataProto,
        return_mode: str = "logits",
    ) -> DataProto:
        """Run teacher forward pass and return outputs.

        Args:
            batch: Must contain ``input_ids`` and optionally ``attention_mask``.
            return_mode: ``"logits"`` for full-vocab logits ``[B, T, V]``,
                ``"hidden"`` for last-layer hidden states ``[B, T, D]``,
                ``"both"`` for both.

        Returns:
            DataProto with the requested teacher outputs.
        """
        if self._model is None:
            raise RuntimeError("init_model() must be called first.")
        if "input_ids" not in batch.tensors:
            raise KeyError("batch must contain 'input_ids'")

        input_ids = batch["input_ids"].to(self._device)
        attention_mask: torch.Tensor | None = None
        if "attention_mask" in batch.tensors:
            attention_mask = batch["attention_mask"].to(self._device)

        result_tensors: dict[str, torch.Tensor] = {
            "input_ids": batch["input_ids"],
        }

        with torch.no_grad():
            outputs = self._model(
                input_ids,
                attention_mask=attention_mask,
                output_hidden_states=(return_mode in ("hidden", "both")),
            )

            if return_mode in ("logits", "both"):
                # Shift to align: logits[:, :-1] predicts input_ids[:, 1:]
                result_tensors["teacher_logits"] = outputs.logits[:, :-1].cpu()

            if return_mode in ("hidden", "both"):
                if (
                    hasattr(outputs, "hidden_states")
                    and outputs.hidden_states is not None
                ):
                    result_tensors["teacher_hidden_states"] = outputs.hidden_states[-1][
                        :, :-1
                    ].cpu()
                else:
                    # Fallback: use the output before lm_head if available
                    last_hidden = getattr(outputs, "last_hidden_state", None)
                    if last_hidden is None:
                        raise RuntimeError(
                            "Model did not return hidden_states. "
                            "Ensure output_hidden_states=True is supported."
                        )
                    result_tensors["teacher_hidden_states"] = last_hidden[:, :-1].cpu()

        if "attention_mask" in batch.tensors:
            result_tensors["attention_mask"] = batch["attention_mask"]

        return DataProto(
            tensors=result_tensors,
            meta=dict(batch.meta),
        )

    def get_lm_head_state(self) -> dict[str, torch.Tensor]:
        """Extract the teacher's lm_head weight for lazy logits reconstruction."""
        if self._model is None:
            raise RuntimeError("init_model() must be called first.")

        state: dict[str, torch.Tensor] = {}
        for name, param in self._model.named_parameters():
            if "lm_head" in name:
                state[name] = param.detach().cpu()
        return state

    def cleanup(self) -> None:
        self._model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        super().cleanup()
