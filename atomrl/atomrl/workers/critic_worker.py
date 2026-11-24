# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Value-function worker for critic-based RL algorithms."""

from __future__ import annotations

import logging
from typing import Any

import torch

from atomrl.core.protocol import DataProto
from atomrl.engine.training.base_engine import BaseEngine, EngineRegistry
from atomrl.workers.base_worker import BaseWorker, get_nested_config

logger = logging.getLogger(__name__)


class CriticWorker(BaseWorker):
    """Computes V(s_t) using an LLM backbone + value head via Engine abstraction."""

    def __init__(
        self, rank: int, world_size: int, config: dict[str, Any] | None = None
    ) -> None:
        super().__init__(rank, world_size, config)
        self._engine: BaseEngine | None = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def init_model(self) -> None:
        """Build critic model (LLM + value head) via EngineRegistry."""
        critic_cfg = get_nested_config(self.config, "critic", default={}) or {}
        policy_cfg = get_nested_config(self.config, "policy", default={}) or {}

        model_name = str(critic_cfg.get("model_name", "")) or str(
            policy_cfg.get("model_name", "")
        )
        backend_raw = str(critic_cfg.get("training_backend", "fsdp2")).lower()

        if backend_raw in ("fsdp", "fsdp2"):
            backend_key = "fsdp2"
        else:
            raise ValueError(f"Unknown critic training_backend: {backend_raw}")

        # Build engine config (similar to actor_worker)
        training_cfg = policy_cfg.get("training", {}) or {}
        engine_config = self._build_engine_config(
            backend_key, training_cfg, critic_cfg, policy_cfg
        )
        optimizer_config = {
            "lr": float(critic_cfg.get("learning_rate", 1e-5)),
            "weight_decay": float(critic_cfg.get("weight_decay", 0.01)),
            "clip_grad": float(critic_cfg.get("max_grad_norm", 1.0)),
            "lr_warmup_steps": int(policy_cfg.get("lr_warmup_steps", 10)),
            "lr_warmup_steps_ratio": float(policy_cfg.get("warmup_ratio", 0.0)),
        }
        model_config = {
            "local_path": model_name,
            "trust_remote_code": True,
            "model_type": "value_model",
        }

        # Try to get value_model engine, fall back to language_model
        try:
            engine_cls = EngineRegistry.get_engine_cls(
                model_type="value_model", backend=backend_key
            )
        except (KeyError, AssertionError):
            engine_cls = EngineRegistry.get_engine_cls(
                model_type="language_model", backend=backend_key
            )

        self._engine = engine_cls(
            model_config=model_config,
            engine_config=engine_config,
            optimizer_config=optimizer_config,
            model_name=model_name,
        )
        self._engine.initialize()
        self._log.info(
            "CriticWorker: initialized %s engine for value model.", backend_key
        )

    def _build_engine_config(
        self,
        backend: str,
        training_cfg: dict,
        critic_cfg: dict,
        policy_cfg: dict,
    ) -> dict[str, Any]:
        """Build engine configuration dict, similar to AtomActorWorker but simpler."""
        if backend in ("fsdp", "fsdp2"):
            fsdp_cfg = training_cfg.get("fsdp_cfg") or {}
            if not isinstance(fsdp_cfg, dict):
                from dataclasses import asdict, is_dataclass

                fsdp_cfg = (
                    asdict(fsdp_cfg) if is_dataclass(fsdp_cfg) else dict(vars(fsdp_cfg))
                )
            return {
                "param_offload": fsdp_cfg.get("param_offload", False),
                "optimizer_offload": fsdp_cfg.get("optimizer_offload", False),
                "grad_offload": fsdp_cfg.get("grad_offload", False),
                "reshard_after_forward": fsdp_cfg.get("reshard_after_forward", True),
                "model_dtype": training_cfg.get("optimizer_dtype", "bf16"),
                "seed": int(policy_cfg.get("seed", 42)),
            }
        return {}

    def compute_values(self, batch: DataProto) -> DataProto:
        """Return per-token value estimates."""
        if self._engine is None:
            raise RuntimeError("init_model() must be called before compute_values().")
        if "input_ids" not in batch.tensors:
            raise KeyError("batch must contain 'input_ids'")

        data: dict[str, Any] = {"input_ids": batch["input_ids"].to(self._device)}
        if "attention_mask" in batch:
            data["attention_mask"] = batch["attention_mask"].to(self._device)

        with self._engine.eval_mode():
            output = self._engine.infer_batch(data)

        # The value model output should have a "values" key or we extract from model_output
        if "model_output" in output and "values" in output["model_output"]:
            values = output["model_output"]["values"]
        elif "values" in output:
            values = output["values"]
        else:
            # Fallback: if the engine returns logits, take the last hidden state projected to scalar
            # This handles the case where we're using a language_model engine
            values = output.get("model_output", {}).get(
                "log_probs",
                torch.zeros(batch["input_ids"].shape[0], batch["input_ids"].shape[1]),
            )

        return DataProto(
            tensors={"values": values.detach().cpu(), "input_ids": batch["input_ids"]},
            meta=dict(batch.meta),
        )

    def train_step(self, batch: DataProto) -> dict[str, float]:
        """Train critic on returns targets using clipped value loss."""
        if self._engine is None:
            raise RuntimeError("init_model() must be called before train_step().")
        if "input_ids" not in batch.tensors:
            raise KeyError("batch must contain 'input_ids'")
        if "returns" not in batch.tensors:
            raise KeyError("critic train_step requires 'returns' tensor")

        from atomrl.algorithms.loss_functions import value_loss

        clip_ratio = float(
            get_nested_config(self.config, "critic", "value_clip_ratio", default=0.2)
            or 0.2
        )

        # Capture batch reference for use inside the closure
        _batch = batch

        def loss_fn(model_output: dict, data: dict) -> tuple:
            values = model_output.get("values") or model_output.get("log_probs")
            if values is None:
                raise RuntimeError("Critic model output has no 'values' key")

            returns = _batch.tensors["returns"].to(values.device)
            old_values = _batch.tensors.get("old_values")
            if old_values is None:
                old_values = _batch.tensors.get("values")
            if old_values is not None:
                old_values = old_values.to(values.device)
            else:
                old_values = values.detach()

            mask = _batch.tensors.get("response_mask") or _batch.tensors.get(
                "attention_mask"
            )
            if mask is not None:
                mask = mask.to(values.device).float()

            # Truncate to match shapes if needed
            min_len = min(values.shape[-1], returns.shape[-1])
            values = values[..., :min_len]
            returns = returns[..., :min_len]
            old_values = old_values[..., :min_len]
            if mask is not None:
                mask = mask[..., :min_len]

            vf_loss = value_loss(values, old_values, returns, clip_ratio, mask=mask)
            return vf_loss, {"critic_loss": float(vf_loss.detach())}

        data: dict[str, Any] = {"input_ids": batch["input_ids"].to(self._device)}
        if "attention_mask" in batch:
            data["attention_mask"] = batch["attention_mask"].to(self._device)

        with self._engine.train_mode():
            output = self._engine.train_batch(data, loss_fn)

        metrics: dict[str, float] = {}
        if "metrics" in output:
            metrics.update(
                {
                    k: float(v) if not isinstance(v, list) else sum(v) / max(len(v), 1)
                    for k, v in output["metrics"].items()
                }
            )
        if "loss" in output:
            loss_val = output["loss"]
            metrics["critic_loss"] = (
                float(loss_val)
                if not isinstance(loss_val, list)
                else sum(loss_val) / max(len(loss_val), 1)
            )

        self._engine.lr_scheduler_step()
        return metrics

    def get_state_dict(self) -> dict[str, torch.Tensor]:
        """Return a CPU copy of the critic model parameters."""
        if self._engine is None:
            raise RuntimeError("init_model() must be called first.")
        params, _ = self._engine.get_per_tensor_param()
        return {k: v.detach().cpu() for k, v in params}

    def cleanup(self) -> None:
        self._engine = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        super().cleanup()
