# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Rollout worker backed by the ATOM inference engine."""

from __future__ import annotations

import logging
from typing import Any

import torch

from atomrl.core.config import AtomConfig, R3Config
from atomrl.core.protocol import DataProto
from atomrl.engine.inference.atom_engine import AtomEngine
from atomrl.engine.inference.kv_cache import FP8KVCacheManager
from atomrl.moe.r3_manager import R3Manager
from atomrl.moe.r3_scope import assert_supported_r3
from atomrl.workers.base_worker import BaseWorker, get_nested_config

logger = logging.getLogger(__name__)


class AtomRolloutWorker(BaseWorker):
    """Generates sequences with optional router logits for MoE / R3."""

    def __init__(
        self, rank: int, world_size: int, config: dict[str, Any] | None = None
    ) -> None:
        super().__init__(rank, world_size, config)
        self._engine: AtomEngine | None = None
        self._kv_manager: FP8KVCacheManager | None = None
        self._r3: R3Manager | None = None
        self._record_r3: bool = False

    def init_model(self) -> None:
        """Construct :class:`~atomrl.engine.inference.atom_engine.AtomEngine`."""
        policy = get_nested_config(self.config, "policy", default={}) or {}
        model_name = str(policy.get("model_name", "stub-model"))
        gen = policy.get("generation", {}) or {}
        atom_dict = gen.get("atom_cfg") or policy.get("atom_cfg") or {}
        atom_cfg = AtomConfig(
            tensor_parallel_size=int(atom_dict.get("tensor_parallel_size", 1)),
            kv_cache_dtype=str(atom_dict.get("kv_cache_dtype", "auto")),
            max_model_len=atom_dict.get("max_model_len"),
        )
        self._engine = AtomEngine(config=atom_cfg, model_name=model_name)
        self._engine.init()
        quant = (
            get_nested_config(self.config, "quantization", "rollout", default={}) or {}
        )
        fp8_kv = str(quant.get("precision", "")).lower() in ("fp8", "float8")
        self._kv_manager = FP8KVCacheManager(enabled=fp8_kv)

        moe = get_nested_config(self.config, "moe", "r3", default={}) or {}
        r3_cfg = R3Config(
            enabled=bool(moe.get("enabled", False)),
            record_router_logits=bool(moe.get("record_router_logits", True)),
            replay_mode=str(moe.get("replay_mode", "hard_assignment")),
        )
        if r3_cfg.enabled:
            assert_supported_r3(
                replay_mode=r3_cfg.replay_mode,
                generation_backend=str(policy.get("generation_backend", "") or ""),
            )
        self._r3 = R3Manager(r3_cfg)
        self._log.info("AtomRolloutWorker: engine ready for %s", model_name)

    def prepare_r3_recording(self) -> None:
        """Enable MoE R3 router capture for the next rollout."""
        self._record_r3 = True

    def clear_r3_recording(self) -> None:
        self._record_r3 = False

    def prepare_for_generation(self) -> None:
        """Recalibrate caches / set sampling defaults before a rollout burst."""
        if self._kv_manager is not None:
            self._kv_manager.recalibrate_scales()
        self._log.info("AtomRolloutWorker.prepare_for_generation")

    def generate(self, batch: DataProto) -> DataProto:
        """Batch decode prompts into ``sequences`` (and optional router tensors).

        ATOM's public rollout contract currently returns tokens only, so
        router distributions are not available on this worker.
        """
        if self._engine is None:
            raise RuntimeError("init_model() must be called before generate().")
        if "input_ids" not in batch.tensors:
            raise KeyError("batch must contain 'input_ids'")

        prompts = [row.tolist() for row in batch["input_ids"].cpu()]
        sampling = dict(batch.meta.get("sampling_params", {}))
        sampling.setdefault("max_new_tokens", 32)
        sampling.setdefault("temperature", 1.0)

        # ATOM's public rollout API returns token results only; it does not
        # expose its worker model for router-hook installation. Do not reach
        # into the subprocess with a private attribute probe.
        if self._record_r3 and self._r3 is not None:
            logger.warning(
                "ATOM public rollout does not expose router recording; "
                "continuing without R3 router distributions"
            )
        token_lists = self._engine.generate(prompts, sampling_params=sampling)
        recorded = {}

        max_len = max(len(seq) for seq in token_lists)
        pad_id = int(batch.meta.get("pad_token_id", 0))
        out = torch.full((len(token_lists), max_len), pad_id, dtype=torch.long)
        for i, seq in enumerate(token_lists):
            out[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)

        meta_out = {**batch.meta, "sampling_params": sampling}
        out_proto = DataProto(tensors={"sequences": out}, meta=meta_out)

        if recorded:
            out_proto = R3Manager.transfer_distributions(out_proto, recorded)
            self._log.info("R3: captured %d layer distributions", len(recorded))

        self._record_r3 = False
        return out_proto

    def update_weights(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Hot-swap rollout weights from training workers."""
        if self._engine is None:
            raise RuntimeError("init_model() must be called before update_weights().")
        self._engine.update_weights(state_dict)
        self._log.info("AtomRolloutWorker.update_weights: %d tensors", len(state_dict))

    def finish_generation(self) -> None:
        """End of rollout phase bookkeeping (does not destroy the engine)."""
        self._log.info("AtomRolloutWorker.finish_generation")

    def cleanup(self) -> None:
        if self._r3 is not None:
            self._r3.clear()
        if self._engine is not None:
            self._engine.shutdown()
        self._engine = None
        self._kv_manager = None
        self._r3 = None
        super().cleanup()
