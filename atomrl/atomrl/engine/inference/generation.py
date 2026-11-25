# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Generation facade over :class:`~atomrl.engine.inference.atom_engine.AtomEngine`."""

from __future__ import annotations

import logging
from typing import Any

import torch

from atomrl.core.config import AtomConfig, AtomRLConfig
from atomrl.core.protocol import DataProto
from atomrl.engine.inference.atom_engine import AtomEngine

logger = logging.getLogger(__name__)


class GenerationInterface:
    """Prepare / generate / finish lifecycle for rollout DataProto batches.

    Amortizes setup costs in ``prepare_for_generation``, streams ``generate``
    calls, then tears down in ``finish_generation``.
    """

    def __init__(self) -> None:
        self._engine: AtomEngine | None = None
        self._sampling_defaults: dict[str, Any] = {}

    def prepare_for_generation(self, config: AtomRLConfig | dict[str, Any]) -> None:
        """Construct (or refresh) the underlying :class:`AtomEngine` from policy config."""
        if isinstance(config, dict):
            policy = config.get("policy", {})
            model_name = str(policy.get("model_name", "stub-model"))
            atom_dict = (
                (policy.get("generation") or {}).get("atom_cfg")
                or policy.get("atom_cfg")
                or {}
            )
            atom_cfg = AtomConfig(
                tensor_parallel_size=int(atom_dict.get("tensor_parallel_size", 1)),
                kv_cache_dtype=str(atom_dict.get("kv_cache_dtype", "auto")),
                max_model_len=atom_dict.get("max_model_len"),
            )
        else:
            model_name = config.policy.model_name or "stub-model"
            atom_cfg = config.policy.generation.atom_cfg

        self._engine = AtomEngine(config=atom_cfg, model_name=model_name)
        self._engine.init()
        self._sampling_defaults = {"max_new_tokens": 64, "temperature": 1.0}
        logger.info(
            "GenerationInterface: prepared engine for model_name=%s", model_name
        )

    def generate(self, batch: DataProto) -> DataProto:
        """Run batch generation; returns sequences and optional router distributions."""
        if self._engine is None:
            raise RuntimeError("Call prepare_for_generation() before generate().")

        if "input_ids" not in batch.tensors:
            raise KeyError("DataProto must contain 'input_ids' for generation.")

        input_ids: torch.Tensor = batch["input_ids"]
        prompts = [row.tolist() for row in input_ids.cpu()]
        meta_sampling = dict(batch.meta.get("sampling_params", {}))
        sampling = {**self._sampling_defaults, **meta_sampling}
        token_lists = self._engine.generate(prompts, sampling_params=sampling)

        max_len = max(len(seq) for seq in token_lists)
        pad_id = int(batch.meta.get("pad_token_id", 0))
        out = torch.full((len(token_lists), max_len), pad_id, dtype=torch.long)
        for i, seq in enumerate(token_lists):
            out[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)

        out_proto = DataProto(
            tensors={"sequences": out},
            meta={**batch.meta, "sampling_params": sampling},
        )

        if batch.meta.get("return_router_dists", False):
            # Placeholder router logits for MoE / R3 pipelines when engine does not supply them
            layers = int(batch.meta.get("num_router_layers", 1))
            batch_size = out.shape[0]
            for layer_idx in range(layers):
                logits = torch.randn(batch_size, 8)
                out_proto.add_router_distributions(layer_idx, logits)
        return out_proto

    def finish_generation(self) -> None:
        """Tear down engine resources after a rollout phase."""
        if self._engine is not None:
            self._engine.shutdown()
            self._engine = None
        logger.info("GenerationInterface.finish_generation: engine released.")
