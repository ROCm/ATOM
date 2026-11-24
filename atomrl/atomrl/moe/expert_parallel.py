# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Expert-parallel sharding helpers for MoE checkpoints."""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from typing import Any

import torch
from torch import Tensor

logger = logging.getLogger(__name__)

_EXPERT_KEY = re.compile(r"(.*\.experts\.)(\d+)(\..+)$")


class ExpertParallelManager:
    """Tracks EP layout metadata and performs conservative checkpoint transforms."""

    def __init__(self, config: Any) -> None:
        self._config = config

    def setup_ep(self, model: Any, ep_size: int) -> None:
        """Record expert-parallel width on the model for downstream kernels."""
        if not isinstance(ep_size, int) or ep_size < 1:
            raise ValueError(f"ep_size must be a positive int, got {ep_size!r}")
        model._atomrl_expert_parallel_size = ep_size
        megatron_ep = getattr(self._config, "expert_parallel_size", None)
        if megatron_ep is not None and int(megatron_ep) != int(ep_size):
            logger.warning(
                "ExpertParallelManager: config expert_parallel_size=%s differs from setup ep_size=%s.",
                megatron_ep,
                ep_size,
            )
        logger.info("Expert parallel size set to %d.", ep_size)

    def reshard_for_inference(
        self,
        state_dict: dict[str, Tensor],
        train_ep_size: int,
        infer_ep_size: int,
    ) -> dict[str, Tensor]:
        """Adjust a training ``state_dict`` for a different inference EP width.

        When widths match, tensors are copied unchanged. When
        ``train_ep_size > infer_ep_size`` and the ratio divides the number of
        per-parameter expert shards in keys matching ``*.experts.<id>.*``,
        consecutive expert tensors are concatenated along dimension ``0``.
        Otherwise tensors are returned unchanged with a warning.
        """
        if train_ep_size < 1 or infer_ep_size < 1:
            raise ValueError("EP sizes must be positive.")

        out: dict[str, Tensor] = {}
        for key, tensor in state_dict.items():
            if not _EXPERT_KEY.match(key):
                out[key] = tensor

        if train_ep_size == infer_ep_size:
            out.update({k: v for k, v in state_dict.items() if _EXPERT_KEY.match(k)})
            return out

        if train_ep_size < infer_ep_size:
            logger.warning(
                "reshard_for_inference: train_ep_size=%s < infer_ep_size=%s is not implemented; "
                "expert tensors are copied unchanged.",
                train_ep_size,
                infer_ep_size,
            )
            out.update({k: v for k, v in state_dict.items() if _EXPERT_KEY.match(k)})
            return out

        if train_ep_size % infer_ep_size != 0:
            logger.warning(
                "reshard_for_inference: EP ratio %s/%s is not integral; expert tensors unchanged.",
                train_ep_size,
                infer_ep_size,
            )
            out.update({k: v for k, v in state_dict.items() if _EXPERT_KEY.match(k)})
            return out

        merge_factor = train_ep_size // infer_ep_size
        groups: dict[tuple[str, str], dict[int, Tensor]] = defaultdict(dict)
        for key, tensor in state_dict.items():
            m = _EXPERT_KEY.match(key)
            if not m:
                continue
            pfx, eid, sfx = m.group(1), int(m.group(2)), m.group(3)
            groups[(pfx, sfx)][eid] = tensor

        if not groups:
            logger.warning(
                "reshard_for_inference: no '*.experts.<id>.*' parameters found; expert tensors omitted."
            )
            return out

        for (pfx, sfx), experts in groups.items():
            ids = sorted(experts)
            tensors = [experts[i] for i in ids]
            if len(tensors) % merge_factor != 0:
                logger.warning(
                    "reshard_for_inference: %d shards for %s... not divisible by merge_factor=%d.",
                    len(tensors),
                    pfx,
                    merge_factor,
                )
                for tid in ids:
                    out[f"{pfx}{tid}{sfx}"] = experts[tid]
                continue
            new_idx = 0
            for start in range(0, len(tensors), merge_factor):
                chunk = tensors[start : start + merge_factor]
                try:
                    cat = torch.cat(chunk, dim=0)
                except RuntimeError as exc:
                    logger.warning(
                        "reshard_for_inference: concat failed for %s...: %s", pfx, exc
                    )
                    for j, tid in enumerate(ids[start : start + merge_factor]):
                        out[f"{pfx}{tid}{sfx}"] = chunk[j]
                    continue
                out[f"{pfx}{new_idx}{sfx}"] = cat
                new_idx += 1

        logger.info(
            "reshard_for_inference: merged expert shards with factor=%d (train_ep=%s infer_ep=%s).",
            merge_factor,
            train_ep_size,
            infer_ep_size,
        )
        return out
