# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Weight loading for diffusion model components."""

import json
import logging
import os

import torch

logger = logging.getLogger(__name__)

_INDEX_NAME = "model.safetensors.index.json"


def _shard_files(path: str) -> dict[str, str]:
    """Map tensor name -> shard filename, for sharded or single-file dirs."""
    index_path = os.path.join(path, _INDEX_NAME)
    if os.path.exists(index_path):
        with open(index_path) as f:
            return json.load(f)["weight_map"]

    single = os.path.join(path, "model.safetensors")
    if not os.path.exists(single):
        raise FileNotFoundError(
            f"no {_INDEX_NAME} and no model.safetensors under {path}"
        )
    from safetensors import safe_open

    with safe_open(single, framework="pt") as f:
        return dict.fromkeys(f.keys(), "model.safetensors")


def load_minimax_h3_dit_weights(
    model: torch.nn.Module,
    path: str,
    *,
    device: torch.device | str = "cpu",
    strict: bool = True,
) -> int:
    """Load a DiT from safetensors shards, applying the QKV reorder.

    The checkpoint stores QKV **interleaved per query group**, not as [Q;K;V]:
    ``num_query_groups`` blocks of ``(heads_per_group + 2) * head_dim`` rows. A
    plain three-way split of the fused tensor is silently wrong.

    Fails loudly on any missing or unexpected tensor -- a partially loaded DiT
    produces plausible noise rather than an error.
    """
    from safetensors import safe_open

    from atom.diffusion.models.minimax_h3.dit import reorder_grouped_qkv_to_qkv

    arch = model.arch
    weight_map = _shard_files(path)
    own = dict(model.state_dict())

    missing = sorted(set(own) - set(weight_map))
    unexpected = sorted(set(weight_map) - set(own))
    if strict and (missing or unexpected):
        raise RuntimeError(
            f"checkpoint/module mismatch: {len(missing)} missing "
            f"(e.g. {missing[:3]}), {len(unexpected)} unexpected "
            f"(e.g. {unexpected[:3]})"
        )

    # Group by shard so each file opens once.
    by_shard: dict[str, list[str]] = {}
    for name, shard in weight_map.items():
        if name in own:
            by_shard.setdefault(shard, []).append(name)

    loaded = 0
    for shard, names in sorted(by_shard.items()):
        with safe_open(os.path.join(path, shard), framework="pt") as f:
            for name in names:
                tensor = f.get_tensor(name)
                if name.endswith("attn.qkv_proj.weight"):
                    tensor = reorder_grouped_qkv_to_qkv(
                        tensor,
                        num_query_groups=arch.num_attention_heads,
                        heads_per_group=1,
                        head_dim=arch.attention_head_dim,
                    )
                target = own[name]
                if tuple(tensor.shape) != tuple(target.shape):
                    raise ValueError(
                        f"{name}: checkpoint shape {tuple(tensor.shape)} != "
                        f"module shape {tuple(target.shape)}"
                    )
                target.data.copy_(
                    tensor.to(device=device, dtype=target.dtype), non_blocking=False
                )
                loaded += 1
        logger.debug("loaded %d tensors from %s", len(names), shard)

    logger.info("loaded %d tensors from %s", loaded, path)
    return loaded
