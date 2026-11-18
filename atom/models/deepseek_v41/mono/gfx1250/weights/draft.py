# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The step kernel's draft-side and step-tail weights (``buffers``: WEIGHT), from the loaded target and DSpark models.

Each comes out in the layout the kernel reads:
- ``emb``: the full bf16 embedding ``[V][H]`` on every rank (the block state's anchor rows; the target's table is
  vocab-sharded, so it is gathered once), ``noise``: the noise token's row;
- ``hw``: this rank's LM head shard ``[V/tp][H]`` in the head kernel's tile order ``[V/16][H/32][2][16][16]``;
- ``mw1``: the Markov head's ``W1`` ``[V][R]`` whole, ``mw2``: this rank's ``W2`` rows ``[V/tp][R]``;
- ``mpw`` / ``mps`` / ``mnw``: stage 0's main_proj (FP8, checkpoint row order) / its 32 x 32 block scales / main_norm;
- ``wkv{j}`` / ``wks{j}`` / ``knw{j}``: stage j's KV projection rows of ``wqkv_a`` (FP8) / scales / kv_norm;
- ``final_nw`` / ``dfinal_nw``: the target's and the draft's final norm (the last layers' fronts' ``anw``).
"""

import torch
import torch.distributed as dist

from atom.models.deepseek_v41.mono.gfx1250.weights.planes import (
    checkpoint_weight,
)
from atom.mono.runtime.consensus import MonoUnsupported


def _gather_rows(shard: torch.Tensor, group) -> torch.Tensor:
    """A row-sharded tensor whole on every rank (the TP group's shards in rank order)."""
    parts = [torch.empty_like(shard) for _ in range(dist.get_world_size(group))]
    dist.all_gather(parts, shard.contiguous(), group=group)
    return torch.cat(parts)


def _tile_head(w: torch.Tensor) -> torch.Tensor:
    """``[V/tp][H]`` bf16 -> the head kernel's ``[V/16][H/32][2][16][16]`` (head: a 16-row vocab tile, its K in
    32-col groups of two 16-col halves, each 16 rows x 16 cols)."""
    v, h = w.shape
    return w.view(v // 16, 16, h // 32, 2, 16).permute(0, 2, 3, 1, 4).contiguous()


def _kv_rows(attn, q_rank: int, head_dim: int):
    """Stage ``attn``'s KV projection: rows ``q_rank .. q_rank + head_dim`` of ``wqkv_a`` and their block scales."""
    w = checkpoint_weight(attn.wqkv_a)
    s = attn.wqkv_a.weight_scale.data.view(torch.uint8)
    if w.shape[0] != q_rank + head_dim or q_rank % 32:
        raise MonoUnsupported(
            f"wqkv_a rows {w.shape[0]} != q_rank {q_rank} + head_dim {head_dim}"
        )
    return w[q_rank:].contiguous(), s[q_rank // 32 :].contiguous()


def bind(target_base, draft, tp_group) -> dict[str, torch.Tensor]:
    """The tensors by kernel name. ``target_base``: the V4.1 base model the draft shares its embedding and head with
    (``share_with_target``; its ``embed``, ``head``, ``norm``);
    ``draft``: the ``DeepseekV41DSpark`` (``mtp`` stages); ``tp_group``: the TP process group.
    """
    cfg = draft.config
    out = {}
    emb = target_base.embed.weight.data
    out["emb"] = emb if emb.shape[0] == cfg.vocab_size else _gather_rows(emb, tp_group)
    out["noise"] = out["emb"][cfg.dspark_noise_token_id].contiguous()
    out["hw"] = _tile_head(target_base.head.weight.data)
    markov = draft.mtp[-1].markov_head
    w1, w2 = markov.markov_w1.weight.data, markov.markov_w2.weight.data
    out["mw1"] = w1 if w1.shape[0] == cfg.vocab_size else _gather_rows(w1, tp_group)
    vr = out["hw"].shape[0] * 16
    rank = dist.get_rank(tp_group)
    out["mw2"] = (
        w2[rank * vr : (rank + 1) * vr].contiguous()
        if w2.shape[0] == cfg.vocab_size
        else w2
    )
    first = draft.mtp[0]
    out["mpw"] = checkpoint_weight(first.main_proj)
    out["mps"] = first.main_proj.weight_scale.data.view(torch.uint8)
    out["mnw"] = first.main_norm.weight.data
    for j, stage in enumerate(draft.mtp):
        out[f"wkv{j}"], out[f"wks{j}"] = _kv_rows(
            stage.attn, cfg.q_lora_rank, cfg.head_dim
        )
        out[f"knw{j}"] = stage.attn.kv_norm.weight.data
    out["final_nw"] = target_base.norm.weight.data
    out["dfinal_nw"] = draft.mtp[-1].norm.weight.data
    return out
