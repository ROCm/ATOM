# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""How the layers split across PP stages: ``ATOM_PP_LAYER_PARTITION``.

Every PP stage pages its KV with the one block table stage 0's scheduler owns,
so the pool is as many blocks as the stage with the most KV bytes per token can
hold (`ModelRunner.get_num_blocks` takes the minimum over the stages). Under
the even split the last stage also holds the drafter's KV: MiniMax-M3.1 PP4 x
TP1 with DSpark has 9,720 B/token on stages 0-2 and 15,000 on stage 3, and
that one stage caps the pool.

``kv_balanced`` picks the contiguous split whose largest per-stage KV bytes per
token is smallest instead, under these limits:

* compute: no stage holds more than ``ATOM_PP_PARTITION_MAX_EXTRA_LAYERS``
  layers above the even split's largest stage;
* splits whose largest stage is within ``BYTES_TOLERANCE`` of the best count as
  equally good: the even split if it is one of them (a model without KV
  imbalance keeps its split), else the one with the smallest largest stage,
  then the smallest sum of squared stage sizes, then the fewest bytes.

The split decides which layers a rank builds, so the bytes are estimated from
the config before anything is built (`minimax_m3_stage_kv_costs`); the pool is
still sized from measured memory. The weights each stage holds are not in the
estimate: M3.1's layers all weigh the same, so a stage with two more layers has
~2.7% less room for its KV, within the tolerance. Ported in spirit from
ROCm/ATOM#3594, which also estimates weights and per-request state for Kimi-K3.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("atom")

EVEN = "even"
KV_BALANCED = "kv_balanced"
PARTITION_MODES = (EVEN, KV_BALANCED)

# Splits whose largest stage is within this fraction of the best are equally
# good, and the most compute-balanced of them is taken.
BYTES_TOLERANCE = 0.02


def even_partition(num_layers: int, pp_size: int) -> list[int]:
    """Layers by count; a remainder goes to the stages before the last, the
    last first (it also holds the final norm and lm_head)."""
    partition = [num_layers // pp_size] * pp_size
    for i in range(2, num_layers % pp_size + 2):
        partition[-i] += 1
    return partition


def stage_bytes(
    layer_bytes: Sequence[int], partition: Sequence[int], last_stage_extra: int = 0
) -> list[int]:
    """KV bytes per token of each stage of ``partition``."""
    out, start = [], 0
    for size in partition:
        out.append(sum(layer_bytes[start : start + size]))
        start += size
    out[-1] += last_stage_extra
    return out


@dataclass(frozen=True)
class PartitionPlan:
    """The chosen split and the even one it is measured against."""

    partition: tuple[int, ...]
    stage_bytes: tuple[int, ...]
    even: tuple[int, ...]
    even_stage_bytes: tuple[int, ...]

    @property
    def pool_gain(self) -> float:
        """Blocks under this split over blocks under the even one, for equal
        free memory on every stage."""
        return max(self.even_stage_bytes) / max(self.stage_bytes)

    def describe(self, first_layer: int = 0) -> str:
        """Each stage's layer range and KV bytes per token."""
        parts, start = [], first_layer
        for stage, (size, nbytes) in enumerate(zip(self.partition, self.stage_bytes)):
            parts.append(f"stage {stage}: layers {start}-{start + size - 1} {nbytes} B")
            start += size
        return "; ".join(parts)


def kv_balanced_partition(
    layer_bytes: Sequence[int],
    pp_size: int,
    last_stage_extra: int = 0,
    max_extra_layers: int | None = 1,
    tolerance: float = BYTES_TOLERANCE,
) -> PartitionPlan:
    """The contiguous split of ``layer_bytes`` (KV bytes per token of each
    layer on one rank) over ``pp_size`` stages with the smallest largest stage,
    the last stage also holding ``last_stage_extra`` (the drafter's KV).

    ``max_extra_layers`` caps a stage at the even split's largest plus that
    many layers (None: no cap). Ties within ``tolerance`` go as the module says.
    """
    n = len(layer_bytes)
    if pp_size < 1 or n < pp_size:
        raise ValueError(f"cannot split {n} layers into {pp_size} non-empty stages")
    if any(b < 0 for b in layer_bytes) or last_stage_extra < 0:
        raise ValueError("KV bytes per token cannot be negative")
    even = even_partition(n, pp_size)
    even_bytes = stage_bytes(layer_bytes, even, last_stage_extra)
    cap = n if max_extra_layers is None else max(even) + max(0, max_extra_layers)
    prefix = [0]
    for b in layer_bytes:
        prefix.append(prefix[-1] + b)

    def cost(i: int, j: int, stage: int) -> int:
        return prefix[j] - prefix[i] + (last_stage_extra if stage == pp_size - 1 else 0)

    def feasible_bytes(limit_layers: int) -> int | None:
        # best[s][j]: the smallest largest stage of layers [0, j) in s stages.
        inf = float("inf")
        best = [[inf] * (n + 1) for _ in range(pp_size + 1)]
        best[0][0] = 0
        for s in range(1, pp_size + 1):
            for j in range(s, n + 1):
                for i in range(max(s - 1, j - limit_layers), j):
                    if best[s - 1][i] < inf:
                        value = max(best[s - 1][i], cost(i, j, s - 1))
                        best[s][j] = min(best[s][j], value)
        result = best[pp_size][n]
        return None if result == inf else int(result)

    best_bytes = feasible_bytes(cap)
    if best_bytes is None:
        raise ValueError(
            f"no split of {n} layers into {pp_size} stages of at most {cap} layers"
        )
    limit = best_bytes * (1 + tolerance)
    if max(even_bytes) <= limit:
        return PartitionPlan(
            tuple(even), tuple(even_bytes), tuple(even), tuple(even_bytes)
        )

    # The smallest largest stage that still fits the byte limit.
    def fits(limit_layers: int) -> bool:
        reach = {0}
        for s in range(pp_size):
            reach = {
                j
                for i in reach
                for j in range(i + 1, min(n, i + limit_layers) + 1)
                if cost(i, j, s) <= limit and (s < pp_size - 1 or j == n)
            }
        return n in reach

    largest = next(size for size in range(1, cap + 1) if fits(size))

    # Then the smallest sum of squares, then the fewest bytes: Pareto sets of
    # (sum of squares, largest bytes, partition) per (stages, layers covered).
    states: dict[int, list[tuple[int, int, tuple[int, ...]]]] = {0: [(0, 0, ())]}
    for s in range(pp_size):
        nxt: dict[int, list[tuple[int, int, tuple[int, ...]]]] = {}
        for i, entries in states.items():
            for j in range(i + 1, min(n, i + largest) + 1):
                c = cost(i, j, s)
                if c > limit or (s == pp_size - 1 and j != n):
                    continue
                for sq, mb, parts in entries:
                    nxt.setdefault(j, []).append(
                        (sq + (j - i) ** 2, max(mb, c), parts + (j - i,))
                    )
        states = {j: _pareto(entries) for j, entries in nxt.items()}
    _, _, partition = min(states[n])
    return PartitionPlan(
        partition,
        tuple(stage_bytes(layer_bytes, partition, last_stage_extra)),
        tuple(even),
        tuple(even_bytes),
    )


def _pareto(entries: list[tuple[int, int, tuple[int, ...]]]) -> list:
    """The entries no other beats on both sum of squares and largest bytes."""
    kept, best_mb = [], None
    for sq, mb, parts in sorted(entries):
        if best_mb is None or mb < best_mb:
            kept.append((sq, mb, parts))
            best_mb = mb
    return kept


def minimax_m3_stage_kv_costs(config: Any) -> tuple[list[int], int] | None:
    """MiniMax-M3's KV bytes per token of each target layer on one rank of a
    PP stage, and of the drafter on the last stage; None for another model.

    Mirrors what the pools charge (`MhaKvPool`, `_index_cache_fields`, the
    draft pool) from config reads only: a sparse layer's K/V in NVFP4 under
    ATOM_M3_SPARSE_KV_NVFP4 (an e2m1 payload and an E4M3 scale per 16
    elements, per head and token), else in ``kv_cache_dtype`` with an fp32
    scale per head and token when it is fp8; its index key in
    ``index_cache_dtype`` (FP4: payload and E4M3 scales); a dense layer's K/V
    like a non-NVFP4 sparse one; the drafter's layers with all their KV heads
    on a TP1 producer (`M3DSparkAttention`).
    """
    from atom.config import _is_minimax_m3_config
    from atom.models.minimax_m3.index_schedule import sparse_attention_layer_ids
    from atom.utils import envs

    hf_config = config.hf_config
    if not _is_minimax_m3_config(hf_config):
        return None
    text = getattr(hf_config, "text_config", None) or hf_config
    tp = max(1, int(getattr(config, "tensor_parallel_size", 1) or 1))

    def heads_per_rank(total: int) -> int:
        return max(1, total // tp)

    def kv_bytes(head_dim: int, dtype: str) -> int:
        if dtype == "fp8":
            return 2 * (head_dim + 4)
        if dtype == "fp4":
            return 2 * (head_dim // 2 + head_dim // 16)
        return 2 * 2 * head_dim

    kv_dtype = str(getattr(config, "kv_cache_dtype", "bf16") or "bf16")
    head_dim = int(
        getattr(text, "head_dim", None) or text.hidden_size // text.num_attention_heads
    )
    heads = heads_per_rank(int(text.num_key_value_heads))
    sparse = sparse_attention_layer_ids(text)
    sparse_cfg = getattr(text, "sparse_attention_config", None) or {}
    index_dim = int(sparse_cfg.get("sparse_index_dim", 0) or 0)
    index_dtype = str(getattr(config, "index_cache_dtype", None) or kv_dtype)
    index_bytes = {
        "fp4": index_dim // 2 + index_dim // 16,
        "fp8": index_dim,
    }.get(index_dtype, 2 * index_dim)
    dcp_config = getattr(config, "dcp_config", None)
    if getattr(dcp_config, "index_shard", False):
        # A rank keeps the index of 1/TP of the blocks (`minimax_m3/index_shard.py`),
        # in HBM and, chunk by chunk, in the CPU tier: per token on average.
        index_bytes = -(-index_bytes // tp)
    sparse_kv = (
        kv_bytes(head_dim, "fp4")
        if envs.ATOM_M3_SPARSE_KV_NVFP4
        else kv_bytes(head_dim, kv_dtype)
    )
    layers = [
        (
            heads * sparse_kv + index_bytes
            if i in sparse
            else heads * kv_bytes(head_dim, kv_dtype)
        )
        for i in range(int(text.num_hidden_layers))
    ]
    draft = 0
    spec = getattr(config, "speculative_config", None)
    draft_cfg = getattr(spec, "draft_model_hf_config", None)
    if draft_cfg is not None:
        draft_text = getattr(draft_cfg, "text_config", None) or draft_cfg
        draft_layers = int(getattr(draft_text, "num_hidden_layers", 0) or 0)
        draft_heads = int(
            getattr(draft_text, "num_key_value_heads", None) or text.num_key_value_heads
        )
        draft_dim = int(getattr(draft_text, "head_dim", None) or head_dim)
        draft = (
            draft_layers * heads_per_rank(draft_heads) * kv_bytes(draft_dim, kv_dtype)
        )
    return layers, draft


def resolve_pp_layer_partition(config: Any) -> tuple[int, ...] | None:
    """The split ``ATOM_PP_LAYER_PARTITION`` asks for, or None for the one
    `get_pp_indices` would take anyway (``VLLM_PP_LAYER_PARTITION`` set, the
    even split asked for, no PP, or a model the estimate does not cover)."""
    import os

    from atom.utils import envs

    pp_size = int(getattr(config, "pipeline_parallel_size", 1) or 1)
    mode = envs.ATOM_PP_LAYER_PARTITION
    if pp_size <= 1 or mode == EVEN:
        return None
    if os.getenv("VLLM_PP_LAYER_PARTITION") is not None:
        logger.info("VLLM_PP_LAYER_PARTITION is set; ATOM_PP_LAYER_PARTITION ignored")
        return None
    costs = minimax_m3_stage_kv_costs(config)
    if costs is None:
        logger.warning(
            "ATOM_PP_LAYER_PARTITION=%s estimates MiniMax-M3 only; this model "
            "keeps the even split",
            mode,
        )
        return None
    layers, draft = costs
    plan = kv_balanced_partition(
        layers, pp_size, draft, envs.ATOM_PP_PARTITION_MAX_EXTRA_LAYERS
    )
    logger.info(
        "ATOM_PP_LAYER_PARTITION=kv_balanced: split %s (%s), KV bytes/token per "
        "stage %s against the even split's %s: %.1f%% more blocks for equal "
        "free memory",
        ",".join(map(str, plan.partition)),
        plan.describe(),
        list(plan.stage_bytes),
        list(plan.even_stage_bytes),
        (plan.pool_gain - 1) * 100,
    )
    return plan.partition
