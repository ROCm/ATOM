# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""A pipeline-parallel layer split that balances the KV pool across stages.

Every stage pages its KV with one shared block table, so the pool has as many
pages as the stage that fits the fewest::

    pool    = min over stages s of pages_s
    pages_s = free_s // page_s
    page_s  = block_size * (per-token KV bytes of the layers on s, plus the
              drafter's on the last stage)
    free_s  = stage budget - weights on s - per-request state on s

The even split (`even_partition`) balances layer counts. On a hybrid model
with a drafter that leaves the last stage holding the most KV per page while
the others leave memory unused: Kimi-K3 at PP8 (24 MLA layers among 93, the
DSpark drafter's 5 MLA layers on the last stage) caps the pool at ~34M tokens.

``ATOM_PP_LAYER_PARTITION=kv_balanced`` picks the contiguous split that
maximises ``pool`` instead, under these limits:

* compute: no stage holds more than ``ATOM_PP_PARTITION_MAX_EXTRA_LAYERS``
  layers above the even split's largest stage;
* every stage holds a layer of each cache kind the model has: a
  linear-attention layer, so it has the recurrent state a PAGE-backed state
  checkpoint images (the runner refuses a stage without one), and a
  per-token-KV layer, whose page is the unit those images are stored in;
* splits whose pool is within ``POOL_TOLERANCE`` of the best count as equally
  good. The even split wins if it is one of them, so a model without KV
  imbalance keeps the split it had; otherwise the one with the smallest
  largest stage, then the smallest sum of squared stage sizes, then the
  largest pool.

The split decides which layers a rank builds, so it is chosen from the config
before anything is built or measured, and every term above is an estimate:
weights from the checkpoint's shard headers, per-token KV and per-request
state from the HF config, the budget from the GPU's memory less a fixed
reserve for what `ModelRunner.get_num_blocks` measures at runtime. The
estimate only picks the split; the pool is still sized from measured memory.
"""

from __future__ import annotations

import collections
import glob
import logging
import math
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass

logger = logging.getLogger("atom")

EVEN = "even"
KV_BALANCED = "kv_balanced"
PARTITION_MODES = (EVEN, KV_BALANCED)

# Splits whose predicted pool is within this fraction of the best are treated
# as equally good, and the most compute-balanced of them is taken. The
# prediction is about this accurate: on Kimi-K3 PP8 it gives 33.8M tokens for
# the even split and 63.8M for 13,13,13,13,13,12,12,4, where 34.4M and 62.4M
# were measured.
POOL_TOLERANCE = 0.02

# What a stage's budget (`gpu_memory_utilization` of the GPU) loses to what
# `get_num_blocks` measures only at runtime, as a fraction of GPU memory: the
# activation peak, memory held outside the PyTorch allocator (RCCL buffers),
# CUDA graphs and its 2% safety margin. Kimi-K3 PP8 on 432 GiB MI455X GPUs
# measured 12 GiB, 17-26 GiB, 0 (eager) and 8.6 GiB: 9-11%.
RUNTIME_RESERVE_FRACTION = 0.10

_KV_DTYPE_BITS = {"bf16": 16, "fp16": 16, "auto": 16, "fp8": 8, "fp4": 4, "nvfp4": 4.5}
_STATE_DTYPE_BYTES = {"fp32": 4, "fp16": 2, "bf16": 2}
# A checkpoint tensor of layer <i>: the target's own unless its prefix names
# another stack (an encoder tower, a separately named MTP stack).
_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")
_NOT_TARGET = ("vision", "visual", "audio", "mm_projector", "multi_modal", "mtp")


@dataclass(frozen=True)
class LayerCost:
    """What one target layer costs the stage that holds it, on one TP rank."""

    weight_bytes: int
    # Paged cache per token: a full-attention layer's KV. 0 for linear attention.
    kv_bytes_per_token: int = 0
    # Recurrent state of one in-flight request, all its slots: a linear-
    # attention layer's. 0 for a layer without one.
    state_bytes_per_request: int = 0


@dataclass(frozen=True)
class PartitionProblem:
    """Everything the split is chosen from."""

    layers: tuple[LayerCost, ...]
    pp_size: int
    # Memory a stage has for its weights, its per-request state and its pages.
    stage_bytes: int
    block_size: int
    max_num_seqs: int
    # Weights outside the layer stack: the embedding (first stage); the final
    # norm, lm_head and the drafter (last stage).
    first_stage_weight_bytes: int = 0
    last_stage_weight_bytes: int = 0
    # The drafter's per-token KV, paged beside the last stage's own.
    last_stage_kv_bytes_per_token: int = 0
    max_extra_layers: int = 1


@dataclass(frozen=True)
class PartitionPlan:
    """The chosen split, and the even one it is measured against."""

    partition: tuple[int, ...]
    pages: tuple[float, ...]
    even: tuple[int, ...]
    even_pages: tuple[float, ...]
    max_layers: int
    block_size: int

    @property
    def pool(self) -> float:
        return min(self.pages)

    @property
    def even_pool(self) -> float:
        return min(self.even_pages)

    def describe(self) -> str:
        def count(pages: float) -> str:
            return "inf" if math.isinf(pages) else str(int(pages))

        def stages(partition: Sequence[int], pages: Sequence[float]) -> str:
            low = min(pages)
            tokens = (
                ""
                if math.isinf(low)
                else f" = {low * self.block_size / 1e6:.1f}M tokens"
            )
            return (
                f"[{','.join(map(str, partition))}]: predicted pages per stage "
                f"[{','.join(map(count, pages))}], pool {count(low)} pages{tokens}"
            )

        text = (
            f"{stages(self.partition, self.pages)}; largest stage "
            f"{max(self.partition)} layers (limit {self.max_layers})"
        )
        if self.partition == self.even:
            return text
        gain = ""
        if 0 < self.even_pool < math.inf:
            gain = f" ({self.pool / self.even_pool - 1:+.0%})"
        return f"{text}; the even split {stages(self.even, self.even_pages)}{gain}"


class PartitionUnavailable(Exception):
    """The config does not say enough to estimate what a stage holds."""


def even_partition(num_layers: int, pp_size: int) -> list[int]:
    """Stage sizes of the default split: layers evenly distributed.

    If the number of layers is not divisible by the number of partitions,
    the remaining layers are evenly distributed across all but the last
    partition. The last partition is excluded because it often contains an
    additional norm layer and we are attempting to balance compute.

    If `pp_size > 2` and the number of remaining layers is
    `0 < x <= pp_size - 2` then the remaining layers are evenly distributed
    across the middle partitions. The first and last partitions are excluded
    because they contain the input and output embeddings respectively and we
    are attempting to reduce maximum memory consumption across partitions.
    """
    partitions = [num_layers // pp_size] * pp_size
    for i in range(2, num_layers % pp_size + 2):
        partitions[-i] += 1
    return partitions


class _Stages:
    """Prefix sums over the layer costs, so a stage's pages cost O(1)."""

    def __init__(self, problem: PartitionProblem):
        self.problem = problem
        self.weights, self.kv, self.state = [0], [0], [0]
        self.kv_layers, self.state_layers = [0], [0]
        for layer in problem.layers:
            self.weights.append(self.weights[-1] + layer.weight_bytes)
            self.kv.append(self.kv[-1] + layer.kv_bytes_per_token)
            self.state.append(self.state[-1] + layer.state_bytes_per_request)
            self.kv_layers.append(self.kv_layers[-1] + (layer.kv_bytes_per_token > 0))
            self.state_layers.append(
                self.state_layers[-1] + (layer.state_bytes_per_request > 0)
            )
        self.needs_kv = self.kv_layers[-1] > 0
        self.needs_state = self.state_layers[-1] > 0

    def holds_every_kind(self, lo: int, hi: int) -> bool:
        return (not self.needs_kv or self.kv_layers[hi] > self.kv_layers[lo]) and (
            not self.needs_state or self.state_layers[hi] > self.state_layers[lo]
        )

    def pages(self, stage: int, lo: int, hi: int) -> float:
        """Pages stage ``stage`` fits holding layers [lo, hi); inf without KV."""
        p = self.problem
        weights = self.weights[hi] - self.weights[lo]
        kv = self.kv[hi] - self.kv[lo]
        if stage == 0:
            weights += p.first_stage_weight_bytes
        if stage == p.pp_size - 1:
            weights += p.last_stage_weight_bytes
            kv += p.last_stage_kv_bytes_per_token
        if kv <= 0:
            return math.inf
        state = (self.state[hi] - self.state[lo]) * p.max_num_seqs
        return max(0, (p.stage_bytes - weights - state) // (kv * p.block_size))

    def of(self, partition: Sequence[int]) -> tuple[float, ...]:
        out, lo = [], 0
        for stage, n in enumerate(partition):
            out.append(self.pages(stage, lo, lo + n))
            lo += n
        return tuple(out)

    def feasible(self, partition: Sequence[int], max_layers: int) -> bool:
        lo = 0
        for n in partition:
            if not 1 <= n <= max_layers or not self.holds_every_kind(lo, lo + n):
                return False
            lo += n
        return True


def predicted_pages(
    problem: PartitionProblem, partition: Sequence[int]
) -> tuple[float, ...]:
    """Each stage's predicted pages under ``partition`` (inf for a stage
    without per-token KV); the pool is their minimum."""
    if len(partition) != problem.pp_size or sum(partition) != len(problem.layers):
        raise ValueError(
            f"{list(partition)} is not a split of {len(problem.layers)} layers "
            f"into {problem.pp_size} stages"
        )
    return _Stages(problem).of(partition)


def _best_pool(stages: _Stages, num_layers: int, pp_size: int, max_layers: int):
    """The largest pool any feasible split reaches; None if there is none."""
    # best[s][lo]: the largest min-pages of stages s.. when stage s starts at lo
    best = [[-1.0] * (num_layers + 1) for _ in range(pp_size + 1)]
    best[pp_size][num_layers] = math.inf
    for s in range(pp_size - 1, -1, -1):
        for lo in range(num_layers):
            for hi in range(lo + 1, min(lo + max_layers, num_layers) + 1):
                if best[s + 1][hi] < 0 or not stages.holds_every_kind(lo, hi):
                    continue
                value = min(stages.pages(s, lo, hi), best[s + 1][hi])
                best[s][lo] = max(best[s][lo], value)
    return None if best[0][0] < 0 else best[0][0]


def _most_balanced(
    stages: _Stages, num_layers: int, pp_size: int, max_layers: int, floor: float
) -> tuple[int, ...] | None:
    """Among feasible splits whose every stage fits ``floor`` pages, the one
    with the smallest sum of squared stage sizes, then the largest pool."""
    # best[s][lo]: (sum of squares, -pool, sizes) of stages s.. from lo
    best: list[list[tuple | None]] = [
        [None] * (num_layers + 1) for _ in range(pp_size + 1)
    ]
    best[pp_size][num_layers] = (0, -math.inf, ())
    for s in range(pp_size - 1, -1, -1):
        for lo in range(num_layers):
            for hi in range(lo + 1, min(lo + max_layers, num_layers) + 1):
                tail = best[s + 1][hi]
                if tail is None or not stages.holds_every_kind(lo, hi):
                    continue
                pages = stages.pages(s, lo, hi)
                if pages < floor:
                    continue
                n = hi - lo
                key = (n * n + tail[0], max(-pages, tail[1]), (n,) + tail[2])
                if best[s][lo] is None or key < best[s][lo]:
                    best[s][lo] = key
    return None if best[0][0] is None else best[0][0][2]


def plan_partition(problem: PartitionProblem) -> PartitionPlan:
    """The split ``problem`` asks for: see the module docstring."""
    num_layers, pp_size = len(problem.layers), problem.pp_size
    if not 1 <= pp_size <= num_layers:
        raise ValueError(f"cannot split {num_layers} layers into {pp_size} stages")
    stages = _Stages(problem)
    even = tuple(even_partition(num_layers, pp_size))
    max_layers = min(num_layers, max(even) + max(0, problem.max_extra_layers))

    def plan(partition: tuple[int, ...]) -> PartitionPlan:
        return PartitionPlan(
            partition=partition,
            pages=stages.of(partition),
            even=even,
            even_pages=stages.of(even),
            max_layers=max_layers,
            block_size=problem.block_size,
        )

    best = _best_pool(stages, num_layers, pp_size, max_layers)
    if best is None or math.isinf(best):
        # No split meets the limits, or there is no per-token KV to balance.
        return plan(even)
    floor = best * (1 - POOL_TOLERANCE)
    if stages.feasible(even, max_layers) and min(stages.of(even)) >= floor:
        return plan(even)
    # The smallest largest stage first: the slowest stage sets the pipeline's
    # pace, so a layer saved there is worth more than any spread elsewhere.
    for largest in range(max(even), max_layers + 1):
        chosen = _most_balanced(stages, num_layers, pp_size, largest, floor)
        if chosen is not None:
            return plan(chosen)
    raise AssertionError("no feasible split reaches the best pool it was found at")


# --------------------------------------------------------------------------
# What a stage holds, estimated from the config
# --------------------------------------------------------------------------


def _text_config(hf_config):
    return getattr(hf_config, "text_config", None) or hf_config


def _layer_kinds(text, num_layers: int) -> list[bool]:
    """Per target layer: True for per-token KV (full attention), False for a
    recurrent state (linear attention). Raises for an unknown layout."""
    from atom.utils.selector import Family, attn_family

    family = attn_family(text)
    lin = getattr(text, "linear_attn_config", None) or {}
    if family is Family.KIMI_MLA and getattr(text, "model_type", "") == "kimi_linear":
        # 1-based, as `_normalize_kimi_config` reads them.
        full = {int(i) - 1 for i in lin.get("full_attn_layers", [])}
        linear = {int(i) - 1 for i in lin.get("kda_layers", [])}
        if full & linear or full | linear != set(range(num_layers)):
            raise PartitionUnavailable(
                "linear_attn_config does not give every layer exactly one kind"
            )
        return [i in full for i in range(num_layers)]
    layer_types = list(getattr(text, "layer_types", None) or [])
    if family is Family.GDN:
        kinds = {"full_attention": True, "linear_attention": False}
        if len(layer_types) != num_layers or not set(layer_types) <= set(kinds):
            raise PartitionUnavailable(f"layer_types {sorted(set(layer_types))}")
        return [kinds[t] for t in layer_types]
    if family in (Family.MLA, Family.MHA) and set(layer_types) <= {"full_attention"}:
        return [True] * num_layers
    what = sorted(set(layer_types)) if layer_types else family.value
    raise PartitionUnavailable(f"no per-layer cache estimate for {what} attention")


def _kv_bytes_per_token(text, kv_cache_dtype: str, tp: int) -> int:
    """One full-attention layer's KV bytes per token, on one TP rank."""
    bits = _KV_DTYPE_BITS.get(str(kv_cache_dtype).lower())
    if bits is None:
        raise PartitionUnavailable(f"kv_cache_dtype {kv_cache_dtype!r}")
    if getattr(text, "kv_lora_rank", None):
        # MLA: one latent row, the same on every TP rank.
        elems = int(text.kv_lora_rank) + int(getattr(text, "qk_rope_head_dim", 0) or 0)
    else:
        heads = getattr(text, "num_key_value_heads", None) or text.num_attention_heads
        head_dim = getattr(text, "head_dim", None) or (
            text.hidden_size // text.num_attention_heads
        )
        elems = 2 * max(1, int(heads) // tp) * int(head_dim)
    return math.ceil(elems * bits / 8)


def _linear_state_bytes_per_request(text, config, tp: int) -> int:
    """One linear-attention layer's recurrent state for one request on one TP
    rank, as `GDNStateMixin.state_spec` sizes it: the conv and temporal state
    of a slot (`GDNStateMixin._state_shape`) times `slots_per_req`, or under
    ReplaySSM one slot and its record buffers (`replayssm_buffer_shapes`)."""
    from atom.utils import envs

    kda = getattr(text, "model_type", "") == "kimi_linear"
    if kda:
        lin = text.linear_attn_config
        k_heads = v_heads = int(lin["num_heads"])
        k_dim = v_dim = int(lin["head_dim"])
        kernel = int(lin.get("short_conv_kernel_size", 4))
    else:
        k_heads, v_heads = text.linear_num_key_heads, text.linear_num_value_heads
        k_dim, v_dim = text.linear_key_head_dim, text.linear_value_head_dim
        kernel = text.linear_conv_kernel_dim
    act = 2  # the model dtype, bf16
    ssm = _STATE_DTYPE_BYTES.get(envs.ATOM_GDN_SSM_DTYPE, 4) if kda else act
    spec = getattr(config, "speculative_config", None)
    num_spec = int(getattr(spec, "num_speculative_tokens", 0) or 0) if spec else 0
    conv = (kernel - 1 + num_spec) * ((2 * k_heads * k_dim + v_heads * v_dim) // tp)
    slot = conv * act + (v_heads // tp) * v_dim * k_dim * ssm
    replayssm = envs.ATOM_ENABLE_REPLAYSSM
    if replayssm is None:
        replayssm = spec is not None
    if not replayssm:
        return slot * (1 + num_spec)
    rows = (v_heads // tp) * max(envs.ATOM_REPLAYSSM_CACHE_LEN, 2 * (num_spec + 1))
    return slot + rows * (k_dim + v_dim) * act + rows * (k_dim if kda else 1) * 4


@dataclass(frozen=True)
class _CheckpointBytes:
    layers: tuple[int, ...]  # each target layer's tensors
    beyond: int  # layer stacks past the target's: an MTP drafter's
    embed: int
    head: int
    other: int  # final norm, vision tower, a drafter's own heads...


def _checkpoint_bytes(model: str, num_layers: int) -> _CheckpointBytes:
    """``model``'s checkpoint bytes, by where they live under PP."""
    from atom.model_loader.weight_utils import checkpoint_tensor_bytes

    tensors = checkpoint_tensor_bytes(model)
    if not tensors:
        raise PartitionUnavailable(f"no safetensors shard headers for {model}")
    layers = [0] * num_layers
    beyond = embed = head = other = 0
    for name, size in tensors.items():
        m = _LAYER_RE.search(name)
        prefix = name[: m.start()] if m else name
        foreign = any(tag in prefix for tag in _NOT_TARGET)
        if m and not foreign and int(m.group(1)) < num_layers:
            layers[int(m.group(1))] += size
        elif m and ("mtp" in prefix or not foreign):
            beyond += size
        elif "embed_tokens" in name and not foreign:
            embed += size
        elif "lm_head" in name:
            head += size
        else:
            other += size
    return _CheckpointBytes(tuple(layers), beyond, embed, head, other)


def _draft_kv_layers(spec) -> int:
    """KV rows the drafter keeps per token, as the MLA builder counts them
    (`AiterMLAMetadataBuilder._global_num_draft_layers`)."""
    draft = spec.draft_model_hf_config
    if spec.use_dspark_with_draft() or spec.method == "eagle3":
        return int(getattr(draft, "num_hidden_layers", 1) or 1)
    for attr in ("num_nextn_predict_layers", "mtp_num_hidden_layers"):
        if getattr(draft, attr, None):
            return int(getattr(draft, attr))
    return 1


def device_memory_bytes() -> int | None:
    """This node's GPU memory size, read without creating a GPU context (the
    split is chosen in the API process, which must not hold one): amdsmi,
    else amdgpu's sysfs. The most common size if the GPUs differ."""
    sizes: list[int] = []
    try:
        import amdsmi

        amdsmi.amdsmi_init()
        try:
            for handle in amdsmi.amdsmi_get_processor_handles():
                vram = amdsmi.AmdSmiMemoryType.VRAM
                sizes.append(int(amdsmi.amdsmi_get_gpu_memory_total(handle, vram)))
        finally:
            amdsmi.amdsmi_shut_down()
    except Exception as exc:  # noqa: BLE001 -- any failure falls back to sysfs
        logger.debug("amdsmi GPU memory query failed: %s", exc)
    if not sizes:
        for path in glob.glob("/sys/class/drm/card*/device/mem_info_vram_total"):
            try:
                with open(path) as f:
                    sizes.append(int(f.read().strip()))
            except (OSError, ValueError):
                continue
    sizes = sorted(s for s in sizes if s > 0)
    return collections.Counter(sizes).most_common(1)[0][0] if sizes else None


def partition_problem(config) -> PartitionProblem:
    """The split problem ``config`` poses, estimated as the module docstring
    describes. Raises `PartitionUnavailable` when a term cannot be estimated."""
    from atom.utils import envs

    text = _text_config(config.hf_config)
    # The count every `get_pp_indices` caller passes.
    num_layers = int(config.hf_config.num_hidden_layers)
    tp = max(1, int(getattr(config, "tensor_parallel_size", 1) or 1))
    kinds = _layer_kinds(text, num_layers)
    kv = _kv_bytes_per_token(text, config.kv_cache_dtype, tp)
    state = 0 if all(kinds) else _linear_state_bytes_per_request(text, config, tp)

    target = _checkpoint_bytes(config.model, num_layers)
    missing = [i for i, size in enumerate(target.layers) if not size]
    if missing:
        raise PartitionUnavailable(
            f"the checkpoint has no weights for layers {missing}"
        )
    head = target.head
    if not head and getattr(text, "tie_word_embeddings", False):
        head = target.embed  # the last stage builds its own lm_head copy
    last_weights, last_kv = head, 0
    spec = getattr(config, "speculative_config", None)
    if spec is not None and spec.draft_model_hf_config is not None:
        draft_text = _text_config(spec.draft_model_hf_config)
        draft_dtype = spec.draft_kv_dtype or config.kv_cache_dtype
        last_kv = _draft_kv_layers(spec) * _kv_bytes_per_token(
            draft_text, draft_dtype, tp
        )
        own = spec.model and os.path.realpath(spec.model) != os.path.realpath(
            config.model
        )
        if own:
            # Its own checkpoint less its embedding: a PP drafter shares the
            # target's (`pp_holds_embedding`, below).
            draft = _checkpoint_bytes(spec.model, 0)
            last_weights += draft.beyond + draft.head + draft.other
        else:
            # An MTP drafter's layers follow the target's in its checkpoint.
            last_weights += target.beyond
        from atom.kv_transfer.disaggregation.types import kv_config_has_producer
        from atom.models.registry import serves_pp_speculation

        if serves_pp_speculation(config) and not kv_config_has_producer(
            config.kv_transfer_config
        ):
            last_weights += target.embed

    total = device_memory_bytes()
    if total is None:
        raise PartitionUnavailable("the GPU memory size is unreadable (amdsmi, sysfs)")
    return PartitionProblem(
        layers=tuple(
            LayerCost(
                weight_bytes=target.layers[i] // tp,
                kv_bytes_per_token=kv if kinds[i] else 0,
                state_bytes_per_request=0 if kinds[i] else state,
            )
            for i in range(num_layers)
        ),
        pp_size=int(config.pipeline_parallel_size),
        stage_bytes=int(
            total * (config.gpu_memory_utilization - RUNTIME_RESERVE_FRACTION)
        ),
        block_size=int(config.kv_cache_block_size),
        max_num_seqs=int(config.max_num_seqs),
        first_stage_weight_bytes=(target.embed + target.other) // tp,
        last_stage_weight_bytes=last_weights // tp,
        last_stage_kv_bytes_per_token=last_kv,
        max_extra_layers=envs.ATOM_PP_PARTITION_MAX_EXTRA_LAYERS,
    )


def resolve_pp_layer_partition(config) -> tuple[int, ...] | None:
    """The stage sizes ``ATOM_PP_LAYER_PARTITION`` asks ``config`` to use, or
    None to keep the split `get_pp_indices` gives without it: the even split,
    or ``VLLM_PP_LAYER_PARTITION``, which always wins."""
    from atom.utils import envs

    mode = envs.ATOM_PP_LAYER_PARTITION
    if mode == EVEN or int(getattr(config, "pipeline_parallel_size", 1) or 1) <= 1:
        return None
    explicit = os.getenv("VLLM_PP_LAYER_PARTITION")
    if explicit is not None:
        logger.info(
            "PP layer split: VLLM_PP_LAYER_PARTITION=%s overrides "
            "ATOM_PP_LAYER_PARTITION=%s",
            explicit,
            mode,
        )
        return None
    from atom.plugin import is_plugin_mode

    if is_plugin_mode():
        logger.warning(
            "ATOM_PP_LAYER_PARTITION=%s is ignored in plugin mode, where the "
            "host framework splits the layers",
            mode,
        )
        return None
    try:
        problem = partition_problem(config)
    except PartitionUnavailable as exc:
        logger.warning(
            "ATOM_PP_LAYER_PARTITION=%s: %s; keeping the even split", mode, exc
        )
        return None
    plan = plan_partition(problem)
    budget = problem.stage_bytes / (1 << 30)
    if plan.partition == plan.even:
        logger.info(
            "PP layer split (ATOM_PP_LAYER_PARTITION=%s): keeping the even split, "
            "as no split within the limits predicts a pool %.0f%% larger: %s; "
            "%.1f GiB per stage for weights, state and pages",
            mode,
            100 * POOL_TOLERANCE,
            plan.describe(),
            budget,
        )
        return None
    logger.info(
        "PP layer split (ATOM_PP_LAYER_PARTITION=%s): %s; %.1f GiB per stage for "
        "weights, state and pages. VLLM_PP_LAYER_PARTITION=%s pins it",
        mode,
        plan.describe(),
        budget,
        ",".join(map(str, plan.partition)),
    )
    return plan.partition
