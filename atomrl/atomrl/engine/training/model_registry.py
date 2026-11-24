# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Architecture registry for the Megatron training path.

Replaces the engine's inline ``_is_dsv4`` / ``_is_moe`` booleans, which needed
another branch at every call site per new family. One :class:`ModelSpec` per
family instead; adding an architecture should not mean editing the engine.

Resolution is ordered first-match, so specific families register before general
ones and a catch-all dense entry is last. ``detect`` sees the engine config too,
because an override there can turn a dense config into a MoE run.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "MODEL_REGISTRY",
    "ModelCaps",
    "ModelRegistry",
    "ModelSpec",
    "hf_num_experts",
    "resolve_head_dim",
]


def hf_num_experts(hf: Mapping[str, Any]) -> int:
    """Routed-expert count declared by an HF config, across the spellings in use.

    Qwen3-MoE uses ``num_experts``, DeepSeek ``n_routed_experts``, Mixtral-style
    configs ``num_local_experts``. Returns 0 for a dense config.
    """
    return int(
        hf.get("num_experts")
        or hf.get("n_routed_experts")
        or hf.get("num_local_experts")
        or 0
    )


def resolve_head_dim(hf: Mapping[str, Any]) -> int:
    """``head_dim`` if the config states it, else hidden / heads."""
    return int(hf.get("head_dim", hf["hidden_size"] // hf["num_attention_heads"]))


@dataclass(frozen=True)
class ModelCaps:
    """What the engine may do with this family.

    Each field replaces an ``if self._is_dsv4`` the engine used to carry.
    Defaults are the permissive ones: a new family opts in, never inherits.
    """

    # Weights move through the HF safetensors bridge. False for DSv4: its
    # block-quantized FP8 is unreadable there, so it uses dist-checkpointing.
    supports_hf_bridge: bool = True
    # Several sequences may share a microbatch. False where attention derives
    # positions from tensor length alone, which would read the pack as one seq.
    supports_dynamic_batch: bool = True
    # Supplies its own TransformerConfig / layer spec instead of the TE default.
    builds_own_config: bool = False
    # Forward must take the pipeline path even at PP=CP=1.
    requires_pipeline_forward: bool = False
    # Reads a packed stream as one sequence (ignores cu_seqlens); constrains
    # sequence alignment and CP shard layout.
    packed_stream_is_single_sequence: bool = False


@dataclass(frozen=True)
class ModelSpec:
    """How to recognise one model family and what the engine needs from it.

    The optional callables are hooks; ``None`` means "use the generic path".
    """

    name: str
    detect: Callable[[Mapping[str, Any], Mapping[str, Any]], bool]
    caps: ModelCaps = field(default_factory=ModelCaps)
    # Dims dataclass for the HF weight bridge; None if the family skips it.
    build_dims: Callable[[Mapping[str, Any]], Any] | None = None
    # Architecture-level routing defaults, still overridable by engine_config.
    # Only reaches the generic config path -- a ``build_config`` family sets its
    # own, so declaring both risks them drifting apart.
    routing_defaults: Mapping[str, Any] = field(default_factory=dict)
    # ``(hf, ec, **parallel) -> TransformerConfig`` when the generic builder
    # cannot describe the family.
    build_config: Callable[..., Any] | None = None
    # ``(tfcfg, ec) -> ModuleSpec`` for heterogeneous layers.
    build_layer_spec: Callable[..., Any] | None = None
    # ``(tfcfg) -> int`` extra sequence-length alignment the family requires.
    sequence_alignment: Callable[[Any], int] | None = None
    # ``(engine) -> None`` run before each forward, to refuse topologies the
    # family gets wrong. Raise to reject; DSv4 uses it for exactly that.
    pre_forward_check: Callable[[Any], None] | None = None
    # ``(engine) -> Iterable[(name, tensor)]``: the rollout weight stream. Gather
    # and renaming are one family decision, so they share a hook. Takes the engine
    # because the gathers are its helpers -- an intentional in-package coupling.
    export_weights: Callable[[Any], Any] | None = None
    # Override the expert check for a family whose config does not state the count
    # in any spelling ``hf_num_experts`` knows. ``None`` uses the count, which is
    # what every current family wants.
    has_experts: Callable[[Mapping[str, Any], Mapping[str, Any]], bool] | None = None

    def resolve_has_experts(self, hf: Mapping[str, Any], ec: Mapping[str, Any]) -> bool:
        """Whether this run has routed experts, honouring the config override.

        Keyed on the effective count, not a per-family flag: engine_config's
        ``num_experts`` can flip a config either way. Same condition the engine
        used before the registry.
        """
        if self.has_experts is not None:
            return bool(self.has_experts(hf, ec))
        return int(ec.get("num_experts") or hf_num_experts(hf) or 0) > 1


class ModelRegistry:
    """Ordered first-match registry of :class:`ModelSpec`."""

    def __init__(self) -> None:
        self._specs: list[ModelSpec] = []

    def register(self, spec: ModelSpec) -> ModelSpec:
        """Append a spec. Order is priority: register specific before general."""
        if any(s.name == spec.name for s in self._specs):
            raise ValueError(f"ModelSpec {spec.name!r} is already registered")
        self._specs.append(spec)
        return spec

    def resolve(
        self, hf: Mapping[str, Any], engine_config: Mapping[str, Any] | None = None
    ) -> ModelSpec:
        """First spec whose ``detect`` accepts this config.

        Raises if nothing matches -- only possible if the catch-all was removed,
        and better loud than a silently wrong dense path.
        """
        ec = engine_config or {}
        for spec in self._specs:
            if spec.detect(hf, ec):
                return spec
        arch = hf.get("architectures") or hf.get("model_type") or "<unknown>"
        raise LookupError(
            f"No ModelSpec matches architecture {arch!r}. "
            f"Registered: {[s.name for s in self._specs]}"
        )

    @property
    def specs(self) -> list[ModelSpec]:
        """Every registered spec, so "for each family" tests cover new ones."""
        return list(self._specs)

    @property
    def names(self) -> list[str]:
        return [s.name for s in self._specs]


MODEL_REGISTRY = ModelRegistry()
