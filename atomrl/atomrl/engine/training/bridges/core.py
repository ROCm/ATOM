# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Shared machinery for the Megatron <-> Hugging Face weight bridges.

A family's export direction is a list of :class:`Rule` objects. Each rule matches
exactly one Megatron parameter name and yields the HF tensor(s) it becomes, so
:meth:`WeightBridge.megatron_to_hf` converts a tensor as soon as it arrives and
never holds the model. That matters because the engines feed it from lazy
gathers: a bridge that looked tensors up by name would have to drain the gather
first, holding every rank's full model at once.

Name templates use ``{L}`` for the layer index and ``{E}`` for the expert index.
"""

from __future__ import annotations

import glob
import logging
import os
import re
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from safetensors.torch import load_file

logger = logging.getLogger(__name__)

HfTensor = tuple[str, torch.Tensor]

MODULE_PREFIXES = ("module.module.", "module.")

# Megatron routed-expert parameter names, grouped-GEMM and sequential layouts.
_EXP_GROUPED = re.compile(r"^(.*\.mlp\.experts\.linear_fc([12]))\.weight(\d+)$")
_EXP_SEQ = re.compile(
    r"^(.*\.mlp\.experts\.local_experts\.)(\d+)(\.linear_fc([12])\.weight)$"
)

_PLACEHOLDERS = {"L": r"(?P<L>\d+)", "E": r"(?P<E>\d+)", "S": r"(?P<S>\d+)"}


def stacked_expert_index(start: int, step: int) -> str:
    """Expert-index token for a stack of experts ``start, start + step, ...``.

    Goes where a single expert index would (``relabel_expert_index``), so a
    stacked tensor keeps its layout's name and gets its own bridge rule.
    """
    return f"{start}::{step}"


def strip_module_prefix(name: str) -> str:
    """Drop the DDP / Float16Module wrapper prefix from a parameter name."""
    for pre in MODULE_PREFIXES:
        if name.startswith(pre):
            return name[len(pre) :]
    return name


def split_gate_up(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a fused SwiGLU ``linear_fc1`` (``[gate; up]`` on dim 0)."""
    gate, up = t.chunk(2, dim=0)
    return gate.contiguous(), up.contiguous()


def expert_local_index(name: str) -> tuple[int, str] | None:
    """Return ``(expert_index, which_fc)`` for a routed-expert param, else None.

    ``which_fc`` is ``"1"`` (fused gate;up) or ``"2"`` (down).
    """
    m = _EXP_GROUPED.match(name)
    if m:
        return int(m.group(3)), m.group(2)
    m = _EXP_SEQ.match(name)
    if m:
        return int(m.group(2)), m.group(4)
    return None


def relabel_expert_index(name: str, new_e: int | str) -> str:
    """Rewrite a routed-expert param name to use expert index ``new_e``."""
    m = _EXP_GROUPED.match(name)
    if m:
        return f"{m.group(1)}.weight{new_e}"
    m = _EXP_SEQ.match(name)
    if m:
        return f"{m.group(1)}{new_e}{m.group(3)}"
    return name


def pp_layer_range(
    d: Any,
    pp_rank: int,
    pp_size: int,
    layers_per_pp_rank: list[int] | None,
) -> tuple[int, int]:
    """Return ``(global_layer_offset, num_local_layers)`` for this PP rank."""
    if pp_size <= 1:
        return 0, d.num_layers
    if layers_per_pp_rank is not None:
        assert len(layers_per_pp_rank) == pp_size
        offset = sum(layers_per_pp_rank[:pp_rank])
        return offset, layers_per_pp_rank[pp_rank]
    per_stage = d.num_layers // pp_size
    return pp_rank * per_stage, per_stage


def load_hf_safetensors(model_dir: str) -> dict[str, torch.Tensor]:
    """Load a full HF state dict from a sharded safetensors directory."""
    state: dict[str, torch.Tensor] = {}
    files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"no safetensors under {model_dir}")
    for f in files:
        state.update(load_file(f))
    return state


@dataclass(frozen=True)
class Site:
    """Where a matched Megatron tensor sits, in HF (global) coordinates.

    Attributes:
        layer: Global layer index, or None for a top-level tensor.
        expert: Global expert index, or None when the name carries none.
        expert_offset: Global index of this rank's first expert, for layouts
            that fuse several experts into one tensor.
        dims: The family's dims dataclass.
        expert_step: Stride between experts of a stacked tensor, else None.
    """

    layer: int | None
    expert: int | None
    expert_offset: int
    dims: Any
    expert_step: int | None = None

    def name(self, template: str) -> str:
        """Fill ``{L}`` / ``{E}`` / ``{S}`` in an HF name template."""
        return template.format(L=self.layer, E=self.expert, S=self.expert_step)


Convert = Callable[[torch.Tensor, Site], Iterable[HfTensor]]


@dataclass(frozen=True)
class Rule:
    """One Megatron parameter name template and how it becomes HF tensors."""

    megatron: str
    convert: Convert
    pattern: re.Pattern[str] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        regex = re.escape(self.megatron)
        for key, group in _PLACEHOLDERS.items():
            regex = regex.replace(re.escape("{" + key + "}"), group)
        object.__setattr__(self, "pattern", re.compile(f"^{regex}$"))


def rename(megatron: str, hf: str) -> Rule:
    """A tensor that only changes name."""
    return Rule(megatron, lambda t, site: [(site.name(hf), t)])


def split(
    megatron: str,
    hf: Sequence[str],
    fn: Callable[[torch.Tensor, Any], Sequence[torch.Tensor]],
) -> Rule:
    """A fused tensor that becomes ``len(hf)`` tensors; ``fn(t, dims)`` splits it."""

    def convert(t: torch.Tensor, site: Site) -> Iterator[HfTensor]:
        parts = fn(t, site.dims)
        assert len(parts) == len(
            hf
        ), f"{megatron}: {len(parts)} parts for {len(hf)} names"
        for template, part in zip(hf, parts):
            yield site.name(template), part

    return Rule(megatron, convert)


def gate_up(megatron: str, hf_prefix: str) -> Rule:
    """A fused SwiGLU ``linear_fc1`` -> ``{hf_prefix}{gate,up}_proj.weight``."""
    return split(
        megatron,
        (hf_prefix + "gate_proj.weight", hf_prefix + "up_proj.weight"),
        lambda t, _d: split_gate_up(t),
    )


def routed_expert_rules(
    hf_prefix: str = "model.layers.{L}.mlp.experts.{E}.",
) -> list[Rule]:
    """Per-expert tensors in both Megatron layouts (grouped GEMM and sequential)."""
    rules = []
    for meg in (
        "decoder.layers.{L}.mlp.experts.linear_fc{fc}.weight{E}",
        "decoder.layers.{L}.mlp.experts.local_experts.{E}.linear_fc{fc}.weight",
    ):
        rules.append(gate_up(meg.replace("{fc}", "1"), hf_prefix))
        rules.append(rename(meg.replace("{fc}", "2"), hf_prefix + "down_proj.weight"))
    return rules


def stacked_expert_rules(
    hf_prefix: str = "model.layers.{L}.mlp.experts.",
) -> list[Rule]:
    """Expert stacks ``[n, out, in]`` from the Megatron-native gather, both layouts.

    Experts ``E, E+S, ...`` of one layer, kept whole: ``linear_fc1`` stays fused
    ``[gate; up]`` per expert, which is the rollout's ``w13`` layout, and the HF
    name carries the stride (``{E}::{S}.gate_up_proj``) for the receiver's
    router. Only the vLLM IPC path asks for stacks.
    """
    stack = "{E}::{S}"
    rules = []
    for meg in (
        "decoder.layers.{L}.mlp.experts.linear_fc{fc}.weight" + stack,
        "decoder.layers.{L}.mlp.experts.local_experts."
        + stack
        + ".linear_fc{fc}.weight",
    ):
        rules.append(
            rename(meg.replace("{fc}", "1"), hf_prefix + stack + ".gate_up_proj")
        )
        rules.append(rename(meg.replace("{fc}", "2"), hf_prefix + stack + ".down_proj"))
    return rules


TOP_LEVEL_RULES = [
    rename("embedding.word_embeddings.weight", "model.embed_tokens.weight"),
    rename("decoder.final_layernorm.weight", "model.norm.weight"),
    rename("output_layer.weight", "lm_head.weight"),
]


class WeightBridge:
    """Streams Megatron named tensors to HF names through a family's rules.

    Args:
        family: Name used in log messages.
        rules: The family's rules. A name must match at most one rule.
    """

    def __init__(self, family: str, rules: Sequence[Rule]) -> None:
        self.family = family
        self.rules = tuple(rules)
        self._exact = {r.megatron: r for r in self.rules if "{" not in r.megatron}
        self._templated = tuple(r for r in self.rules if "{" in r.megatron)

    def match(self, name: str) -> tuple[Rule, dict[str, int]] | None:
        """Return the rule for a (prefix-stripped) Megatron name and its indices."""
        rule = self._exact.get(name)
        if rule is not None:
            return rule, {}
        for rule in self._templated:
            m = rule.pattern.match(name)
            if m:
                return rule, {k: int(v) for k, v in m.groupdict().items()}
        return None

    def megatron_to_hf(
        self,
        named_params: Iterable[tuple[str, torch.Tensor]],
        dims: Any = None,
        *,
        layer_offset: int = 0,
        expert_offset: int = 0,
    ) -> Iterator[HfTensor]:
        """Yield ``(hf_name, tensor)`` for each Megatron tensor, in input order.

        Args:
            named_params: ``(megatron_name, tensor)`` pairs, possibly lazy.
            dims: The family's dims, handed to rules that reshape.
            layer_offset: Added to every layer index; nonzero when the names
                are stage-local rather than global.
            expert_offset: Added to every expert index; nonzero when the names
                are EP-rank-local rather than global.
        """
        unmatched: list[str] = []
        for raw, t in named_params:
            name = strip_module_prefix(raw)
            found = self.match(name)
            if found is None:
                unmatched.append(name)
                continue
            rule, idx = found
            site = Site(
                layer=idx["L"] + layer_offset if "L" in idx else None,
                expert=idx["E"] + expert_offset if "E" in idx else None,
                expert_offset=expert_offset,
                dims=dims,
                expert_step=idx.get("S"),
            )
            yield from rule.convert(t, site)
        if unmatched:
            logger.warning(
                "%s bridge: %d Megatron tensor(s) match no rule and were not exported: %s",
                self.family,
                len(unmatched),
                unmatched[:8],
            )
