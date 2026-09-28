# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Checkpoint-name rewriting for the model loader.

Turning an on-disk tensor name into the parameter name it belongs to is a
sequence of model-declared rules (prefix/suffix/substring maps, MTP filters,
shared-expert fusion).  Keeping it here — free of AITER and of any GPU state —
means the rules can be unit-tested directly instead of only through a full
`load_model` run.
"""

import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

# WeightsMapper is adapted from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/utils.py
WeightsMapping = Mapping[str, str | None]
"""If a key maps to a value of `None`, the corresponding weight is ignored."""


@dataclass
class WeightsMapper:
    """Maps the name of each weight if they match the following patterns."""

    orig_to_new_substr: WeightsMapping = field(default_factory=dict)
    orig_to_new_prefix: WeightsMapping = field(default_factory=dict)
    orig_to_new_suffix: WeightsMapping = field(default_factory=dict)

    def __or__(self, other: "WeightsMapper") -> "WeightsMapper":
        """Combine two `WeightsMapper`s by merging their mappings."""
        return WeightsMapper(
            orig_to_new_substr={**self.orig_to_new_substr, **other.orig_to_new_substr},
            orig_to_new_prefix={**self.orig_to_new_prefix, **other.orig_to_new_prefix},
            orig_to_new_suffix={**self.orig_to_new_suffix, **other.orig_to_new_suffix},
        )

    def _map_name(self, key: str) -> str | None:
        for substr, new_key in self.orig_to_new_substr.items():
            if substr in key:
                if new_key is None:
                    return None

                key = key.replace(substr, new_key, 1)

        for prefix, new_key in self.orig_to_new_prefix.items():
            if key.startswith(prefix):
                if new_key is None:
                    return None

                key = key.replace(prefix, new_key, 1)

        for suffix, new_key in self.orig_to_new_suffix.items():
            if key.endswith(suffix):
                if new_key is None:
                    return None

                key = new_key.join(key.rsplit(suffix, 1))

        return key

    def apply(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> Iterable[tuple[str, torch.Tensor]]:
        return (
            (out_name, data)
            for name, data in weights
            if (out_name := self._map_name(name)) is not None
        )

    def apply_list(self, values: list[str]) -> list[str]:
        return [
            out_name
            for name in values
            if (out_name := self._map_name(name)) is not None
        ]

    def apply_dict(self, values: dict[str, Any]) -> dict[str, Any]:
        return {
            out_name: value
            for name, value in values.items()
            if (out_name := self._map_name(name)) is not None
        }

    def get_unstacked_mapper(self) -> "WeightsMapper":
        """Return the mapper variant suitable for quantization config names.

        ATOM keeps packed-module handling separate from checkpoint-name
        rewriting, so this mapper never contains vLLM-style stacked mappings
        and is already unstacked.
        """
        return self


def have_shared_expert(name: str) -> str | None:
    """Return the `...shared_expert(s).` substring in `name`, if any.

    Matches both `mlp.` (GLM4, Qwen, ...) and `ffn.` (DeepSeek-V4) module
    naming. The matched substring is replaced by the caller with
    `<prefix>experts.{n_routed}.` so the shared expert lands in the fused
    MoE buffer's extra slot. Returning the full prefix (incl. mlp./ffn.)
    lets the rewrite preserve the module-naming style.
    """
    maybe_matching_list = [
        "block_sparse_moe.shared_experts.",
        "block_sparse_moe.shared_expert.",
        "mlp.shared_experts.",
        "mlp.shared_expert.",
        "ffn.shared_experts.",
        "ffn.shared_expert.",
    ]
    for maybe_matching_name in maybe_matching_list:
        if maybe_matching_name in name:
            return maybe_matching_name
    return None


def shared_expert_prefixes(name: str, matching_name: str) -> tuple[str, str]:
    """Split `name` into the (shared expert, routed expert) module prefixes.

    Both are needed to decide whether the two are quantized identically and can
    therefore share one fused MoE buffer.
    """
    layer_prefix = name.split(matching_name, 1)[0]
    module_prefix = matching_name.split("shared_expert", 1)[0]
    return (
        layer_prefix + matching_name.rstrip("."),
        layer_prefix + f"{module_prefix}experts",
    )


@dataclass
class CheckpointNameRewriter:
    """Turns an on-disk tensor name into the parameter name it belongs to.

    Nine model-declared rules applied in a fixed order; `rewrite` returns None
    for a tensor this model does not want. Order matters and is not obvious --
    the MTP filter has to run before `mtp_remap`, because a drafter checkpoint
    remaps `mtp.*` onto names that no longer contain "mtp" -- so it lives in
    one place rather than being spread through the loading loop.

    `fuse_shared_expert(shared_prefix, routed_prefix) -> bool` is injected: the
    real decision reads the quantization config, which would drag AITER into
    this module and out of reach of the unit test gate.
    """

    weights_mapper: WeightsMapper | None = None
    weights_mapping: Mapping[str, str] = field(default_factory=dict)
    skip_weight_prefixes: Sequence[str] = ()
    mtp_remap: Callable[[str], str | None] | None = None
    spec_decode: bool = False
    num_hidden_layers: int = 0
    n_routed_experts: int | None = None
    fuse_shared_expert: Callable[[str, str], bool] = lambda *_: False
    disable_fused_shared_loading: bool = False

    def rewrite(self, name: str) -> str | None:
        name = self._apply_mapper(name)
        if name is None:
            return None
        # Draft models remap ckpt-side `mtp.*` entries into params whose names
        # do not themselves contain `mtp` (Qwen3.5 MTP rewrites `mtp.*` ->
        # `model.*`), so gate only on `spec_decode`: dropping every `mtp` name
        # first would discard the whole drafter checkpoint before the
        # model-specific remap ever runs.
        if "mtp" in name and not self.spec_decode:
            return None
        if name.endswith("kv_scale") or "inv_freq" in name:
            return None
        # Model-declared prefixes to ignore, e.g. the vision encoder of a
        # multimodal checkpoint being served text-only.
        if any(name.startswith(p) for p in self.skip_weight_prefixes):
            return None
        if self.spec_decode and self.mtp_remap is not None:
            name = self.mtp_remap(name)
            if name is None:
                return None
        for part, replacement in self.weights_mapping.items():
            if part in name:
                name = name.replace(part, replacement)
        if "weight_scale_inv" in name:
            name = name.replace("weight_scale_inv", "weight_scale")
        if self._is_past_last_layer(name):
            return None
        return self._maybe_fuse_shared_expert(name)

    # ── steps ─────────────────────────────────────────────────────────────

    def _apply_mapper(self, name: str) -> str | None:
        if self.weights_mapper is None:
            return name
        return self.weights_mapper._map_name(name)

    def _is_past_last_layer(self, name: str) -> bool:
        if not self.num_hidden_layers or self.spec_decode:
            return False
        match = re.search(r"model\.layers\.(\d+)\.", name)
        return match is not None and int(match.group(1)) >= self.num_hidden_layers

    def _maybe_fuse_shared_expert(self, name: str) -> str:
        matching_name = have_shared_expert(name)
        if matching_name is None:
            return name
        # Some models keep shared experts unfused (V4-Pro with FP4 routed vs
        # FP8 shared, or DP + mori all2all); their shared weights must land on
        # the standalone Expert module instead of the fused slot.
        if self.disable_fused_shared_loading:
            return name
        if not self.fuse_shared_expert(*shared_expert_prefixes(name, matching_name)):
            return name
        if self.n_routed_experts is None:
            raise AttributeError(
                "Cannot remap shared expert weights without n_routed_experts, "
                "num_local_experts, or num_experts on the model config."
            )
        # Keep the module-naming prefix (mlp. / ffn.) so the rewritten name
        # matches this model's routed-expert parameter naming. The id follows
        # the *logical* routed experts, matching
        # `FusedMoE.make_expert_params_mapping`; FusedMoE translates it to a
        # physical slot, which under EPLB also has to clear the replicas.
        module_prefix = matching_name.split("shared_expert", 1)[0]
        return name.replace(
            matching_name, f"{module_prefix}experts.{self.n_routed_experts}."
        )


def extract_expert_target_and_id(name: str) -> tuple[str, int] | None:
    """Extract fused parameter name and expert id from expert checkpoint name.
    like 'model.layers.10.mlp.experts.100.w2_bias' -> model.layers.10.mlp.experts.w2_bias and 100
    """
    if "experts" not in name:
        return None
    parts = name.split(".")
    ids = [s for s in parts if s.isdigit()]
    if len(ids) != 2:
        return None
    expert_id = int(ids[-1])
    expert_token = str(expert_id)
    if expert_token not in parts:
        return None
    fused_parts = parts.copy()
    fused_parts.pop(len(parts) - 1 - parts[::-1].index(expert_token))
    return ".".join(fused_parts), expert_id
