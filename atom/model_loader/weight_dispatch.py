# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Routing a checkpoint tensor to the parameter (or parameters) it feeds.

By the time a tensor gets here its name has already been rewritten
(`CheckpointNameRewriter`); what remains is deciding *how* it is written, which
depends on how the checkpoint packs its weights:

  packed        one checkpoint tensor covers several parameters, or several
                cover one (`packed_modules_mapping`)
  fused expert  all routed experts of a layer in a single stacked tensor
  per expert    one tensor per (expert, shard), the batched staging path
  merged        `experts.<id>.<name>` feeding a merged parameter
  generic       everything else

The branches are ordered, and falling through one means "not mine, try the
next". Two of them also carry state across tensors: the fused-expert format is
detected from the first tensor that looks like it, and that detection then
picks the mapping used for every tensor after it.
"""

import logging
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch
from torch import nn

from atom.model_loader.weight_names import extract_expert_target_and_id

logger = logging.getLogger("atom")


class WeightDispatcher:
    """Writes checkpoint tensors into model parameters.

    Accumulates `loaded_weights_record` (parameter names that were written, for
    the plugin hosts' own load checks) and `dropped_ckpt_keys` (tensors whose
    rewritten name matches no parameter, which is almost always a rewrite-rule
    bug and is reported after loading).
    """

    def __init__(
        self,
        *,
        model: nn.Module,
        params_dict: dict[str, nn.Parameter],
        hf_config: Any,
        prefix: str,
        spec_decode: bool,
        submit: Callable[..., None],
        staging_pool: Any,
        batching_enabled: bool,
        batching_excluded: Callable[[nn.Parameter], bool] | None,
        default_weight_loader: Callable,
        packed_modules_mapping: Mapping[str, Any],
        expert_index: Mapping[str, tuple[str, int, str]],
        expert_weight_prefixes: Sequence[str],
        has_expert_mapping: bool,
        detect_fused_expert_fn: Callable[[str], bool] | None,
        get_fused_expert_mapping_fn: Callable[[], Sequence[tuple]] | None,
        load_fused_expert_weights_fn: Callable | None,
        on_fused_param: Callable[[nn.Parameter], None] | None = None,
    ):
        self.model = model
        self.params_dict = params_dict
        self.hf_config = hf_config
        self.prefix = prefix
        self.spec_decode = spec_decode
        self.submit = submit
        self.staging_pool = staging_pool
        self.batching_enabled = batching_enabled
        self.batching_excluded = batching_excluded
        self.default_weight_loader = default_weight_loader
        self.packed_modules_mapping = packed_modules_mapping
        self.expert_index = expert_index
        self.expert_weight_prefixes = expert_weight_prefixes
        self.has_expert_mapping = has_expert_mapping
        self.detect_fused_expert_fn = detect_fused_expert_fn
        self.get_fused_expert_mapping_fn = get_fused_expert_mapping_fn
        self.load_fused_expert_weights_fn = load_fused_expert_weights_fn
        # Fused writes bypass `submit` and need storage prepared first.
        self.on_fused_param = on_fused_param

        self.loaded_weights_record: set[str] = set()
        self.dropped_ckpt_keys: list[tuple[str, str]] = []
        # Latched on the first tensor that looks like a fused-expert one, then
        # used for every tensor after it. Per-load state on purpose: the same
        # model class loads a fused-expert target checkpoint and a per-expert
        # drafter block in two separate passes.
        self._is_fused_expert = False
        self._fused_expert_params_mapping: Sequence[tuple] = ()

    def dispatch(self, orig_ckpt_name: str, name: str, tensor: torch.Tensor) -> None:
        if not self._dispatch_packed(orig_ckpt_name, name, tensor):
            self._dispatch_expert(orig_ckpt_name, name, tensor)

    # ── packed modules ────────────────────────────────────────────────────

    def _dispatch_packed(
        self, orig_ckpt_name: str, name: str, tensor: torch.Tensor
    ) -> bool:
        """True when a packed rule claimed this tensor, handled or abandoned.

        Abandoned counts as claimed: a rule that matched but whose target
        parameter is missing stops the search rather than falling through to
        the expert paths, which is what the original `break` did.
        """
        for k in self.packed_modules_mapping:
            # We handle the experts below in expert_params_mapping
            if (
                "mlp.experts." in name
                or "ffn.experts." in name
                or "block_sparse_moe.experts." in name
            ) and name not in self.params_dict:
                continue
            if k not in name:
                continue
            packed_value = self.packed_modules_mapping[k]
            # Handle both tuple (fuse parameter) and list (shard parameter)
            if isinstance(packed_value, list):
                # Checkpoint has fused weight, split into separate params
                for shard_idx, target_name in enumerate(packed_value):
                    param_name = name.replace(k, target_name)
                    if "output_scale" in param_name:
                        continue
                    param = self._parameter(orig_ckpt_name, param_name)
                    if param is None:
                        continue
                    self.submit(param.weight_loader, param, tensor, shard_idx)
                    self._record(param_name)
            else:
                # Checkpoint has separate weights, load into fused param
                v, shard_id = packed_value
                param_name = name.replace(k, v)
                # FIXME output_scale has a value, so accuracy is incorrect. this should be loaded and used in llfp4.
                if "output_scale" not in param_name:
                    param = self._parameter(orig_ckpt_name, param_name)
                    if param is None:
                        return True
                    self.submit(param.weight_loader, param, tensor, shard_id)
                    self._record(param_name)
            return True
        return False

    # ── expert paths ──────────────────────────────────────────────────────

    def _dispatch_expert(
        self, orig_ckpt_name: str, name: str, tensor: torch.Tensor
    ) -> None:
        self._detect_fused_expert_format(name)
        if not self.has_expert_mapping:
            self._dispatch_generic(orig_ckpt_name, name, tensor)
            return
        if self._dispatch_fused_expert(name, tensor):
            return
        matched, name = self._dispatch_per_expert(name, tensor)
        if not matched:
            self._dispatch_merged_or_generic(orig_ckpt_name, name, tensor)

    def _detect_fused_expert_format(self, name: str) -> None:
        if self.detect_fused_expert_fn is None or self._is_fused_expert:
            return
        self._is_fused_expert = self.detect_fused_expert_fn(name)
        if self._is_fused_expert and self.get_fused_expert_mapping_fn is not None:
            self._fused_expert_params_mapping = self.get_fused_expert_mapping_fn()

    def _dispatch_fused_expert(self, name: str, tensor: torch.Tensor) -> bool:
        """All routed experts of a layer arriving as one stacked tensor."""
        if not (
            self._is_fused_expert
            and self.load_fused_expert_weights_fn is not None
            and self._fused_expert_params_mapping
        ):
            return False
        for mapping_entry in self._fused_expert_params_mapping:
            param_name, weight_name, shard_id = mapping_entry[:3]
            if weight_name not in name:
                continue
            name_mapped = name.replace(weight_name, param_name)
            if name_mapped not in self.params_dict:
                continue

            # Writes the routed experts straight into the fused parameter, so
            # the staging pool must not also own it -- see ExpertStagingPool's
            # ownership rule.
            self.staging_pool.decline(self.params_dict[name_mapped])
            if self.on_fused_param is not None:
                self.on_fused_param(self.params_dict[name_mapped])

            # Generic call - model provides implementation details
            num_experts = getattr(self.hf_config, "n_routed_experts", 0) or getattr(
                self.hf_config, "num_experts", 0
            )
            if self.load_fused_expert_weights_fn(
                name,  # Original checkpoint name
                name_mapped,  # Mapped parameter name
                self.params_dict,
                tensor,
                shard_id,
                num_experts,
            ):
                self._record(name_mapped)
                return True
        return False

    def _dispatch_per_expert(self, name: str, tensor: torch.Tensor) -> tuple[bool, str]:
        """One tensor per (expert, shard); the batched staging path.

        Returns the rewritten name alongside the verdict: the expert prefix is
        substituted in place, and the merged/generic fallbacks below work on
        the substituted name.
        """
        for wm_name in self.expert_weight_prefixes:
            if wm_name not in name:
                continue
            pm_name, expert_id, shard_id = self.expert_index[wm_name]
            name = name.replace(wm_name, pm_name)
            if name.endswith((".bias", "_bias")) and name not in self.params_dict:
                return True, name
            if "mtp" in name and not self.spec_decode:
                return True, name
            param = self.params_dict.get(name)
            if param is None:
                # Parameter absent from model (e.g. weight scales for an
                # unquantized drafter MTP block); skip silently.
                return True, name
            batching_excluded = (
                self.batching_excluded is not None and self.batching_excluded(param)
            )
            if (
                self.batching_enabled
                and not batching_excluded
                and self.staging_pool.is_batchable(param, name)
            ):
                self.submit(
                    self.staging_pool.stage, param, name, shard_id, expert_id, tensor
                )
            else:
                self.submit(
                    param.weight_loader, param, tensor, name, shard_id, expert_id
                )
            self._record(name)
            return True, name
        return False, name

    def _dispatch_merged_or_generic(
        self, orig_ckpt_name: str, name: str, tensor: torch.Tensor
    ) -> None:
        if "mtp" in name and not self.spec_decode:
            return
        if merged_target := extract_expert_target_and_id(name):
            fused_name, expert_id = merged_target
            param = self._parameter(orig_ckpt_name, fused_name)
            if param is None:
                return
            # Merged loader writes expert slots directly; same ownership rule
            # as the fused path above.
            self.staging_pool.decline(param)
            weight_loader = getattr(param, "weight_loader", self.default_weight_loader)
            self.submit(
                weight_loader,
                param,
                tensor,
                "",  # use merged moe loader
                "",
                expert_id,
            )
            self._record(fused_name)
            return
        self._dispatch_generic(orig_ckpt_name, name, tensor)

    def _dispatch_generic(
        self, orig_ckpt_name: str, name: str, tensor: torch.Tensor
    ) -> None:
        param = self._parameter(orig_ckpt_name, name)
        if param is None:
            return
        weight_loader = getattr(param, "weight_loader", self.default_weight_loader)
        self.submit(weight_loader, param, tensor)
        self._record(name)

    # ── helpers ───────────────────────────────────────────────────────────

    def _parameter(self, orig_ckpt_name: str, param_name: str) -> nn.Parameter | None:
        try:
            return self.model.get_parameter(param_name)
        except AttributeError:
            self.dropped_ckpt_keys.append((orig_ckpt_name, param_name))
            return None

    def _record(self, param_name: str) -> None:
        self.loaded_weights_record.add(self.prefix + param_name)
