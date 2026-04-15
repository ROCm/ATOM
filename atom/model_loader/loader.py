# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import concurrent.futures
import os
import logging
import re
from glob import glob
from typing import Generator, Tuple
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import safetensors
import torch
from torch import nn
from tqdm import tqdm
from transformers import AutoConfig

from atom.utils import envs
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME

from atom.model_loader.weight_utils import (
    download_weights_from_hf,
    filter_duplicate_safetensors_files,
)
from atom.models.deepseek_mtp import (
    get_spec_layer_idx_from_weight_name,
    rewrite_spec_layer_name,
)
from atom.model_ops.base_config import QuantizeMethodBase
from atom.model_ops.moe import (
    FusedMoEMethodBase,
    is_rocm_aiter_fusion_shared_expert_enabled,
)
from aiter.dist.parallel_state import get_tp_group
from atom.models.qwen3_next_mtp import remap_mtp_weight_name

from atom.plugin.prepare import is_sglang

logger = logging.getLogger("atom")


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


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    if loaded_weight.numel() == param.data.numel():
        param.data.copy_(loaded_weight)
    elif loaded_weight.numel() // get_tp_group().world_size == param.data.numel():
        loaded_weight_per_rank = loaded_weight.numel() // get_tp_group().world_size
        tp_rank_start = loaded_weight_per_rank * get_tp_group().rank
        tp_rank_end = tp_rank_start + loaded_weight_per_rank
        param.data.copy_(loaded_weight.view(-1)[tp_rank_start:tp_rank_end])


def safetensors_weights_iterator(
    model_name_or_path: str,
    disable_mmap: bool = False,
) -> Generator[Tuple[str, torch.Tensor], None, None]:
    """Iterate over the weights in the model safetensor files."""
    logger.info(f"disable_mmap: {disable_mmap}")
    path = (
        model_name_or_path
        if os.path.isdir(model_name_or_path)
        else download_weights_from_hf(
            model_name_or_path, None, ["*.safetensors"], ignore_patterns=["original/*"]
        )
    )
    hf_weights_files = filter_duplicate_safetensors_files(
        glob(os.path.join(path, "*.safetensors")), path, SAFE_WEIGHTS_INDEX_NAME
    )
    enable_tqdm = (
        not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
    )

    iters = tqdm(
        hf_weights_files,
        desc=f"Loading safetensors shards[{model_name_or_path}]",
        disable=not enable_tqdm,
    )
    for st_file in iters:
        # Advise kernel for sequential read-ahead (mmap optimization)
        if not disable_mmap and hasattr(os, "posix_fadvise"):
            try:
                fd = os.open(st_file, os.O_RDONLY)
                file_size = os.fstat(fd).st_size
                os.posix_fadvise(
                    fd,
                    0,
                    file_size,
                    os.POSIX_FADV_SEQUENTIAL | os.POSIX_FADV_WILLNEED,
                )
                os.close(fd)
            except OSError:
                pass

        if disable_mmap:
            with open(st_file, "rb") as f:
                result = safetensors.torch.load(f.read())
                for name, param in result.items():
                    yield name, param
        else:
            with safetensors.safe_open(st_file, framework="pt", device="cpu") as f:
                for name in f.keys():
                    yield name, f.get_tensor(name)


# when plugin mode, model loader method is bind to model implementation
# thus call this interface to load the model, which leverages the load_model
# method
def load_model_in_plugin_mode(
    model,
    config,
    prefix: str = "",
    weights_mapper: WeightsMapper | None = None,
    load_fused_expert_weights_fn=None,
) -> set[str]:

    # during loading model, the outplace operation may consume more
    # GPU mem, which cached in torch caching allocator, here actively
    # call empty cache to free the extra reserved but not used memory
    def _empty_cache():
        import gc

        gc.collect()
        torch.cuda.empty_cache()

    assert (
        config.plugin_config is not None and config.plugin_config.is_plugin_mode
    ), "ATOM is not running in plugin mode"
    if config.plugin_config.is_vllm:
        model_name_or_path = config.plugin_config.model_config.model
    elif config.plugin_config.is_sglang:
        model_name_or_path = config.plugin_config.model_config.model_path

    _empty_cache()
    config_for_loading = (
        config.hf_config.text_config
        if hasattr(config.hf_config, "text_config")
        else config.hf_config
    )
    loaded_weights_record = load_model(
        model=model,
        model_name_or_path=model_name_or_path,
        hf_config=config_for_loading,
        load_dummy=config.load_dummy,
        spec_decode=False,
        prefix=prefix,
        is_plugin_mode=True,
        weights_mapper=weights_mapper,
        load_fused_expert_weights_fn=load_fused_expert_weights_fn,
    )
    _empty_cache()
    return loaded_weights_record


def load_model(
    model: nn.Module,
    model_name_or_path: str,
    hf_config: AutoConfig,
    load_dummy: bool = False,
    spec_decode: bool = False,
    prefix: str = "",
    is_plugin_mode: bool = False,
    weights_mapper: WeightsMapper | None = None,
    load_fused_expert_weights_fn=None,
):
    def have_shared_expert(name):
        maybe_matching_list = ["mlp.shared_experts.", "mlp.shared_expert."]
        for maybe_matching_name in maybe_matching_list:
            if maybe_matching_name in name:
                return maybe_matching_name
        return None

    def extract_expert_target_and_id(name: str) -> Tuple[str, int] | None:
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

    # need to record the loaded weight name for vllm load check
    # it is only used in plugin mode for vllm
    loaded_weights_record: set[str] = set()

    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    weights_mapping = getattr(model, "weights_mapping", {})
    skip_weight_prefixes = getattr(model, "skip_weight_prefixes", [])
    params_dict = dict(model.named_parameters())

    # Pre-index expert_mapping by weight_name_part for O(1) lookup.
    # Original code does O(N) scan of expert_mapping (768 entries) per tensor,
    # causing ~19s of CPU time for 90k expert tensors. This reduces it to O(1).
    has_expert_mapping = hasattr(model, "get_expert_mapping")
    expert_index = {}  # {weight_name_part: (param_name_part, expert_id, shard_id)}
    expert_weight_prefixes = []  # sorted longest-first for prefix matching
    if has_expert_mapping:
        for (
            param_name_part,
            weight_name_part,
            expert_id,
            shard_id,
        ) in model.get_expert_mapping():
            expert_index[weight_name_part] = (param_name_part, expert_id, shard_id)
        # Sort by length descending so longer (more specific) prefixes match first
        expert_weight_prefixes = sorted(expert_index.keys(), key=len, reverse=True)

    # Get fused expert mapping from model if it provides one
    is_fused_expert = False
    fused_expert_params_mapping = []
    detect_fused_expert_fn = getattr(model, "detect_fused_expert_format", None)
    get_fused_expert_mapping_fn = getattr(model, "get_fused_expert_mapping", None)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        disable_mmap = envs.ATOM_DISABLE_MMAP
        for name, weight_tensor in safetensors_weights_iterator(
            model_name_or_path, disable_mmap=disable_mmap
        ):
            if weights_mapper is not None:
                mapped_name = weights_mapper._map_name(name)
                if mapped_name is None:
                    continue
                name = mapped_name
            if load_dummy:
                continue
            if "mtp" in name and not spec_decode:
                continue
            if name.endswith("kv_scale") or "inv_freq" in name:
                continue
            # Skip weights matching model-defined prefixes (e.g. vision encoder
            # weights in multimodal checkpoints that are not needed for text-only
            # inference).
            if skip_weight_prefixes and any(
                name.startswith(p) for p in skip_weight_prefixes
            ):
                continue
            if spec_decode:
                if hf_config.model_type == "deepseek_mtp":
                    spec_layer = get_spec_layer_idx_from_weight_name(hf_config, name)
                    if spec_layer is None:
                        continue
                    name = rewrite_spec_layer_name(spec_layer, name)
                elif hf_config.model_type == "qwen3_next_mtp":
                    remapped_name = remap_mtp_weight_name(name)
                    if remapped_name is None:
                        continue
                    name = remapped_name
            for mapping_part in weights_mapping.keys():
                if mapping_part in name:
                    name = name.replace(mapping_part, weights_mapping[mapping_part])
            if "weight_scale_inv" in name:
                name = name.replace("weight_scale_inv", "weight_scale")

            layerId_ = re.search(r"model\.layers\.(\d+)\.", name)
            layerId = int(layerId_.group(1)) if layerId_ else 0
            if (
                hf_config.num_hidden_layers
                and layerId >= hf_config.num_hidden_layers
                and not spec_decode
            ):
                continue
            maybe_matching_name = have_shared_expert(name)
            if (
                is_rocm_aiter_fusion_shared_expert_enabled()
                and maybe_matching_name is not None
            ):
                name = name.replace(
                    maybe_matching_name,
                    f"mlp.experts.{hf_config.n_routed_experts}.",
                )
            for k in packed_modules_mapping:
                # We handle the experts below in expert_params_mapping
                if "mlp.experts." in name and name not in params_dict:
                    continue
                if k in name:
                    packed_value = packed_modules_mapping[k]
                    # Handle both tuple (fuse parameter) and list (shard parameter)
                    if isinstance(packed_value, list):
                        # Checkpoint has fused weight, split into separate params
                        for shard_idx, target_name in enumerate(packed_value):
                            param_name = name.replace(k, target_name)
                            if "output_scale" not in param_name:
                                try:
                                    param = model.get_parameter(param_name)
                                except AttributeError:
                                    continue
                                weight_loader = getattr(param, "weight_loader")
                                futures.append(
                                    executor.submit(
                                        weight_loader, param, weight_tensor, shard_idx
                                    )
                                )
                                loaded_weights_record.add(prefix + param_name)
                    else:
                        # Checkpoint has separate weights, load into fused param
                        v, shard_id = packed_value
                        param_name = name.replace(k, v)
                        # FIXME output_scale has a value, so accuracy is incorrect. this should be loaded and used in llfp4.
                        if "output_scale" not in param_name:
                            try:
                                param = model.get_parameter(param_name)
                            except AttributeError:
                                break
                            weight_loader = getattr(param, "weight_loader")
                            # weight_loader(param, weight_tensor, shard_id)
                            futures.append(
                                executor.submit(
                                    weight_loader, param, weight_tensor, shard_id
                                )
                            )
                            loaded_weights_record.add(prefix + param_name)
                    break
            else:
                # Detect fused expert format if model provides detection function
                if detect_fused_expert_fn is not None and not is_fused_expert:
                    is_fused_expert = detect_fused_expert_fn(name)
                    if is_fused_expert and get_fused_expert_mapping_fn is not None:
                        fused_expert_params_mapping = get_fused_expert_mapping_fn()

                # Check if model has expert mapping before processing
                if has_expert_mapping:
                    # Handle fused expert format
                    # Model-specific detection and handling via callback functions
                    if (
                        is_fused_expert
                        and load_fused_expert_weights_fn is not None
                        and fused_expert_params_mapping
                    ):
                        matched = False
                        for mapping_entry in fused_expert_params_mapping:
                            param_name, weight_name, shard_id = mapping_entry[:3]
                            if weight_name not in name:
                                continue
                            name_mapped = name.replace(weight_name, param_name)
                            if name_mapped not in params_dict:
                                continue

                            # Generic call - model provides implementation details
                            num_experts = getattr(
                                hf_config, "n_routed_experts", 0
                            ) or getattr(hf_config, "num_experts", 0)
                            matched = load_fused_expert_weights_fn(
                                name,  # Original checkpoint name
                                name_mapped,  # Mapped parameter name
                                params_dict,
                                weight_tensor,
                                shard_id,
                                num_experts,
                            )

                            if matched:
                                loaded_weights_record.add(prefix + name)
                                break

                        if matched:
                            continue

                    matched = False
                    for wm_name in expert_weight_prefixes:
                        if wm_name not in name:
                            continue
                        pm_name, expert_id, shard_id = expert_index[wm_name]
                        name = name.replace(wm_name, pm_name)
                        if (
                            name.endswith(".bias") or name.endswith("_bias")
                        ) and name not in params_dict:
                            matched = True
                            break
                        if "mtp" in name and not spec_decode:
                            matched = True
                            break
                        try:
                            param = model.get_parameter(name)
                        except AttributeError:
                            # Parameter absent from model (e.g. weight scales for
                            # an unquantized drafter MTP block); skip silently.
                            matched = True
                            break
                        weight_loader = getattr(param, "weight_loader")
                        futures.append(
                            executor.submit(
                                weight_loader,
                                param,
                                weight_tensor,
                                name,
                                shard_id,
                                expert_id,
                            )
                        )
                        loaded_weights_record.add(prefix + name)
                        matched = True
                        break
                    if not matched:
                        if "mtp" in name and not spec_decode:
                            continue
                        if merged_target := extract_expert_target_and_id(name):
                            fused_name, expert_id = merged_target
                            try:
                                param = model.get_parameter(fused_name)
                            except AttributeError:
                                continue
                            weight_loader = getattr(
                                param, "weight_loader", default_weight_loader
                            )
                            futures.append(
                                executor.submit(
                                    weight_loader,
                                    param,
                                    weight_tensor,
                                    "",  # use merged moe loader
                                    "",
                                    expert_id,
                                )
                            )
                            loaded_weights_record.add(prefix + name)
                        try:
                            param = model.get_parameter(name)
                        except AttributeError:
                            continue
                        weight_loader = getattr(
                            param, "weight_loader", default_weight_loader
                        )
                        futures.append(
                            executor.submit(weight_loader, param, weight_tensor)
                        )
                        loaded_weights_record.add(prefix + name)
                else:
                    # Model doesn't have expert mapping, use generic loading
                    try:
                        param = model.get_parameter(name)
                    except AttributeError:
                        continue
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    # weight_loader(param, weight_tensor)
                    futures.append(executor.submit(weight_loader, param, weight_tensor))
                    loaded_weights_record.add(prefix + name)
        # Wait for all tasks to complete and raise any exceptions.
        for future in concurrent.futures.as_completed(futures):
            future.result()

    # Avoid holding stale Parameter refs that prevent storage release.
    del params_dict

    for _, module in model.named_modules():
        if hasattr(module, "process_weights_after_loading"):
            module.process_weights_after_loading()
        quant_method = getattr(module, "quant_method", None)

        # when running plugin mode for sglang, don't do the post process here
        # since sglang will call this func automatically after finishing loading
        if isinstance(quant_method, QuantizeMethodBase) and not is_sglang():
            quant_method.process_weights_after_loading(module)
        if isinstance(quant_method, FusedMoEMethodBase):
            quant_method.init_prepare_finalize(module)

    if is_plugin_mode:
        return loaded_weights_record
