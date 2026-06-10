# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import concurrent.futures
import json
import os
import logging
import re
import time
from glob import glob
from typing import Generator, Tuple
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import safetensors
import safetensors.torch
import torch
from torch import nn
from tqdm import tqdm
from transformers import AutoConfig

# safetensors<=0.7.0 ships a Python `_TYPES` dict missing the `F8_E8M0`
# (MX scale) entry, even though both torch and the safetensors-rust binary
# support it. The mmap'd `safe_open` path goes through Rust and works, but
# the `safetensors.torch.load(bytes)` path used when `ATOM_DISABLE_MMAP=true`
# raises `KeyError: 'F8_E8M0'` on DeepSeek-V4-Pro shards. Register the
# missing dtype string so both paths behave identically.
if "F8_E8M0" not in safetensors.torch._TYPES and hasattr(torch, "float8_e8m0fnu"):
    safetensors.torch._TYPES["F8_E8M0"] = torch.float8_e8m0fnu

from atom.utils import envs
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME

from atom.model_loader.weight_utils import (
    download_weights_from_hf,
    filter_duplicate_safetensors_files,
)
from atom.model_ops.base_config import QuantizeMethodBase
from atom.model_ops.moe import FusedMoEMethodBase
from atom.model_ops.topK import (
    is_rocm_aiter_fusion_shared_expert_enabled,
    is_rocm_aiter_fusion_shared_expert_enabled_for_quant_config,
)
from aiter.dist.parallel_state import get_tp_group

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
    else:
        # Shape mismatch we cannot resolve — leaving the destination at its init
        # value is almost always a bug. The post-load check in load_model() will
        # catch this and warn (param will be in `unloaded` set since this loader
        # never wrote to it). Raise here so the failure is loud at copy time
        # too, instead of being masked by the default ones-init of RMSNorm etc.
        raise RuntimeError(
            f"default_weight_loader: shape mismatch — param={tuple(param.shape)} "
            f"loaded={tuple(loaded_weight.shape)}. Cannot copy."
        )


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
    spec_decode: bool = False,
    hf_config_override: AutoConfig | None = None,
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
    if hf_config_override is not None:
        config_for_loading = getattr(
            hf_config_override, "hf_config", hf_config_override
        )
        if hasattr(config_for_loading, "text_config"):
            config_for_loading = config_for_loading.text_config
    else:
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
        spec_decode=spec_decode,
        prefix=prefix,
        is_plugin_mode=True,
        weights_mapper=weights_mapper,
        load_fused_expert_weights_fn=load_fused_expert_weights_fn,
    )
    _empty_cache()
    return loaded_weights_record


def _save_online_quant_info(
    oq_layers: list[dict],
    model_name_or_path: str,
    elapsed_seconds: float,
    online_quant_config: dict,
):
    """Save online quantization info to a JSON file (rank 0 only)."""
    if get_tp_group().rank_in_group != 0:
        return
    output_dir = envs.ATOM_TORCH_PROFILER_DIR or os.getcwd()
    os.makedirs(output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    timestamp_ns = time.time_ns() % 1_000_000_000
    filepath = os.path.join(
        output_dir, f"online_quant_info_{timestamp}_{timestamp_ns:09d}.json"
    )

    payload = {
        "model": model_name_or_path,
        "online_quant_config": online_quant_config,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "num_layers": len(oq_layers),
        "layers": oq_layers,
    }
    with open(filepath, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    logger.info("Online quantization info saved to %s", filepath)


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
        # Match both `mlp.` (GLM4, Qwen, ...) and `ffn.` (DeepSeek-V4) module
        # naming. The matched substring is replaced by the caller with
        # `<prefix>experts.{n_routed}.` so the shared expert lands in the fused
        # MoE buffer's extra slot. Returning the full prefix (incl. mlp./ffn.)
        # lets the rewrite preserve the module-naming style.
        maybe_matching_list = [
            "mlp.shared_experts.",
            "mlp.shared_expert.",
            "ffn.shared_experts.",
            "ffn.shared_expert.",
        ]
        for maybe_matching_name in maybe_matching_list:
            if maybe_matching_name in name:
                return maybe_matching_name
        return None

    def should_fuse_shared_expert_weight(name: str, matching_name: str) -> bool:
        layer_prefix = name.split(matching_name, 1)[0]
        module_prefix = matching_name.split("shared_expert", 1)[0]
        shared_expert_prefix = layer_prefix + matching_name.rstrip(".")
        routed_expert_prefix = layer_prefix + f"{module_prefix}experts"
        model_quant_config = getattr(
            getattr(model, "atom_config", None), "quant_config", None
        )
        if model_quant_config is None:
            model_quant_config = getattr(model, "quant_config", None)
        if model_quant_config is not None and hasattr(
            model_quant_config, "get_layer_quant_config"
        ):
            return is_rocm_aiter_fusion_shared_expert_enabled_for_quant_config(
                model_quant_config,
                shared_expert_prefix=shared_expert_prefix,
                routed_expert_prefix=routed_expert_prefix,
            )
        return is_rocm_aiter_fusion_shared_expert_enabled(
            shared_expert_prefix=shared_expert_prefix,
            routed_expert_prefix=routed_expert_prefix,
        )

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

    # Auto-detect weight mapper from model if not provided explicitly
    if weights_mapper is None:
        model_mapper = getattr(model, "hf_to_atom_mapper", None)
        if isinstance(model_mapper, dict):
            weights_mapper = WeightsMapper(orig_to_new_prefix=model_mapper)
        elif isinstance(model_mapper, WeightsMapper):
            weights_mapper = model_mapper

    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    weights_mapping = getattr(model, "weights_mapping", {})
    skip_weight_prefixes = getattr(model, "skip_weight_prefixes", [])
    mtp_remap = getattr(model, "remap_mtp_weight_name", None)
    # Models can also expose a `weights_mapper` (WeightsMapper instance) for
    # precise prefix/suffix-anchored renames that the dumb substring-substitution
    # `weights_mapping` dict cannot express safely. If both are set they are
    # composed: weights_mapper applies first, then the legacy substring map.
    if weights_mapper is None:
        weights_mapper = getattr(model, "weights_mapper", None)
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

    # Track ckpt names that were silently dropped at `get_parameter`
    # AttributeError sites — these indicate weights_mapping bugs where the
    # rewritten name doesn't correspond to any model param. (orig, mapped) pairs.
    dropped_ckpt_keys: list[tuple[str, str]] = []

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
    use_threadpool = envs.ATOM_LOADER_USE_THREADPOOL
    if use_threadpool:
        executor = concurrent.futures.ThreadPoolExecutor()
    else:
        executor = None
    futures = []

    def _submit(fn, *args):
        if executor is not None:
            futures.append(executor.submit(fn, *args))
        else:
            fn(*args)

    try:
        disable_mmap = envs.ATOM_DISABLE_MMAP
        for name, weight_tensor in safetensors_weights_iterator(
            model_name_or_path, disable_mmap=disable_mmap
        ):
            _orig_ckpt_name = name  # preserve for ckpt-side coverage report
            if weights_mapper is not None:
                mapped_name = weights_mapper._map_name(name)
                if mapped_name is None:
                    continue
                name = mapped_name
            if load_dummy:
                continue
            # Draft models may remap ckpt-side `mtp.*` entries into params
            # whose names do not themselves contain `mtp` (e.g. Qwen3.5 MTP
            # rewrites `mtp.*` -> `model.*`). Gate only on `spec_decode`,
            # otherwise we can drop the entire drafter checkpoint before the
            # model-specific remap logic has a chance to run.
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
            if spec_decode and mtp_remap is not None:
                remapped = mtp_remap(name)
                if remapped is None:
                    continue
                name = remapped
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
                maybe_matching_name is not None
                # When the model keeps shared experts unfused (e.g. V4-Pro with
                # FP4 routed vs FP8 shared, or DP + mori all2all), do NOT rewrite
                # the shared weights into the fused slot — they must load into the
                # standalone Expert module. Stays True for models without this
                # attr (GLM4 etc.) so their fused-shared path is unchanged.
                and not getattr(model, "disable_fused_shared_loading", False)
                and should_fuse_shared_expert_weight(name, maybe_matching_name)
            ):
                # Preserve the module-naming prefix (mlp. / ffn.) so the rewritten
                # name matches this model's routed-expert param naming.
                module_prefix = maybe_matching_name.split("shared_expert", 1)[0]
                name = name.replace(
                    maybe_matching_name,
                    f"{module_prefix}experts.{hf_config.n_routed_experts}.",
                )
            for k in packed_modules_mapping:
                # We handle the experts below in expert_params_mapping
                if (
                    "mlp.experts." in name or "ffn.experts." in name
                ) and name not in params_dict:
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
                                    dropped_ckpt_keys.append(
                                        (_orig_ckpt_name, param_name)
                                    )
                                    continue
                                weight_loader = getattr(param, "weight_loader")
                                _submit(weight_loader, param, weight_tensor, shard_idx)
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
                                dropped_ckpt_keys.append((_orig_ckpt_name, param_name))
                                break
                            weight_loader = getattr(param, "weight_loader")
                            _submit(weight_loader, param, weight_tensor, shard_id)
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
                        _submit(
                            weight_loader,
                            param,
                            weight_tensor,
                            name,
                            shard_id,
                            expert_id,
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
                                dropped_ckpt_keys.append((_orig_ckpt_name, fused_name))
                                continue
                            weight_loader = getattr(
                                param, "weight_loader", default_weight_loader
                            )
                            _submit(
                                weight_loader,
                                param,
                                weight_tensor,
                                "",  # use merged moe loader
                                "",
                                expert_id,
                            )
                            loaded_weights_record.add(prefix + fused_name)
                            continue
                        try:
                            param = model.get_parameter(name)
                        except AttributeError:
                            dropped_ckpt_keys.append((_orig_ckpt_name, name))
                            continue
                        weight_loader = getattr(
                            param, "weight_loader", default_weight_loader
                        )
                        _submit(weight_loader, param, weight_tensor)
                        loaded_weights_record.add(prefix + name)
                else:
                    # Model doesn't have expert mapping, use generic loading
                    try:
                        param = model.get_parameter(name)
                    except AttributeError:
                        dropped_ckpt_keys.append((_orig_ckpt_name, name))
                        continue
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    _submit(weight_loader, param, weight_tensor)
                    loaded_weights_record.add(prefix + name)
    finally:
        if executor is not None:
            concurrent.futures.wait(futures)
            executor.shutdown(wait=True)

    # Verify every model parameter actually got loaded from the checkpoint.
    # Without this check, weights_mapping bugs (e.g. a substring rule
    # accidentally rewriting `attn_norm.weight` → `attn_model.norm.weight`)
    # silently leave the destination parameter at its init value (all-ones for
    # RMSNorm, all-zeros for newly-allocated buffers), corrupting forward
    # outputs in ways that are extremely hard to diagnose. WARN loudly here
    # so the failure surfaces at load time instead of at generation time.
    loaded_param_names = {
        n.removeprefix(prefix) if prefix else n for n in loaded_weights_record
    }
    expected_param_names = set(params_dict.keys())
    unloaded = sorted(expected_param_names - loaded_param_names)
    # Filter known-OK skips: post-load-derived params (e.g. FusedMoE shuffle
    # output buffers, weight_scale params merged from multiple checkpoint scales).
    # Heuristic: anything ending in `_shuffled`, `_packed`, etc. Conservative
    # default = report everything else.
    suppressed_suffixes = ("_shuffled", "_packed", "_meta_for_quant", "weight_scale_2")
    truly_unloaded = [
        n for n in unloaded if not any(n.endswith(s) for s in suppressed_suffixes)
    ]
    if truly_unloaded:
        # Only report from rank 0 (other ranks have the same view).
        try:
            _is_rank0 = get_tp_group().rank == 0
        except Exception:
            _is_rank0 = True
        if _is_rank0:
            sample = truly_unloaded[:20]
            logger.warning(
                "load_model: %d/%d model parameters were NOT loaded from "
                "checkpoint and remain at their init values. This is almost "
                "always a bug (typically a `weights_mapping` substring rule "
                "that accidentally renames a param to something the model "
                "doesn't have). Fix the mapping or the on-disk → param name "
                "translation. First %d unloaded names: %s",
                len(truly_unloaded),
                len(expected_param_names),
                len(sample),
                sample,
            )

    # Reverse direction: ckpt names that were silently dropped by
    # `get_parameter` AttributeError. These are the actionable bug class —
    # the mapping rewrote the ckpt name to something the model has no slot for,
    # so legitimate ckpt data was thrown away. Filter known-benign families
    # (output_scale, kv_scale, etc.) so the warning is signal, not noise.
    if dropped_ckpt_keys:
        benign_substrings = (
            "output_scale",
            "kv_scale",
            "inv_freq",
            "weight_scale_2",
        )
        actionable_drops = [
            (orig, mapped)
            for orig, mapped in dropped_ckpt_keys
            if not any(s in orig or s in mapped for s in benign_substrings)
        ]
        try:
            _is_rank0 = get_tp_group().rank == 0
        except Exception:
            _is_rank0 = True
        if actionable_drops and _is_rank0:
            sample = actionable_drops[:20]
            logger.warning(
                "load_model: %d checkpoint tensors were silently dropped "
                "because the rewritten name has no matching model parameter. "
                "This is a `weights_mapping` / `WeightsMapper` bug — real "
                "ckpt data is being thrown away. Fix the rewrite rule. "
                "First %d (orig_ckpt_name → rewritten_name): %s",
                len(actionable_drops),
                len(sample),
                sample,
            )

    # Avoid holding stale Parameter refs that prevent storage release.
    del params_dict
    has_online_quant = any(
        getattr(m, "online_quant", False)
        or (
            getattr(m, "quant_config", None) is not None
            and getattr(m.quant_config, "online_quant", False)
        )
        for _, m in model.named_modules()
    )
    if has_online_quant:
        logger.info("Weight post-processing started (includes online quantization)")
    pp_start = time.perf_counter()

    for module_name, module in model.named_modules():
        if hasattr(module, "process_weights_after_loading"):
            module.process_weights_after_loading()
        quant_method = getattr(module, "quant_method", None)

        # when running plugin mode for sglang, don't do the post process here
        # since sglang will call this func automatically after finishing loading
        if isinstance(quant_method, QuantizeMethodBase) and not is_sglang():
            quant_method.process_weights_after_loading(module)
        if isinstance(quant_method, FusedMoEMethodBase):
            quant_method.init_prepare_finalize(module)

        # Online quantization creates new params (e.g. weight_scale) that are
        # not present in the source checkpoint. Record them as "loaded" so the
        # plugin host's strict weight tracking (e.g. vLLM's default loader)
        # does not flag them as uninitialized.
        if getattr(module, "_online_quant_info", None) is not None:
            for param_name, _ in module.named_parameters(recurse=False):
                full_name = f"{module_name}.{param_name}" if module_name else param_name
                loaded_weights_record.add(prefix + full_name)

    if has_online_quant:
        pp_elapsed = time.perf_counter() - pp_start
        oq_layers = []
        raw_online_quant_config = None
        for _, module in model.named_modules():
            info = getattr(module, "_online_quant_info", None)
            if info is not None:
                oq_layers.append(info)
            if raw_online_quant_config is None:
                qc = getattr(module, "quant_config", None)
                if qc is not None and hasattr(qc, "online_quant_config_raw"):
                    raw_online_quant_config = qc.online_quant_config_raw
        logger.info(
            "Weight post-processing done: %.2f seconds, " "%d layers online-quantized",
            pp_elapsed,
            len(oq_layers),
        )
        _save_online_quant_info(
            oq_layers,
            model_name_or_path,
            pp_elapsed,
            raw_online_quant_config or {},
        )

    return loaded_weights_record
