# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import concurrent.futures
import os
import logging
import re
from glob import glob
from typing import Generator, Tuple

import safetensors
import torch
from torch import nn
from tqdm import tqdm
from transformers import AutoConfig
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
    loaded_weights_record = load_model(
        model=model,
        model_name_or_path=model_name_or_path,
        hf_config=config.hf_config,
        load_dummy=config.load_dummy,
        spec_decode=False,
        prefix=prefix,
        is_plugin_mode=True,
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
):
    def have_shared_expert(name):
        maybe_matching_list = ["mlp.shared_experts.", "mlp.shared_expert."]
        for maybe_matching_name in maybe_matching_list:
            if maybe_matching_name in name:
                return maybe_matching_name
        return None

    # need to record the loaded weight name for vllm load check
    # it is only used in plugin mode for vllm
    loaded_weights_record: set[str] = set()

    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    weights_mapping = getattr(model, "weights_mapping", {})
    params_dict = dict(model.named_parameters())
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        disable_mmap = os.environ.get("ATOM_DISABLE_MMAP", "false").lower() == "true"
        for name, weight_tensor in safetensors_weights_iterator(
            model_name_or_path, disable_mmap=disable_mmap
        ):
            if load_dummy:
                continue
            if name.endswith("kv_scale") or "inv_freq" in name:
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
            name_suffix = name.split(".")[-1]
            if name_suffix in weights_mapping.keys():
                name = name.replace(name_suffix, weights_mapping[name_suffix])
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
                if "mtp" in name and not spec_decode:
                    continue
                if k in name:
                    v, shard_id = packed_modules_mapping[k]
                    param_name = name.replace(k, v)
                    # FIXME output_scale has a value, so accuracy is incorrect. this should be loaded and used in llfp4.
                    if "output_scale" not in param_name:
                        param = model.get_parameter(param_name)
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
                # Check if model has expert mapping before processing
                if hasattr(model, "get_expert_mapping"):
                    for k in model.get_expert_mapping():
                        param_name, weight_name, expert_id, shard_id = k
                        if weight_name not in name:
                            continue
                        name = name.replace(weight_name, param_name)
                        if (
                            name.endswith(".bias") or name.endswith("_bias")
                        ) and name not in dict(model.named_parameters()):
                            continue
                        if "mtp" in name and not spec_decode:
                            continue
                        param = model.get_parameter(name)
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
                        # weight_loader(
                        #     param,
                        #     weight_tensor,
                        #     name,
                        #     shard_id=shard_id,
                        #     expert_id=expert_id,
                        # )
                        break
                    else:
                        if "mtp" in name and not spec_decode:
                            continue
                        param = model.get_parameter(name)
                        weight_loader = getattr(
                            param, "weight_loader", default_weight_loader
                        )
                        futures.append(
                            executor.submit(weight_loader, param, weight_tensor)
                        )
                        loaded_weights_record.add(prefix + name)
                        # weight_loader(param, weight_tensor)
                else:
                    # Model doesn't have expert mapping, use generic loading
                    param = model.get_parameter(name)
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    # weight_loader(param, weight_tensor)
                    futures.append(executor.submit(weight_loader, param, weight_tensor))
                    loaded_weights_record.add(prefix + name)
        # Wait for all tasks to complete and raise any exceptions.
        for future in concurrent.futures.as_completed(futures):
            future.result()
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
