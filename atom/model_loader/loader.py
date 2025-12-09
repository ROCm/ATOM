import concurrent.futures
import os
import re
from glob import glob
from typing import Generator, List, Tuple

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
from atom.model_ops.base_config import QuantizeMethodBase
from atom.model_ops.moe import is_rocm_aiter_fusion_shared_expert_enabled
from aiter.dist.parallel_state import get_tp_group


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


def load_model(
    model: nn.Module,
    model_name_or_path: str,
    hf_config: AutoConfig,
    load_dummy: bool = False,
):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    weights_mapping = getattr(model, "weights_mapping", {})
    params_dict = dict(model.named_parameters())
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for name, weight_tensor in safetensors_weights_iterator(model_name_or_path):
            if load_dummy:
                continue
            if name.endswith("kv_scale"):
                continue
            name_suffix = name.split(".")[-1]
            if name_suffix in weights_mapping.keys():
                name = name.replace(name_suffix, weights_mapping[name_suffix])
            if "weight_scale_inv" in name:
                name = name.replace("weight_scale_inv", "weight_scale")

            layerId_ = re.search(r"model\.layers\.(\d+)\.", name)
            layerId = int(layerId_.group(1)) if layerId_ else 0
            if hf_config.num_hidden_layers and layerId >= hf_config.num_hidden_layers:
                # print(f"Skipping loading {name} as layerId {layerId} >= num_hidden_layers {hf_config.num_hidden_layers}")
                continue
            if (
                is_rocm_aiter_fusion_shared_expert_enabled()
                and "mlp.shared_experts" in name
            ):
                name = name.replace(
                    "mlp.shared_experts",
                    f"mlp.experts.{hf_config.n_routed_experts}",
                )
            for k in packed_modules_mapping:
                # We handle the experts below in expert_params_mapping
                if "mlp.experts." in name and name not in params_dict:
                    continue
                if k in name:
                    v, shard_id = packed_modules_mapping[k]
                    param_name = name.replace(k, v)
                    if ("output_scale" not in param_name):
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader")
                        # weight_loader(param, weight_tensor, shard_id)
                        futures.append(
                            executor.submit(weight_loader, param, weight_tensor, shard_id)
                        )
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
                        # weight_loader(
                        #     param,
                        #     weight_tensor,
                        #     name,
                        #     shard_id=shard_id,
                        #     expert_id=expert_id,
                        # )
                        break
                    else:
                        param = model.get_parameter(name)
                        weight_loader = getattr(
                            param, "weight_loader", default_weight_loader
                        )
                        futures.append(
                            executor.submit(weight_loader, param, weight_tensor)
                        )
                        # weight_loader(param, weight_tensor)
                else:
                    # Model doesn't have expert mapping, use generic loading
                    param = model.get_parameter(name)
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    # weight_loader(param, weight_tensor)
                    futures.append(executor.submit(weight_loader, param, weight_tensor))
        # Wait for all tasks to complete and raise any exceptions.
        for future in concurrent.futures.as_completed(futures):
            future.result()
    for _, module in model.named_modules():
        if hasattr(module, "process_weights_after_loading"):
            module.process_weights_after_loading()
        quant_method = getattr(module, "quant_method", None)
        if isinstance(quant_method, QuantizeMethodBase):
            quant_method.process_weights_after_loading(module)
