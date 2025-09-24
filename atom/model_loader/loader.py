import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME
from atom.model_loader.weight_utils import download_weights_from_hf, filter_duplicate_safetensors_files
from atom.model_ops.base_config import QuantizeMethodBase
from tqdm import tqdm


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, model_name_or_path: str):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    is_path = os.path.isdir(model_name_or_path)
    path = (
        model_name_or_path
        if is_path
        else download_weights_from_hf(
            model_name_or_path, None, ["*.safetensors"], ignore_patterns=["original/*"]
        )
    )
    hf_weights_files = filter_duplicate_safetensors_files(glob(os.path.join(path, "*.safetensors")), path, SAFE_WEIGHTS_INDEX_NAME)
    for file in tqdm(hf_weights_files):
        print(f"Loading weights from {file}")
        with safe_open(file, "pt", "cpu") as f:
            for name in f.keys():
                if  name.endswith(
                    "kv_scale"
                ):
                    continue
                weight_tensor = f.get_tensor(name)
                if "weight_scale_inv" in name:
                    name = name.replace(
                        "weight_scale_inv", "weight_scale"
                    )
                for k in packed_modules_mapping:
                    if k in name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = name.replace(k, v)
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, weight_tensor, shard_id)
                        break
                else:
                    # Check if model has expert mapping before processing
                    if hasattr(model, "get_expert_mapping"):
                        for k in model.get_expert_mapping():
                            param_name, weight_name, expert_id, shard_id = k
                            if weight_name not in name:
                                continue
                            name = name.replace(weight_name, param_name)
                            if ((name.endswith(".bias") or name.endswith("_bias"))
                                    and name not in dict(model.named_parameters())):
                                continue
                            param = model.get_parameter(name)
                            weight_loader = getattr(param, "weight_loader")
                            weight_loader(param,
                                        weight_tensor,
                                        name,
                                        shard_id=shard_id,
                                        expert_id=expert_id)
                            break
                        else:
                            param = model.get_parameter(name)
                            weight_loader = getattr(
                                param, "weight_loader", default_weight_loader
                            )
                            weight_loader(param, weight_tensor)
                    else:
                        # Model doesn't have expert mapping, use generic loading
                        param = model.get_parameter(name)
                        weight_loader = getattr(
                            param, "weight_loader", default_weight_loader
                        )
                        weight_loader(param, weight_tensor)
    for _, module in model.named_modules():
        if hasattr(module, "process_weights_after_loading"):
            module.process_weights_after_loading()
        quant_method = getattr(module, "quant_method", None)
        if isinstance(quant_method, QuantizeMethodBase):
            quant_method.process_weights_after_loading(module)
