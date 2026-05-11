import torch

from atom.config import get_current_atom_config


def _init_vllm_layer_state(
    layer,
    *,
    layer_name: str,
    kv_cache_dtype: str,
    calculate_kv_scales: bool,
    quant_config,
) -> None:
    from vllm.model_executor.layers.attention.attention import _init_kv_cache_quant
    from vllm.utils.torch_utils import kv_cache_dtype_str_to_dtype

    atom_config = get_current_atom_config()
    vllm_config = atom_config.plugin_config.vllm_config

    layer.layer_name = layer_name
    layer.kv_cache_dtype = kv_cache_dtype
    layer.kv_cache_torch_dtype = kv_cache_dtype_str_to_dtype(
        kv_cache_dtype, vllm_config.model_config
    )
    layer.calculate_kv_scales = calculate_kv_scales
    layer.quant_config = quant_config
    layer.kv_cache = torch.tensor([])

    _init_kv_cache_quant(layer, quant_config, layer_name)


def _register_vllm_static_forward_context(layer) -> None:
    atom_config = get_current_atom_config()
    static_forward_context = (
        atom_config.plugin_config.vllm_config.compilation_config.static_forward_context
    )
    if layer.layer_name in static_forward_context:
        raise ValueError(f"Duplicate layer name: {layer.layer_name}")
    static_forward_context[layer.layer_name] = layer


def _set_default_scales(layer) -> None:
    from vllm.model_executor.layers.attention.attention import set_default_quant_scales

    set_default_quant_scales(layer, register_buffer=False)
    if hasattr(layer, "_o_scale_float"):
        layer._o_scale_float = None
