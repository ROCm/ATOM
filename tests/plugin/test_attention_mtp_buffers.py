# SPDX-License-Identifier: MIT

import importlib
import re
import sys
import types
from types import SimpleNamespace

import torch


def _install_attention_import_stubs(monkeypatch):
    aiter = types.ModuleType("aiter")
    aiter.__path__ = []
    aiter.dtypes = SimpleNamespace()
    aiter.get_mla_metadata_info_v1 = lambda *args, **kwargs: None
    aiter.get_mla_metadata_v1 = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "aiter", aiter)

    aiter_dist = types.ModuleType("aiter.dist")
    aiter_dist.__path__ = []
    monkeypatch.setitem(sys.modules, "aiter.dist", aiter_dist)
    aiter.dist = aiter_dist

    parallel_state = types.ModuleType("aiter.dist.parallel_state")
    parallel_state.get_tp_group = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "aiter.dist.parallel_state", parallel_state)
    aiter_dist.parallel_state = parallel_state

    atom = types.ModuleType("atom")
    atom.__path__ = []
    monkeypatch.setitem(sys.modules, "atom", atom)

    atom_config = types.ModuleType("atom.config")
    monkeypatch.setitem(sys.modules, "atom.config", atom_config)
    atom.config = atom_config
    monkeypatch.setattr(atom_config,
                        "get_current_atom_config",
                        lambda: None,
                        raising=False)

    atom_model_ops = types.ModuleType("atom.model_ops")
    atom_model_ops.__path__ = []
    monkeypatch.setitem(sys.modules, "atom.model_ops", atom_model_ops)
    atom.model_ops = atom_model_ops

    attention_mha = types.ModuleType("atom.model_ops.attention_mha")
    attention_mha.PagedAttentionImpl = object
    monkeypatch.setitem(sys.modules, "atom.model_ops.attention_mha", attention_mha)
    atom_model_ops.attention_mha = attention_mha

    attention_mla = types.ModuleType("atom.model_ops.attention_mla")
    attention_mla.MLAAttention = object
    attention_mla._MLA_MIN_HEADS = 1
    monkeypatch.setitem(sys.modules, "atom.model_ops.attention_mla", attention_mla)
    atom_model_ops.attention_mla = attention_mla

    atom_utils = types.ModuleType("atom.utils")
    atom_utils.__path__ = []
    monkeypatch.setitem(sys.modules, "atom.utils", atom_utils)
    atom.utils = atom_utils

    block_convert = types.ModuleType("atom.utils.block_convert")
    block_convert.kv_indices_generate_triton = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "atom.utils.block_convert", block_convert)
    atom_utils.block_convert = block_convert

    forward_context = types.ModuleType("atom.utils.forward_context")
    forward_context.Context = object
    forward_context.AttentionMetaData = object
    monkeypatch.setitem(sys.modules, "atom.utils.forward_context", forward_context)
    atom_utils.forward_context = forward_context


def _install_vllm_init_stubs(monkeypatch):
    class VllmConfig:
        pass

    vllm_config = types.ModuleType("vllm.config")
    vllm_config.VllmConfig = VllmConfig
    monkeypatch.setitem(sys.modules, "vllm.config", vllm_config)

    platform_utils = types.ModuleType("vllm.utils.platform_utils")
    platform_utils.num_compute_units = lambda device_index: 1
    monkeypatch.setitem(sys.modules, "vllm.utils.platform_utils", platform_utils)

    cp_utils = types.ModuleType("vllm.v1.worker.cp_utils")
    cp_utils.get_total_cp_world_size = lambda: 1
    monkeypatch.setitem(sys.modules, "vllm.v1.worker.cp_utils", cp_utils)

    math_utils = types.ModuleType("vllm.utils.math_utils")
    math_utils.cdiv = lambda a, b: (a + b - 1) // b
    monkeypatch.setitem(sys.modules, "vllm.utils.math_utils", math_utils)

    models_utils = types.ModuleType("atom.models.utils")

    def extract_layer_index(layer_name):
        match = re.search(r"\d+", layer_name)
        if match is None:
            raise ValueError(
                f"Expected layer_name to contain a numeric layer index, got: {layer_name!r}"
            )
        return int(match.group(0))

    models_utils.extract_layer_index = extract_layer_index
    monkeypatch.setitem(sys.modules, "atom.models.utils", models_utils)

    return VllmConfig


def test_mtp_draft_layer_arange_buffer_uses_token_capacity(monkeypatch):
    _install_attention_import_stubs(monkeypatch)
    VllmConfig = _install_vllm_init_stubs(monkeypatch)
    sys.modules.pop("atom.plugin.attention", None)

    attention = importlib.import_module("atom.plugin.attention")

    class BaseBuilder:
        def __init__(self, *args, **kwargs):
            pass

        def _init_reorder_batch_threshold(self, *args, **kwargs):
            pass

    config = VllmConfig()
    config.scheduler_config = SimpleNamespace(
        max_num_batched_tokens=4,
        max_num_seqs=2,
    )
    config.model_config = SimpleNamespace(
        max_model_len=4096,
        hf_config=SimpleNamespace(num_hidden_layers=10),
    )
    config.speculative_config = SimpleNamespace(num_speculative_tokens=1)

    builder = BaseBuilder()
    init = attention.create_mla_sparse_indexer_metadata_builder_init_method(BaseBuilder)
    init(
        builder,
        kv_cache_spec=SimpleNamespace(block_size=16),
        layer_names=["model.layers.10.self_attn"],
        config=config,
        device=torch.device("cpu"),
    )

    assert builder.num_speculative_tokens == 0
    assert (
        builder.arange_buffer.numel() == config.scheduler_config.max_num_batched_tokens
    )
