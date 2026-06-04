# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Tests for the extracted Qwen3.5 SGLang wrapper module."""

import importlib
import sys
from types import ModuleType
from unittest.mock import patch

import torch


class _Obj:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class _WeightsMapper:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def _package(name: str) -> ModuleType:
    module = ModuleType(name)
    module.__path__ = []
    return module


def _module(name: str, **attrs) -> ModuleType:
    module = ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def _make_fake_modules() -> dict[str, ModuleType]:
    return {
        "sglang": _package("sglang"),
        "sglang.srt": _package("sglang.srt"),
        "sglang.srt.layers": _package("sglang.srt.layers"),
        "sglang.srt.layers.quantization": _package("sglang.srt.layers.quantization"),
        "sglang.srt.model_executor": _package("sglang.srt.model_executor"),
        "sglang.srt.models": _package("sglang.srt.models"),
        "aiter": _package("aiter"),
        "aiter.dist": _package("aiter.dist"),
        "sglang.srt.layers.quantization.base_config": _module(
            "sglang.srt.layers.quantization.base_config",
            QuantizationConfig=object,
        ),
        "sglang.srt.model_executor.forward_batch_info": _module(
            "sglang.srt.model_executor.forward_batch_info",
            ForwardBatch=object,
            PPProxyTensors=object,
        ),
        "sglang.srt.models.qwen3_5": _module(
            "sglang.srt.models.qwen3_5",
            Qwen3_5ForConditionalGeneration=type(
                "_SglQwen35VL",
                (),
                {"__init__": lambda self, *args, **kwargs: None},
            ),
            Qwen3_5MoeForConditionalGeneration=type(
                "_SglQwen35MoeVL",
                (),
                {"__init__": lambda self, *args, **kwargs: None},
            ),
        ),
        "aiter.dist.parallel_state": _module(
            "aiter.dist.parallel_state",
            get_pp_group=lambda: _Obj(is_first_rank=True, is_last_rank=True),
        ),
        "atom.model_loader.loader": _module(
            "atom.model_loader.loader",
            WeightsMapper=_WeightsMapper,
        ),
        "atom.models.qwen3_5": _module(
            "atom.models.qwen3_5",
            Qwen3_5ForCausalLM=type("Qwen3_5ForCausalLM", (), {}),
            Qwen3_5ForCausalLMBase=type("Qwen3_5ForCausalLMBase", (), {}),
            Qwen3_5Model=type("Qwen3_5Model", (), {}),
            Qwen3_5MoeForCausalLM=type("Qwen3_5MoeForCausalLM", (), {}),
            detect_fused_expert_format=lambda *_a, **_k: False,
            get_fused_expert_mapping=lambda: [],
            load_fused_expert_weights=lambda *_a, **_k: True,
        ),
        "atom.models.utils": _module(
            "atom.models.utils",
            IntermediateTensors=dict,
        ),
        "atom.plugin.config": _module(
            "atom.plugin.config",
            generate_atom_config_for_plugin_mode=lambda config: config,
        ),
        "atom.plugin.sglang.attention_backend.attention_gdn": _module(
            "atom.plugin.sglang.attention_backend.attention_gdn",
            SGLangGDNForwardContext=type(
                "SGLangGDNForwardContext",
                (),
                {},
            ),
        ),
        "atom.plugin.sglang.models.base_model_wrapper": _module(
            "atom.plugin.sglang.models.base_model_wrapper",
            SGLangForwardBatchMetadata=type(
                "SGLangForwardBatchMetadata",
                (),
                {},
            ),
            load_model_weights_for_sglang=lambda *_a, **_k: set(),
        ),
    }


def test_qwen35_bf16_mapping_uses_fused_in_proj_layout():
    with patch.dict(sys.modules, _make_fake_modules()):
        sys.modules.pop("atom.plugin.sglang.models.qwen3_5", None)
        module = importlib.import_module("atom.plugin.sglang.models.qwen3_5")
        atom_config = _Obj(
            quant_config=_Obj(global_quant_config=_Obj(quant_dtype=torch.bfloat16))
        )
        remapped = module._apply_bf16_in_proj_mapping(
            dict(module._PACKED_MODULES_MAPPING), atom_config
        )

    assert "in_proj_qkvzba" in remapped
    assert remapped["in_proj_qkv"] == ("in_proj_qkvzba", (0, 1, 2))
    assert remapped["in_proj_z"] == ("in_proj_qkvzba", 3)
    assert remapped["in_proj_b"] == ("in_proj_qkvzba", 4)
    assert remapped["in_proj_a"] == ("in_proj_qkvzba", 5)
    assert "in_proj_qkvz" not in remapped
    assert "in_proj_ba" not in remapped


def test_qwen35_prepare_adaptations_remap_quant_config():
    with patch.dict(sys.modules, _make_fake_modules()):
        sys.modules.pop("atom.plugin.sglang.models.qwen3_5", None)
        module = importlib.import_module("atom.plugin.sglang.models.qwen3_5")

        calls = {}

        def remap_layer_name(
            hf_config, packed_modules_mapping=None, weights_mapper=None
        ):
            calls["hf_config"] = hf_config
            calls["packed_modules_mapping"] = packed_modules_mapping
            calls["weights_mapper"] = weights_mapper

        text_config = _Obj(model_type="qwen3_5_moe", num_experts=256)
        atom_config = _Obj(
            hf_config=_Obj(text_config=text_config),
            quant_config=_Obj(
                global_quant_config=_Obj(quant_dtype=torch.float8_e4m3fnuz),
                remap_layer_name=remap_layer_name,
            ),
        )

        module.apply_prepare_model_adaptations(
            atom_config, "Qwen3_5MoeForConditionalGeneration"
        )

    assert text_config.n_shared_experts == 1
    assert text_config.n_routed_experts == 256
    assert calls["hf_config"] is atom_config.hf_config
    assert calls["packed_modules_mapping"]["in_proj_b"] == ("in_proj_ba", 0)
    assert (
        calls["weights_mapper"].orig_to_new_prefix["model.language_model."] == "model."
    )
