# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Tests for the extracted Qwen3.5 SGLang wrapper module."""

import importlib
import sys
from contextlib import nullcontext
from types import ModuleType
from unittest.mock import patch

import pytest
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
            get_pp_group=lambda: _Obj(
                is_first_rank=True, is_last_rank=True, world_size=1
            ),
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
            get_fused_expert_mapping=list,
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
        "atom.plugin.sglang.runtime": _module(
            "atom.plugin.sglang.runtime",
            SGLangForwardBatchMetadata=object,
            SGLangPluginRuntime=object,
            get_model_arch_spec=lambda _arch: _Obj(bind_cache_views=None),
            plugin_runtime_scope=lambda **_kwargs: nullcontext(),
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


@pytest.mark.parametrize(
    "model_arch",
    [
        "Qwen3_5MoeForCausalLM",
        "Qwen3_5MoeForConditionalGeneration",
    ],
)
def test_qwen35_prepare_adaptations_remap_quant_config(model_arch):
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

        module.apply_prepare_model_adaptations(atom_config, model_arch)

    assert text_config.n_shared_experts == 1
    assert text_config.n_routed_experts == 256
    assert calls["hf_config"] is atom_config.hf_config
    assert calls["packed_modules_mapping"]["in_proj_b"] == ("in_proj_ba", 0)
    assert (
        calls["weights_mapper"].orig_to_new_prefix["model.language_model."] == "model."
    )


class _FakeLayer:
    def __call__(self, positions, hidden_states, residual):
        del positions
        next_residual = (
            hidden_states.clone() if residual is None else hidden_states + residual
        )
        return hidden_states + 1, next_residual


class _FakeNorm:
    def __call__(self, hidden_states, residual):
        return hidden_states + residual, None


class _FakeDecoderStack:
    def __init__(self):
        self.config = _Obj(hidden_size=2)
        self.start_layer = 0
        self.end_layer = 2
        self.layers = [_FakeLayer(), _FakeLayer()]
        self.norm = _FakeNorm()

    def get_input_embeddings(self, input_ids):
        return input_ids.to(torch.float32)


def test_decoder_stack_dflash_capture_uses_shifted_residual_stream():
    with patch.dict(sys.modules, _make_fake_modules()):
        sys.modules.pop("atom.plugin.sglang.models.qwen3_5", None)
        module = importlib.import_module("atom.plugin.sglang.models.qwen3_5")
        stack = _FakeDecoderStack()
        inputs_embeds = torch.tensor([[1.0, 2.0]])
        positions = torch.tensor([0])

        final_hidden, aux_hidden = module._forward_qwen35_decoder_stack(
            stack,
            input_ids=None,
            positions=positions,
            inputs_embeds=inputs_embeds,
            dflash_capture_points=(1,),
        )

    assert len(aux_hidden) == 1
    torch.testing.assert_close(aux_hidden[0], torch.tensor([[3.0, 5.0]]))
    assert final_hidden.shape == inputs_embeds.shape


def test_decoder_stack_dflash_capture_includes_previous_deepstack():
    with patch.dict(sys.modules, _make_fake_modules()):
        sys.modules.pop("atom.plugin.sglang.models.qwen3_5", None)
        module = importlib.import_module("atom.plugin.sglang.models.qwen3_5")
        stack = _FakeDecoderStack()
        inputs_embeds = torch.tensor([[1.0, 2.0]])
        deepstack = torch.tensor([[10.0, 20.0, 0.0, 0.0, 0.0, 0.0]])

        _, aux_hidden = module._forward_qwen35_decoder_stack(
            stack,
            input_ids=None,
            positions=torch.tensor([0]),
            inputs_embeds=inputs_embeds,
            input_deepstack_embeds=deepstack,
            dflash_capture_points=(1,),
        )

    torch.testing.assert_close(aux_hidden[0], torch.tensor([[13.0, 25.0]]))


def test_dflash_setter_keeps_outer_wrapper_shifted_points():
    with patch.dict(sys.modules, _make_fake_modules()):
        sys.modules.pop("atom.plugin.sglang.models.qwen3_5", None)
        module = importlib.import_module("atom.plugin.sglang.models.qwen3_5")
        adapter_cls = module._get_qwen35_language_model_stack_cls(
            module.Qwen3_5ForCausalLM
        )
        adapter = object.__new__(adapter_cls)
        adapter.model = _Obj(start_layer=0, end_layer=60)
        adapter.atom_config = _Obj(enable_dp_attention=False)

        adapter.set_dflash_layers_to_capture([2, 10, 18, 26, 34, 42, 50, 58])

    assert adapter.dflash_capture_points == (2, 10, 18, 26, 34, 42, 50, 58)
