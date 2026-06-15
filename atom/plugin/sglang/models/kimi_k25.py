from __future__ import annotations

from typing import Any, Iterable, Optional

import torch
from torch import nn

from atom.model_loader.loader import WeightsMapper, load_model_in_plugin_mode
from atom.plugin.sglang.runtime import (
    SGLangForwardBatchMetadata,
    SGLangPluginRuntime,
    plugin_runtime_scope,
)
from sglang.srt.distributed import get_pp_group
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.models.kimi_k25 import (
    KimiK25ForConditionalGeneration as _SglangKimiK25ForConditionalGeneration,
)
from sglang.srt.utils import LazyValue

_KIMI_K25_PACKED_MODULES_MAPPING = {
    "q_a_proj": ("fused_qkv_a_proj", 0),
    "kv_a_proj_with_mqa": ("fused_qkv_a_proj", 1),
    "gate_proj": ("gate_up_proj", 0),
    "up_proj": ("gate_up_proj", 1),
}

_KIMI_K25_QUANT_EXCLUDE_NAME_MAPPING = {
    "language_model.model.": "model.",
    "language_model.lm_head": "lm_head",
}

_KIMI_K25_HF_TO_ATOM_MAPPER = WeightsMapper(
    orig_to_new_substr={
        "mm_projector.proj.0": "mm_projector.linear_1",
        "mm_projector.proj.2": "mm_projector.linear_2",
    },
    orig_to_new_prefix={
        # Kimi checkpoints seen in the wild use both nested and flattened
        # language-model prefixes. Keep all of them pointed at the SGLang-style
        # wrapper slots backed by the ATOM text model.
        "model.language_model.": "language_model.model.",
        "language_model.layers.": "language_model.model.layers.",
        "lm_head.": "language_model.lm_head.",
    },
)


def remap_kimi_k25_quant_config_for_sglang_plugin(
    atom_config: Any, model_arch: str
) -> None:
    del model_arch
    quant_config = getattr(atom_config, "quant_config", None)
    if quant_config is None:
        return

    quant_config.remap_layer_name(
        atom_config.hf_config,
        packed_modules_mapping=dict(_KIMI_K25_PACKED_MODULES_MAPPING),
        weights_mapper=_KIMI_K25_HF_TO_ATOM_MAPPER,
        quant_exclude_name_mapping=_KIMI_K25_QUANT_EXCLUDE_NAME_MAPPING,
    )


class _AtomKimiK25LanguageModelAdapter(nn.Module):
    """Expose ATOM's Kimi text model through SGLang's language-model API."""

    def __init__(
        self,
        atom_lm: nn.Module,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.pp_group = get_pp_group()
        self.atom_config = atom_lm.atom_config
        self.config = atom_lm.config
        self.quant_config = quant_config or self.atom_config.quant_config

        atom_text_lm = atom_lm.language_model
        atom_text_lm.atom_config = self.atom_config
        self.model = atom_text_lm.model
        self.lm_head = atom_text_lm.lm_head
        self.make_empty_intermediate_tensors = atom_lm.make_empty_intermediate_tensors
        self.packed_modules_mapping = atom_text_lm.packed_modules_mapping
        self._routed_experts_weights_of_layer = LazyValue(
            lambda: {
                layer_id: layer.mlp.get_moe_weights()
                for layer_id, layer in enumerate(self.model.layers)
                if hasattr(getattr(layer, "mlp", None), "get_moe_weights")
            }
        )
        self.logits_processor = LogitsProcessor(
            self.config,
            skip_all_gather=bool(self.atom_config.enable_dp_attention),
        )
        self.__dict__["_atom_lm"] = atom_lm

        from atom.plugin.sglang.models.deepseek_mla import setup_deepseek_for_sglang

        with plugin_runtime_scope(framework="sglang", atom_config=self.atom_config):
            setup_deepseek_for_sglang(atom_text_lm)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    @property
    def start_layer(self) -> int:
        return self.model.start_layer

    @property
    def end_layer(self) -> int:
        return self.model.end_layer

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self._atom_lm.get_expert_mapping()

    @torch.no_grad()
    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
        **model_kwargs: Any,
    ):
        with plugin_runtime_scope(framework="sglang", atom_config=self.atom_config):
            with SGLangPluginRuntime(
                atom_config=self.atom_config,
                forward_batch=forward_batch,
                positions=positions,
                input_ids=input_ids,
                input_embeds=input_embeds,
            ) as runtime:
                metadata = SGLangForwardBatchMetadata.build(
                    runtime.forward_batch,
                    pp_proxy_tensors=pp_proxy_tensors,
                    save_kv_cache=model_kwargs.get("save_kv_cache"),
                )
                intermediate_tensors = (
                    SGLangForwardBatchMetadata.to_intermediate_tensors(
                        pp_proxy_tensors,
                        metadata,
                    )
                )
                with SGLangForwardBatchMetadata.bind(metadata):
                    hidden_states = self._atom_lm(
                        input_ids=runtime.input_ids,
                        positions=runtime.positions,
                        intermediate_tensors=intermediate_tensors,
                        inputs_embeds=runtime.input_embeds,
                    )
                hidden_states = runtime.trim_output(hidden_states)

                if self.pp_group.is_last_rank:
                    return self.logits_processor(
                        runtime.input_ids,
                        hidden_states,
                        self.lm_head,
                        runtime.forward_batch,
                    )
                return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        # The outer Kimi wrapper owns loading so vision/projector weights are
        # loaded together with the ATOM text backbone.
        del weights
        return set()

    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight

    def set_embed_and_head(self, embed, head):
        del self.model.embed_tokens.weight
        del self.lm_head.weight
        self.model.embed_tokens.weight = embed
        self.lm_head.weight = head
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def set_embed(self, embed):
        del self.model.embed_tokens.weight
        self.model.embed_tokens.weight = embed
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


class KimiK25ForConditionalGeneration(_SglangKimiK25ForConditionalGeneration):
    hf_to_atom_mapper = _KIMI_K25_HF_TO_ATOM_MAPPER
    packed_modules_mapping = _KIMI_K25_PACKED_MODULES_MAPPING

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        **kwargs: Any,
    ) -> None:
        from atom.plugin.sglang.prepare import prepare_model

        with plugin_runtime_scope(framework="sglang"):
            atom_lm = prepare_model(config=config)

        original_language_model_cls = (
            _SglangKimiK25ForConditionalGeneration.__init__.__globals__[
                "DeepseekV3ForCausalLM"
            ]
        )
        _SglangKimiK25ForConditionalGeneration.__init__.__globals__[
            "DeepseekV3ForCausalLM"
        ] = lambda *args, **kwargs: _AtomKimiK25LanguageModelAdapter(
            atom_lm,
            quant_config=quant_config,
        )
        try:
            super().__init__(
                config=config,
                quant_config=quant_config,
                prefix=prefix,
                **kwargs,
            )
        finally:
            _SglangKimiK25ForConditionalGeneration.__init__.__globals__[
                "DeepseekV3ForCausalLM"
            ] = original_language_model_cls

        self.atom_config = atom_lm.atom_config
        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )
        self.packed_modules_mapping = self.language_model.packed_modules_mapping

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.language_model.get_expert_mapping()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        del weights
        with plugin_runtime_scope(framework="sglang", atom_config=self.atom_config):
            return load_model_in_plugin_mode(
                model=self,
                config=self.atom_config,
                prefix="",
                weights_mapper=self.hf_to_atom_mapper,
            )


EntryClass = [KimiK25ForConditionalGeneration]
