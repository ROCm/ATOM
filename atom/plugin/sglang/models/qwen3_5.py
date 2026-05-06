# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from contextlib import contextmanager
from collections.abc import Iterable
from typing import Any, Optional

import torch
from torch import nn

from aiter.dist.parallel_state import get_pp_group as _aiter_pp_group
from sglang.srt.layers.quantization.base_config import (
    QuantizationConfig as SGLangQuantizationConfig,
)
from sglang.srt.model_executor.forward_batch_info import (
    ForwardBatch,
    PPProxyTensors,
)
from sglang.srt.models.qwen3_5 import (
    Qwen3_5ForConditionalGeneration as _SglQwen35VL,
    Qwen3_5MoeForConditionalGeneration as _SglQwen35MoeVL,
)

from atom.model_loader.loader import WeightsMapper
from atom.models.qwen3_5 import (
    Qwen3_5ForCausalLM,
    Qwen3_5ForCausalLMBase,
    Qwen3_5Model,
    Qwen3_5MoeForCausalLM,
    detect_fused_expert_format,
    get_fused_expert_mapping,
    load_fused_expert_weights,
)
from atom.models.utils import IntermediateTensors
from atom.plugin.sglang.attention_backend.attention_gdn import (
    SGLangGDNForwardContext,
)
from atom.plugin.sglang.models.base_model_wrapper import (
    SGLangForwardBatchMetadata,
)

_PACKED_MODULES_MAPPING = {
    "qkv_proj": ("qkv_proj", None),
    "in_proj_qkvz": ("in_proj_qkvz", None),
    "in_proj_ba": ("in_proj_ba", None),
    "q_proj": ("qkv_proj", "q"),
    "k_proj": ("qkv_proj", "k"),
    "v_proj": ("qkv_proj", "v"),
    "gate_up_proj": ["gate_proj", "up_proj"],
    "gate_proj": ("gate_up_proj", 0),
    "up_proj": ("gate_up_proj", 1),
    "in_proj_qkv": ("in_proj_qkvz", (0, 1, 2)),
    "in_proj_z": ("in_proj_qkvz", 3),
    "in_proj_b": ("in_proj_ba", 0),
    "in_proj_a": ("in_proj_ba", 1),
    ".gate.": (".gate.", 0),
    "shared_expert_gate": ("gate", 1),
}


_BF16_IN_PROJ_MAPPING = {
    "in_proj_qkv": ("in_proj_qkvzba", (0, 1, 2)),
    "in_proj_z": ("in_proj_qkvzba", 3),
    "in_proj_b": ("in_proj_qkvzba", 4),
    "in_proj_a": ("in_proj_qkvzba", 5),
}


def _apply_bf16_in_proj_mapping(mapping: dict, atom_config: Any) -> dict:
    if atom_config.quant_config.global_quant_config.quant_dtype != torch.bfloat16:
        return mapping

    mapping.pop("in_proj_qkvz", None)
    mapping.pop("in_proj_ba", None)
    mapping["in_proj_qkvzba"] = ("in_proj_qkvzba", None)
    mapping.update(_BF16_IN_PROJ_MAPPING)
    return mapping


def _make_qwen35_hf_mapper(language_model_prefix: str) -> WeightsMapper:
    # Qwen3.5 uses the same HF checkpoint keys for both paths below, but the
    # language-model subtree differs: the SGLang outer wrapper nests it under
    # `model.model.*`, while the ATOM inner model uses `model.*`.
    return WeightsMapper(
        orig_to_new_substr={"attn.qkv.": "attn.qkv_proj."},
        orig_to_new_prefix={
            "model.visual.": "visual.",
            "model.language_model.model.": language_model_prefix,
            "model.language_model.lm_head.": "lm_head.",
            "model.language_model.": language_model_prefix,
            "lm_head.": "lm_head.",
        },
    )


_QWEN35_SGLANG_VL_HF_MAPPER = _make_qwen35_hf_mapper("model.model.")
_QWEN35_ATOM_HF_MAPPER = _make_qwen35_hf_mapper("model.")


def _patch_qwen35_moe_text_for_sparse_moe_block(hf_config: Any) -> None:
    tc = getattr(hf_config, "text_config", None)
    if tc is None:
        tc = hf_config
    mt = getattr(tc, "model_type", "") or ""
    if "qwen3_5" not in mt or "moe" not in mt:
        return
    tc.n_shared_experts = 1
    if hasattr(tc, "num_experts"):
        tc.n_routed_experts = tc.num_experts


def remap_qwen35_quant_config_for_sglang_plugin(atom_config: Any) -> None:
    atom_config.quant_config.remap_layer_name(
        atom_config.hf_config,
        packed_modules_mapping=_apply_bf16_in_proj_mapping(
            dict(_PACKED_MODULES_MAPPING), atom_config
        ),
        weights_mapper=_QWEN35_ATOM_HF_MAPPER,
    )


def apply_prepare_model_adaptations(atom_config: Any, model_arch: str) -> None:
    if model_arch == "Qwen3_5MoeForConditionalGeneration":
        _patch_qwen35_moe_text_for_sparse_moe_block(atom_config.hf_config)
    if model_arch in {
        "Qwen3_5ForConditionalGeneration",
        "Qwen3_5MoeForConditionalGeneration",
    }:
        remap_qwen35_quant_config_for_sglang_plugin(atom_config)


def _forward_qwen35_decoder_stack(
    decoder_stack: Qwen3_5Model,
    input_ids: Optional[torch.Tensor],
    positions: torch.Tensor,
    intermediate_tensors: IntermediateTensors | None = None,
    inputs_embeds: Optional[torch.Tensor] = None,
    input_deepstack_embeds: Optional[torch.Tensor] = None,
) -> torch.Tensor | IntermediateTensors:
    if input_deepstack_embeds is None or input_deepstack_embeds.numel() == 0:
        return decoder_stack(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
        )

    if _aiter_pp_group().is_first_rank:
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = decoder_stack.get_input_embeddings(input_ids)
        residual = None
    else:
        assert intermediate_tensors is not None
        hidden_states = intermediate_tensors["hidden_states"]
        residual = intermediate_tensors["residual"]

    hs = decoder_stack.config.hidden_size
    for local_i, layer in enumerate(
        decoder_stack.layers[decoder_stack.start_layer : decoder_stack.end_layer]
    ):
        hidden_states, residual = layer(positions, hidden_states, residual)
        layer_num = decoder_stack.start_layer + local_i
        if (
            input_deepstack_embeds is not None
            and input_deepstack_embeds.numel() > 0
            and layer_num < 3
        ):
            sep = hs * layer_num
            hidden_states.add_(input_deepstack_embeds[:, sep : sep + hs])

    if not _aiter_pp_group().is_last_rank:
        return IntermediateTensors(
            {"hidden_states": hidden_states, "residual": residual}
        )
    hidden_states, _ = decoder_stack.norm(hidden_states, residual)
    return hidden_states


_QWEN35_SGLANG_LANGUAGE_MODEL_STACKS: dict[
    type[Qwen3_5ForCausalLMBase], type[nn.Module]
] = {}


def _get_qwen35_language_model_stack_cls(
    atom_model_cls: type[Qwen3_5ForCausalLMBase],
) -> type[nn.Module]:
    stack_cls = _QWEN35_SGLANG_LANGUAGE_MODEL_STACKS.get(atom_model_cls)
    if stack_cls is not None:
        return stack_cls

    class _AtomQwen35LanguageModelAdapter(atom_model_cls):
        _pending_vlm_root_config: Any = None

        def __init__(
            self,
            config: Any,
            quant_config: Optional[SGLangQuantizationConfig] = None,
            prefix: str = "",
        ) -> None:
            del prefix
            import atom

            nn.Module.__init__(self)
            root_config = type(self)._pending_vlm_root_config
            if root_config is None:
                root_config = config
            atom_lm = atom.prepare_model(config=root_config, engine="sglang")
            if atom_lm is None:
                arch = getattr(root_config, "architectures", ["unknown"])[0]
                raise ValueError(f"ATOM failed to build language model for {arch}")

            self.config = atom_lm.config
            self.quant_config = quant_config or atom_lm.atom_config.quant_config
            self.atom_config = atom_lm.atom_config
            self.model = atom_lm.model
            self.make_empty_intermediate_tensors = (
                atom_lm.make_empty_intermediate_tensors
            )
            # Keep a strong reference to the full ATOM LM without registering it
            # as another submodule tree, so parameter names still flow through
            # the SGLang wrapper's expected `model.*` / `lm_head.*` prefixes.
            self.__dict__["_atom_lm"] = atom_lm

        @property
        def embed_tokens(self) -> nn.Module:
            return self.model.embed_tokens

        @property
        def layers(self) -> nn.Module:
            return self.model.layers

        @property
        def norm(self) -> nn.Module:
            return self.model.norm

        @property
        def start_layer(self) -> int:
            return self.model.start_layer

        @property
        def end_layer(self) -> int:
            return self.model.end_layer

        @property
        def vocab_size(self) -> int:
            return self.model.vocab_size

        @property
        def lm_head(self) -> nn.Module:
            return self._atom_lm.lm_head

        def get_input_embeddings(
            self, input_ids: Optional[torch.Tensor] = None
        ) -> torch.Tensor | nn.Module:
            if input_ids is None:
                return self.model.embed_tokens
            return self.model.get_input_embeddings(input_ids)

        def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
            return self.model.embed_input_ids(input_ids)

        def forward(
            self,
            input_ids: Optional[torch.Tensor],
            positions: torch.Tensor,
            intermediate_tensors: IntermediateTensors | None = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            forward_batch: Optional[ForwardBatch] = None,
            input_embeds: Optional[torch.Tensor] = None,
            pp_proxy_tensors: Optional[PPProxyTensors] = None,
            input_deepstack_embeds: Optional[torch.Tensor] = None,
            save_kv_cache: bool = True,
            **kwargs: Any,
        ):
            kwargs = dict(kwargs)
            metadata = SGLangForwardBatchMetadata.build(
                forward_batch,
                pp_proxy_tensors=pp_proxy_tensors,
                save_kv_cache=save_kv_cache,
            )
            if inputs_embeds is None:
                inputs_embeds = input_embeds
            if inputs_embeds is None:
                inputs_embeds = kwargs.pop("inputs_embeds", None)
            if intermediate_tensors is None:
                intermediate_tensors = (
                    SGLangForwardBatchMetadata.to_intermediate_tensors(
                        pp_proxy_tensors,
                        metadata,
                    )
                )
            del kwargs
            with SGLangForwardBatchMetadata.bind(metadata):
                with SGLangGDNForwardContext.bind(metadata):
                    out = _forward_qwen35_decoder_stack(
                        self.model,
                        input_ids,
                        positions,
                        intermediate_tensors=intermediate_tensors,
                        inputs_embeds=inputs_embeds,
                        input_deepstack_embeds=input_deepstack_embeds,
                    )
            if isinstance(out, IntermediateTensors):
                return PPProxyTensors(dict(out.tensors))
            return out

    _QWEN35_SGLANG_LANGUAGE_MODEL_STACKS[atom_model_cls] = (
        _AtomQwen35LanguageModelAdapter
    )
    return _AtomQwen35LanguageModelAdapter


class _Qwen3_5ConditionalGenerationSglangBase:
    packed_modules_mapping = _PACKED_MODULES_MAPPING
    atom_language_model_cls: type[Qwen3_5ForCausalLMBase] = Qwen3_5ForCausalLM
    hf_to_sglang_mapper = _QWEN35_SGLANG_VL_HF_MAPPER

    def _prepare_sglang_root_config(self, config: Any) -> None:
        del config

    def __init__(
        self,
        config: Any,
        quant_config: Optional[SGLangQuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        self._prepare_sglang_root_config(config)
        stack_cls = _get_qwen35_language_model_stack_cls(
            type(self).atom_language_model_cls
        )
        stack_cls._pending_vlm_root_config = config
        try:
            super().__init__(
                config,
                quant_config,
                prefix,
                language_model_cls=stack_cls,
            )
        finally:
            stack_cls._pending_vlm_root_config = None
        self.language_model = self.model
        self.atom_config = self.model.atom_config
        self.lm_head = self.model.lm_head
        self.packed_modules_mapping = _apply_bf16_in_proj_mapping(
            dict(type(self).packed_modules_mapping), self.atom_config
        )
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )
        if quant_config is not None and hasattr(quant_config, "packed_modules_mapping"):
            quant_config.packed_modules_mapping = self.packed_modules_mapping

    def _load_weights_in_plugin_mode(
        self,
        *,
        weights_mapper: Optional[WeightsMapper] = None,
        load_fused_expert_weights_fn=None,
    ) -> set[str]:
        from atom.model_loader.loader import load_model_in_plugin_mode

        return load_model_in_plugin_mode(
            model=self,
            config=self.atom_config,
            prefix="",
            weights_mapper=weights_mapper,
            load_fused_expert_weights_fn=load_fused_expert_weights_fn,
        )

    @contextmanager
    def _maybe_disable_shared_expert_fusion_for_load(self):
        # Some Qwen3.5 FP8 checkpoints keep `shared_expert.*` as standalone
        # modules. In that case, the generic loader must not rewrite those keys
        # into `mlp.experts.<n_routed_experts>.*` during load.
        has_standalone_shared_expert = any(
            ".shared_expert." in name for name, _ in self.named_parameters()
        )
        if not has_standalone_shared_expert:
            yield
            return

        import atom.model_loader.loader as atom_loader

        original = atom_loader.is_rocm_aiter_fusion_shared_expert_enabled
        atom_loader.is_rocm_aiter_fusion_shared_expert_enabled = lambda: False
        try:
            yield
        finally:
            atom_loader.is_rocm_aiter_fusion_shared_expert_enabled = original


class Qwen3_5ForConditionalGeneration(
    _Qwen3_5ConditionalGenerationSglangBase, _SglQwen35VL
):
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        del weights
        return self._load_weights_in_plugin_mode(
            weights_mapper=self.hf_to_sglang_mapper
        )


class Qwen3_5MoeForConditionalGeneration(
    _Qwen3_5ConditionalGenerationSglangBase, _SglQwen35MoeVL
):
    atom_language_model_cls = Qwen3_5MoeForCausalLM

    def _prepare_sglang_root_config(self, config: Any) -> None:
        _patch_qwen35_moe_text_for_sparse_moe_block(config)

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.model.get_expert_mapping()

    def detect_fused_expert_format(self, weight_name: str) -> bool:
        return detect_fused_expert_format(weight_name)

    def get_fused_expert_mapping(self) -> list[tuple[str, str, str]]:
        return get_fused_expert_mapping()

    def load_fused_expert_weights(
        self,
        original_name: str,
        name: str,
        params_dict: dict,
        loaded_weight: torch.Tensor,
        shard_id: str,
        num_experts: int,
    ) -> bool:
        return load_fused_expert_weights(
            original_name,
            name,
            params_dict,
            loaded_weight,
            shard_id,
            num_experts,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        del weights
        with self._maybe_disable_shared_expert_fusion_for_load():
            return self._load_weights_in_plugin_mode(
                weights_mapper=self.hf_to_sglang_mapper,
                load_fused_expert_weights_fn=self.load_fused_expert_weights,
            )


# SGLang discovers these multimodal wrappers from this module's `EntryClass`.
# They are not covered by `base_model_wrapper.py`, whose generated entries only
# handle the plain causal-LM architectures in `_MODEL_ARCH_SPECS`.
EntryClass = [Qwen3_5ForConditionalGeneration, Qwen3_5MoeForConditionalGeneration]
