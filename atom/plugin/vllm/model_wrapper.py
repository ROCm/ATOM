from collections.abc import Iterable

import importlib
import torch
import torch.nn as nn
from aiter.dist.parallel_state import (
    get_pp_group,
    get_tp_group,
)
from vllm.config import VllmConfig
from vllm.model_executor.models.interfaces import (
    SupportsPP,
    SupportsQuant,
    SupportsMultiModal,
    SupportsMRoPE,
    MultiModalEmbeddings,
)
from vllm.model_executor.models.interfaces_base import (
    VllmModel,
    VllmModelForTextGeneration,
)
from vllm.sequence import IntermediateTensors
from vllm.forward_context import (
    get_forward_context as get_vllm_forward_context,
    is_forward_context_available,
)

import atom  # noqa: F401
from atom.plugin.config import generate_atom_config_for_plugin_mode
from atom.plugin.prepare import _set_framework_backbone

import logging

logger = logging.getLogger("atom")


_ATOM_MODEL_CLASSES: dict[str, str] = {
    "LlamaForCausalLM": "atom.models.llama:LlamaForCausalLM",
    "Qwen3ForCausalLM": "atom.models.qwen3:Qwen3ForCausalLM",
    "Qwen3MoeForCausalLM": "atom.models.qwen3_moe:Qwen3MoeForCausalLM",
    "GptOssForCausalLM": "atom.models.gpt_oss:GptOssForCausalLM",
    "DeepseekV3ForCausalLM": "atom.models.deepseek_v2:DeepseekV3ForCausalLM",
    "DeepseekV32ForCausalLM": "atom.models.deepseek_v2:DeepseekV3ForCausalLM",
    "Glm4MoeForCausalLM": "atom.models.glm4_moe:Glm4MoeForCausalLM",
    "GlmMoeDsaForCausalLM": "atom.models.deepseek_v2:GlmMoeDsaForCausalLM",
    "Qwen3NextForCausalLM": "atom.models.qwen3_next:Qwen3NextForCausalLM",
    "Qwen3_5MoeForConditionalGeneration": "atom.models.qwen3_5:Qwen3_5MoeForConditionalGeneration_",
    "Qwen3_5ForConditionalGeneration": "atom.models.qwen3_5:Qwen3_5ForConditionalGeneration_",
    "KimiK25ForConditionalGeneration": "atom.plugin.vllm.models.kimi_k25:KimiK25ForConditionalGeneration_",
    "MiniMaxM2ForCausalLM": "atom.models.minimax_m2:MiniMaxM2ForCausalLM",
}


def _get_atom_model_cls(model_arch: str) -> type:
    if model_arch is not None and model_arch in _ATOM_MODEL_CLASSES:
        model_ref = _ATOM_MODEL_CLASSES[model_arch]
    else:
        raise ValueError(f"The {model_arch} is not supported by ATOM OOT backend")

    module_path, class_name = model_ref.split(":", 1)
    return getattr(importlib.import_module(module_path), class_name)


def _prepare_env(atom_config) -> None:
    from atom.plugin.register import set_attn_cls, init_aiter_dist

    # set global attention class
    logger.info("Set global attention class")
    set_attn_cls()

    # init aiter dist for using aiter custom collective ops
    logger.info("Init aiter dist for using aiter custom collective ops")
    init_aiter_dist(config=atom_config)


class ATOMModelBase(nn.Module, VllmModel, SupportsQuant, SupportsPP):
    def __init_subclass__(cls, *args, **kwargs):
        super().__init_subclass__(*args, **kwargs)

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        _set_framework_backbone("vllm")

        self.config = vllm_config.model_config.hf_config
        self.text_config = self.config.get_text_config()
        self.cache_config = vllm_config.cache_config
        self.device_config = vllm_config.device_config
        self.model_config = vllm_config.model_config
        self.parallel_config = vllm_config.parallel_config
        self.quant_config = vllm_config.quant_config
        self.vllm_compilation_config = vllm_config.compilation_config

        # Weights to skip in `self.load_weights`
        self.skip_prefixes: list[str] = []
        self.skip_substrs: list[str] = []
        self.ignore_unexpected_prefixes: list[str] = []
        self.ignore_unexpected_suffixes: list[str] = []

        self.vllm_config = vllm_config
        self.atom_config = generate_atom_config_for_plugin_mode(vllm_config)

        _prepare_env(atom_config=self.atom_config)

        model_arch = vllm_config.model_config.architectures[0]
        model_cls = _get_atom_model_cls(model_arch)
        module_remapping = getattr(model_cls, "packed_modules_mapping", {})
        weights_mapper = getattr(model_cls, "hf_to_atom_mapper", {})
        self.atom_config.quant_config.remap_layer_name(
            self.atom_config.hf_config,
            packed_modules_mapping=module_remapping,
            weights_mapper=weights_mapper,
        )

        # In ATOM, quant_exclude_name_mapping is used to translate the HF module names
        # to ATOM's format. It is invoked in ATOM's model_runner initialization, but
        # lacks correspondences in vLLM. So we invoke the translation here for vLLM OOT.
        exclude_mapping = getattr(model_cls, "quant_exclude_name_mapping", {})
        if exclude_mapping and self.atom_config.quant_config is not None:
            self.atom_config.quant_config.apply_exclude_name_mapping(exclude_mapping)

        default_excludes = getattr(model_cls, "quant_default_exclude_layers", [])
        if default_excludes and self.atom_config.quant_config is not None:
            self.atom_config.quant_config.apply_default_exclude_layers(default_excludes)

        logger.info(f"Construct ATOM model {model_arch} for vLLM plugin mode")
        self.model = model_cls(self.atom_config)

        # For sparse MLA, register the Indexer's DeepseekV32IndexerCache as
        # a virtual subclass of vLLM's AttentionLayerBase so vLLM can discover
        # it and allocate KV cache.
        self._register_indexer_caches_with_vllm()

        if self.model is None:
            model_arch = vllm_config.model_config.architectures[0]
            raise ValueError(
                f"The model {model_arch} is not supported by model impl backend atom"
            )

        # here init aiter dist for using aiter custom collective ops
        self.pp_group = get_pp_group()
        self.tp_group = get_tp_group()

    def _register_indexer_caches_with_vllm(self):
        """Register DeepseekV32IndexerCache instances with vLLM so that:
        1. vLLM discovers them via isinstance(AttentionLayerBase) for KV cache
           allocation (get_kv_cache_spec iterates static_forward_context)
        2. bind_kv_cache() can find them in vLLM's static_forward_context to
           assign the allocated KV cache tensor
        3. The indexer's metadata lookup uses the correct prefix in vLLM's
           attn_metadata dict

        ATOM's DeepseekV32IndexerCache inherits from nn.Module (not vLLM's
        AttentionLayerBase), so we register it as a virtual subclass.
        We also register each instance in vLLM's static_forward_context using
        the same prefix convention as other attention layers (the prefix
        parameter passed at construction, e.g. 'model.layers.0...k_cache').
        """
        from atom.models.deepseek_v2 import DeepseekV32IndexerCache

        # Find indexer cache instances. module.prefix is the ATOM-internal
        # prefix set during __init__ (e.g. "model.layers.0.self_attn.indexer.k_cache").
        indexer_caches = []
        for _name, module in self.model.named_modules():
            if isinstance(module, DeepseekV32IndexerCache):
                indexer_caches.append(module)

        if not indexer_caches:
            return

        try:
            from vllm.model_executor.layers.attention_layer_base import (
                AttentionLayerBase,
            )

            # Register DeepseekV32IndexerCache as a virtual subclass of
            # AttentionLayerBase so vLLM's isinstance() check passes.
            AttentionLayerBase.register(DeepseekV32IndexerCache)
            logger.info(
                "Registered DeepseekV32IndexerCache as AttentionLayerBase "
                "virtual subclass for vLLM KV cache allocation"
            )
        except ImportError:
            logger.warning(
                "Could not import AttentionLayerBase from vLLM. "
                "Indexer cache will not be managed by vLLM."
            )
            return

        # Register each indexer cache in vLLM's static_forward_context.
        # Use module.prefix (the ATOM-internal prefix), which follows the same
        # convention as vLLM's MLAAttention layers that self-register with
        # their prefix parameter (e.g. "model.layers.0.self_attn.attn").
        vllm_sfc = self.vllm_compilation_config.static_forward_context
        for module in indexer_caches:
            prefix = module.prefix
            if prefix not in vllm_sfc:
                vllm_sfc[prefix] = module
                logger.info(
                    f"Registered indexer cache in vLLM static_forward_context: "
                    f"{prefix}"
                )
            else:
                logger.warning(
                    f"Indexer cache {prefix} already in vLLM "
                    f"static_forward_context, skipping"
                )

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **model_kwargs,
    ) -> torch.Tensor | IntermediateTensors:
        if not self.pp_group.is_first_rank:
            assert intermediate_tensors is not None
            input_ids = None
            inputs_embeds = intermediate_tensors["hidden_states"]

        # pass positions from vLLM to OOT execution path via vLLM's per-forward context
        if is_forward_context_available():
            get_vllm_forward_context().additional_kwargs["atom_positions"] = positions
        elif "positions" in self.atom_config.compilation_config.static_forward_context:
            buf = self.atom_config.compilation_config.static_forward_context[
                "positions"
            ]
            buf[: positions.numel()].copy_(positions)

        hidden_states = self.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **model_kwargs,
        )

        if not self.pp_group.is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states})

        return hidden_states

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> set[str]:
        # prevent circular import
        from atom.model_loader.loader import load_model_in_plugin_mode

        loaded_weights_record = load_model_in_plugin_mode(
            model=self.model, config=self.atom_config, prefix="model."
        )
        return loaded_weights_record

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.model.compute_logits(hidden_states)
        return logits


class ATOMForCausalLM(ATOMModelBase, VllmModelForTextGeneration): ...


class ATOMMoEForCausalLM(ATOMModelBase, VllmModelForTextGeneration): ...


class ATOMForConditionalGeneration(
    ATOMModelBase, VllmModelForTextGeneration, SupportsMultiModal, SupportsMRoPE
):

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        """
        Get the placeholder text for the `i`th `modality` item in the prompt.
        """
        raise NotImplementedError

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        return self.model.embed_multimodal(**kwargs)

    def configure_mm_token_handling(self, vocab_size, mm_token_ids):
        return self.model.configure_mm_token_handling(vocab_size, mm_token_ids)

    def get_language_model(self):
        return self.model.get_language_model()

    def get_num_mm_encoder_tokens(self, num_image_tokens):
        return self.model.get_num_mm_encoder_tokens(num_image_tokens)

    def get_num_mm_connector_tokens(self, num_vision_tokens):
        return self.model.get_num_mm_connector_tokens(num_vision_tokens)

    def embed_input_ids(
        self, input_ids, multimodal_embeddings=None, *, is_multimodal=None
    ):
        return self.model.embed_input_ids(
            input_ids,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
        )

    def _embed_text_input_ids(self, input_ids, embed_input_ids, *, is_multimodal):
        return self.model._embed_text_input_ids(
            input_ids, embed_input_ids, is_multimodal=is_multimodal
        )

    def get_mrope_input_positions(self, input_tokens, mm_features):
        return self.model.get_mrope_input_positions(input_tokens, mm_features)
