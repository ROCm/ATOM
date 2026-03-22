# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Inference-only Kimi-K2.5 model (text-only backbone).

Kimi-K2.5 is a multimodal model whose language backbone is a DeepseekV3-style
MoE transformer with MLA attention.  For text-only serving we load only the
``language_model.*`` weights and delegate to the existing
:class:`DeepseekV2ForCausalLM` implementation.

Vision encoder and multimodal projector weights are skipped during loading
via :pyattr:`skip_weight_prefixes`.
"""

from typing import Optional, Union

import torch
from torch import nn

from atom.config import Config
from atom.models.deepseek_v2 import DeepseekV2ForCausalLM, DeepseekV2Model, DeepseekV2DecoderLayer
from atom.models.utils import IntermediateTensors
from atom.plugin.prepare import is_vllm

from atom.models.utils import (
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
    extract_layer_index,
    PPMissingLayer,
)
from atom.utils.decorators import support_torch_compile
from aiter.dist.parallel_state import get_pp_group
from atom.model_ops.embed_head import VocabParallelEmbedding, ParallelLMHead
from atom.model_ops.layernorm import LayerNorm, RMSNorm


class KimiK25ForCausalLM(nn.Module):
    """Kimi-K2.5 text-only wrapper around :class:`DeepseekV2ForCausalLM`.

    The HuggingFace checkpoint stores the LLM weights under the
    ``language_model.*`` prefix.  By placing the underlying model as
    ``self.language_model``, PyTorch's parameter naming automatically
    matches the checkpoint layout so no explicit prefix stripping is needed.

    Vision tower and multimodal projector weights are excluded via
    :pyattr:`skip_weight_prefixes` which the model loader respects.
    """

    # Weight prefixes that should be silently skipped during loading
    # (these belong to the vision encoder / MM projector that we don't use).
    skip_weight_prefixes = [
        "vision_tower.",
        "mm_projector.",
    ]

    def __init__(
        self,
        atom_config: Config,
        prefix: str = "",
    ):
        super().__init__()
        self.config = atom_config.hf_config

        # The underlying LLM – named ``language_model`` so that its parameter
        # names match the ``language_model.*`` keys in the checkpoint.
        self.language_model = DeepseekV2ForCausalLM(
            atom_config=atom_config,
            prefix="",
        )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    # ---- properties forwarded to the inner model ----

    @property
    def packed_modules_mapping(self):
        return self.language_model.packed_modules_mapping

    # ---- forward / inference API ----

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.language_model.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        return self.language_model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        return self.language_model.compute_logits(hidden_states)

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.language_model.get_expert_mapping()




@support_torch_compile
class KimiK25Model(DeepseekV2Model):
    def __init__(
        self,
        atom_config: Config,
        prefix: str = "",
        layer_type: type[nn.Module] = DeepseekV2DecoderLayer,
    ):
        super(DeepseekV2Model, self).__init__()

        config = atom_config.hf_config.text_config
        cache_config = atom_config.kv_cache_dtype
        quant_config = atom_config.quant_config
        self.config = config

        self.vocab_size = config.vocab_size

        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix, layer_num=None: DeepseekV2DecoderLayer(
                config,
                prefix,
                topk_indices_buffer=None,
                cache_config=cache_config,
                quant_config=quant_config,
                layer_num=layer_num,
            ),
            prefix=f"{prefix}.layers",
            layer_num_offset=0,
        )

        # fused_allreduce will have to be turned off here if the fuse_ar_input_norm variable is False in the last layer
        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(
                config.hidden_size,
                eps=config.rms_norm_eps,
                fused_allreduce=self.layers[self.end_layer - 1].fuse_ar_input_norm,
            )
        else:
            self.norm = PPMissingLayer()
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )


class KimiK25ForCausalLM_(nn.Module):

    def __init__(
        self,
        atom_config: Config,
        prefix: str = "",
        layer_type: type[nn.Module] = DeepseekV2DecoderLayer,
    ):
        super().__init__()
        config = atom_config.hf_config.text_config
        quant_config = atom_config.quant_config
        self.config = config
        self.quant_config = quant_config

        if hasattr(config, "q_lora_rank") and config.q_lora_rank is not None:
            self.packed_modules_mapping = {
                "q_a_proj": ("fused_qkv_a_proj", 0),
                "kv_a_proj_with_mqa": ("fused_qkv_a_proj", 1),
                "gate_proj": ("gate_up_proj", 0),
                "up_proj": ("gate_up_proj", 1),
            }
        else:
            self.packed_modules_mapping = {
                "gate_proj": ("gate_up_proj", 0),
                "up_proj": ("gate_up_proj", 1),
            }

        self.model = KimiK25Model(
            atom_config=atom_config,
            prefix=maybe_prefix(prefix, "model"),
            layer_type=layer_type,
        )
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                org_num_embeddings=config.vocab_size,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        else:
            self.lm_head = PPMissingLayer()

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        logits = self.lm_head(hidden_states)
        return logits

    def make_empty_intermediate_tensors(
        self, batch_size: int, dtype: torch.dtype, device: torch.device
    ) -> IntermediateTensors:
        return IntermediateTensors(
            {
                "hidden_states": torch.zeros(
                    (batch_size, self.config.hidden_size), dtype=dtype, device=device
                ),
                "residual": torch.zeros(
                    (batch_size, self.config.hidden_size), dtype=dtype, device=device
                ),
            }
        )

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.model.get_expert_mapping()

if is_vllm():
    from vllm.model_executor.models.kimi_k25 import KimiK25ForConditionalGeneration as vLLMKimiK25
    from vllm.model_executor.models.kimi_k25_vit import MoonViT3dPretrainedModel, KimiK25MultiModalProjector
    from atom.models.utils import maybe_prefix
    from typing import Iterable
    from atom.model_loader.loader import WeightsMapper, load_model_in_plugin_mode
    from atom.plugin.vllm.model_wrapper import ATOMForConditionalGeneration

    from vllm.multimodal import MULTIMODAL_REGISTRY
    from vllm.model_executor.models.kimi_k25 import KimiK25ProcessingInfo, KimiK25DummyInputsBuilder, KimiK25MultiModalProcessor
    from atom.model_config.kimi_k25 import KimiK25Config, KimiK25VisionConfig
    @MULTIMODAL_REGISTRY.register_processor(
        KimiK25MultiModalProcessor,
        info=KimiK25ProcessingInfo,
        dummy_inputs=KimiK25DummyInputsBuilder,
    )
    class KimiK25ForConditionalGeneration_(vLLMKimiK25):
        packed_modules_mapping: dict[str, tuple[str, int]] = {
            "q_a_proj": ("fused_qkv_a_proj", 0),
            "kv_a_proj_with_mqa": ("fused_qkv_a_proj", 1),
            "gate_proj": ("gate_up_proj", 0),
            "up_proj": ("gate_up_proj", 1),
        }
        hf_to_atom_mapper = WeightsMapper(
            orig_to_new_prefix={
                "model.visual.": "visual.",
                "lm_head.": "language_model.lm_head.",
                "model.language_model.": "language_model.model.",
            }
        )

        def __init__(self, atom_config: Config, prefix: str = "model"):
            # protocols have not __init__ method, so we need to use nn.Module.__init__
            nn.Module.__init__(self)
            config: KimiK25Config = atom_config.hf_config

            vllm_config = atom_config.plugin_config.vllm_config
            quant_config = vllm_config.quant_config
            multimodal_config = vllm_config.model_config.multimodal_config
            self.atom_config = atom_config

            self.config = config
            self.multimodal_config = multimodal_config
            self.use_data_parallel = multimodal_config.mm_encoder_tp_mode == "data"
            self.video_pruning_rate = multimodal_config.video_pruning_rate
            self.is_multimodal_pruning_enabled = (
                multimodal_config.is_multimodal_pruning_enabled()
            )

            with self._mark_tower_model(vllm_config, "vision_chunk"):
                self.vision_tower = MoonViT3dPretrainedModel(
                    config.vision_config,
                    quant_config=self._maybe_ignore_quant_config(quant_config),
                    prefix=maybe_prefix(prefix, "vision_tower"),
                )
                # self.vision_tower = self.vision_tower.to(
                #     device=self.device, dtype=model_config.dtype
                # )

                self.mm_projector = KimiK25MultiModalProjector(
                    config=config.vision_config,
                    use_data_parallel=self.use_data_parallel,
                    quant_config=self._maybe_ignore_quant_config(quant_config),
                    prefix=maybe_prefix(prefix, "mm_projector"),
                )
                # self.mm_projector = self.mm_projector.to(
                #     device=self.device, dtype=model_config.dtype
                # )

            self.quant_config = quant_config
            with self._mark_language_model(vllm_config):
                self.language_model = KimiK25ForCausalLM_(
                    atom_config=atom_config,
                    prefix=maybe_prefix(prefix, "language_model"),
                )
                # self.language_model = init_vllm_registered_model(
                #     vllm_config=vllm_config,
                #     hf_config=config.text_config,
                #     prefix=maybe_prefix(prefix, "language_model"),
                #     architectures=["DeepseekV2ForCausalLM"],
                # )
            self.make_empty_intermediate_tensors = (
                self.language_model.make_empty_intermediate_tensors
            )

        def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
            # load weights in plugin mode and discard passed weights generator
            # here prefix is "model." because Qwen3ForCausalLM is constructed in model
            # wrapper class, so the name of loaded weights are prefixed with "model.".
            # The vLLM will check the name of the loaded weights to make sure all the
            # weights are loaded correctly
            loaded_weights_record = load_model_in_plugin_mode(
                model=self,
                config=self.atom_config,
                prefix="model.",
                weights_mapper=self.hf_to_atom_mapper,
            )
            return loaded_weights_record

    @MULTIMODAL_REGISTRY.register_processor(
        KimiK25MultiModalProcessor,
        info=KimiK25ProcessingInfo,
        dummy_inputs=KimiK25DummyInputsBuilder,
    )
    class KimiK25ForConditionalGeneration(ATOMForConditionalGeneration):
        def load_weights(
            self,
            weights: Iterable[tuple[str, torch.Tensor]],
        ) -> set[str]:
            return self.model.load_weights(weights)
