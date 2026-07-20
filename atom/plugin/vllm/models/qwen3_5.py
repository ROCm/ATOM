from collections.abc import Iterable

import torch
from torch import nn

from aiter.dist.parallel_state import get_tensor_model_parallel_rank
from atom.config import Config
from atom.models import qwen3_5 as qwen3_5_base
from atom.models.qwen3_5 import (
    Qwen3_5Config,
    Qwen3_5ForCausalLM as Qwen3_5ForCausalLMBase,
    Qwen3_5GatedDeltaNet,
    Qwen3_5MoeConfig,
    Qwen3_5MoeForCausalLM as Qwen3_5MoeForCausalLMBase,
    detect_fused_expert_format,
    get_fused_expert_mapping,
    load_fused_expert_weights,
    maybe_prefix,
)
from atom.plugin.vllm.model_wrapper import ATOMForConditionalGeneration
from atom.model_loader.loader import WeightsMapper, load_model_in_plugin_mode
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc,
    MambaStateCopyFuncCalculator,
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.models.interfaces import IsHybrid
from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5ForConditionalGeneration as vLLMQwen3_5,
    Qwen3_5MoeForConditionalGeneration as vLLMQwen3_5Moe,
    Qwen3_5MoeProcessingInfo,
    Qwen3_5ProcessingInfo,
)
from vllm.model_executor.models.qwen3_vl import (
    Qwen3VLDummyInputsBuilder,
    Qwen3VLMultiModalProcessor,
    Qwen3_VisionTransformer,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum


class Qwen3_5GatedDeltaNetVllm(Qwen3_5GatedDeltaNet, MambaBase):
    def __init__(
        self,
        atom_config: Config,
        quant_config=None,
        speculative_config=None,
        prefix: str = "",
    ) -> None:
        super().__init__(
            atom_config=atom_config,
            quant_config=quant_config,
            speculative_config=speculative_config,
            prefix=prefix,
        )
        self.model_config = atom_config.plugin_config.vllm_config.model_config
        self.cache_config = atom_config.plugin_config.vllm_config.cache_config
        self.tp_rank = get_tensor_model_parallel_rank()
        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    def get_state_dtype(self) -> tuple[torch.dtype, torch.dtype]:
        return MambaStateDtypeCalculator.gated_delta_net_state_dtype(
            self.model_config.dtype,
            self.cache_config.mamba_cache_dtype,
            self.cache_config.mamba_ssm_cache_dtype,
        )

    def get_state_shape(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return MambaStateShapeCalculator.gated_delta_net_state_shape(
            self.tp_size,
            self.num_k_heads,
            self.num_v_heads,
            self.head_k_dim,
            self.head_v_dim,
            self.conv_kernel_size,
            self.num_spec,
        )

    @property
    def mamba_type(self) -> MambaAttentionBackendEnum:
        return MambaAttentionBackendEnum.GDN_ATTN


class Qwen3_5ForCausalLM(Qwen3_5ForCausalLMBase):
    def __init__(self, *args, **kwargs):
        original_gdn_cls = qwen3_5_base.Qwen3_5GatedDeltaNet
        qwen3_5_base.Qwen3_5GatedDeltaNet = Qwen3_5GatedDeltaNetVllm
        try:
            super().__init__(*args, **kwargs)
        finally:
            qwen3_5_base.Qwen3_5GatedDeltaNet = original_gdn_cls


class Qwen3_5MoeForCausalLM(Qwen3_5MoeForCausalLMBase):
    def __init__(self, *args, **kwargs):
        original_gdn_cls = qwen3_5_base.Qwen3_5GatedDeltaNet
        qwen3_5_base.Qwen3_5GatedDeltaNet = Qwen3_5GatedDeltaNetVllm
        try:
            super().__init__(*args, **kwargs)
        finally:
            qwen3_5_base.Qwen3_5GatedDeltaNet = original_gdn_cls


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3VLMultiModalProcessor,
    info=Qwen3_5ProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
class Qwen3_5ForConditionalGeneration_(vLLMQwen3_5):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
        "gate_up_proj": ["gate_proj", "up_proj"],
        "in_proj_qkv": ("in_proj_qkvz", (0, 1, 2)),
        "in_proj_z": ("in_proj_qkvz", 3),
        "in_proj_b": ("in_proj_ba", 0),
        "in_proj_a": ("in_proj_ba", 1),
        ".gate.": (".gate.", 0),
        "shared_expert_gate": ("gate", 1),
    }

    hf_to_atom_mapper = WeightsMapper(
        orig_to_new_prefix={
            "model.visual.": "visual.",
            "lm_head.": "language_model.lm_head.",
            "model.language_model.": "language_model.model.",
        }
    )
    hf_to_vllm_mapper = hf_to_atom_mapper

    def __init__(self, atom_config: Config, prefix: str = "model"):
        nn.Module.__init__(self)
        config: Qwen3_5Config = atom_config.hf_config
        vllm_config = atom_config.plugin_config.vllm_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config
        self.atom_config = atom_config
        if (
            self.atom_config.quant_config.global_quant_config.quant_dtype
            == torch.bfloat16
        ):
            self.packed_modules_mapping.pop("in_proj_qkv")
            self.packed_modules_mapping.pop("in_proj_b")
            self.packed_modules_mapping.pop("in_proj_a")
            self.packed_modules_mapping["in_proj_qkv"] = (
                "in_proj_qkvzba",
                (0, 1, 2),
            )
            self.packed_modules_mapping["in_proj_z"] = ("in_proj_qkvzba", (3))
            self.packed_modules_mapping["in_proj_b"] = ("in_proj_qkvzba", (4))
            self.packed_modules_mapping["in_proj_a"] = ("in_proj_qkvzba", (5))

        self.config = config
        self.multimodal_config = multimodal_config
        self.use_data_parallel = multimodal_config.mm_encoder_tp_mode == "data"
        self.video_pruning_rate = multimodal_config.video_pruning_rate
        self.is_multimodal_pruning_enabled = (
            multimodal_config.is_multimodal_pruning_enabled()
        )
        with self._mark_tower_model(vllm_config, {"image", "video"}):
            self.visual = Qwen3_VisionTransformer(
                config.vision_config,
                norm_eps=getattr(config, "rms_norm_eps", 1e-6),
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "visual"),
            )

        with self._mark_language_model(vllm_config):
            self.language_model = Qwen3_5ForCausalLM(
                atom_config=atom_config,
                prefix=maybe_prefix("", "language_model"),
            )
        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loaded_weights_record = load_model_in_plugin_mode(
            model=self,
            config=self.atom_config,
            prefix="model.",
            weights_mapper=self.hf_to_atom_mapper,
        )
        return loaded_weights_record


class Qwen3_5MoeForConditionalGeneration_(vLLMQwen3_5Moe):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
        "gate_up_proj": ["gate_proj", "up_proj"],
        "in_proj_qkv": ("in_proj_qkvz", (0, 1, 2)),
        "in_proj_z": ("in_proj_qkvz", 3),
        "in_proj_b": ("in_proj_ba", 0),
        "in_proj_a": ("in_proj_ba", 1),
        ".gate.": (".gate.", 0),
        "shared_expert_gate": ("gate", 1),
    }

    hf_to_atom_mapper = WeightsMapper(
        orig_to_new_prefix={
            "model.visual.": "visual.",
            "lm_head.": "language_model.lm_head.",
            "model.language_model.": "language_model.model.",
        }
    )

    def __init__(self, atom_config: Config, prefix: str = "model"):
        nn.Module.__init__(self)
        self.atom_config = atom_config
        vllm_config = atom_config.plugin_config.vllm_config
        atom_config.hf_config.text_config.n_shared_experts = 1
        atom_config.hf_config.text_config.n_routed_experts = (
            atom_config.hf_config.text_config.num_experts
        )
        config: Qwen3_5MoeConfig = atom_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config
        if (
            self.atom_config.quant_config.global_quant_config.quant_dtype
            == torch.bfloat16
        ):
            self.packed_modules_mapping.pop("in_proj_qkv")
            self.packed_modules_mapping.pop("in_proj_b")
            self.packed_modules_mapping.pop("in_proj_a")
            self.packed_modules_mapping["in_proj_qkv"] = (
                "in_proj_qkvzba",
                (0, 1, 2),
            )
            self.packed_modules_mapping["in_proj_z"] = ("in_proj_qkvzba", (3))
            self.packed_modules_mapping["in_proj_b"] = ("in_proj_qkvzba", (4))
            self.packed_modules_mapping["in_proj_a"] = ("in_proj_qkvzba", (5))

        self.config = config
        self.multimodal_config = multimodal_config
        self.use_data_parallel = multimodal_config.mm_encoder_tp_mode == "data"
        self.video_pruning_rate = multimodal_config.video_pruning_rate
        self.is_multimodal_pruning_enabled = (
            multimodal_config.is_multimodal_pruning_enabled()
        )

        with self._mark_tower_model(vllm_config, {"image", "video"}):
            self.visual = Qwen3_VisionTransformer(
                config.vision_config,
                norm_eps=getattr(config, "rms_norm_eps", 1e-6),
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "visual"),
            )

        with self._mark_language_model(vllm_config):
            self.language_model = Qwen3_5MoeForCausalLM(
                atom_config=atom_config, prefix=maybe_prefix("", "language_model")
            )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

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
        loaded_weights_record = load_model_in_plugin_mode(
            model=self,
            config=self.atom_config,
            prefix="model.",
            weights_mapper=self.hf_to_atom_mapper,
            load_fused_expert_weights_fn=self.load_fused_expert_weights,
        )
        return loaded_weights_record

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.language_model.get_expert_mapping()

    def embed_multimodal(self, **kwargs):
        return super().embed_multimodal(**kwargs)


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3VLMultiModalProcessor,
    info=Qwen3_5ProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
class Qwen3_5ForConditionalGeneration(ATOMForConditionalGeneration, IsHybrid):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
        "gate_up_proj": ["gate_proj", "up_proj"],
        "in_proj_qkv": ("in_proj_qkvz", (0, 1, 2)),
        "in_proj_z": ("in_proj_qkvz", 3),
        "in_proj_b": ("in_proj_ba", 0),
        "in_proj_a": ("in_proj_ba", 1),
        ".gate.": (".gate.", 0),
        "shared_expert_gate": ("gate", 1),
    }

    hf_to_atom_mapper = WeightsMapper(
        orig_to_new_prefix={
            "lm_head.": "language_model.lm_head.",
            "model.language_model.": "language_model.model.",
        }
    )
    hf_to_vllm_mapper = hf_to_atom_mapper

    def embed_multimodal(self, **kwargs):
        return self.model.embed_multimodal(**kwargs)

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return "<|vision_start|><|image_pad|><|vision_end|>"
        if modality.startswith("video"):
            return "<|vision_start|><|video_pad|><|vision_end|>"

        raise ValueError("Only image or video modality is supported")

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls,
        vllm_config: VllmConfig,
    ) -> tuple[torch.dtype, torch.dtype]:
        return MambaStateDtypeCalculator.gated_delta_net_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
            vllm_config.cache_config.mamba_ssm_cache_dtype,
        )

    @classmethod
    def get_mamba_state_shape_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        parallel_config = vllm_config.parallel_config
        hf_config = vllm_config.model_config.hf_text_config
        tp_size = parallel_config.tensor_parallel_size
        num_spec = (
            vllm_config.speculative_config.num_speculative_tokens
            if vllm_config.speculative_config
            else 0
        )
        return MambaStateShapeCalculator.gated_delta_net_state_shape(
            tp_size,
            hf_config.linear_num_key_heads,
            hf_config.linear_num_value_heads,
            hf_config.linear_key_head_dim,
            hf_config.linear_value_head_dim,
            hf_config.linear_conv_kernel_dim,
            num_spec,
        )

    @classmethod
    def get_mamba_state_copy_func(
        cls,
    ) -> tuple[MambaStateCopyFunc, MambaStateCopyFunc]:
        return MambaStateCopyFuncCalculator.gated_delta_net_state_copy_func()

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> set[str]:
        return self.model.load_weights(weights)


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3VLMultiModalProcessor,
    info=Qwen3_5MoeProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
class Qwen3_5MoeForConditionalGeneration(Qwen3_5ForConditionalGeneration, IsHybrid):
    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.model.get_expert_mapping()
