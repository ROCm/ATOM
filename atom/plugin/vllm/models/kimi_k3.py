from types import SimpleNamespace
from typing import Any, ClassVar

import torch
from torch import nn
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.forward_context import get_forward_context as get_vllm_forward_context
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc,
    MambaStateCopyFuncCalculator,
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.models.interfaces import IsHybrid
from vllm.model_executor.models.kimi_k25_vit import (
    KimiK25MultiModalProjector,
    MoonViT3dPretrainedModel,
)
from vllm.model_executor.models.vision import is_vit_use_data_parallel
from vllm.models.kimi_k3 import (
    KimiK3ForConditionalGeneration as vLLMKimiK3,
)
from vllm.models.kimi_k3.common.mm_preprocess import (
    KimiK3DummyInputsBuilder,
    KimiK3MultiModalProcessor,
    KimiK3ProcessingInfo,
)
from vllm.models.kimi_k3.nvidia.kda_metadata import KimiK3KDAMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum

from atom.config import Config, QuantizationConfig
from atom.model_loader.loader import WeightsMapper, load_model_in_plugin_mode
from atom.models import kimi_k3 as kimi_k3_base
from atom.models.kimi_k3 import (
    KimiK3ForCausalLM as KimiK3ForCausalLMBase,
)
from atom.models.kimi_k3 import (
    KimiKDAAttention,
    _normalize_kimi_config,
)
from atom.models.utils import maybe_prefix
from atom.plugin.vllm.kda_backend import AtomKimiK3KDAAttentionBackend
from atom.plugin.vllm.model_wrapper import ATOMForConditionalGeneration
from atom.utils.forward_context import get_forward_context as get_atom_forward_context


def _get_k3_state_shape(
    vllm_config: VllmConfig,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    config = vllm_config.model_config.hf_text_config
    _normalize_kimi_config(config)
    num_spec = (
        vllm_config.speculative_config.num_speculative_tokens
        if vllm_config.speculative_config
        else 0
    )
    return MambaStateShapeCalculator.kda_state_shape(
        tp_world_size=vllm_config.parallel_config.tensor_parallel_size,
        num_heads=config.linear_num_value_heads,
        head_dim=config.linear_value_head_dim,
        num_k_heads=config.linear_num_key_heads,
        head_k_dim=config.linear_key_head_dim,
        conv_kernel_size=config.linear_conv_kernel_dim,
        num_spec=num_spec,
    )


def _get_k3_state_dtype(vllm_config: VllmConfig) -> tuple[torch.dtype, torch.dtype]:
    return MambaStateDtypeCalculator.kda_state_dtype(
        vllm_config.model_config.dtype,
        vllm_config.cache_config.mamba_cache_dtype,
    )


class KimiKDAAttentionVllm(KimiKDAAttention, MambaBase):
    """Kimi-K3 KDA layer backed by vLLM-owned recurrent state."""

    def __init__(self, atom_config, quant_config=None, prefix: str = "") -> None:
        super().__init__(
            atom_config=atom_config,
            quant_config=quant_config,
            prefix=prefix,
        )
        vllm_config = atom_config.plugin_config.vllm_config
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.num_k_heads = self.num_heads
        self.num_v_heads = self.num_heads
        self.head_k_dim = self.head_dim
        self.head_v_dim = self.head_dim
        self.num_spec = (
            vllm_config.speculative_config.num_speculative_tokens
            if vllm_config.speculative_config
            else 0
        )

        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

        self._atom_metadata = SimpleNamespace(kda_metadata=None)
        self._atom_cache = SimpleNamespace(k_cache=None, v_cache=None)
        self._atom_kv_cache_data = {f"layer_{self.layer_num}": self._atom_cache}

    def process_weights_after_loading(self, *args, **kwargs) -> None:
        """Accept vLLM's activation dtype and run native KDA post-load folding."""
        return super().process_weights_after_loading()

    def get_state_dtype(self) -> tuple[torch.dtype, torch.dtype]:
        return _get_k3_state_dtype(self.vllm_config)

    def get_state_shape(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return MambaStateShapeCalculator.kda_state_shape(
            tp_world_size=self.tp_size,
            num_heads=self.num_v_heads,
            head_dim=self.head_v_dim,
            num_k_heads=self.num_k_heads,
            head_k_dim=self.head_k_dim,
            conv_kernel_size=self.conv_kernel_size,
            num_spec=self.num_spec,
        )

    def get_attn_backend(self) -> type[AttentionBackend]:
        return AtomKimiK3KDAAttentionBackend

    @property
    def mamba_type(self) -> MambaAttentionBackendEnum:
        # KDA shares vLLM's GDN cache specification, but uses its own metadata
        # backend via get_attn_backend().
        return MambaAttentionBackendEnum.GDN_ATTN

    def _forward_impl(
        self,
        hidden_states: torch.Tensor,
        hidden_states_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        vllm_context = get_vllm_forward_context()
        attn_metadata = vllm_context.attn_metadata
        if attn_metadata is None:
            return torch.zeros(
                hidden_states.shape, dtype=torch.bfloat16, device=hidden_states.device
            )

        if not isinstance(attn_metadata, dict):
            raise TypeError("Kimi-K3 vLLM attention metadata must be layer-indexed")
        kda_metadata = attn_metadata[self.layer_name]
        if not isinstance(kda_metadata, KimiK3KDAMetadata):
            raise TypeError(
                f"Expected KimiK3KDAMetadata for {self.layer_name}, "
                f"got {type(kda_metadata).__name__}"
            )

        vllm_layer = vllm_context.no_compile_layers[self.layer_name]
        conv_state, ssm_state = vllm_layer.kv_cache
        self._atom_metadata.kda_metadata = kda_metadata
        self._atom_cache.k_cache = conv_state
        self._atom_cache.v_cache = ssm_state

        atom_context = get_atom_forward_context()
        previous_metadata = atom_context.attn_metadata
        previous_kv_cache_data = atom_context.kv_cache_data
        atom_context.attn_metadata = self._atom_metadata
        atom_context.kv_cache_data = self._atom_kv_cache_data
        try:
            output = super()._forward_impl(hidden_states, hidden_states_scale)
        finally:
            atom_context.attn_metadata = previous_metadata
            atom_context.kv_cache_data = previous_kv_cache_data

        # vLLM pads token rows to the selected piecewise/full graph bucket,
        # while KDA metadata tracks only real tokens. The native KDA path
        # intentionally slices to num_actual_tokens; restore the graph bucket
        # width so this custom op matches its fake implementation's output shape.
        if output.shape[0] < hidden_states.shape[0]:
            output = torch.nn.functional.pad(
                output,
                (0, 0, 0, hidden_states.shape[0] - output.shape[0]),
            )
        return output


class KimiK3ForCausalLM(KimiK3ForCausalLMBase):
    def __init__(self, *args, **kwargs):
        original_kda_cls = kimi_k3_base.KimiKDAAttention
        kimi_k3_base.KimiKDAAttention = KimiKDAAttentionVllm
        try:
            super().__init__(*args, **kwargs)
        finally:
            kimi_k3_base.KimiKDAAttention = original_kda_cls

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        # Required by vLLM SupportsMultiModal.get_language_model discovery.
        return self.get_input_embeddings(input_ids)


@MULTIMODAL_REGISTRY.register_processor(
    KimiK3MultiModalProcessor,
    info=KimiK3ProcessingInfo,
    dummy_inputs=KimiK3DummyInputsBuilder,
)
class KimiK3ForConditionalGeneration_(vLLMKimiK3):
    hf_to_atom_mapper = WeightsMapper(
        orig_to_new_prefix={
            # ATOM nests language_model.language_model (ATOM KimiK3ForCausalLM
            # wraps KimiLinearForCausalLM); checkpoint stores language_model.model.
            "language_model.": "language_model.language_model.",
            "mm_projector.proj.0": "mm_projector.linear_1",
            "mm_projector.proj.2": "mm_projector.linear_2",
        }
    )
    # ATOMModelBase.__init__ reads these off the inner model class to drive
    # quant remapping. This inner subclasses vLLM's upstream MM class (not the
    # native K3 class), so the attributes must be declared explicitly or the
    # fused-projection remap and exclude translation silently no-op on a
    # quantized checkpoint. Suffix-based and nesting-agnostic, so reuse native.
    packed_modules_mapping = KimiK3ForCausalLMBase.packed_modules_mapping
    # Native K3's exclude values use NO `model.` prefix (identity-style, relative
    # to the inner-class root). This inner adds ONE extra `language_model.` level
    # because self.language_model is the ATOM KimiK3ForCausalLM, which itself
    # nests a KimiLinearForCausalLM at .language_model — so text-layer paths are
    # language_model.language_model.model.* / .lm_head.
    quant_exclude_name_mapping: ClassVar[dict[str, str]] = {
        "language_model.model.": "language_model.language_model.model.",
        "language_model.lm_head": "language_model.language_model.lm_head",
    }

    def __init__(self, atom_config: Config, prefix: str = "model"):
        # Protocols have no __init__; skip vLLMKimiK3.__init__ (it would build
        # vLLM's KimiLinear language model) and set up nn.Module directly.
        nn.Module.__init__(self)
        hf_config = getattr(atom_config, "hf_config", None)
        assert hf_config is not None, "hf_config is not found in atom_config"
        vision_config = hf_config.vision_config
        self.config = hf_config
        self.atom_config = atom_config

        vllm_config = atom_config.plugin_config.vllm_config
        quant_config = vllm_config.quant_config
        atom_quant_config = atom_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config
        self.multimodal_config = multimodal_config
        self.use_data_parallel = is_vit_use_data_parallel(
            vision_config.num_attention_heads
        )

        with self._mark_tower_model(vllm_config, "image"):
            self.vision_tower = MoonViT3dPretrainedModel(
                vision_config,
                quant_config=self._maybe_ignore_quant_config(
                    quant_config,
                    atom_quant_config.exclude_layers or [],
                    "vision_tower",
                ),
                prefix=maybe_prefix(prefix, "vision_tower"),
            )
            self.mm_projector = KimiK25MultiModalProjector(
                config=vision_config,
                use_data_parallel=self.use_data_parallel,
                quant_config=self._maybe_ignore_quant_config(
                    quant_config,
                    atom_quant_config.exclude_layers or [],
                    "mm_projector",
                ),
                prefix=maybe_prefix(prefix, "mm_projector"),
            )

        self.quant_config = quant_config
        with self._mark_language_model(vllm_config):
            self.language_model = KimiK3ForCausalLM(
                atom_config=atom_config,
                prefix=maybe_prefix(prefix, "language_model"),
            )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )
        self.media_placeholder = self.config.media_placeholder_token_id

        # load_model reads packed_modules_mapping / weights_mapping off the top
        # model (this inner class), but the KDA-aware fusion is only known at the
        # nested language model's instance level: KimiK3ForCausalLM sets
        # packed_modules_mapping = _kda_packed_modules_mapping(kimi_kda_layers)
        # (fuses self_attn.{q,k,v,g}_proj -> in_proj) and carries the
        # compressed-tensors weight_packed->weight rename. Surface both here so
        # the plugin loader applies them (the class-level packed_modules_mapping
        # kept for ATOMModelBase's pre-init quant remap has an empty KDA list).
        self.packed_modules_mapping = self.language_model.packed_modules_mapping
        self.weights_mapping = self.language_model.weights_mapping

    def _maybe_ignore_quant_config(
        self, quant_config: Any, exclude_layers: list[str], layer_name: str
    ):
        for exclude_layer in exclude_layers:
            if QuantizationConfig._matches_exclude(
                layer_name, exclude_layer, check_contains=True
            ):
                return None
        return quant_config

    def load_weights(self, weights):
        # weights generator is discarded; ATOM loads from disk in plugin mode.
        # prefix="model." because ATOMModelBase constructs this as `.model`.
        return load_model_in_plugin_mode(
            model=self,
            config=self.atom_config,
            prefix="model.",
            weights_mapper=self.hf_to_atom_mapper,
        )

    def get_expert_mapping(self):
        return self.language_model.get_expert_mapping()


@MULTIMODAL_REGISTRY.register_processor(
    KimiK3MultiModalProcessor,
    info=KimiK3ProcessingInfo,
    dummy_inputs=KimiK3DummyInputsBuilder,
)
class KimiK3ForCausalLMVllm(ATOMForConditionalGeneration, IsHybrid):
    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality == "image":
            return "<|kimi_image_placeholder|>"
        raise ValueError(f"Unsupported modality: {modality}")

    def load_weights(self, weights):
        return self.model.load_weights(weights)

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls,
        vllm_config: VllmConfig,
    ) -> tuple[torch.dtype, torch.dtype]:
        return _get_k3_state_dtype(vllm_config)

    @classmethod
    def get_mamba_state_shape_from_config(
        cls,
        vllm_config: VllmConfig,
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return _get_k3_state_shape(vllm_config)

    @classmethod
    def get_mamba_state_copy_func(
        cls,
    ) -> tuple[MambaStateCopyFunc, MambaStateCopyFunc]:
        return MambaStateCopyFuncCalculator.kda_state_copy_func()
