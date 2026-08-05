from dataclasses import replace
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
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
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
from atom.models.utils import IntermediateTensors, maybe_prefix
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
    return MambaStateShapeCalculator.gated_delta_net_state_shape(
        vllm_config.parallel_config.tensor_parallel_size,
        config.linear_num_key_heads,
        config.linear_num_value_heads,
        config.linear_key_head_dim,
        config.linear_value_head_dim,
        config.linear_conv_kernel_dim,
        num_spec,
    )


def _get_k3_state_dtype(vllm_config: VllmConfig) -> tuple[torch.dtype, torch.dtype]:
    conv_dtype, _ = MambaStateDtypeCalculator.gated_delta_net_state_dtype(
        vllm_config.model_config.dtype,
        vllm_config.cache_config.mamba_cache_dtype,
        vllm_config.cache_config.mamba_ssm_cache_dtype,
    )
    # FLA KDA accumulates the recurrent matrix in fp32 and rejects lower
    # precision initial state. Native ATOM uses the same fp32 state contract.
    return conv_dtype, torch.float32


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

        self._atom_metadata = SimpleNamespace(gdn_metadata=None)
        self._atom_cache = SimpleNamespace(k_cache=None, v_cache=None)
        self._atom_kv_cache_data = {f"layer_{self.layer_num}": self._atom_cache}

    def process_weights_after_loading(self, *args, **kwargs) -> None:
        # Newer vLLM (model_loader/utils.py) calls
        # process_weights_after_loading(model_config.dtype) on every
        # AttentionLayerBase; native KimiKDAAttention takes no argument. Absorb
        # the extra positional so KDA post-load folding still runs.
        return super().process_weights_after_loading()

    def get_state_dtype(self) -> tuple[torch.dtype, torch.dtype]:
        return _get_k3_state_dtype(self.vllm_config)

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

    def _forward_segments(
        self,
        hidden_states: torch.Tensor,
        hidden_states_scale: torch.Tensor | None,
        gdn_metadata: GDNAttentionMetadata,
    ) -> torch.Tensor:
        """Run the native KDA layer once per request class in the batch.

        The native layer takes one branch per call -- prefill, decode, or
        speculative decode -- and writes only that class's rows into an
        uninitialized output. Its own runtime batches by class, so a single
        branch always covers everything; vLLM's continuous batching does not.
        It never mixes plain and speculative decodes (it folds the former into
        the prefill counts), but a request still prefilling while another
        drafts is routine, and there the speculative rows would come back as
        whatever the allocator last left there.

        The builder already splits every per-class input -- token indices,
        zero-based ``query_start_loc``, state indices -- so each class runs as
        if it were the whole batch and the rows are scattered back afterwards.
        """
        mixed = gdn_metadata.num_spec_decodes > 0 and (
            gdn_metadata.num_prefills > 0 or gdn_metadata.num_decodes > 0
        )
        if not mixed:
            self._atom_metadata.gdn_metadata = gdn_metadata
            return super()._forward_impl(hidden_states, hidden_states_scale)

        spec_indx = gdn_metadata.spec_token_indx
        non_spec_indx = gdn_metadata.non_spec_token_indx
        assert spec_indx is not None and non_spec_indx is not None
        rows = hidden_states[: gdn_metadata.num_actual_tokens]

        def _scale_rows(index: torch.Tensor) -> torch.Tensor | None:
            if hidden_states_scale is None:
                return None
            return hidden_states_scale[: gdn_metadata.num_actual_tokens].index_select(
                0, index
            )

        spec_out = self._forward_one_segment(
            rows.index_select(0, spec_indx),
            _scale_rows(spec_indx),
            replace(
                gdn_metadata,
                num_prefills=0,
                num_prefill_tokens=0,
                num_decodes=0,
                num_decode_tokens=0,
                num_actual_tokens=gdn_metadata.num_spec_decode_tokens,
            ),
        )
        non_spec_out = self._forward_one_segment(
            rows.index_select(0, non_spec_indx),
            _scale_rows(non_spec_indx),
            replace(
                gdn_metadata,
                num_spec_decodes=0,
                num_spec_decode_tokens=0,
                num_actual_tokens=(
                    gdn_metadata.num_prefill_tokens + gdn_metadata.num_decode_tokens
                ),
            ),
        )

        merged = spec_out.new_empty((rows.shape[0], spec_out.shape[-1]))
        merged.index_copy_(0, spec_indx, spec_out)
        merged.index_copy_(0, non_spec_indx, non_spec_out)
        return merged

    def _forward_one_segment(
        self,
        hidden_states: torch.Tensor,
        hidden_states_scale: torch.Tensor | None,
        gdn_metadata: GDNAttentionMetadata,
    ) -> torch.Tensor:
        self._atom_metadata.gdn_metadata = gdn_metadata
        return super()._forward_impl(hidden_states, hidden_states_scale)

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
        gdn_metadata = attn_metadata[self.layer_name]
        if not isinstance(gdn_metadata, GDNAttentionMetadata):
            raise TypeError(
                f"Expected GDNAttentionMetadata for {self.layer_name}, "
                f"got {type(gdn_metadata).__name__}"
            )

        vllm_layer = vllm_context.no_compile_layers[self.layer_name]
        conv_state, ssm_state = vllm_layer.kv_cache
        self._atom_cache.k_cache = conv_state
        self._atom_cache.v_cache = ssm_state

        atom_context = get_atom_forward_context()
        previous_metadata = atom_context.attn_metadata
        previous_kv_cache_data = atom_context.kv_cache_data
        atom_context.attn_metadata = self._atom_metadata
        atom_context.kv_cache_data = self._atom_kv_cache_data
        try:
            output = self._forward_segments(
                hidden_states, hidden_states_scale, gdn_metadata
            )
        finally:
            atom_context.attn_metadata = previous_metadata
            atom_context.kv_cache_data = previous_kv_cache_data

        # vLLM pads token rows to the selected piecewise/full graph bucket,
        # while GDN metadata tracks only real tokens. The native KDA path
        # intentionally slices to num_actual_tokens; restore the graph bucket
        # width so this custom op matches its fake implementation's output shape.
        if output.shape[0] < hidden_states.shape[0]:
            output = torch.nn.functional.pad(
                output,
                (0, 0, 0, hidden_states.shape[0] - output.shape[0]),
            )
        return output


def _aux_hidden_state_tap(original_forward, sink: list, slot: int):
    """Record the hidden state entering a layer, then run it unchanged.

    ``pending_add`` carries a residual the native loop defers to the next
    layer, so the value actually entering this layer is the sum of the two.
    """

    def forward(positions, hidden_states, block_residual=None, pending_add=None):
        sink[slot] = (
            hidden_states if pending_add is None else hidden_states + pending_add
        )
        return original_forward(
            positions, hidden_states, block_residual, pending_add=pending_add
        )

    return forward


class KimiLinearModelVllm(kimi_k3_base.KimiLinearModel):
    """Native Kimi-K3 body that can also emit Eagle3/DSpark auxiliary states.

    The DSpark draft is trained on the outputs of five target layers. Those are
    collected by tapping the chosen layers once, when the layer set is
    configured, rather than by restating the native forward -- whose deferred
    residual and attention-residual bookkeeping is easy to get subtly wrong and
    would silently drift from the original.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.aux_hidden_state_layers: tuple[int, ...] = ()
        # One slot per layer, allocated once. The taps close over this list, so
        # replacing it would leave them writing somewhere nothing reads --
        # including inside an already-traced graph.
        self._aux_hidden_states: list[torch.Tensor | None] = [None] * len(self.layers)

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        for layer in self.layers:
            original = layer.__dict__.pop("_atom_aux_untapped_forward", None)
            if original is not None:
                layer.forward = original

        self.aux_hidden_state_layers = tuple(layers)
        # Fixed slots rather than appends: the collection order then matches the
        # configured layer order no matter how the taps fire.
        for slot in range(len(self._aux_hidden_states)):
            self._aux_hidden_states[slot] = None
        for slot, idx in enumerate(self.aux_hidden_state_layers):
            layer = self.layers[idx]
            layer._atom_aux_untapped_forward = layer.forward
            layer.forward = _aux_hidden_state_tap(
                layer.forward, self._aux_hidden_states, slot
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ):
        # The signature must stay explicit: ATOM's compile decorator marks the
        # dynamic token dimension by binding these argument names, and a
        # *args/**kwargs override would silently compile the model at a single
        # static shape.
        hidden_states = super().forward(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )
        if not self.aux_hidden_state_layers:
            return hidden_states
        if isinstance(hidden_states, IntermediateTensors):
            return hidden_states
        collected = self._aux_hidden_states[: len(self.aux_hidden_state_layers)]
        missing = [
            self.aux_hidden_state_layers[i]
            for i, h in enumerate(collected)
            if h is None
        ]
        if missing:
            raise RuntimeError(
                f"Kimi-K3 aux hidden states were not produced by layers {missing}."
            )
        return hidden_states, collected


class KimiK3ForCausalLM(KimiK3ForCausalLMBase):
    def __init__(self, *args, **kwargs):
        original_kda_cls = kimi_k3_base.KimiKDAAttention
        original_model_cls = kimi_k3_base.KimiLinearModel
        kimi_k3_base.KimiKDAAttention = KimiKDAAttentionVllm
        kimi_k3_base.KimiLinearModel = KimiLinearModelVllm
        try:
            super().__init__(*args, **kwargs)
        finally:
            kimi_k3_base.KimiKDAAttention = original_kda_cls
            kimi_k3_base.KimiLinearModel = original_model_cls

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        # Required by vLLM SupportsMultiModal.get_language_model discovery.
        return self.get_input_embeddings(input_ids)

    @property
    def model(self) -> nn.Module:
        """The body the DSpark loader reaches for ``embed_tokens`` on.

        ``load_dspark_model`` reads ``get_language_model().model.embed_tokens``
        and ``get_language_model().lm_head``. Multimodal discovery stops at this
        class, one level above the causal LM that actually owns both, so expose
        them here rather than redirecting ``get_language_model`` -- which the
        vision path also uses. A property, so it stays out of ``named_modules``
        and the checkpoint prefixes are unchanged.
        """
        return self.language_model.model

    @property
    def lm_head(self) -> nn.Module:
        """The head the DSpark draft borrows to score its block."""
        return self.language_model.lm_head

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        self.language_model.model.set_aux_hidden_state_layers(layers)

    def get_eagle3_aux_hidden_state_layers(self) -> tuple[int, ...]:
        """Fallback only; the DSpark checkpoint names its own target layers."""
        num_layers = len(self.language_model.model.layers)
        return (2, num_layers // 2, num_layers - 3)


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

    # ATOMModelBase probes the inner model for the ATOM server-mode aux-hidden
    # -state interface when the target drives EAGLE3 or DSpark. The taps live on
    # the text body, one level further in.
    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        self.language_model.set_aux_hidden_state_layers(layers)

    def get_eagle3_aux_hidden_state_layers(self) -> tuple[int, ...]:
        return self.language_model.get_eagle3_aux_hidden_state_layers()


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
        return MambaStateCopyFuncCalculator.gated_delta_net_state_copy_func()
