from dataclasses import replace
from types import SimpleNamespace
from typing import Any, ClassVar

import torch
from aiter.dist.parallel_state import get_pp_group
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
    KimiDecoderLayer,
    KimiKDAAttention,
    _normalize_kimi_config,
)
from atom.models.kimi_k3 import (
    KimiK3ForCausalLM as KimiK3ForCausalLMBase,
)
from atom.models.utils import IntermediateTensors, maybe_prefix
from atom.plugin.vllm.kda_backend import AtomKimiK3KDAAttentionBackend
from atom.plugin.vllm.model_wrapper import ATOMForConditionalGeneration
from atom.utils.forward_context import get_forward_context as get_atom_forward_context

_KIMI_K3_QUANT_EXCLUDE_NAME_MAPPING: dict[str, str] = {
    "language_model.model.": "language_model.language_model.model.",
    "language_model.lm_head": "language_model.language_model.lm_head",
}


def _adapt_kda_metadata_for_atom(kda_metadata: KimiK3KDAMetadata) -> None:
    """Provide the state-read index field expected by native ATOM KDA."""
    kda_metadata.non_spec_state_indices_in_tensor = (  # type: ignore[attr-defined]
        kda_metadata.non_spec_state_indices_tensor
    )


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


from vllm.compilation.breakable_cudagraph import eager_break_during_capture

from atom.config import get_current_atom_config
from atom.utils import mark_spliting_op


def _kda_attention_vllm_fake(
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor | None,
    layer_name: str,
    output: torch.Tensor,
) -> None:
    return None


@mark_spliting_op(
    is_custom=True,
    gen_fake=_kda_attention_vllm_fake,
    mutates_args=["output"],
)
def kda_attention_vllm(
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor | None,
    layer_name: str,
    output: torch.Tensor,
) -> None:
    """Opaque splitting-op boundary for the KDA mixer, vLLM plugin flavour.

    Mirrors ``aiter.kda_attention_with_output`` but takes an ``output`` buffer
    instead of returning one, which ``@eager_break_during_capture`` requires.
    Registering it here rather than widening the shared op keeps the native
    ATOM and SGLang paths on the original signature.
    """
    self = get_current_atom_config().compilation_config.static_forward_context[
        layer_name
    ]
    self._kda_run(hidden_states, hidden_states_scale, output)


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
        self._packed_start_loc = None

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

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states_scale = None
        if isinstance(hidden_states, tuple):
            hidden_states, hidden_states_scale = hidden_states
        output = torch.empty(
            hidden_states.shape, dtype=torch.bfloat16, device=hidden_states.device
        )
        torch.ops.aiter.kda_attention_vllm(
            hidden_states, hidden_states_scale, self.layer_name, output
        )
        return output

    @eager_break_during_capture
    def _kda_run(
        self,
        hidden_states: torch.Tensor,
        hidden_states_scale: torch.Tensor | None,
        output: torch.Tensor,
    ) -> None:
        """Run the mixer outside the piecewise cudagraph, writing into ``output``.

        A capture batch under DSpark carries ``1 + num_spec`` tokens per request,
        which ``decode_threshold=1`` reads as prefill, so capture takes the
        chunked KDA path. That path builds ``chunk_indices`` from this batch's
        ``cu_seqlens`` via a host readback -- illegal mid-capture, and frozen
        into every replay if the readback is dodged. Breaking out re-runs the
        mixer per replay against live metadata instead, which is what vLLM's own
        Kimi GDN linear attention does.

        Full decode graphs are untouched: they reach the builder through
        ``build_for_cudagraph_capture``, which synthesises the draft counts and
        lands on the fused spec path.
        """
        output.copy_(self._forward_impl(hidden_states, hidden_states_scale))

    def _forward_segments(
        self,
        hidden_states: torch.Tensor,
        hidden_states_scale: torch.Tensor | None,
        kda_metadata: KimiK3KDAMetadata,
    ) -> torch.Tensor:
        """Run the native KDA layer once per request class in the batch.

        The native layer takes one branch per call -- prefill, decode, or
        speculative decode -- and writes only that class's rows into an
        uninitialized output, which suffices for a runtime that batches by class.
        Under vLLM's continuous batching a request prefilling while another
        drafts is routine, and the speculative rows would come back as whatever
        the allocator last left behind.

        The builder splits every per-class input (token indices, zero-based
        ``query_start_loc``, state indices), so each class runs as if it were the
        whole batch and the rows are scattered back afterwards.
        """
        mixed = kda_metadata.num_spec_decodes > 0 and (
            kda_metadata.num_prefills > 0 or kda_metadata.num_decodes > 0
        )
        if not mixed:
            return self._forward_one_segment(
                hidden_states, hidden_states_scale, kda_metadata
            )

        spec_indx = kda_metadata.spec_token_indx
        non_spec_indx = kda_metadata.non_spec_token_indx
        assert spec_indx is not None and non_spec_indx is not None
        rows = hidden_states[: kda_metadata.num_actual_tokens]

        def _scale_rows(index: torch.Tensor) -> torch.Tensor | None:
            if hidden_states_scale is None:
                return None
            return hidden_states_scale[: kda_metadata.num_actual_tokens].index_select(
                0, index
            )

        spec_out = self._forward_one_segment(
            rows.index_select(0, spec_indx),
            _scale_rows(spec_indx),
            replace(
                kda_metadata,
                num_prefills=0,
                num_prefill_tokens=0,
                num_decodes=0,
                num_decode_tokens=0,
                num_actual_tokens=kda_metadata.num_spec_decode_tokens,
            ),
        )
        non_spec_out = self._forward_one_segment(
            rows.index_select(0, non_spec_indx),
            _scale_rows(non_spec_indx),
            replace(
                kda_metadata,
                num_spec_decodes=0,
                num_spec_decode_tokens=0,
                num_actual_tokens=(
                    kda_metadata.num_prefill_tokens + kda_metadata.num_decode_tokens
                ),
                non_spec_query_start_loc=(
                    kda_metadata.non_spec_query_start_loc
                    if kda_metadata.non_spec_query_start_loc is not None
                    else self._packed_decode_start_loc(
                        kda_metadata.num_decodes, hidden_states.device
                    )
                ),
            ),
        )

        merged = spec_out.new_empty((rows.shape[0], spec_out.shape[-1]))
        merged.index_copy_(0, spec_indx, spec_out)
        merged.index_copy_(0, non_spec_indx, non_spec_out)
        return merged

    def _packed_decode_start_loc(
        self, num_decodes: int, device: torch.device
    ) -> torch.Tensor:
        """``cu_seqlens`` for a decode-only segment: one token per request.

        vLLM leaves ``non_spec_query_start_loc`` unset there, its packed-decode
        kernel taking no cumulative lengths where ATOM's fused recurrence does.
        Allocated at scheduler capacity and only sliced, so a captured graph keeps
        pointing at the same memory.
        """
        if self._packed_start_loc is None:
            self._packed_start_loc = torch.arange(
                self.vllm_config.scheduler_config.max_num_seqs + 1,
                dtype=torch.int32,
                device=device,
            )
        return self._packed_start_loc[: num_decodes + 1]

    def _forward_one_segment(
        self,
        hidden_states: torch.Tensor,
        hidden_states_scale: torch.Tensor | None,
        kda_metadata: KimiK3KDAMetadata,
    ) -> torch.Tensor:
        self._atom_metadata.kda_metadata = kda_metadata
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
        kda_metadata = attn_metadata[self.layer_name]
        if not isinstance(kda_metadata, KimiK3KDAMetadata):
            raise TypeError(
                f"Expected KimiK3KDAMetadata for {self.layer_name}, "
                f"got {type(kda_metadata).__name__}"
            )
        _adapt_kda_metadata_for_atom(kda_metadata)

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
                hidden_states, hidden_states_scale, kda_metadata
            )
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


def _k3_residual_stream(
    hidden_states: torch.Tensor,
    pending_add: torch.Tensor | None,
    pending_add2: torch.Tensor | None,
    block_residual: torch.Tensor | None,
) -> torch.Tensor:
    """The plain residual stream at a layer boundary, reconstructed by the layer
    itself -- its return protocol defines which tensors are deferred addends."""
    return KimiDecoderLayer.aux_hidden_state(
        (hidden_states, pending_add, pending_add2, block_residual)
    )


class KimiLinearModelVllm(kimi_k3_base.KimiLinearModel):
    """Native Kimi-K3 body that can also emit Eagle3/DSpark auxiliary states.

    Collected inline in the layer loop: tapping wrapped layer ``forward``s splits
    this function across a Dynamo graph break, and ATOM's compile backend accepts
    exactly one graph.

    ``aux_hidden_state_layers`` indexes the residual stream, not the layers:
    entry ``i`` is the stream entering layer ``i``, i.e. the reference model's
    ``output.hidden_states[i]``. vLLM adds one to the draft's
    ``target_layer_ids`` before handing them over.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.aux_hidden_state_layers: tuple[int, ...] = ()

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        self.aux_hidden_state_layers = tuple(layers)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ):
        # The signature must stay explicit: ATOM's compile decorator marks the
        # dynamic token dimension by binding these argument names, so a
        # *args/**kwargs override would compile at a single static shape.
        if not self.aux_hidden_state_layers:
            return super().forward(
                input_ids, positions, intermediate_tensors, inputs_embeds
            )

        if get_pp_group().is_first_rank:
            hidden_states = (
                inputs_embeds
                if inputs_embeds is not None
                else self.embed_tokens(input_ids)
            )
            block_residual = (
                hidden_states.new_zeros(
                    hidden_states.shape[0], 0, hidden_states.shape[1]
                )
                if getattr(self.config, "attn_res_block_size", None) is not None
                else None
            )
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            block_residual = intermediate_tensors["block_residual"]

        aux_hidden_states: list[torch.Tensor] = []
        pending_add = pending_add2 = None
        for idx in range(self.start_layer, self.end_layer):
            if idx in self.aux_hidden_state_layers:
                aux_hidden_states.append(
                    _k3_residual_stream(
                        hidden_states, pending_add, pending_add2, block_residual
                    )
                )
            hidden_states, pending_add, pending_add2, block_residual = self.layers[idx](
                positions,
                hidden_states,
                block_residual,
                pending_add=pending_add,
                pending_add2=pending_add2,
            )

        if not get_pp_group().is_last_rank:
            hidden_states = _k3_residual_stream(
                hidden_states, pending_add, pending_add2, block_residual
            )
            return IntermediateTensors(
                {"hidden_states": hidden_states, "block_residual": block_residual}
            )

        if self.end_layer in self.aux_hidden_state_layers:
            aux_hidden_states.append(
                _k3_residual_stream(
                    hidden_states, pending_add, pending_add2, block_residual
                )
            )
        hidden_states, _ = self.output_attn_res(
            hidden_states, block_residual, pending_add, pending_add2
        )
        return hidden_states, aux_hidden_states


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
        """Fallback only; a DSpark checkpoint names its own target layers."""
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
    packed_modules_mapping = KimiK3ForCausalLMBase.packed_modules_mapping
    quant_exclude_name_mapping: ClassVar[dict[str, str]] = (
        _KIMI_K3_QUANT_EXCLUDE_NAME_MAPPING
    )

    def __init__(self, atom_config: Config, prefix: str = "model"):
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
        return load_model_in_plugin_mode(
            model=self,
            config=self.atom_config,
            prefix="model.",
            weights_mapper=self.hf_to_atom_mapper,
        )

    def get_expert_mapping(self):
        return self.language_model.get_expert_mapping()

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
        return MambaStateCopyFuncCalculator.kda_state_copy_func()
