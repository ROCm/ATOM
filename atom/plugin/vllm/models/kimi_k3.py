from dataclasses import replace
from types import SimpleNamespace

import torch
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
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum

from atom.models import kimi_k3 as kimi_k3_base
from atom.models.kimi_k3 import (
    KimiK3ForCausalLM as KimiK3ForCausalLMBase,
)
from atom.models.kimi_k3 import (
    KimiKDAAttention,
    _normalize_kimi_config,
)
from atom.models.utils import IntermediateTensors
from atom.plugin.vllm.model_wrapper import ATOMMoEForCausalLM
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
        self, hidden_states: torch.Tensor, gdn_metadata: GDNAttentionMetadata
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
            return super()._forward_impl(hidden_states)

        spec_indx = gdn_metadata.spec_token_indx
        non_spec_indx = gdn_metadata.non_spec_token_indx
        assert spec_indx is not None and non_spec_indx is not None
        rows = hidden_states[: gdn_metadata.num_actual_tokens]

        spec_out = self._forward_one_segment(
            rows.index_select(0, spec_indx),
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
        self, hidden_states: torch.Tensor, gdn_metadata: GDNAttentionMetadata
    ) -> torch.Tensor:
        self._atom_metadata.gdn_metadata = gdn_metadata
        return super()._forward_impl(hidden_states)

    def _forward_impl(self, hidden_states: torch.Tensor) -> torch.Tensor:
        vllm_context = get_vllm_forward_context()
        attn_metadata = vllm_context.attn_metadata
        if attn_metadata is None:
            return hidden_states.new_zeros(hidden_states.shape)

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
            output = self._forward_segments(hidden_states, gdn_metadata)
        finally:
            atom_context.attn_metadata = previous_metadata
            atom_context.kv_cache_data = previous_kv_cache_data

        # vLLM pads token rows to the selected piecewise/full graph bucket,
        # while GDN metadata tracks only real tokens. The native KDA path
        # intentionally slices to num_actual_tokens; restore the graph bucket
        # width so this custom op matches its empty_like fake implementation.
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

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        self.language_model.model.set_aux_hidden_state_layers(layers)

    def get_eagle3_aux_hidden_state_layers(self) -> tuple[int, ...]:
        """Fallback only; the DSpark checkpoint names its own target layers."""
        num_layers = len(self.language_model.model.layers)
        return (2, num_layers // 2, num_layers - 3)


class KimiK3ForCausalLMVllm(ATOMMoEForCausalLM, IsHybrid):
    def get_language_model(self) -> torch.nn.Module:
        """Expose the inner causal-LM so draft loaders can find ``embed_tokens``.

        vLLM's DSpark loader reaches for ``target.get_language_model().model
        .embed_tokens``; without this it would stop at the multimodal wrapper
        and silently skip sharing the embedding with the draft.
        """
        return self.model.language_model

    @property
    def lm_head(self) -> torch.nn.Module:
        """The head the DSpark draft borrows to score its block."""
        return self.model.language_model.lm_head

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
