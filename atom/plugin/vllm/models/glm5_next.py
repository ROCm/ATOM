# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""GLM-5.3-Flash on the vllm-atom backend.

GLM-5.3-Flash is the same shape as Kimi-K3 -- MLA full-attention layers
interleaved with KDA linear-attention layers -- so this mirrors
``atom/plugin/vllm/models/kimi_k3.py`` and reuses its KDA metadata adapter
rather than growing a second one. What differs is only the geometry (11 MLA +
34 KDA over 45 layers) and that GLM's MLA layers additionally run the DSA
sparse indexer, which is already handled by the shared sparse-MLA path.

The recurrent state for the KDA layers is owned by vLLM, not ATOM: subclassing
``MambaBase`` is what makes vLLM's KV-cache manager hand those layers a mamba
state spec instead of an attention spec.
"""

import torch
from vllm.config import VllmConfig
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc,
    MambaStateCopyFuncCalculator,
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.models.interfaces import IsHybrid

from atom.model_ops.attentions.aiter_mla import aligned_index_cache_dim
from atom.model_ops.glm5_next import kpool as kpool_ops
from atom.model_ops.glm5_next.geometry import (
    effective_kpool_size,
    pooled_path_enabled,
)
from atom.models import glm5_next as glm5_next_base
from atom.models.glm5_next import (
    Glm5NextForConditionalGeneration as Glm5NextBase,
)
from atom.models.glm5_next import (
    Glm5NextIndexer,
    Glm5NextKDAAttention,
    _normalize_glm5_next_config,
)
from atom.models.kimi_k3 import _text_config
from atom.plugin.vllm.model_wrapper import ATOMMoEForCausalLM
from atom.plugin.vllm.models.kimi_k3 import KimiKDAAttentionVllm


def _glm5_text_config(vllm_config: VllmConfig):
    """Normalized text config, the single source of the KDA dims below."""
    config = _text_config(vllm_config.model_config.hf_config)
    _normalize_glm5_next_config(config)
    return config


def _get_glm5_state_shape(
    vllm_config: VllmConfig,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    config = _glm5_text_config(vllm_config)
    # GLM-5.3-Flash rejects speculative decoding (see _check_unsupported), so
    # there is no draft-token dimension to size for.
    return MambaStateShapeCalculator.kda_state_shape(
        tp_world_size=vllm_config.parallel_config.tensor_parallel_size,
        num_heads=config.linear_num_value_heads,
        head_dim=config.linear_value_head_dim,
        num_k_heads=config.linear_num_key_heads,
        head_k_dim=config.linear_key_head_dim,
        conv_kernel_size=config.linear_conv_kernel_dim,
        num_spec=0,
    )


def _get_glm5_state_dtype(vllm_config: VllmConfig) -> tuple[torch.dtype, torch.dtype]:
    return MambaStateDtypeCalculator.kda_state_dtype(
        vllm_config.model_config.dtype,
        vllm_config.cache_config.mamba_cache_dtype,
    )


def _check_unsupported(vllm_config: VllmConfig) -> None:
    """Re-raise the guards `Config.__post_init__` applies natively.

    Plugin mode builds a `PluginConfig` (atom/plugin/config.py) and never runs
    ATOM `Config.__post_init__`, so the GLM-5.3-Flash feature guards there are
    simply absent here -- an unsupported combination would start and produce
    wrong output instead of refusing. Keep this list in step with
    `atom/config.py`.
    """
    unsupported = []
    parallel = vllm_config.parallel_config
    if getattr(parallel, "prefill_context_parallel_size", 1) > 1:
        unsupported.append("PCP")
    if getattr(parallel, "decode_context_parallel_size", 1) > 1:
        unsupported.append("DCP")
    if vllm_config.speculative_config is not None:
        unsupported.append("speculative decoding")
    if unsupported:
        raise ValueError(
            "GLM-5.3-Flash on the vllm-atom backend does not yet support "
            f"{', '.join(unsupported)}"
        )

    # Prefix caching moves the KDA state off `mamba_cache_mode == "none"`, where
    # each live request owns one state slot for its lifetime. Every pooled-index
    # structure here is keyed on that slot, and vLLM's cross-block state
    # migration knows nothing about them, so the state would be carried forward
    # and the pooled rows silently left behind.
    cache_config = vllm_config.cache_config
    if getattr(cache_config, "enable_prefix_caching", False):
        raise ValueError(
            "GLM-5.3-Flash on the vllm-atom backend does not support prefix "
            "caching; pass --no-enable-prefix-caching."
        )
    # KV offloading/transfer assumes one cached row per token, which the pooled
    # index cache (tokens_per_state == index_kpool) is not.
    if getattr(vllm_config, "kv_transfer_config", None) is not None:
        raise ValueError(
            "GLM-5.3-Flash on the vllm-atom backend does not support KV "
            "transfer/offloading."
        )

    # The pooled indexer needs the cache to hold one row per `index_kpool`
    # tokens, which `MLAAttentionSpec.tokens_per_state` declares. That requires
    # the pool size to divide the block size vLLM settled on (and keeps
    # chunked-prefill boundaries pool-aligned); it always does for a
    # power-of-two pool and a block size that is a multiple of 128, but say so
    # rather than mis-address the cache if that ever stops holding.
    config = _glm5_text_config(vllm_config)
    index_kpool = int(getattr(config, "index_kpool", 1) or 1)
    if pooled_path_enabled(index_kpool):
        block_size = vllm_config.cache_config.block_size
        if block_size % index_kpool != 0:
            raise ValueError(
                f"GLM-5.3-Flash's pooled indexer needs index_kpool "
                f"({index_kpool}) to divide the KV block size ({block_size}). "
                "Pass an explicit --block-size that it divides, or set "
                "ATOM_GLM5_KPOOL=0 to run the dense path at or below "
                f"index_topk ({getattr(config, 'index_topk', 2048)})."
            )


class Glm5NextKDAAttentionVllm(KimiKDAAttentionVllm, Glm5NextKDAAttention):
    """GLM-5.3-Flash KDA layer backed by vLLM-owned recurrent state.

    Everything about running a KDA layer under vLLM is inherited rather than
    restated: reading the layer's KDA metadata, binding vLLM's conv/ssm state
    into ATOM's cache view, swapping the ATOM forward context around the native
    mixer, splitting a mixed prefill/decode/spec batch by request class, and the
    custom op that keeps the stateful recurrence out of the traced graph. That
    is all in `KimiKDAAttentionVllm`, and it applies unchanged because
    `Glm5NextKDAAttention` IS a `KimiKDAAttention` -- the two differ only in how
    the output gate is factorised, which the native layer folds at load.

    Declaring only the cache spec (state shape/dtype/backend) is not enough and
    fails silently: the layer would still take ATOM's native forward path, read
    an ATOM forward context that plugin mode never populates, and return
    garbage that first shows up as a wrong first prefill token.

    Base order matters. The vLLM adapter must come FIRST so its overrides win;
    with GLM's native class first, `process_weights_after_loading` resolves to
    the native no-argument fold and vLLM's dtype argument raises. In this order
    each adapter method still reaches GLM's specialization through `super()`.
    """

    # (k_cache_prefix, index_kpool, head_dim) of every pooled indexer whose tail
    # buffer is keyed on THIS layer's state slots; set on one layer only, by
    # `_lend_kda_state_slots_to_indexers`.
    pooled_tail_indexers: tuple[tuple[str, int, int], ...] = ()

    def get_state_dtype(self) -> tuple[torch.dtype, torch.dtype]:
        return _get_glm5_state_dtype(self.vllm_config)

    def bind_kv_cache(self, kv_cache: torch.Tensor) -> None:
        """Bind the KDA state, then size the pooled tail buffers off its slots.

        Here rather than on the first forward: vLLM binds the KV cache before it
        warms up kernels and captures CUDA graphs, and a buffer first allocated
        inside a capture would have its zero-fill replayed every step.
        """
        super().bind_kv_cache(kv_cache)
        if not self.pooled_tail_indexers:
            return
        from atom.plugin.vllm.attention.glm5_kpool_indexer import (
            allocate_tail_cache,
        )

        state = self.kv_cache[0]
        for prefix, index_kpool, head_dim in self.pooled_tail_indexers:
            allocate_tail_cache(
                prefix, state.shape[0], index_kpool, head_dim, state.device
            )


class Glm5NextIndexerVllm(Glm5NextIndexer):
    """GLM-5.3-Flash's DSA indexer, run from vLLM's metadata.

    The dense path needs nothing here: the shared indexer already dispatches to
    the plugin op (`IndexerDecoratorForPluginMode`). The pooled (kpool) path
    keeps the native projections and kernels but replaces the op that consumes
    them, because the native one reads ATOM's forward context and ATOM's own
    tail buffer, neither of which exists under vLLM.
    """

    def __init__(self, atom_config, config, *args, **kwargs):
        super().__init__(atom_config, config, *args, **kwargs)
        # A NoPE indexer (rope_dim 0, head_dim 128) turns on the shared
        # indexer's nope-cache fusion, which hands the op the BF16 query and
        # leaves norm and quantization to the native fused kernel. The plugin's
        # dense op (`sparse_attn_indexer_plugin_mode`) has no such kernel: it
        # takes an FP8 query and writes the cache itself, and fails to compile
        # on a BF16 one. Turn the fusion off so the dense path (ATOM_GLM5_KPOOL=0)
        # quantizes before the op, as GLM-5.2's roped indexer always does.
        self.use_nope_cache_fusion = False
        # The cache module's head_dim becomes the MLAAttentionSpec head_size,
        # the bytes vLLM budgets per cached row. Declare the 16-byte-aligned
        # width native ATOM allocates (144), not index_head_dim + 4 (132): 144
        # divides the 576-byte MLA row, so the indexer page divides the MLA
        # page and is never padded -- aiter's index-cache write takes sizes,
        # not stride(0), and would write a padded page to the wrong block.
        self.k_cache.head_dim = aligned_index_cache_dim(config)
        # Tokens per cached row, which `_deepseek_v32_indexer_get_kv_cache_spec`
        # publishes as tokens_per_state; 1 when the pooled path is off.
        self.k_cache.tokens_per_state = effective_kpool_size(self.index_kpool)
        # Bound here, not looked up in the traced forward, and importing the
        # module is what registers the op: left unimported,
        # `torch.ops.aiter.<op>` raises inside the traced frame, Dynamo falls
        # back to eager for it, and the extra graph trips ATOM's "VllmBackend
        # can only be called once", which names neither the op nor the model.
        from atom.plugin.vllm.attention import glm5_kpool_indexer  # noqa: F401

        self._kpool_plugin_op = torch.ops.aiter.sparse_attn_indexer_kpool_plugin_mode
        # The KDA layer whose state slots key the pooled tail buffer; set by
        # `_lend_kda_state_slots_to_indexers` once the model is built.
        self.kda_layer_name: str | None = None

    def forward_impl(
        self,
        hidden_states: torch.Tensor,
        qr: torch.Tensor,
        qr_scale: torch.Tensor | None,
        positions,
        rotary_emb=None,
        *,
        projected_k_weights=None,
    ) -> None:
        if not self.use_kpool():
            return super().forward_impl(
                hidden_states,
                qr,
                qr_scale,
                positions,
                rotary_emb,
                projected_k_weights=projected_k_weights,
            )
        # The pooled query, key and weights, exactly as the native pooled path
        # computes them (`Glm5NextIndexer.forward_impl`); keep the two in step.
        q = self.wq_b(qr, qr_scale)
        q = q.view(-1, self.n_head, self.head_dim)
        k = self.k_norm(self.wk(hidden_states))
        weights = self.weights_proj(hidden_states)
        q_fp8, q_scale = kpool_ops.fwht128_quant_fp8(q.view(-1, self.head_dim))
        q_fp8 = q_fp8.view(-1, self.n_head, self.head_dim)
        q_scale = q_scale.view(-1, self.n_head, 1)
        weights = (weights.unsqueeze(-1) * q_scale * self._weights_scale).squeeze(-1)
        self._kpool_plugin_op(
            hidden_states,
            self.k_cache.prefix,
            self.kda_layer_name,
            self.k_cache.kv_cache[0],
            q_fp8,
            k,
            self.index_kpool_compress_gate(hidden_states),
            weights,
            self.index_kpool_compress_ape,
            positions,
            self.sparse_kv_indices_buffer,
            self.topk_tokens,
            self.index_kpool,
            self.head_dim,
            self.max_model_len,
            self.topk_out_width,
            self.scale_fmt,
            self.stable_topk,
        )


class Glm5NextForConditionalGeneration_(Glm5NextBase):
    """ATOM's GLM-5.3-Flash with its KDA layers and indexers swapped for the
    vLLM-backed ones.

    The model resolves ``Glm5NextKDAAttention`` and ``Glm5NextIndexer`` as
    module globals, so rebinding them for the duration of construction is
    enough; they are restored afterwards, so a native ATOM engine in the same
    process is unaffected.
    """

    def __init__(self, *args, **kwargs):
        original_kda_cls = glm5_next_base.Glm5NextKDAAttention
        original_indexer_cls = glm5_next_base.Glm5NextIndexer
        glm5_next_base.Glm5NextKDAAttention = Glm5NextKDAAttentionVllm
        glm5_next_base.Glm5NextIndexer = Glm5NextIndexerVllm
        try:
            super().__init__(*args, **kwargs)
        finally:
            glm5_next_base.Glm5NextKDAAttention = original_kda_cls
            glm5_next_base.Glm5NextIndexer = original_indexer_cls
        _lend_kda_state_slots_to_indexers(self)


def _lend_kda_state_slots_to_indexers(model) -> None:
    """Point every indexer at ONE KDA layer, whose state slots key its tail buffer.

    The pooled indexer holds each request's still-open pool in a side buffer
    indexed by that request's KDA state slot. vLLM splits the 34 KDA layers
    across several KV-cache groups with independent block tables, so a slot id
    only means something within one group -- every indexer therefore borrows the
    same layer's, and the buffer is sized from that layer's state pool.
    """
    kda_layer = next(
        (
            module
            for module in model.modules()
            if isinstance(module, Glm5NextKDAAttentionVllm)
        ),
        None,
    )
    if kda_layer is None:
        return
    pooled = []
    for module in model.modules():
        if isinstance(module, Glm5NextIndexerVllm):
            module.kda_layer_name = kda_layer.layer_name
            if module.use_kpool():
                pooled.append(
                    (module.k_cache.prefix, module.index_kpool, module.head_dim)
                )
    kda_layer.pooled_tail_indexers = tuple(pooled)


class Glm5NextForCausalLMVllm(ATOMMoEForCausalLM, IsHybrid):
    """vLLM-facing entry point for GLM-5.3-Flash."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        _check_unsupported(vllm_config)
        super().__init__(vllm_config=vllm_config, prefix=prefix)

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls,
        vllm_config: VllmConfig,
    ) -> tuple[torch.dtype, torch.dtype]:
        return _get_glm5_state_dtype(vllm_config)

    @classmethod
    def get_mamba_state_shape_from_config(
        cls,
        vllm_config: VllmConfig,
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return _get_glm5_state_shape(vllm_config)

    @classmethod
    def get_mamba_state_copy_func(
        cls,
    ) -> tuple[MambaStateCopyFunc, MambaStateCopyFunc]:
        return MambaStateCopyFuncCalculator.kda_state_copy_func()
