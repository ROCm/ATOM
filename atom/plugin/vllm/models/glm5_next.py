import copy
from contextlib import contextmanager

import torch
from vllm.config import VllmConfig
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc,
    MambaStateCopyFuncCalculator,
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.models.interfaces import IsHybrid

from atom.model_ops.glm5_next import kpool as kpool_ops
from atom.models import glm5_next as glm5_next_base
from atom.models import glm5_next_mtp as glm5_next_mtp_base
from atom.models.glm5_next import _ROPE_PAD, _normalize_glm5_next_config
from atom.plugin.vllm import glm5_kpool
from atom.plugin.vllm.model_wrapper import ATOMMoEForCausalLM
from atom.plugin.vllm.models.kimi_k3 import KimiKDAAttentionVllm

_Glm5NextKDAAttention = glm5_next_base.Glm5NextKDAAttention
_Glm5NextIndexer = glm5_next_base.Glm5NextIndexer


def _text_config(config):
    config = getattr(config, "text_config", config)
    _normalize_glm5_next_config(config)
    return config


def _get_glm5_state_shape(vllm_config: VllmConfig) -> tuple[tuple[int, ...], ...]:
    config = _text_config(vllm_config.model_config.hf_text_config)
    return MambaStateShapeCalculator.kda_state_shape(
        tp_world_size=vllm_config.parallel_config.tensor_parallel_size,
        num_heads=config.linear_num_value_heads,
        head_dim=config.linear_value_head_dim,
        num_k_heads=config.linear_num_key_heads,
        head_k_dim=config.linear_key_head_dim,
        conv_kernel_size=config.linear_conv_kernel_dim,
        num_spec=glm5_kpool.num_speculative_tokens(vllm_config),
    )


def _get_glm5_state_dtype(vllm_config: VllmConfig) -> tuple[torch.dtype, ...]:
    return MambaStateDtypeCalculator.kda_state_dtype(
        vllm_config.model_config.dtype,
        vllm_config.cache_config.mamba_cache_dtype,
    )


def _num_draft_layers(vllm_config: VllmConfig, text_config) -> int:
    spec = vllm_config.speculative_config
    if spec is None:
        return 0
    return int(getattr(text_config, "num_nextn_predict_layers", 0) or 0)


def _check_supported(vllm_config: VllmConfig) -> None:
    unsupported = []
    spec = vllm_config.speculative_config
    if spec is not None and spec.method != "mtp":
        unsupported.append(f"speculative method {spec.method!r} (only mtp)")
    if vllm_config.parallel_config.pipeline_parallel_size > 1:
        unsupported.append("pipeline parallelism")
    if unsupported:
        raise NotImplementedError(
            "GLM-5.3-Flash on the vLLM plugin does not support "
            + ", ".join(unsupported)
        )


@contextmanager
def _swap_layer_classes():
    originals = (glm5_next_base.Glm5NextKDAAttention, glm5_next_base.Glm5NextIndexer)
    glm5_next_base.Glm5NextKDAAttention = Glm5NextKDAAttentionVllm
    glm5_next_base.Glm5NextIndexer = Glm5NextIndexerVllm
    try:
        yield
    finally:
        glm5_next_base.Glm5NextKDAAttention, glm5_next_base.Glm5NextIndexer = originals


class Glm5NextKDAAttentionVllm(_Glm5NextKDAAttention, KimiKDAAttentionVllm):
    def process_weights_after_loading(self, *args, **kwargs) -> None:
        return _Glm5NextKDAAttention.process_weights_after_loading(self)


class Glm5NextIndexerVllm(_Glm5NextIndexer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if not self.use_kpool():
            raise NotImplementedError(
                "GLM-5.3-Flash on the vLLM plugin needs the pooled indexer "
                "(ATOM_GLM5_KPOOL=1)"
            )
        del self.k_cache
        self.kpool_index_cache: torch.Tensor | None = None
        self.kpool_index_slot = -1
        self.kpool_subs_per_page = 0
        self.mla_layer_name = ""

    def bind_kpool_index_cache(
        self, index_cache: torch.Tensor, slot: int, subs_per_page: int
    ) -> None:
        self.kpool_index_cache = index_cache
        self.kpool_index_slot = slot
        self.kpool_subs_per_page = subs_per_page

    def forward_impl(
        self,
        hidden_states: torch.Tensor,
        qr: torch.Tensor,
        qr_scale: torch.Tensor | None,
        positions,
        rotary_emb=None,
    ) -> torch.Tensor:
        q = self.wq_b(qr, qr_scale).view(-1, self.head_dim)
        k = self.k_norm(self.wk(hidden_states))
        weights = self.weights_proj(hidden_states)
        q_fp8, q_scale = kpool_ops.fwht128_quant_fp8(q)
        q_fp8 = q_fp8.view(-1, self.n_head, self.head_dim)
        weights = (
            weights.unsqueeze(-1)
            * q_scale.view(-1, self.n_head, 1)
            * self._weights_scale
        ).squeeze(-1)
        torch.ops.aiter.glm5_kpool_indexer(
            k,
            self.index_kpool_compress_gate(hidden_states),
            q_fp8,
            weights,
            positions,
            self.prefix,
            self.sparse_kv_indices_buffer,
        )
        return weights


def _sparse_attentions(attn_modules):
    attns = list(attn_modules)
    for attn in attns:
        attn.indexer.mla_layer_name = attn.mla_attn.layer_name
    return attns


class Glm5NextForConditionalGeneration(glm5_next_base.Glm5NextForConditionalGeneration):
    def __init__(self, atom_config, prefix: str = "") -> None:
        vllm_config = atom_config.plugin_config.vllm_config
        _check_supported(vllm_config)
        if _ROPE_PAD != glm5_kpool.GLM5_NEXT_MLA_ROPE_PAD:
            raise RuntimeError("glm5_kpool.GLM5_NEXT_MLA_ROPE_PAD is out of date")
        with _swap_layer_classes():
            super().__init__(atom_config, prefix=prefix)
        attns = _sparse_attentions(
            layer.self_attn for layer in self.model.layers if not layer.is_linear_attn
        )
        glm5_kpool.register_kpool_index_proxy(
            vllm_config,
            self.config,
            attns[0].mla_attn,
            [attn.indexer for attn in attns],
            num_draft_layers=_num_draft_layers(vllm_config, self.config),
        )


class Glm5NextMTP(glm5_next_mtp_base.Glm5NextMTP):
    def __init__(self, atom_config, prefix: str = "") -> None:
        vllm_config = atom_config.plugin_config.vllm_config
        text_config = copy.copy(_text_config(atom_config.hf_config))
        atom_config.hf_config = text_config
        spec = atom_config.speculative_config
        if spec is not None:
            atom_config.speculative_config = copy.copy(spec)
            atom_config.speculative_config.draft_model_hf_config = text_config
        with _swap_layer_classes():
            super().__init__(atom_config, prefix=prefix)
        attns = _sparse_attentions(
            layer.mtp_block.self_attn for layer in self.model.layers.values()
        )
        sfc = vllm_config.compilation_config.static_forward_context
        sfc[glm5_kpool.index_proxy_layer_name(text_config)].add_indexers(
            [attn.indexer for attn in attns]
        )

    # Top-k reuse needs a per-token index layout the plugin's ragged buffer lacks.
    def set_skip_topk(self, skip: bool) -> None:
        return None

    def compact_topk_indices(self, slot_ids: torch.Tensor) -> None:
        return None


class Glm5NextForConditionalGenerationVllm(ATOMMoEForCausalLM, IsHybrid):
    @classmethod
    def get_mamba_state_dtype_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[torch.dtype, ...]:
        return _get_glm5_state_dtype(vllm_config)

    @classmethod
    def get_mamba_state_shape_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[tuple[int, ...], ...]:
        return _get_glm5_state_shape(vllm_config)

    @classmethod
    def get_mamba_state_copy_func(cls) -> tuple[MambaStateCopyFunc, ...]:
        return MambaStateCopyFuncCalculator.kda_state_copy_func()
