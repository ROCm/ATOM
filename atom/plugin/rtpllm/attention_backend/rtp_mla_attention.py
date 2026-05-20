"""RTP-style MLA adapter for GLM5 rtp-llm plugin mode."""

from __future__ import annotations

import inspect
from typing import Optional

import torch


def _resolve_index_topk(attn) -> int:
    for obj, attr in (
        (getattr(attn, "indexer", None), "index_topk"),
        (getattr(attn, "indexer", None), "topk_tokens"),
        (attn, "index_topk"),
        (getattr(attn, "config", None), "index_topk"),
    ):
        value = getattr(obj, attr, None) if obj is not None else None
        if value is not None:
            return int(value)
    raise AttributeError("GLM5 RTP MLA M1 indexer requires index_topk/topk_tokens")


def _get_topk_indices_buffer(attn) -> torch.Tensor:
    indexer = getattr(attn, "indexer", None)
    buffer = getattr(indexer, "topk_indices_buffer", None) if indexer is not None else None
    if buffer is None:
        buffer = getattr(attn, "topk_indices_buffer", None)
    if buffer is None:
        buffer = getattr(attn, "_topk_indices_buffer", None)
    if buffer is None:
        raise AttributeError("GLM5 RTP MLA M1 indexer requires topk_indices_buffer")
    return buffer


def _should_emit_topk_indices(attn) -> bool:
    try:
        from atom.utils.forward_context import get_forward_context

        forward_context = get_forward_context()
    except Exception:
        return True

    context = getattr(forward_context, "context", None)
    if getattr(context, "is_dummy_run", False):
        return False
    attn_metadata = getattr(forward_context, "attn_metadata", None)
    if getattr(context, "is_prefill", False) and attn_metadata is not None:
        max_seqlen_k = getattr(attn_metadata, "max_seqlen_k", None)
        if max_seqlen_k is not None:
            try:
                return int(max_seqlen_k) > _get_topk_indices_buffer(attn).shape[1]
            except AttributeError:
                return True
    return True


class RTPMLAAttention:
    """Dense RTP MLA adapter for the native GLM5 MLA call contract."""

    use_mla = True

    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs
        mla_modules = kwargs.get("mla_modules")
        self.mla_modules = mla_modules
        self.q_proj = getattr(mla_modules, "q_proj", None)
        self.o_proj = getattr(mla_modules, "o_proj", None)
        self.kv_b_proj = getattr(mla_modules, "kv_b_proj", None)
        self.indexer = getattr(mla_modules, "indexer", None)
        self.qk_head_dim = getattr(mla_modules, "qk_head_dim", None)
        self.v_head_dim = getattr(mla_modules, "v_head_dim", None)
        self.q_lora_rank = getattr(mla_modules, "q_lora_rank", None)
        self.kv_lora_rank = getattr(mla_modules, "kv_lora_rank", None)
        self.num_heads = getattr(mla_modules, "num_heads", None)
        self.num_local_heads = getattr(mla_modules, "num_local_heads", self.num_heads)
        self.index_topk = getattr(mla_modules, "index_topk", None)
        self.topk_indices_buffer = (
            getattr(self.indexer, "topk_indices_buffer", None)
            if self.indexer is not None
            else None
        )
        injected_backend = kwargs.get("dense_backend")
        if injected_backend is not None:
            self.dense_backend = injected_backend
        elif mla_modules is not None:
            from atom.plugin.rtpllm.attention_backend.rtp_dense_mla_backend import (
                RTPDenseMlaBackend,
            )
            from atom.plugin.rtpllm.attention_backend.rtp_sparse_mla_backend import (
                RTPSparseMlaBackend,
            )

            self.dense_backend = RTPSparseMlaBackend(
                dense_backend=RTPDenseMlaBackend(mla_modules=mla_modules),
                v_head_dim=mla_modules.v_head_dim,
                mla_modules=mla_modules,
                scale=kwargs.get("scale"),
            )
        else:
            self.dense_backend = None
        self.kv_cache = kwargs.get("kv_cache")
        self.layer_id = int(kwargs.get("layer_id", kwargs.get("layer_num", 0)))

    @staticmethod
    def _backend_accepts_positions(backend: object) -> bool:
        try:
            signature = inspect.signature(backend.forward)
        except (AttributeError, TypeError, ValueError):
            return False
        return "positions" in signature.parameters or any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )

    def _project_query(
        self, query: torch.Tensor, q_scale: Optional[torch.Tensor]
    ) -> tuple[torch.Tensor, bool]:
        if query.ndim == 3:
            return query, False
        if self.q_proj is None:
            return query, False

        q = self.q_proj(query, q_scale)
        if q.ndim == 3:
            return q, True

        num_heads = self.num_local_heads if self.num_local_heads is not None else self.num_heads
        if num_heads is None:
            if self.qk_head_dim is None:
                raise AttributeError("GLM5 RTP MLA native contract requires num_heads")
            num_heads = q.shape[-1] // int(self.qk_head_dim)
        if self.qk_head_dim is None:
            self.qk_head_dim = q.shape[-1] // int(num_heads)
        return q.reshape(-1, int(num_heads), int(self.qk_head_dim)), True

    def _resolve_topk_indices(
        self,
        query: torch.Tensor,
        q_scale: Optional[torch.Tensor],
        positions: Optional[torch.Tensor],
        explicit_topk_indices: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if explicit_topk_indices is not None:
            return explicit_topk_indices
        if self.indexer is None:
            return None

        if not _should_emit_topk_indices(self):
            return None
        index_topk = _resolve_index_topk(self)
        return _get_topk_indices_buffer(self)[: query.shape[0], :index_topk]

    def forward(
        self,
        query: torch.Tensor,
        compressed_kv: torch.Tensor,
        k_pe: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        q_scale: Optional[torch.Tensor] = None,
        topk_indices: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        if self.dense_backend is None:
            raise NotImplementedError(
                "RTPMLAAttention requires an attention backend for contract execution"
            )
        q, native_projected = self._project_query(query, q_scale)
        topk_indices = self._resolve_topk_indices(
            query,
            q_scale,
            positions,
            kwargs.get("topk_indices", topk_indices),
        )
        forward_kwargs = {"topk_indices": topk_indices}
        if self._backend_accepts_positions(self.dense_backend):
            forward_kwargs["positions"] = positions
        attn_output = self.dense_backend.forward(
            q,
            compressed_kv,
            k_pe,
            self.kv_cache,
            self.layer_id,
            **forward_kwargs,
        )
        if native_projected and self.o_proj is not None:
            attn_output = attn_output.reshape(attn_output.shape[0], -1).contiguous()
            return self.o_proj(attn_output)
        return attn_output

    __call__ = forward


def apply_attention_mla_rtpllm_patch() -> None:
    """Switch ATOM's generic Attention symbol to the RTP MLA adapter."""

    import atom.model_ops as ops

    ops.RTPMLAAttention = RTPMLAAttention
    ops.Attention = RTPMLAAttention

