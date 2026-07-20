"""RTP-style MLA adapter for GLM5 rtp-llm plugin mode."""

from __future__ import annotations

import inspect
from types import MethodType
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
    raise AttributeError("GLM5 RTP MLA indexer requires index_topk/topk_tokens")


def _get_topk_indices_buffer(attn) -> torch.Tensor:
    indexer = getattr(attn, "indexer", None)
    buffer = (
        getattr(indexer, "topk_indices_buffer", None) if indexer is not None else None
    )
    if buffer is None:
        buffer = getattr(attn, "topk_indices_buffer", None)
    if buffer is None:
        buffer = getattr(attn, "_topk_indices_buffer", None)
    if buffer is None:
        raise AttributeError("GLM5 RTP MLA indexer requires topk_indices_buffer")
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
    return True


def _use_rtp_sparse_attn_indexer(indexer: object | None) -> None:
    if indexer is None or not hasattr(indexer, "sparse_attn_indexer_impl"):
        return
    __import__("atom.plugin.rtpllm.attention_backend.rtp_sparse_mla_backend")
    indexer.sparse_attn_indexer_impl = torch.ops.aiter.rtp_sparse_attn_indexer
    if getattr(indexer, "_atom_rtp_topk_buffer_patched", False) or not hasattr(
        indexer, "forward"
    ):
        return
    original_forward = indexer.forward

    def _forward_with_topk_buffer(self, hidden_states, *args, **kwargs):
        num_tokens = int(hidden_states.shape[0])
        topk_tokens = getattr(self, "topk_tokens", None)
        if topk_tokens is None:
            topk_tokens = getattr(self, "index_topk")
        topk_tokens = int(topk_tokens)
        buffer = getattr(self, "topk_indices_buffer", None)
        needs_new_buffer = (
            buffer is None
            or buffer.dim() != 2
            or buffer.device != hidden_states.device
            or int(buffer.shape[0]) < num_tokens
            or int(buffer.shape[1]) < topk_tokens
        )
        if needs_new_buffer:
            buffer = torch.empty(
                num_tokens,
                topk_tokens,
                dtype=torch.int32,
                device=hidden_states.device,
            )
            self.topk_indices_buffer = buffer
        self.sparse_kv_indices_buffer = self.topk_indices_buffer
        return original_forward(hidden_states, *args, **kwargs)

    indexer.forward = MethodType(_forward_with_topk_buffer, indexer)
    indexer._atom_rtp_topk_buffer_patched = True


class RTPMLAAttention:
    """RTP MLA adapter for the native GLM5 MLA call contract."""

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
        _use_rtp_sparse_attn_indexer(self.indexer)
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
        injected_backend = kwargs.get("sparse_backend")
        if injected_backend is not None:
            self.sparse_backend = injected_backend
        elif mla_modules is not None:
            from atom.plugin.rtpllm.attention_backend.rtp_sparse_mla_backend import (
                RTPSparseMlaBackend,
            )

            self.sparse_backend = RTPSparseMlaBackend(
                v_head_dim=mla_modules.v_head_dim,
                mla_modules=mla_modules,
                scale=kwargs.get("scale"),
            )
        else:
            self.sparse_backend = None
        self.kv_cache = kwargs.get("kv_cache")
        self.layer_id = int(kwargs.get("layer_id", kwargs.get("layer_num", 0)))
        self._sparse_backend_accepts_positions = (
            self._backend_accepts_positions(self.sparse_backend)
            if self.sparse_backend is not None
            else False
        )

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

        num_heads = (
            self.num_local_heads if self.num_local_heads is not None else self.num_heads
        )
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
        if self.sparse_backend is None:
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
        if self._sparse_backend_accepts_positions:
            forward_kwargs["positions"] = positions
        attn_output = self.sparse_backend.forward(
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

    import importlib
    import sys

    ops = importlib.import_module("atom.model_ops")
    base_attention = importlib.import_module("atom.model_ops.base_attention")

    ops.RTPMLAAttention = RTPMLAAttention
    ops.Attention = RTPMLAAttention
    base_attention.Attention = RTPMLAAttention

    deepseek_v2 = sys.modules.get("atom.models.deepseek_v2")
    if deepseek_v2 is None:
        try:
            import atom.models.deepseek_v2 as deepseek_v2
        except (ImportError, ModuleNotFoundError):
            return
    deepseek_v2.Attention = RTPMLAAttention
