# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""DeepSeek-V4 attention adaptations for SGLang plugin mode."""

from __future__ import annotations

import contextvars
import copy
import types

import torch
from torch import nn

_draft_extend_fused_swa_ctx = contextvars.ContextVar(
    "atom_sglang_dsv4_draft_extend_fused_swa_ctx",
    default=None,
)


def _install_draft_extend_fused_swa_patch() -> None:
    """Patch ATOM DSV4 symbols only while SGLang graph integration needs them."""

    import atom.models.deepseek_v4 as dsv4

    if getattr(dsv4, "_atom_sglang_draft_extend_fused_swa_patched", False):
        return

    original_qk_norm_rope_maybe_quant = dsv4.qk_norm_rope_maybe_quant
    original_swa_write = dsv4.swa_write
    original_indexer_score_topk = dsv4.Indexer.indexer_score_topk
    original_score_topk_decode = dsv4.Indexer._score_topk_decode

    def qk_norm_rope_maybe_quant(*args, **kwargs):
        ctx = _draft_extend_fused_swa_ctx.get()
        if ctx is not None and kwargs.get("swa_kv") is None:
            attn = ctx["attn"]
            attn_md = ctx["attn_md"]
            cache_size = int(attn.swa_kv.shape[1])
            kwargs.update(
                swa_kv=attn.swa_kv,
                state_slot_mapping=attn_md.state_slot_mapping,
                batch_id_per_token=attn_md.batch_id_per_token,
                swa_cu_seqlens_q=attn_md.cu_seqlens_q,
                swa_cache_size=cache_size,
                swa_write_per_batch=min(int(attn_md.max_seqlen_q), cache_size),
            )
        return original_qk_norm_rope_maybe_quant(*args, **kwargs)

    def swa_write(*args, **kwargs):
        if _draft_extend_fused_swa_ctx.get() is not None:
            return None
        return original_swa_write(*args, **kwargs)

    def indexer_score_topk(self, q_fp8, weights, topk):
        fc = dsv4.get_forward_context()
        if bool(
            getattr(fc.attn_metadata, "use_decode_indexer_for_verify_graph", False)
        ):
            indexer_meta = fc.attn_metadata.indexer_meta
            block_tables = fc.attn_metadata.block_tables
            return self._score_topk_decode(
                q_fp8, weights, block_tables, indexer_meta, topk
            )
        return original_indexer_score_topk(self, q_fp8, weights, topk)

    def _score_topk_decode(self, q_fp8, weights, block_tables, indexer_meta, topk):
        fc = dsv4.get_forward_context()
        if not bool(
            getattr(fc.attn_metadata, "use_decode_indexer_for_verify_graph", False)
        ):
            return original_score_topk_decode(
                self, q_fp8, weights, block_tables, indexer_meta, topk
            )

        total_tokens = q_fp8.size(0)
        n_committed_per_seq_gpu = indexer_meta["n_committed_per_seq_gpu"]
        next_n = max(1, int(fc.attn_metadata.max_seqlen_q))
        bs = total_tokens // next_n
        q_4d = q_fp8.view(bs, next_n, self.n_heads, self.head_dim)
        kv_cache_4d = self.kv_cache.unsqueeze(-2)
        logits = torch.empty(
            total_tokens,
            self._max_model_len_idx,
            dtype=torch.float32,
            device=q_fp8.device,
        )
        dsv4.deepgemm_fp8_paged_mqa_logits(
            q_4d,
            kv_cache_4d,
            weights,
            logits,
            n_committed_per_seq_gpu,
            block_tables,
            self._max_model_len_idx,
            KVBlockSize=self.kv_cache.size(1),
            Preshuffle=True,
        )

        cu_starts = indexer_meta.get("cu_starts_gpu")
        cu_ends = indexer_meta.get("cu_ends_gpu")
        if cu_starts is None or cu_ends is None:
            return original_score_topk_decode(
                self, q_fp8, weights, block_tables, indexer_meta, topk
            )

        topk_local = torch.empty(
            total_tokens,
            self.index_topk,
            dtype=torch.int32,
            device=q_fp8.device,
        )
        local_starts = torch.zeros_like(cu_starts)
        local_ends = (cu_ends - cu_starts).clamp_min_(0)
        dsv4.top_k_per_row_prefill(
            logits,
            local_starts,
            local_ends,
            topk_local,
            None,
            total_tokens,
            logits.stride(0),
            logits.stride(1),
            k=topk,
        )
        return topk_local

    dsv4.qk_norm_rope_maybe_quant = qk_norm_rope_maybe_quant
    dsv4.swa_write = swa_write
    dsv4.Indexer.indexer_score_topk = indexer_score_topk
    dsv4.Indexer._score_topk_decode = _score_topk_decode
    dsv4._atom_sglang_draft_extend_fused_swa_patched = True


def patch_deepseek_v4_attention_for_sglang(attn: nn.Module) -> None:
    """Patch ATOM V4 attention for SGLang's padded prefill execution.

    SGLang can present padded prefill tensors (e.g. bucket width 256) while the
    ATOM V4 metadata built by the proxy bridge describes only real tokens.  Run
    native ATOM attention on the real token prefix, then pad the output back so
    the surrounding dense graph still sees the original tensor shape.
    """
    if hasattr(attn, "_sglang_v4_forward_impl"):
        return

    _install_draft_extend_fused_swa_patch()
    original_forward_impl = attn.forward_impl
    attn._sglang_v4_forward_impl = original_forward_impl

    def _forward_impl(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        from atom.utils.forward_context import AttnState, get_forward_context

        fc = get_forward_context()
        if fc.context.is_dummy_run:
            return self._sglang_v4_forward_impl(x, positions)

        attn_md = fc.attn_metadata
        is_draft_extend_graph = bool(
            getattr(attn_md, "is_dsv4_draft_extend_graph", False)
        )

        def call_original(
            x_arg: torch.Tensor, positions_arg: torch.Tensor
        ) -> torch.Tensor:
            if not is_draft_extend_graph:
                return self._sglang_v4_forward_impl(x_arg, positions_arg)
            token = _draft_extend_fused_swa_ctx.set(
                {"attn": self, "attn_md": fc.attn_metadata}
            )
            try:
                return self._sglang_v4_forward_impl(x_arg, positions_arg)
            finally:
                _draft_extend_fused_swa_ctx.reset(token)

        if attn_md is not None and attn_md.state is not AttnState.DECODE:
            batch_id_per_token = getattr(attn_md, "batch_id_per_token", None)
            is_verify_graph = bool(
                getattr(attn_md, "use_decode_indexer_for_verify_graph", False)
            )
            indptr = getattr(attn_md, "kv_indptr_extend", None)
            if torch.is_tensor(indptr) and indptr.dim() > 0:
                # Avoid GPU->CPU reads such as cu_seqlens_q[-1].item() under
                # CUDA graph capture. Prefill/target-verify metadata carries
                # true per-token extend indptrs; its shape is the safest source
                # of real token count when SGLang presents padded graph tensors.
                num_real = int(indptr.shape[0]) - 1
            elif is_verify_graph:
                state_slots = getattr(attn_md, "state_slot_mapping", None)
                num_reqs = (
                    int(state_slots.shape[0])
                    if torch.is_tensor(state_slots)
                    else int(getattr(fc.context, "batch_size", 1))
                )
                num_real = int(getattr(attn_md, "max_seqlen_q", 1)) * num_reqs
            else:
                num_real = (
                    int(batch_id_per_token.shape[0])
                    if torch.is_tensor(batch_id_per_token)
                    else x.shape[0]
                )
            if 0 <= num_real < x.shape[0]:
                sliced_md = copy.copy(attn_md)

                def slice_attr(name: str, n: int) -> None:
                    value = getattr(sliced_md, name, None)
                    if torch.is_tensor(value):
                        setattr(sliced_md, name, value[:n])
                    elif value is not None:
                        try:
                            setattr(sliced_md, name, value[:n])
                        except Exception:
                            pass

                for name in (
                    "batch_id_per_token",
                    "batch_id_per_token_cpu",
                    "skip_prefix_len_csa",
                ):
                    slice_attr(name, num_real)
                for name in (
                    "kv_indptr_extend",
                    "kv_indptr_prefix_swa",
                    "kv_indptr_prefix_csa",
                    "kv_indptr_prefix_hca",
                ):
                    slice_attr(name, num_real + 1)
                indexer_meta = getattr(sliced_md, "indexer_meta", None)
                if isinstance(indexer_meta, dict):
                    indexer_meta = dict(indexer_meta)
                    for key in (
                        "batch_id_per_token_gpu",
                        "seq_base_per_token_gpu",
                        "cu_starts_gpu",
                        "cu_ends_gpu",
                    ):
                        value = indexer_meta.get(key)
                        if torch.is_tensor(value):
                            indexer_meta[key] = value[:num_real]
                    sliced_md.indexer_meta = indexer_meta
                original_md = fc.attn_metadata
                fc.attn_metadata = sliced_md
                try:
                    out = call_original(x[:num_real], positions[:num_real])
                finally:
                    fc.attn_metadata = original_md
                return torch.nn.functional.pad(out, (0, 0, 0, x.shape[0] - num_real))
        return call_original(x, positions)

    attn.forward_impl = types.MethodType(_forward_impl, attn)
