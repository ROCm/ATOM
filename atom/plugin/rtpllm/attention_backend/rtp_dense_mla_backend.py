"""Dense MLA fallback for GLM5 rtp-llm plugin mode."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch


_FP8_CACHE_DTYPES = tuple(
    dtype
    for dtype in (
        getattr(torch, "float8_e4m3fnuz", None),
        getattr(torch, "float8_e4m3fn", None),
        torch.uint8,
    )
    if dtype is not None
)


def _raise_cache_error(message: str) -> None:
    raise RuntimeError(message)


@dataclass(frozen=True)
class _DenseMlaMetadata:
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor | None
    block_table: torch.Tensor | None
    slot_mapping: torch.Tensor | None
    is_prefill: bool
    block_size: int


class RTPDenseMlaBackend:
    """Small dense MLA backend used before the sparse kernel is wired.

    This backend intentionally avoids vLLM plugin metadata. It consumes the
    native GLM5 five-tuple already prepared by DeepseekV2MLAAttention and uses
    RTPForwardContext metadata only to recover per-sequence token ranges.
    """

    def __init__(self, *, mla_modules: Any) -> None:
        self.mla_modules = mla_modules
        self.kv_b_proj = getattr(mla_modules, "kv_b_proj", None)
        self.rotary_emb = getattr(mla_modules, "rotary_emb", None)
        self.v_head_dim = int(getattr(mla_modules, "v_head_dim"))
        self.qk_nope_head_dim = getattr(mla_modules, "qk_nope_head_dim", None)
        self.qk_rope_head_dim = getattr(mla_modules, "qk_rope_head_dim", None)
        self._projection_checked = False

    @staticmethod
    def _read_is_prefill(context: Any) -> bool:
        if context is None or not hasattr(context, "is_prefill"):
            raise ValueError(
                "GLM5 RTP dense MLA requires explicit context.is_prefill metadata."
            )
        return bool(getattr(context, "is_prefill"))

    @staticmethod
    def _get_metadata(num_tokens: int, device: torch.device) -> _DenseMlaMetadata:
        attn_metadata = None
        context = None
        rtp_seq_size_per_block = 1
        try:
            from atom.utils.forward_context import get_forward_context

            forward_context = get_forward_context()
            attn_metadata = getattr(forward_context, "attn_metadata", None)
            context = getattr(forward_context, "context", None)
            rtp_seq_size_per_block = int(
                getattr(attn_metadata, "rtp_seq_size_per_block", 0)
                or getattr(attn_metadata, "rtp_kernel_seq_size_per_block", 0)
                or 0
            )
            plugin_metadata = getattr(attn_metadata, "plugin_metadata", None)
            query_start_loc = getattr(plugin_metadata, "query_start_loc", None)
            if query_start_loc is None:
                query_start_loc = getattr(plugin_metadata, "rtp_cu_seqlens_q", None)
            if query_start_loc is None:
                query_start_loc = getattr(attn_metadata, "cu_seqlens_q", None)
            if query_start_loc is None:
                decode_metadata = getattr(plugin_metadata, "decode_metadata", None)
                query_start_loc = getattr(decode_metadata, "query_start_loc", None)
            seq_lens = getattr(plugin_metadata, "seq_lens", None)
            if seq_lens is None:
                seq_lens = getattr(attn_metadata, "context_lens", None)
            block_table = getattr(plugin_metadata, "block_table", None)
            if block_table is None:
                block_table = getattr(attn_metadata, "block_tables", None)
            slot_mapping = getattr(plugin_metadata, "slot_mapping", None)
            if slot_mapping is None:
                slot_mapping = getattr(attn_metadata, "slot_mapping", None)
        except Exception:
            query_start_loc = None
            seq_lens = None
            block_table = None
            slot_mapping = None

        if (
            context is not None
            and hasattr(context, "is_prefill")
            and not bool(getattr(context, "is_prefill"))
            and isinstance(seq_lens, torch.Tensor)
            and int(seq_lens.numel()) == num_tokens
            and isinstance(block_table, torch.Tensor)
            and isinstance(slot_mapping, torch.Tensor)
        ):
            query_start_loc = torch.arange(
                num_tokens + 1, dtype=torch.int64, device=device
            )

        if query_start_loc is not None and int(query_start_loc.numel()) >= 2:
            query_start_loc = query_start_loc.to(device=device, dtype=torch.int64)
            if int(query_start_loc[0].item()) == 0 and int(query_start_loc[-1].item()) == num_tokens:
                is_prefill = RTPDenseMlaBackend._read_is_prefill(context)
                return _DenseMlaMetadata(
                    query_start_loc=query_start_loc,
                    seq_lens=(
                        seq_lens.to(device=device, dtype=torch.int64)
                        if isinstance(seq_lens, torch.Tensor)
                        else None
                    ),
                    block_table=(
                        block_table.to(device=device, dtype=torch.int64)
                        if isinstance(block_table, torch.Tensor)
                        else None
                    ),
                    slot_mapping=(
                        slot_mapping.to(device=device, dtype=torch.int64)
                        if isinstance(slot_mapping, torch.Tensor)
                        else None
                    ),
                    is_prefill=is_prefill,
                    block_size=max(1, rtp_seq_size_per_block),
                )
        if num_tokens != 1:
            raise ValueError(
                "GLM5 RTP dense MLA requires query_start_loc metadata for "
                f"multi-token batches (num_tokens={num_tokens})."
            )
        is_prefill = RTPDenseMlaBackend._read_is_prefill(context)
        return _DenseMlaMetadata(
            query_start_loc=torch.tensor([0, num_tokens], dtype=torch.int64, device=device),
            seq_lens=None,
            block_table=None,
            slot_mapping=None,
            is_prefill=is_prefill,
            block_size=max(1, rtp_seq_size_per_block),
        )

    @staticmethod
    def _unwrap_linear_output(value: Any) -> torch.Tensor:
        if isinstance(value, tuple):
            value = value[0]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Expected kv_b_proj to return Tensor, got {type(value)!r}.")
        return value

    def _apply_current_rope(
        self,
        q: torch.Tensor,
        k_pe: torch.Tensor,
        positions: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rope_dim = int(self.qk_rope_head_dim or k_pe.shape[-1])
        if rope_dim == 0:
            return q, k_pe
        if self.rotary_emb is None:
            raise ValueError("GLM5 RTP dense MLA requires rotary_emb for RoPE dimensions.")
        if positions is None or int(positions.numel()) != int(q.shape[0]):
            got = None if positions is None else int(positions.numel())
            raise ValueError(
                "GLM5 RTP dense MLA requires per-token absolute positions for RoPE "
                f"(positions={got}, tokens={int(q.shape[0])})."
            )
        if int(q.shape[-1]) < rope_dim:
            raise ValueError(
                f"GLM5 RTP dense MLA invalid q shape for RoPE: q={tuple(q.shape)}, "
                f"rope_dim={rope_dim}."
            )

        q_rope = q.clone()
        k_pe_rope = k_pe.clone()
        # RotaryEmbedding.forward rotates the full tensor it receives. Passing
        # only q_pe/k_pe is equivalent to the fused MLA path's nope-first layout.
        rotated_q_pe, rotated_k_pe = self.rotary_emb(
            positions.to(device=q.device, dtype=torch.long),
            q_rope[..., -rope_dim:],
            k_pe_rope,
        )
        q_rope[..., -rope_dim:] = rotated_q_pe
        return q_rope, rotated_k_pe

    def _project_kv(
        self,
        q: torch.Tensor,
        compressed_kv: torch.Tensor,
        k_pe: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if self.kv_b_proj is None:
            return None
        _, num_heads, qk_head_dim = q.shape
        num_kv_tokens = int(compressed_kv.shape[0])
        rope_dim = int(self.qk_rope_head_dim or k_pe.shape[-1])
        nope_dim = int(self.qk_nope_head_dim or (qk_head_dim - rope_dim))
        if nope_dim <= 0:
            raise ValueError(
                f"Invalid MLA qk dims: qk_head_dim={qk_head_dim}, rope_dim={rope_dim}."
            )

        compressed_kv = compressed_kv.contiguous()
        kv_nope = self._unwrap_linear_output(self.kv_b_proj(compressed_kv))
        if kv_nope.numel() == 0:
            raise ValueError("GLM5 RTP dense MLA kv_b_proj returned an empty tensor.")
        expected_last_dim = num_heads * (nope_dim + self.v_head_dim)
        if kv_nope.shape[-1] != expected_last_dim:
            raise ValueError(
                "GLM5 RTP dense MLA kv_b_proj output shape mismatch "
                f"(got={tuple(kv_nope.shape)}, expected_last_dim={expected_last_dim}, "
                f"num_heads={num_heads}, qk_nope_head_dim={nope_dim}, "
                f"v_head_dim={self.v_head_dim})."
            )
        if not self._projection_checked:
            self._projection_checked = True

        kv_nope = kv_nope.reshape(num_kv_tokens, num_heads, nope_dim + self.v_head_dim)
        k_nope, value = kv_nope.split([nope_dim, self.v_head_dim], dim=-1)
        if k_pe.dim() == 2:
            k_pe = k_pe.unsqueeze(1)
        k_pe = k_pe.expand(num_kv_tokens, num_heads, rope_dim)
        key = torch.cat((k_nope, k_pe), dim=-1)
        return key, value

    @staticmethod
    def _causal_attention(
        q: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        query_start_loc: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        pieces: list[torch.Tensor] = []
        for start_tensor, end_tensor in zip(query_start_loc[:-1], query_start_loc[1:]):
            start = int(start_tensor.item())
            end = int(end_tensor.item())
            if end <= start:
                continue
            q_seg = q[start:end].float()
            k_seg = key[start:end].float()
            v_seg = value[start:end].float()
            scores = torch.einsum("tnd,snd->nts", q_seg, k_seg) * scale
            seq_len = end - start
            causal_mask = torch.ones(
                (seq_len, seq_len), dtype=torch.bool, device=q.device
            ).tril()
            scores = scores.masked_fill(~causal_mask.unsqueeze(0), float("-inf"))
            probs = torch.softmax(scores, dim=-1)
            pieces.append(torch.einsum("nts,snd->tnd", probs, v_seg))
        if not pieces:
            return value.new_empty((0, value.shape[1], value.shape[2]))
        return torch.cat(pieces, dim=0)

    @staticmethod
    def _cross_causal_attention(
        q: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        q_len = int(q.shape[0])
        k_len = int(key.shape[0])
        if q_len == 0:
            return value.new_empty((0, value.shape[1], value.shape[2]))
        if k_len < q_len:
            raise ValueError(
                f"GLM5 RTP dense MLA got invalid cross attention lengths: q={q_len}, k={k_len}."
            )
        scores = torch.einsum("tnd,snd->nts", q.float(), key.float()) * scale
        q_pos = torch.arange(q_len, device=q.device).unsqueeze(1)
        k_pos = torch.arange(k_len, device=q.device).unsqueeze(0)
        causal_mask = k_pos <= (k_len - q_len + q_pos)
        scores = scores.masked_fill(~causal_mask.unsqueeze(0), float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        return torch.einsum("nts,snd->tnd", probs, value.float())

    @staticmethod
    def _flatten_latent_cache(
        layer_cache: Any,
        *,
        block_size: int,
        kv_dim: int,
    ) -> torch.Tensor | None:
        kv_cache_base = getattr(layer_cache, "kv_cache_base", None)
        if not isinstance(kv_cache_base, torch.Tensor) or kv_cache_base.numel() == 0:
            return None
        if kv_cache_base.dtype in _FP8_CACHE_DTYPES:
            raise NotImplementedError(
                "GLM5 RTP dense MLA reference path requires BF16/FP16 latent KV cache; "
                "FP8 KV cache layout/dequant is not supported yet."
            )
        if kv_cache_base.dim() == 3 and int(kv_cache_base.shape[-1]) == kv_dim:
            return kv_cache_base.reshape(-1, kv_dim)
        if kv_cache_base.dim() == 2 and int(kv_cache_base.shape[1]) % block_size == 0:
            per_token_dim = int(kv_cache_base.shape[1]) // block_size
            if per_token_dim == kv_dim:
                return kv_cache_base.view(kv_cache_base.shape[0], block_size, kv_dim).reshape(
                    -1, kv_dim
                )
        return None

    @staticmethod
    def _write_current_to_cache(
        *,
        layer_cache: Any,
        compressed_kv: torch.Tensor,
        k_pe: torch.Tensor,
        metadata: _DenseMlaMetadata,
        kv_dim: int,
    ) -> None:
        if metadata.slot_mapping is None:
            return
        flat_cache = RTPDenseMlaBackend._flatten_latent_cache(
            layer_cache, block_size=metadata.block_size, kv_dim=kv_dim
        )
        if flat_cache is None:
            return
        latent = torch.cat((compressed_kv, k_pe), dim=-1)
        if latent.shape[0] != metadata.slot_mapping.shape[0]:
            return
        slots = metadata.slot_mapping[: latent.shape[0]].long()
        flat_size = int(flat_cache.shape[0])
        non_negative = slots >= 0
        in_bounds = non_negative & (slots < flat_size)
        if bool((non_negative & (slots >= flat_size)).any().item()):
            bad_slots = slots[non_negative & (slots >= flat_size)]
            _raise_cache_error(
                "GLM5 RTP dense MLA refuses to write out-of-bounds slot_mapping "
                f"(block_size={metadata.block_size}, flat_tokens={flat_size}, "
                f"slot_min={int(bad_slots.min().item())}, "
                f"slot_max={int(bad_slots.max().item())})."
            )
        if not bool(in_bounds.any().item()):
            return
        flat_cache[slots[in_bounds]] = latent[in_bounds].to(dtype=flat_cache.dtype)

    @staticmethod
    def _resolve_layer_cache(kv_cache: object, layer_id: int) -> object:
        if kv_cache is not None:
            return kv_cache
        try:
            from atom.utils.forward_context import get_forward_context

            forward_context = get_forward_context()
            kv_cache_data = getattr(forward_context, "kv_cache_data", None)
            if kv_cache_data is None:
                return None
            layer_cache_entry = kv_cache_data.get(f"layer_{int(layer_id)}")
            if layer_cache_entry is None:
                return None
            return getattr(layer_cache_entry, "k_cache", layer_cache_entry)
        except Exception:
            return None

    @staticmethod
    def _gather_latent_history(
        *,
        layer_cache: Any,
        metadata: _DenseMlaMetadata,
        batch_idx: int,
        kv_dim: int,
    ) -> torch.Tensor | None:
        if metadata.block_table is None or metadata.seq_lens is None:
            return None
        flat_cache = RTPDenseMlaBackend._flatten_latent_cache(
            layer_cache, block_size=metadata.block_size, kv_dim=kv_dim
        )
        if flat_cache is None:
            return None
        seq_len = int(metadata.seq_lens[batch_idx].item())
        if seq_len <= 0:
            return None
        block_row = metadata.block_table[batch_idx].long()
        positions = torch.arange(seq_len, dtype=torch.long, device=flat_cache.device)
        block_cols = torch.div(positions, metadata.block_size, rounding_mode="floor")
        block_col_max = int(block_cols.max().item())
        if block_col_max >= int(block_row.numel()):
            return None
        offsets = positions.remainder(metadata.block_size)
        slots = block_row[block_cols] * metadata.block_size + offsets
        flat_size = int(flat_cache.shape[0])
        if bool(((slots < 0) | (slots >= flat_size)).any().item()):
            bad_slots = slots[(slots < 0) | (slots >= flat_size)]
            _raise_cache_error(
                "GLM5 RTP dense MLA refuses to gather out-of-bounds KV history "
                f"(batch_idx={batch_idx}, seq_len={seq_len}, "
                f"block_size={metadata.block_size}, flat_tokens={flat_size}, "
                f"slot_min={int(bad_slots.min().item())}, "
                f"slot_max={int(bad_slots.max().item())})."
            )
        return flat_cache[slots]

    @staticmethod
    def _require_decode_cache_metadata(
        *,
        layer_cache: Any,
        metadata: _DenseMlaMetadata,
        kv_dim: int,
    ) -> None:
        missing = []
        if metadata.block_table is None:
            missing.append("block_table")
        if metadata.seq_lens is None:
            missing.append("seq_lens")
        if metadata.slot_mapping is None:
            missing.append("slot_mapping")
        if missing:
            raise ValueError(
                "GLM5 RTP dense MLA decode requires RTP KV metadata: "
                + ", ".join(missing)
                + "."
            )
        flat_cache = RTPDenseMlaBackend._flatten_latent_cache(
            layer_cache, block_size=metadata.block_size, kv_dim=kv_dim
        )
        if flat_cache is None:
            raise ValueError(
                "GLM5 RTP dense MLA decode requires a readable BF16/FP16 kv_cache_base."
            )

    def forward(
        self,
        q: torch.Tensor,
        compressed_kv: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: object,
        layer_id: int,
        topk_indices: Optional[torch.Tensor] = None,
        positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del topk_indices
        if self.kv_b_proj is None:
            return q.new_zeros((q.shape[0], q.shape[1], self.v_head_dim))
        layer_cache = self._resolve_layer_cache(kv_cache, layer_id)
        q, k_pe = self._apply_current_rope(q, k_pe, positions)
        projected = self._project_kv(q, compressed_kv, k_pe)
        if projected is None:
            return q.new_zeros((q.shape[0], q.shape[1], self.v_head_dim))
        metadata = self._get_metadata(q.shape[0], q.device)
        kv_dim = int(compressed_kv.shape[-1]) + int(k_pe.shape[-1])
        self._write_current_to_cache(
            layer_cache=layer_cache,
            compressed_kv=compressed_kv,
            k_pe=k_pe,
            metadata=metadata,
            kv_dim=kv_dim,
        )
        key, value = projected
        query_start_loc = metadata.query_start_loc
        scale = float(q.shape[-1] ** -0.5)
        if metadata.is_prefill:
            output = self._causal_attention(q, key, value, query_start_loc, scale)
            return output.to(dtype=compressed_kv.dtype)

        self._require_decode_cache_metadata(
            layer_cache=layer_cache,
            metadata=metadata,
            kv_dim=kv_dim,
        )
        pieces: list[torch.Tensor] = []
        for batch_idx, (start_tensor, end_tensor) in enumerate(
            zip(query_start_loc[:-1], query_start_loc[1:])
        ):
            start = int(start_tensor.item())
            end = int(end_tensor.item())
            if end <= start:
                continue
            q_seg = q[start:end]
            latent_history = self._gather_latent_history(
                layer_cache=layer_cache,
                metadata=metadata,
                batch_idx=batch_idx,
                kv_dim=kv_dim,
            )
            if latent_history is None:
                raise ValueError(
                    "GLM5 RTP dense MLA decode failed to gather latent KV history."
                )
            hist_compressed_kv, hist_k_pe = latent_history.split(
                [compressed_kv.shape[-1], k_pe.shape[-1]], dim=-1
            )
            hist_key, hist_value = self._project_kv(q_seg, hist_compressed_kv, hist_k_pe)
            pieces.append(
                self._cross_causal_attention(q_seg, hist_key, hist_value, scale)
            )
        output = torch.cat(pieces, dim=0) if pieces else value.new_empty((0, *value.shape[1:]))
        return output.to(dtype=compressed_kv.dtype)
