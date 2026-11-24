# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Eagle3 draft model for speculative decoding.

Architecture aligned with lightseekorg/kimi-k2.5-eagle3 (HuggingFace format):
- RMSNorm (no bias) instead of LayerNorm
- 3-input fusion: fc(cat(aux_layer_1, aux_layer_mid, aux_layer_late)) from teacher
- Dual-norm decoder: hidden_norm + input_layernorm, cat → attention
- Separate Q/K/V projections with 2×hidden input dimension
- YaRN RoPE scaling (rope_theta=1000000, factor=64)

Architecture aligned with lightseekorg/kimi-k2.5-eagle3 (HuggingFace format).
"""

from __future__ import annotations

import logging
import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

logger = logging.getLogger(__name__)


def _make_causal_mask(
    bsz: int,
    tgt_len: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(tgt_len, device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(tgt_len, 1), 0)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len)


def _prepare_attention_mask(
    attention_mask: Tensor,
    bsz: int,
    tgt_len: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    """Build [B, 1, T, T] causal + padding mask from [B, T] attention_mask."""
    causal = _make_causal_mask(bsz, tgt_len, dtype, device)
    expanded = (
        attention_mask[:, None, None, :].expand(bsz, 1, tgt_len, tgt_len).to(dtype)
    )
    inverted = (1.0 - expanded).masked_fill(
        (1.0 - expanded).bool(), torch.finfo(dtype).min
    )
    return causal + inverted


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x.to(input_dtype)


# -- YaRN RoPE helpers --


def _yarn_find_correction_dim(
    num_rotations: float,
    dim: int,
    base: float = 10000,
    max_position_embeddings: int = 2048,
) -> float:
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (
        2 * math.log(base)
    )


def _yarn_find_correction_range(
    low_rot: float,
    high_rot: float,
    dim: int,
    base: float = 10000,
    max_position_embeddings: int = 2048,
) -> tuple[int, int]:
    low = math.floor(
        _yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings)
    )
    high = math.ceil(
        _yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings)
    )
    return max(low, 0), min(high, dim - 1)


def _yarn_get_mscale(scale: float = 1.0, mscale: float = 1.0) -> float:
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def _yarn_linear_ramp_mask(min_val: float, max_val: float, dim: int) -> Tensor:
    if min_val == max_val:
        max_val += 0.001
    linear_func = (torch.arange(dim, dtype=torch.float32) - min_val) / (
        max_val - min_val
    )
    return torch.clamp(linear_func, 0, 1)


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 262144,
        base: float = 1000000.0,
        scaling_factor: float = 1.0,
        original_max_position_embeddings: int = 4096,
        beta_fast: float = 32.0,
        beta_slow: float = 1.0,
        mscale: float = 1.0,
        mscale_all_dim: float = 0.0,
        rope_type: str = "yarn",
        low_freq_factor: float = 1.0,
        high_freq_factor: float = 4.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.scaling_factor = scaling_factor
        self.original_max_position_embeddings = original_max_position_embeddings
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.mscale = mscale
        self.mscale_all_dim = mscale_all_dim
        self.rope_type = rope_type
        self.low_freq_factor = low_freq_factor
        self.high_freq_factor = high_freq_factor

        self._cos_cached: Tensor | None = None
        self._sin_cached: Tensor | None = None
        self._cached_seq_len = 0

    def _base_inv_freq(self, device: torch.device) -> Tensor:
        return 1.0 / (
            self.base
            ** (
                torch.arange(0, self.dim, 2, device=device, dtype=torch.float32)
                / self.dim
            )
        )

    def _build_inv_freq(self, device: torch.device) -> Tensor:
        if self.scaling_factor <= 1.0:
            return self._base_inv_freq(device)

        if self.rope_type == "llama3":
            inv_freq = self._base_inv_freq(device)
            old_ctx = float(self.original_max_position_embeddings)
            low_wl = old_ctx / self.low_freq_factor
            high_wl = old_ctx / self.high_freq_factor
            wavelen = 2 * math.pi / inv_freq

            inv_freq_llama = torch.where(
                wavelen > low_wl, inv_freq / self.scaling_factor, inv_freq
            )
            smooth = (old_ctx / wavelen - self.low_freq_factor) / (
                self.high_freq_factor - self.low_freq_factor
            )
            smoothed = (
                1 - smooth
            ) * inv_freq_llama / self.scaling_factor + smooth * inv_freq_llama
            in_smooth_band = (wavelen >= high_wl) & (wavelen <= low_wl)
            return torch.where(in_smooth_band, smoothed, inv_freq_llama)

        freq_extra = self._base_inv_freq(device)
        freq_inter = freq_extra / self.scaling_factor

        low, high = _yarn_find_correction_range(
            self.beta_fast,
            self.beta_slow,
            self.dim,
            self.base,
            self.original_max_position_embeddings,
        )
        inv_freq_mask = 1.0 - _yarn_linear_ramp_mask(low, high, self.dim // 2).to(
            device=device
        )
        inv_freq = freq_inter * (1 - inv_freq_mask) + freq_extra * inv_freq_mask
        return inv_freq

    def _compute_mscale(self) -> float:
        if self.scaling_factor <= 1.0:
            return 1.0
        if self.rope_type == "llama3":
            return 1.0
        return _yarn_get_mscale(self.scaling_factor, self.mscale) / _yarn_get_mscale(
            self.scaling_factor, self.mscale_all_dim
        )

    def _update_cache(self, seq_len: int, device: torch.device) -> None:
        if seq_len <= self._cached_seq_len and self._cos_cached is not None:
            return
        self._cached_seq_len = max(seq_len, self.max_position_embeddings)
        inv_freq = self._build_inv_freq(device)
        t = torch.arange(self._cached_seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)

        ms = self._compute_mscale()
        self._cos_cached = (emb.cos() * ms).unsqueeze(0).unsqueeze(0)
        self._sin_cached = (emb.sin() * ms).unsqueeze(0).unsqueeze(0)

    def forward(self, x: Tensor, seq_len: int) -> tuple[Tensor, Tensor]:
        self._update_cache(seq_len, x.device)
        return (
            self._cos_cached[:, :, :seq_len, :].to(x.dtype),
            self._sin_cached[:, :, :seq_len, :].to(x.dtype),
        )


def _rotate_half(x: Tensor) -> Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb(
    q: Tensor,
    k: Tensor,
    cos: Tensor,
    sin: Tensor,
    position_ids: Tensor,
) -> tuple[Tensor, Tensor]:
    cos = cos.squeeze(1).squeeze(0)
    sin = sin.squeeze(1).squeeze(0)
    cos = cos[position_ids].unsqueeze(1)
    sin = sin[position_ids].unsqueeze(1)
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


class Eagle3Attention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        rope_theta: float = 1000000.0,
        max_position_embeddings: int = 262144,
        rope_scaling: dict | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_kv_groups = num_heads // num_kv_heads

        input_dim = hidden_size * 2
        self.q_proj = nn.Linear(input_dim, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(input_dim, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(input_dim, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

        scaling_factor = 1.0
        original_max_pos = 4096
        beta_fast = 32.0
        beta_slow = 1.0
        mscale = 1.0
        mscale_all_dim = 0.0
        rope_type = "yarn"
        low_freq_factor = 1.0
        high_freq_factor = 4.0

        if rope_scaling is not None:
            rope_type = rope_scaling.get("rope_type", rope_scaling.get("type", "yarn"))
            scaling_factor = rope_scaling.get("factor", 1.0)
            original_max_pos = rope_scaling.get(
                "original_max_position_embeddings", 4096
            )
            beta_fast = rope_scaling.get("beta_fast", 32.0)
            beta_slow = rope_scaling.get("beta_slow", 1.0)
            mscale = rope_scaling.get("mscale", 1.0)
            mscale_all_dim = rope_scaling.get("mscale_all_dim", 0.0)
            low_freq_factor = rope_scaling.get("low_freq_factor", 1.0)
            high_freq_factor = rope_scaling.get("high_freq_factor", 4.0)

        self.rotary_emb = RotaryEmbedding(
            head_dim,
            max_position_embeddings=max_position_embeddings,
            base=rope_theta,
            scaling_factor=scaling_factor,
            original_max_position_embeddings=original_max_pos,
            beta_fast=beta_fast,
            beta_slow=beta_slow,
            mscale=mscale,
            mscale_all_dim=mscale_all_dim,
            rope_type=rope_type,
            low_freq_factor=low_freq_factor,
            high_freq_factor=high_freq_factor,
        )

        # YaRN scales softmax by mscale^2/sqrt(d); llama3 / no-scaling use SDPA default.
        self._softmax_scale: float | None = None
        if rope_scaling is not None and scaling_factor > 1.0 and rope_type == "yarn":
            ms = _yarn_get_mscale(scaling_factor, mscale_all_dim)
            self._softmax_scale = (ms * ms) / math.sqrt(head_dim)

    def forward(
        self,
        hidden_states: Tensor,
        position_ids: Tensor,
        attn_mask: Tensor | None = None,
        cache_keys: Tensor | None = None,
        cache_values: Tensor | None = None,
        use_cache: bool = False,
    ) -> tuple[Tensor, Tensor | None, Tensor | None]:
        B, T, _ = hidden_states.shape

        q = (
            self.q_proj(hidden_states)
            .view(B, T, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(hidden_states)
            .view(B, T, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(hidden_states)
            .view(B, T, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )

        if not use_cache:
            cos, sin = self.rotary_emb(q, T)
            q, k = _apply_rotary_pos_emb(q, k, cos, sin, position_ids)

            if self.num_kv_groups > 1:
                k = (
                    k.unsqueeze(2)
                    .expand(-1, -1, self.num_kv_groups, -1, -1)
                    .reshape(B, self.num_heads, T, self.head_dim)
                )
                v = (
                    v.unsqueeze(2)
                    .expand(-1, -1, self.num_kv_groups, -1, -1)
                    .reshape(B, self.num_heads, T, self.head_dim)
                )

            attn_out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                is_causal=(attn_mask is None),
                scale=self._softmax_scale,
            )
            attn_out = (
                attn_out.transpose(1, 2)
                .contiguous()
                .view(B, T, self.num_heads * self.head_dim)
            )
            return self.o_proj(attn_out), None, None

        # --- Cached path: diagonal attention (matches TorchSpec) ---
        lck = 0 if cache_keys is None else cache_keys.shape[2]

        cos, sin = self.rotary_emb(q, T + lck)
        q, k = _apply_rotary_pos_emb(q, k, cos, sin, position_ids + lck)

        if self.num_kv_groups > 1:
            k = (
                k.unsqueeze(2)
                .expand(-1, -1, self.num_kv_groups, -1, -1)
                .reshape(B, self.num_heads, T, self.head_dim)
            )
            v = (
                v.unsqueeze(2)
                .expand(-1, -1, self.num_kv_groups, -1, -1)
                .reshape(B, self.num_heads, T, self.head_dim)
            )

        if cache_keys is None:
            cache_keys = k.unsqueeze(2)
            cache_values = v.unsqueeze(2)
        else:
            cache_keys = torch.cat([cache_keys, k.unsqueeze(2)], dim=2)
            cache_values = torch.cat([cache_values, v.unsqueeze(2)], dim=2)

        lck = cache_keys.shape[2]
        k0 = cache_keys[:, :, 0]
        v0 = cache_values[:, :, 0]

        scale = self._softmax_scale or (1.0 / math.sqrt(self.head_dim))

        from atomrl.utils.flash_attn import is_available as _fa_available

        if _fa_available() and attn_mask is not None and attn_mask.dim() == 2:
            attn_out = self._cached_attn_flash(
                q,
                k0,
                v0,
                cache_keys,
                cache_values,
                lck,
                attn_mask,
                scale,
                B,
                T,
            )
        else:
            attn_out = self._cached_attn_matmul(
                q,
                k0,
                v0,
                cache_keys,
                cache_values,
                lck,
                attn_mask,
                scale,
                B,
                T,
            )

        attn_out = (
            attn_out.transpose(1, 2)
            .contiguous()
            .view(B, T, self.num_heads * self.head_dim)
        )
        return self.o_proj(attn_out), cache_keys, cache_values

    def _cached_attn_flash(
        self,
        q,
        k0,
        v0,
        cache_keys,
        cache_values,
        lck,
        attn_mask,
        scale,
        B,
        T,
    ):
        """Flash attention for k0 + online softmax combination with diagonal steps."""
        from atomrl.utils.flash_attn import varlen_causal_attn_with_lse

        out0, lse0 = varlen_causal_attn_with_lse(q, k0, v0, attn_mask, scale)

        if lck <= 1:
            return out0

        # Diagonal scores for steps 1..lck-1
        pad_mask = ~attn_mask.bool()  # [B, T]
        diag_scores = []
        for i in range(1, lck):
            ki = cache_keys[:, :, i]
            s_i = (q * ki).sum(-1) * scale  # [B, H, T]
            s_i = s_i.masked_fill(pad_mask.unsqueeze(1), float("-inf"))
            diag_scores.append(s_i)

        # Combined LSE: logsumexp(lse0, s_1, ..., s_{lck-1})
        all_lse = torch.stack([lse0] + diag_scores, dim=-1).float()  # [B, H, T, lck]
        lse_all = torch.logsumexp(all_lse, dim=-1)  # [B, H, T]
        lse_all = lse_all.masked_fill(lse_all == float("-inf"), 0.0)

        # Reweight flash attention output
        w0 = torch.exp(lse0 - lse_all)  # [B, H, T]
        attn_out = out0 * w0.unsqueeze(-1).to(out0.dtype)

        # Add diagonal value contributions
        for i in range(1, lck):
            vi = cache_values[:, :, i]
            wi = torch.exp(diag_scores[i - 1] - lse_all).to(vi.dtype)
            attn_out = attn_out + wi.unsqueeze(-1) * vi

        return attn_out

    def _cached_attn_matmul(
        self,
        q,
        k0,
        v0,
        cache_keys,
        cache_values,
        lck,
        attn_mask,
        scale,
        B,
        T,
    ):
        """Fallback: manual matmul with O(T^2) 4D mask."""
        if attn_mask is not None and attn_mask.dim() == 2:
            attn_mask = _prepare_attention_mask(attn_mask, B, T, q.dtype, q.device)

        attn_weights = torch.matmul(q, k0.transpose(2, 3)) * scale
        if attn_mask is not None:
            attn_weights = attn_weights + attn_mask

        for i in range(1, lck):
            ki = cache_keys[:, :, i]
            w_i = (q * ki).sum(-1) * scale
            attn_weights = torch.cat((attn_weights, w_i[..., None]), dim=-1)

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)

        attn_out = torch.matmul(attn_weights[..., :T], v0)
        for i in range(1, lck):
            vi = cache_values[:, :, i]
            wi = attn_weights[..., T + i - 1]
            attn_out = attn_out + wi[..., None] * vi

        return attn_out


class Eagle3MLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Eagle3DecoderLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        intermediate_size: int,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 1000000.0,
        rope_scaling: dict | None = None,
    ) -> None:
        super().__init__()
        self.hidden_norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.self_attn = Eagle3Attention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
        )
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.mlp = Eagle3MLP(hidden_size, intermediate_size)

    def forward(
        self,
        input_emb: Tensor,
        hidden_states: Tensor,
        position_ids: Tensor,
        attn_mask: Tensor | None = None,
        cache_keys: Tensor | None = None,
        cache_values: Tensor | None = None,
        use_cache: bool = False,
    ) -> tuple[Tensor, Tensor | None, Tensor | None]:
        residual = hidden_states

        normed_hidden = self.hidden_norm(hidden_states)
        normed_emb = self.input_layernorm(input_emb)
        concat = torch.cat((normed_emb, normed_hidden), dim=-1)

        hidden_states, cache_keys, cache_values = self.self_attn(
            concat,
            position_ids,
            attn_mask=attn_mask,
            cache_keys=cache_keys,
            cache_values=cache_values,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, cache_keys, cache_values


class Eagle3Model(nn.Module):
    """Eagle3 speculative decoding draft model.

    Architecture aligned with lightseekorg/kimi-k2.5-eagle3.
    FC input: cat(3 aux teacher hidden states) — NOT cat(embed, teacher, prev_draft).
    FC is called ONCE before the speculative loop; only token_embeds change per step.
    """

    def __init__(
        self,
        hidden_dim: int,
        vocab_size: int,
        num_heads: int = 64,
        num_layers: int = 1,
        length: int = 5,
        ffn_dim: int | None = None,
        head_dim: int = 128,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 1000000.0,
        num_kv_heads: int | None = None,
        rope_scaling: dict | None = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.length = length

        num_kv_heads = num_kv_heads or num_heads
        ffn_dim = ffn_dim or hidden_dim * 4

        self.fc = nn.Linear(hidden_dim * 3, hidden_dim, bias=False)

        self.layers = nn.ModuleList(
            [
                Eagle3DecoderLayer(
                    hidden_size=hidden_dim,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                    intermediate_size=ffn_dim,
                    rms_norm_eps=rms_norm_eps,
                    rope_theta=rope_theta,
                    rope_scaling=rope_scaling,
                )
                for _ in range(num_layers)
            ]
        )

        self.out_norm = RMSNorm(hidden_dim, eps=rms_norm_eps)

    def forward(
        self,
        token_embeds: Tensor,
        aux_hidden_states: Tensor,
        teacher_lm_head_weight: Tensor,
        embed_weight: Tensor | None = None,
        loss_mask: Tensor | None = None,
        target_ids: Tensor | None = None,
        loss_type: str = "cross_entropy",
        target_hidden_states: Tensor | None = None,
        attention_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """
        Aligned with TorchSpec Eagle3Model.forward:
        - Off-policy: input_ids left-shifted each step (ground truth, not draft predictions)
        - loss_mask left-shifted each step
        - target_hidden_states indexed by step (not step+1)
        - hidden_states (h) pass between steps (not reset)

        Args:
            token_embeds: [B, T, H] — from F.embedding(input_ids, embed_weight)
            aux_hidden_states: [B, T, 3*H] — concatenated 3 aux teacher hidden states
            teacher_lm_head_weight: [V, H] — teacher lm_head (frozen, for teacher logits only)
            embed_weight: [V, H] — teacher embed_tokens (for off-policy embedding each step)
            loss_mask: [B, T] — mask for valid positions (left-shifted each step)
            target_ids: [B, T] — input token ids (left-shifted each step for off-policy)
            loss_type: "cross_entropy" or "forward_kl"
            target_hidden_states: [B, T+length, H] — teacher last hidden states (post-norm)
            attention_mask: [B, T] — 1 for real tokens, 0 for padding
        """
        B, T, D = token_embeds.shape
        logits_list: list[Tensor] = []
        losses: list[Tensor] = []
        accuracies: list[Tensor] = []

        position_ids = (
            torch.arange(T, device=token_embeds.device).unsqueeze(0).expand(B, -1)
        )

        # Pass raw 2D mask — Eagle3Attention routes to flash attention or builds
        # 4D mask internally as needed
        attn_mask = attention_mask

        if target_hidden_states is not None:
            target_hidden_states = F.pad(
                target_hidden_states, (0, 0, 0, self.length), value=0.0
            )

        h = self.fc(aux_hidden_states)

        current_ids = target_ids
        if current_ids is not None and embed_weight is not None:
            current_ids = current_ids.clamp(min=0, max=embed_weight.shape[0] - 1)
        current_mask = loss_mask

        cache_keys = None
        cache_values = None

        for step in range(self.length):
            for layer in self.layers:
                h, cache_keys, cache_values = layer(
                    input_emb=token_embeds,
                    hidden_states=h,
                    position_ids=position_ids,
                    attn_mask=attn_mask,
                    cache_keys=cache_keys,
                    cache_values=cache_values,
                    use_cache=True,
                )

            _fused_fn = getattr(self, "_fused_kl_loss_fn", None)
            if (
                _fused_fn is not None
                and loss_type == "forward_kl"
                and target_hidden_states is not None
                and current_mask is not None
            ):
                ths = target_hidden_states[:, step : step + T]
                loss, acc = _fused_fn(
                    draft_hidden_states=h,
                    target_hidden_states=ths,
                    loss_mask=current_mask[:, :T],
                    norm_weight=self.out_norm.weight,
                    draft_lm_head_weight=teacher_lm_head_weight,
                    target_lm_head_weight=teacher_lm_head_weight,
                    norm_eps=getattr(
                        self.out_norm,
                        "eps",
                        getattr(self.out_norm, "variance_epsilon", 1e-6),
                    ),
                )
                losses.append(loss)
                accuracies.append(acc)
            else:
                normed = self.out_norm(h)

                if current_mask is not None:
                    hs_flat = normed.reshape(-1, D)
                    mask_flat = current_mask[:, :T].reshape(-1).bool()
                    valid_idx = mask_flat.nonzero(as_tuple=True)[0]

                    N_valid = valid_idx.shape[0]
                    if N_valid > 0:
                        normed_valid = hs_flat.index_select(0, valid_idx)

                        draft_lm_head_w = teacher_lm_head_weight

                        if (
                            loss_type == "forward_kl"
                            and target_hidden_states is not None
                        ):
                            ths = target_hidden_states[:, step : step + T]
                            ths_flat = ths.reshape(-1, target_hidden_states.shape[-1])
                            ths_valid = ths_flat.index_select(0, valid_idx)

                            CHUNK = 512
                            kl_parts: list[Tensor] = []
                            acc_parts: list[Tensor] = []
                            for cs in range(0, N_valid, CHUNK):
                                ce = min(cs + CHUNK, N_valid)
                                with torch.no_grad():
                                    teacher_logits = F.linear(
                                        ths_valid[cs:ce], teacher_lm_head_weight
                                    )
                                    tp = F.softmax(teacher_logits.float(), dim=-1)
                                    teacher_pred = teacher_logits.argmax(dim=-1)
                                    del teacher_logits
                                draft_logits = F.linear(
                                    normed_valid[cs:ce], draft_lm_head_w
                                )
                                logits_f32 = draft_logits.float()
                                kl_parts.append(
                                    torch.logsumexp(logits_f32, dim=-1)
                                    - (tp * logits_f32).sum(-1)
                                )
                                acc_parts.append(
                                    (
                                        draft_logits.argmax(dim=-1) == teacher_pred
                                    ).float()
                                )
                                del tp, draft_logits, logits_f32
                            losses.append(torch.cat(kl_parts).sum() / max(N_valid, 1))
                            accuracies.append(
                                torch.cat(acc_parts).sum() / max(N_valid, 1)
                            )
                        elif current_ids is not None:
                            tgt_flat = current_ids[:, :T].reshape(-1)
                            tgt_valid = tgt_flat.index_select(0, valid_idx)
                            logits_valid = F.linear(normed_valid, draft_lm_head_w)
                            ce_valid = F.cross_entropy(
                                logits_valid.float(), tgt_valid, reduction="none"
                            )
                            acc_valid = (
                                logits_valid.argmax(dim=-1) == tgt_valid
                            ).float()
                            losses.append(ce_valid.sum() / max(N_valid, 1))
                            accuracies.append(acc_valid.sum() / max(N_valid, 1))
                            del logits_valid

            if step < self.length - 1:
                # Off-policy left-shift (TorchSpec: padding(input_ids, left=False))
                if current_ids is not None:
                    current_ids = torch.cat(
                        (current_ids[:, 1:], torch.zeros_like(current_ids[:, :1])),
                        dim=1,
                    )
                if current_mask is not None:
                    current_mask = torch.cat(
                        (current_mask[:, 1:], torch.zeros_like(current_mask[:, :1])),
                        dim=1,
                    )
                if embed_weight is not None and current_ids is not None:
                    token_embeds = F.embedding(current_ids[:, :T], embed_weight)
                else:
                    token_embeds = torch.cat(
                        (token_embeds[:, 1:], torch.zeros_like(token_embeds[:, :1])),
                        dim=1,
                    )

        result: dict[str, Tensor] = {"logits_list": logits_list}
        if losses:
            result["losses"] = losses
            result["accuracies"] = accuracies
        else:
            # Zero valid tokens across all steps (e.g. FSDP rank with all-padding
            # micro-batch).  Return a zero loss connected to the compute graph so
            # backward still runs and FSDP all-reduce doesn't hang.
            zero = h.sum() * 0.0
            result["losses"] = [zero] * self.length
            result["accuracies"] = [torch.zeros(1, device=h.device)] * self.length
        return result
