# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""DFlash draft model for speculative decoding.

DFlash draft model.  DFlash uses multi-layer teacher hidden
states with anchor sampling and block-causal masking.

Architecture
------------
1. Receive hidden states from *K* intermediate teacher layers.
2. Fuse multi-layer hidden states via a learned projection.
3. Pass through lightweight Transformer blocks with block-causal masking.
4. Project to vocabulary logits via the shared teacher ``lm_head``.

Loss uses cross-entropy with exponential decay weighting:
``weight(i) = exp(-(i - 1) / gamma)`` where *i* is the prediction position.
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F
from torch import Tensor, nn

logger = logging.getLogger(__name__)


class DFlashModel(nn.Module):
    """DFlash speculative decoding draft model.

    Parameters
    ----------
    hidden_dim : int
        Dimension matching the target model's hidden size.
    num_target_layers : int
        Number of intermediate teacher layers whose hidden states are consumed.
    num_heads : int
        Attention heads in each Transformer block.
    num_layers : int
        Number of stacked Transformer blocks.
    block_size : int
        Block size for block-causal masking during training.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_target_layers: int = 1,
        num_heads: int = 16,
        num_layers: int = 2,
        block_size: int = 8,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_target_layers = num_target_layers
        self.block_size = block_size

        # Fuse multi-layer teacher hidden states into a single vector
        if num_target_layers > 1:
            self.layer_proj = nn.Linear(
                hidden_dim * num_target_layers, hidden_dim, bias=False
            )
        else:
            self.layer_proj = None

        self.input_norm = nn.LayerNorm(hidden_dim)
        self.blocks = nn.ModuleList()
        for _ in range(num_layers):
            self.blocks.append(_DFlashTransformerBlock(hidden_dim, num_heads))
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        teacher_hidden_states: list[Tensor] | Tensor,
        lm_head_weight: Tensor,
        loss_mask: Tensor | None = None,
        target_ids: Tensor | None = None,
        loss_decay_gamma: float = 7.0,
    ) -> dict[str, Tensor]:
        """Forward pass.

        Args:
            teacher_hidden_states: Either a list of ``[B, T, D]`` tensors (one
                per target layer) or a single pre-concatenated ``[B, T, D*K]``
                tensor.
            lm_head_weight: ``[V, D]`` teacher lm_head weight.
            loss_mask: ``[B, T]`` mask.
            target_ids: ``[B, T+1]`` ground-truth tokens.  Position *t* predicts
                ``target_ids[:, t+1]``.
            loss_decay_gamma: Exponential decay factor for position weighting.

        Returns:
            Dict with ``"logits"``, and optionally ``"loss"`` and ``"accuracy"``.
        """
        if isinstance(teacher_hidden_states, list):
            if len(teacher_hidden_states) > 1:
                x = torch.cat(teacher_hidden_states, dim=-1)
            else:
                x = teacher_hidden_states[0]
        else:
            x = teacher_hidden_states

        if self.layer_proj is not None:
            x = self.layer_proj(x)

        x = self.input_norm(x)

        # Build block-causal mask
        B, T, _D = x.shape
        attn_mask = self._build_block_causal_mask(T, x.device)

        for block in self.blocks:
            x = block(x, attn_mask=attn_mask)

        x = self.out_norm(x)
        logits = F.linear(x, lm_head_weight)  # [B, T, V]

        result: dict[str, Tensor] = {"logits": logits}

        if target_ids is not None:
            tgt = target_ids[:, 1 : T + 1]  # [B, T]
            ce = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                tgt.reshape(-1),
                reduction="none",
            ).reshape(B, T)

            # Exponential decay weights
            positions = torch.arange(T, device=x.device, dtype=torch.float32)
            decay_weights = torch.exp(-positions / loss_decay_gamma)

            if loss_mask is not None:
                mask_f = loss_mask.to(dtype=ce.dtype)
                weights = mask_f * decay_weights.unsqueeze(0)
                denom = weights.sum().clamp(min=1.0)
                result["loss"] = (ce * weights).sum() / denom
                preds = logits.argmax(dim=-1)
                result["accuracy"] = (
                    (preds == tgt).float() * mask_f
                ).sum() / mask_f.sum().clamp(min=1.0)
            else:
                weights = decay_weights.unsqueeze(0).expand(B, -1)
                result["loss"] = (ce * weights).mean()
                result["accuracy"] = (logits.argmax(dim=-1) == tgt).float().mean()

        return result

    def _build_block_causal_mask(self, seq_len: int, device: torch.device) -> Tensor:
        """Create block-causal attention mask.

        Each block of ``block_size`` tokens can only attend to positions
        strictly before the block's start, plus positions within the block.
        """
        bs = self.block_size
        mask = torch.zeros(seq_len, seq_len, device=device, dtype=torch.bool)
        for i in range(seq_len):
            block_start = (i // bs) * bs
            # Can attend to everything before this block + within this block up to i
            mask[i, :block_start] = True
            mask[i, block_start : i + 1] = True
        # Convert to float mask for nn.MultiheadAttention (0 = attend, -inf = mask)
        float_mask = torch.where(mask, 0.0, float("-inf"))
        return float_mask


class _DFlashTransformerBlock(nn.Module):
    """Transformer block with pre-norm and SwiGLU FFN."""

    def __init__(self, hidden_dim: int, num_heads: int) -> None:
        super().__init__()
        ffn_dim = hidden_dim * 4
        self.attn_norm = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.w1 = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.w2 = nn.Linear(ffn_dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, ffn_dim, bias=False)

    def forward(self, x: Tensor, attn_mask: Tensor | None = None) -> Tensor:
        h = self.attn_norm(x)
        h, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + h
        h = self.ffn_norm(x)
        x = x + self.w2(F.silu(self.w1(h)) * self.w3(h))
        return x
