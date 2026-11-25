# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Lazy teacher logits reconstruction from cached hidden states.

Loads only the teacher's ``lm_head`` (and optionally final layer norm) so that
full-vocabulary logits can be reconstructed on the training side without loading
the entire teacher model.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from safetensors import safe_open
from torch import nn

logger = logging.getLogger(__name__)


class TeacherLMHead(nn.Module):
    """Frozen projection: hidden states -> full-vocabulary logits.

    Given teacher hidden states ``[B, T, D]``, produces logits ``[B, T, V]``
    using only the teacher's output projection weight (and optional RMSNorm).
    All parameters are frozen -- no gradients flow through this module.
    """

    def __init__(
        self,
        lm_head_weight: torch.Tensor,
        norm_weight: torch.Tensor | None = None,
        norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        vocab_size, hidden_dim = lm_head_weight.shape
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.lm_head.weight = nn.Parameter(lm_head_weight, requires_grad=False)

        self.norm: nn.Module | None = None
        if norm_weight is not None:
            self.norm = _RMSNorm(hidden_dim, eps=norm_eps)
            self.norm.weight = nn.Parameter(norm_weight, requires_grad=False)

        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)

    @classmethod
    def from_pretrained(
        cls,
        model_path: str | Path,
        *,
        lm_head_key: str = "lm_head.weight",
        norm_key: str = "model.norm.weight",
        load_norm: bool = False,
        norm_eps: float = 1e-6,
        device: str | torch.device = "cpu",
    ) -> TeacherLMHead:
        """Load from a HuggingFace-style checkpoint directory.

        Scans safetensors shards for the required weight keys.
        """
        model_dir = Path(model_path)
        lm_head_weight = None
        norm_weight = None

        shard_files = sorted(model_dir.glob("*.safetensors"))
        if not shard_files:
            raise FileNotFoundError(f"No safetensors files found in {model_dir}")

        for shard_path in shard_files:
            with safe_open(str(shard_path), framework="pt", device=str(device)) as f:
                keys = f.keys()
                if lm_head_key in keys and lm_head_weight is None:
                    lm_head_weight = f.get_tensor(lm_head_key)
                if load_norm and norm_key in keys and norm_weight is None:
                    norm_weight = f.get_tensor(norm_key)

            if lm_head_weight is not None and (
                not load_norm or norm_weight is not None
            ):
                break

        if lm_head_weight is None:
            raise KeyError(f"Key {lm_head_key!r} not found in {model_dir}")

        logger.info(
            "TeacherLMHead loaded: vocab_size=%d, hidden_dim=%d, has_norm=%s",
            lm_head_weight.shape[0],
            lm_head_weight.shape[1],
            norm_weight is not None,
        )
        return cls(lm_head_weight, norm_weight, norm_eps=norm_eps)

    @torch.no_grad()
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Reconstruct logits from teacher hidden states.

        Args:
            hidden_states: ``[B, T, D]`` teacher last-layer hidden states.

        Returns:
            Logits tensor ``[B, T, V]``.
        """
        x = hidden_states
        if self.norm is not None:
            x = self.norm(x)
        return self.lm_head(x)


class _RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * norm).to(x.dtype) * self.weight
