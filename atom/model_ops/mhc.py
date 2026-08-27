# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Manifold-Constrained Hyper-Connections (mHC) as a reusable sub-layer wrapper.

An mHC model does not carry one residual vector per token but ``hc_mult`` of
them.  Every sub-layer (attention, FFN) is sandwiched between two learned
projections:

  ``hc_pre``   ``[T, hc, dim]`` -> ``[T, dim]``   reduce the stack to one
               sub-layer input, and emit the post-gate / combination matrix
               that ``hc_post`` will need.
  ``hc_post``  ``[T, dim]`` -> ``[T, hc, dim]``   gate the sub-layer's output
               back into the stack, mixing the previous stack through a
               doubly-stochastic (Sinkhorn-projected) combination matrix.

The three parameter groups per sub-layer are ``fn`` (the mixing projection),
``base`` (its bias) and ``scale``; ``hc_split_sinkhorn`` turns their product
into the ``(pre, post, comb)`` triple.  See ``atom/model_ops/sparse_attn_v4.py``
for that projection and Xie et al. (2026) for the formulation.

Two consecutive sub-layers meet as "previous ``hc_post``" immediately followed
by "next ``hc_pre``", so aiter exposes them fused as ``mhc_fused_post_pre``.
Callers therefore *defer* each ``hc_post`` and hand its inputs to the next
``hc_pre``; only the final sub-layer materializes its ``hc_post`` alone.

DeepSeek-V4 (``atom/models/deepseek_v4.py``) carries its own copy of this math,
fused into its heavily-tuned ``Block``.  This module is the standalone version
used by GLM-5.3-Flash; the two should converge once V4's block can be
refactored without disturbing its tuning.
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F
from torch import nn

try:  # aiter provides fused mHC kernels on ROCm; absence falls back to torch.
    import aiter
except ImportError:  # pragma: no cover - aiter is present in every ATOM image
    aiter = None

from atom.model_ops.sparse_attn_v4 import hc_split_sinkhorn


def hc_expand(x: torch.Tensor, hc_mult: int) -> torch.Tensor:
    """``[T, dim]`` -> ``[T, hc, dim]`` by replication (model entry)."""
    return x.unsqueeze(-2).repeat(1, hc_mult, 1)


def hc_contract(x: torch.Tensor) -> torch.Tensor:
    """``[T, hc, dim]`` -> ``[T, dim]`` by averaging the streams (model exit).

    GLM-5.3-Flash ships no learned read-out head (no ``hc_head_*`` tensors), so
    the stack collapses by a plain mean.  DeepSeek-V4 instead learns that
    reduction -- do not "share" the two.
    """
    return x.mean(dim=-2)


class HyperConnection(nn.Module):
    """The ``fn``/``base``/``scale`` parameter group for ONE sub-layer.

    ``dim`` is the model width, ``hc_mult`` the number of residual streams.
    Parameters are fp32 (the reference sets ``set_dtype(torch.float32)``); the
    residual stream itself stays bf16.
    """

    def __init__(self, dim: int, hc_mult: int) -> None:
        super().__init__()
        from atom.model_ops.utils import atom_parameter

        self.dim = dim
        self.hc_mult = hc_mult
        mix_hc = (2 + hc_mult) * hc_mult
        self.fn = atom_parameter(
            torch.empty(mix_hc, hc_mult * dim, dtype=torch.float32)
        )
        self.base = atom_parameter(torch.empty(mix_hc, dtype=torch.float32))
        self.scale = atom_parameter(torch.empty(3, dtype=torch.float32))


class MHCOps:
    """Kernel-bound mHC operations shared by every sub-layer of one model.

    Holds the resolved aiter entry points and the scalar hyper-parameters, so a
    decoder layer only supplies the per-sub-layer ``HyperConnection`` tensors.
    """

    # `hc_post_mult_value`: the post gate is `2.0 * sigmoid(...)`.
    HC_POST_MULT = 2.0

    def __init__(
        self,
        dim: int,
        hc_mult: int,
        norm_eps: float,
        hc_eps: float,
        sinkhorn_iters: int,
        post_mult: float = HC_POST_MULT,
    ) -> None:
        self.dim = dim
        self.hc_mult = hc_mult
        self.norm_eps = norm_eps
        self.hc_eps = hc_eps
        self.sinkhorn_iters = sinkhorn_iters
        self.post_mult = post_mult
        # aiter's mhc kernels trap (mhc_kernels.cu __builtin_trap) unless the
        # model width is a multiple of 256 or 512; bind them only when legal so
        # an unsupported width silently takes the torch path instead of dying.
        dim_ok = dim % 512 == 0 or dim % 256 == 0
        # ATOM_MHC_FORCE_TORCH=1 drops to the torch reference. The fused kernels
        # and the reference must agree; when they disagree the output degrades
        # quietly rather than failing, so flipping this is the cheapest way to
        # tell an mHC bug apart from one elsewhere in the model.
        if os.environ.get("ATOM_MHC_FORCE_TORCH", "0") == "1":
            dim_ok = False
        get = (
            (lambda n: getattr(aiter, n, None))
            if aiter is not None
            else (lambda n: None)
        )
        self._pre = get("mhc_pre") if dim_ok else None
        self._post = get("mhc_post") if dim_ok else None
        self._fused_post_pre = get("mhc_fused_post_pre") if dim_ok else None

    @property
    def has_fused_post_pre(self) -> bool:
        return self._fused_post_pre is not None

    def pre(
        self,
        residual: torch.Tensor,  # [T, hc, dim]
        hc: HyperConnection,
        norm_weight: torch.Tensor | None = None,
        norm_eps: float = 1e-6,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns ``(y [T, dim], post [T, hc], comb [T, hc, hc])``."""
        if self._pre is not None:
            post, comb, y = self._pre(
                residual,
                hc.fn,
                hc.scale,
                hc.base,
                float(self.norm_eps),
                float(self.hc_eps),
                float(self.hc_eps),
                self.post_mult,
                int(self.sinkhorn_iters),
                norm_weight,
                norm_eps,
            )
            return y, post.squeeze(-1), comb

        dtype = residual.dtype
        x_flat = residual.flatten(-2)
        x_normed = F.rms_norm(x_flat.float(), (x_flat.shape[-1],), None, self.norm_eps)
        mixes = F.linear(x_normed, hc.fn)
        pre, post, comb = hc_split_sinkhorn(
            mixes, hc.scale, hc.base, self.hc_mult, self.sinkhorn_iters, self.hc_eps
        )
        y = torch.sum(pre.unsqueeze(-1) * residual, dim=-2)
        if norm_weight is not None:
            y = F.rms_norm(y.float(), (y.shape[-1],), norm_weight.float(), norm_eps).to(
                dtype
            )
        return y.to(dtype), post, comb

    def post(
        self,
        x: torch.Tensor,  # [T, dim] sub-layer output
        residual: torch.Tensor,  # [T, hc, dim] pre-sub-layer stack
        post_gate: torch.Tensor,  # [T, hc]
        comb: torch.Tensor,  # [T, hc, hc]
    ) -> torch.Tensor:
        """Returns the new residual stack ``[T, hc, dim]``."""
        if self._post is not None:
            out = torch.empty_like(residual)
            self._post(out, x, residual, post_gate.unsqueeze(-1), comb)
            return out
        y = post_gate.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(
            comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=-3
        )
        return y.type_as(x)

    def fused_post_pre(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post_gate: torch.Tensor,
        comb: torch.Tensor,
        hc: HyperConnection,
        norm_weight: torch.Tensor | None = None,
        norm_eps: float = 1e-6,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Previous sub-layer's ``post`` + this sub-layer's ``pre``, in one call.

        Returns ``(residual, post_gate, comb, y)``.
        """
        if self._fused_post_pre is not None:
            # aiter returns (post, comb, y, residual) -- a DIFFERENT order from
            # this method's contract. Reorder here so callers never have to know
            # which path ran; getting this wrong is silent until a shape
            # mismatch surfaces layers later.
            new_post, new_comb, y, new_residual = self._fused_post_pre(
                x,
                residual,
                post_gate,
                comb,
                hc.fn,
                hc.scale,
                hc.base,
                float(self.norm_eps),
                float(self.hc_eps),
                float(self.hc_eps),
                self.post_mult,
                int(self.sinkhorn_iters),
                norm_weight,
                norm_eps,
            )
            if new_post.dim() == 3 and new_post.shape[-1] == 1:
                new_post = new_post.squeeze(-1)
            return new_residual, new_post, new_comb, y
        residual = self.post(x, residual, post_gate, comb)
        y, post_gate, comb = self.pre(residual, hc, norm_weight, norm_eps)
        return residual, post_gate, comb, y
