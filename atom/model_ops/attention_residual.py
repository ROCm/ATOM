# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Attention-residual mixing layer (Kimi-K3).

The layer wrapper lives here; the Triton kernel it dispatches lives in
``atom.model_ops.kimi_k3.attention_residual``, which follows
flash-linear-attention's ``fused_attnres`` (that module's docstring lists the
deltas). Attention Residuals: https://arxiv.org/abs/2603.15031

This wrapper mirrors how the reference KDA layer drives that op
(``fla/models/kda/modeling_kda.py``): ``proj`` and ``norm`` here are its
``attn_res_proj``/``attn_res_norm``, and ``out_norm`` is its ``attn_norm``,
which fla passes as ``output_rms_weight`` -- hence the result coming back
already normed rather than the caller norming it.
"""

from __future__ import annotations

import torch
from torch import nn

from atom.model_ops.layernorm import RMSNorm
from atom.model_ops.linear import ReplicatedLinear

__all__ = ["AttnRes"]


def _rms_eps(norm: RMSNorm) -> float:
    return getattr(norm, "variance_epsilon", getattr(norm, "eps", 1e-6))


class AttnRes(nn.Module):
    """One attention-residual mixing site.

    Mixes the B candidates of ``block_residual`` with a running ``prefix_sum``:
    rmsnorm each of the B+1, score = <normed, score_weight>, softmax over B+1,
    weighted sum. ``proj`` and ``norm`` define that scoring; their product is a
    load-time constant folded into a single [H] vector (see ``score_weight``).

    Three independent things decide what forward() actually runs, and all three
    are settled here rather than at the call site:

    * ``enabled`` -- whether this model uses attention residuals at all. When
      False there is no mixing and no block state; forward degenerates to
      ``out_norm(prefix_sum + addends)``, i.e. the ordinary pre-norm residual
      step. ``proj``/``norm`` are then unused and may be None.
    * ``block_residual`` empty vs populated -- with no candidates yet the
      softmax is a no-op, so the same degenerate path applies.
    * ``out_norm`` -- the caller's rmsnorm OF THE RESULT. Passing one is what
      decides the fusion: it is folded into the kernel's store and the returned
      mix comes back already normed and scaled, so the caller must not norm it
      again. Given None, the mix is returned raw.

    The upshot for callers is that forward() has one shape in every mode:
    hand it the prefix, the block, and any pending addends; get back
    ``(mixed_output, prefix_out)``. It never returns a mix that still needs
    norming, and never asks the caller which path it took.

    ``proj``/``norm``/``out_norm`` are passed in already constructed and stay
    owned by the caller. That is deliberate: weights load by exact
    ``named_parameters()`` path and a miss only WARNs, silently leaving an
    RMSNorm at all-ones, so re-parenting them under this module would corrupt
    the model quietly. Torch dedups a shared parameter to the name it was FIRST
    registered under, so aliasing here is invisible as long as the owner
    constructs them before handing them over.
    """

    def __init__(
        self,
        proj: ReplicatedLinear | None = None,
        norm: RMSNorm | None = None,
        out_norm: RMSNorm | None = None,
        enabled: bool = True,
        block_size: int | None = None,
        layer_idx: int = 0,
    ):
        super().__init__()
        if enabled and (proj is None or norm is None):
            raise ValueError("an enabled AttnRes needs both proj and norm")
        self.enabled = enabled
        self.proj = proj
        self.norm = norm
        self.out_norm = out_norm
        self.eps = 1e-6 if norm is None else _rms_eps(norm)
        self.out_eps = 1e-6 if out_norm is None else _rms_eps(out_norm)
        self.score_weight: torch.Tensor | None = None
        # Set only on the site that closes out blocks (see maybe_close_block).
        self.block_size = block_size
        self.layer_idx = layer_idx

    def process_weights_after_loading(self) -> None:
        # Fold the static rmsnorm gain and the scoring projection into one [H]
        # vector. Both operands are load-time constants, so the kernel reads a
        # single vector per row instead of reloading and multiplying two.
        # The loader calls this for every module, after the proj's own hook.
        if not self.enabled:
            return
        self.score_weight = (
            self.norm.weight.float() * self.proj.weight.squeeze(0).float()
        ).contiguous()

    def forward(
        self,
        prefix_sum: torch.Tensor | None,
        block_residual: torch.Tensor | None = None,
        add_hidden: torch.Tensor | None = None,
        add_hidden2: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns ``(mixed_output, prefix_out)``.

        The addends are the caller's ``prefix_sum = prefix_sum + ...``, folded
        into the kernel's on-load so no separate [T, H] elementwise kernel runs;
        ``prefix_out`` is that sum. A None ``prefix_sum`` means the block was
        just closed out and this site starts a fresh one, so the first addend
        IS the prefix.
        """
        if prefix_sum is None:
            prefix_sum, add_hidden, add_hidden2 = add_hidden, add_hidden2, None
        assert prefix_sum is not None

        if self.enabled and block_residual is not None and block_residual.shape[1] > 0:
            score_weight = self.score_weight
            if score_weight is None:  # loader hook did not run (plugin hosts)
                self.process_weights_after_loading()
                score_weight = self.score_weight
            from atom.model_ops.kimi_k3 import apply_attn_res

            return apply_attn_res(
                prefix_sum,
                block_residual,
                score_weight,
                self.eps,
                add_hidden,
                None if self.out_norm is None else self.out_norm.weight,
                self.out_eps,
                add_hidden2,
            )

        # Nothing to mix (residuals off, or no candidates yet). Apply by hand
        # what the kernel would otherwise have folded into its load and store.
        if add_hidden is not None:
            prefix_sum = prefix_sum + add_hidden
            if add_hidden2 is not None:
                prefix_sum = prefix_sum + add_hidden2
        mixed = prefix_sum if self.out_norm is None else self.out_norm(prefix_sum)
        return mixed, prefix_sum

    def maybe_close_block(
        self,
        prefix_sum: torch.Tensor,
        block_residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Append ``prefix_sum`` as a candidate every ``block_size`` layers.

        Returns the new ``(block_residual, prefix_sum)``. A closed-out block
        leaves prefix_sum None: the running sum has been banked as a candidate
        and the next site starts a fresh one from whatever it is handed.
        Residuals disabled, or a layer mid-block, means no change.
        """
        if not self.enabled or self.block_size is None:
            return block_residual, prefix_sum
        if self.layer_idx % self.block_size != 0:
            return block_residual, prefix_sum
        assert block_residual is not None
        block_residual = torch.cat([block_residual, prefix_sum.unsqueeze(1)], dim=1)
        return block_residual, None
