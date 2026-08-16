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


def _block_residual_has_headroom(block_residual: torch.Tensor) -> bool:
    """Whether ``block_residual``'s *storage* -- not just its logical shape --
    has room for one more row in dim 1, i.e. widening the view by one and
    storing into that new row touches only real, already-allocated memory.

    Checking this from the tensor's own stride/shape/storage-offset (rather
    than needing a separate handle to some "parent" buffer) is what lets
    ``_grow_block_residual_in_place`` work as a self-contained operation on
    whatever ``block_residual`` object it is handed.

    A pure stride-ratio check (capacity = stride(0) // stride(1)) is NOT
    enough: PyTorch gives a dim-1-empty tensor the same stride(0) ==
    stride(1) convention whether it is a genuine prefix slice of a padded
    ``[T, N_cap, H]`` buffer (real spare storage) or a tensor freshly made via
    e.g. ``torch.zeros(T, 0, H)`` (no storage at all) -- at ``B == 0`` stride
    alone cannot tell those apart. Comparing against the tensor's actual
    ``untyped_storage().nbytes()`` is unambiguous in every case, including
    that one.
    """
    t, b, h = block_residual.shape
    if t == 0 or block_residual.dim() != 3:
        return False
    s0, s1, s2 = block_residual.stride()
    if s2 != 1 or s1 == 0:
        return False
    last_needed_elem = (
        block_residual.storage_offset() + (t - 1) * s0 + b * s1 + (h - 1) * s2 + 1
    )
    needed_bytes = last_needed_elem * block_residual.element_size()
    return needed_bytes <= block_residual.untyped_storage().nbytes()


def _grow_block_residual_in_place(
    block_residual: torch.Tensor, new_row: torch.Tensor
) -> torch.Tensor | None:
    """Append ``new_row`` as one more candidate without moving existing ones.

    Widens ``block_residual``'s view over its own backing storage and stores
    only the new row -- no read of the existing B rows, no relocation.
    Returns the widened view, or ``None`` if there is no spare capacity (see
    ``_block_residual_has_headroom``), signaling the caller to fall back to
    ``torch.cat`` instead.
    """
    if not _block_residual_has_headroom(block_residual):
        return None
    t, b, h = block_residual.shape
    widened = block_residual.as_strided(
        (t, b + 1, h), block_residual.stride(), block_residual.storage_offset()
    )
    widened[:, b, :] = new_row
    return widened


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
        # Kernel-fused cat([block_residual, prefix_out], 1) from the attn_res
        # call inside forward(), stashed for maybe_close_block() to consume
        # instead of re-reading block_residual via a separate torch.cat. Reset
        # at the top of every forward() and cleared once consumed; None means
        # "no fused block was produced this call" (residuals disabled, no
        # candidates yet, a call that wasn't a close-block layer, or -- see
        # _block_residual_capacity -- block_residual had spare storage to
        # grow into for free, so forward() didn't bother asking the kernel to
        # relocate it at all). Either way maybe_close_block() handles it:
        # first the in-place grow, then the plain torch.cat as a last resort.
        # This is purely a perf side-channel between the two calls that
        # kimi_k3.py always makes back-to-back on the same instance.
        self._pending_block_residual: torch.Tensor | None = None

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

        self._pending_block_residual = None  # reset every call, see the field doc

        if self.enabled and block_residual is not None and block_residual.shape[1] > 0:
            score_weight = self.score_weight
            if score_weight is None:  # loader hook did not run (plugin hosts)
                self.process_weights_after_loading()
                score_weight = self.score_weight
            from atom.model_ops.kimi_k3 import apply_attn_res

            # Ask the kernel to also emit the block-banking concat when this is
            # a close-block layer, so maybe_close_block() (called right after
            # this, on this same instance) can skip its torch.cat -- but only
            # when block_residual has no spare storage to grow into for free
            # (_block_residual_capacity). When it does, relocating all B
            # existing rows through the kernel is strictly more expensive than
            # letting maybe_close_block() append the new candidate with a
            # single in-place [T, H] store, so skip the fused relocate here.
            will_close = (
                self.block_size is not None and self.layer_idx % self.block_size == 0
            )
            needs_relocate = will_close and not _block_residual_has_headroom(
                block_residual
            )
            if not needs_relocate:
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
            mixed_output, prefix_out, block_out = apply_attn_res(
                prefix_sum,
                block_residual,
                score_weight,
                self.eps,
                add_hidden,
                None if self.out_norm is None else self.out_norm.weight,
                self.out_eps,
                add_hidden2,
                close_block=True,
            )
            self._pending_block_residual = block_out
            return mixed_output, prefix_out

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

        When the immediately preceding ``forward()`` call on this instance ran
        the attn_res kernel and had to relocate (no spare storage to grow into
        for free -- see ``_block_residual_capacity``), it already computed this
        same concat in-kernel (``_pending_block_residual``); reuse that instead
        of re-reading ``block_residual`` from HBM via ``torch.cat``. Otherwise
        try ``_grow_block_residual_in_place`` first -- this covers both the
        common case (block_residual has spare storage, forward() deliberately
        skipped the fused relocate) and the very first block (block_residual
        still empty; whether this succeeds then depends on whether the caller
        gave it any spare storage up front). Only a tensor with no spare
        storage at all (e.g. one a pipeline-parallel recv just rematerialized)
        falls through to the plain cat, which is the same cost either way.
        """
        if not self.enabled or self.block_size is None:
            return block_residual, prefix_sum
        if self.layer_idx % self.block_size != 0:
            return block_residual, prefix_sum
        assert block_residual is not None
        if self._pending_block_residual is not None:
            block_residual = self._pending_block_residual
            self._pending_block_residual = None
        else:
            grown = _grow_block_residual_in_place(block_residual, prefix_sum)
            block_residual = (
                grown
                if grown is not None
                else torch.cat([block_residual, prefix_sum.unsqueeze(1)], dim=1)
            )
        return block_residual, None
