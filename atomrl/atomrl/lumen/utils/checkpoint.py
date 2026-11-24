# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""FP8-aware activation checkpointing for Lumen.

Wraps PyTorch's gradient checkpointing with proper FP8 scaling manager
context preservation and RNG state tracking for stochastic rounding.
"""

import logging
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

import torch
from torch.utils.checkpoint import checkpoint as torch_checkpoint

logger = logging.getLogger(__name__)


class _FP8ScalingContext:
    """Captures and restores FP8 scaling manager state for recompute."""

    def __init__(self):
        self._saved_state: dict | None = None

    def save(self, scaling_manager):
        """Save current scaling manager state."""
        if scaling_manager is None:
            self._saved_state = None
            return

        state = {}
        if hasattr(scaling_manager, "amax_history"):
            state["amax_history"] = {
                k: [t.clone() for t in v]
                for k, v in scaling_manager.amax_history.items()
            }
        if hasattr(scaling_manager, "scale_cache"):
            state["scale_cache"] = {
                k: v.clone() if isinstance(v, torch.Tensor) else v
                for k, v in scaling_manager.scale_cache.items()
            }
        self._saved_state = state

    def restore(self, scaling_manager):
        """Restore saved scaling manager state."""
        if scaling_manager is None or self._saved_state is None:
            return

        state = self._saved_state
        if "amax_history" in state and hasattr(scaling_manager, "amax_history"):
            scaling_manager.amax_history.clear()
            for k, saved_deque_list in state["amax_history"].items():
                live = scaling_manager.amax_history[k]
                for t in saved_deque_list:
                    live.append(t.clone())
        if "scale_cache" in state and hasattr(scaling_manager, "scale_cache"):
            scaling_manager.scale_cache = {
                k: v.clone() if isinstance(v, torch.Tensor) else v
                for k, v in state["scale_cache"].items()
            }


@contextmanager
def _fp8_recompute_context(scaling_manager=None):
    """Context manager that saves/restores FP8 state during recompute."""
    fp8_ctx = _FP8ScalingContext()
    fp8_ctx.save(scaling_manager)

    # Save RNG state for stochastic rounding
    rng_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None

    try:
        yield
    finally:
        # Restore FP8 state after recompute
        fp8_ctx.restore(scaling_manager)
        if rng_state is not None:
            torch.cuda.set_rng_state(rng_state)


def lumen_checkpoint(
    forward_fn: Callable,
    *args,
    scaling_manager=None,
    use_reentrant: bool = False,
    preserve_rng_state: bool = True,
    **kwargs,
) -> Any:
    """FP8-aware gradient checkpointing.

    Wraps torch.utils.checkpoint.checkpoint with FP8 scaling manager
    context preservation. Ensures that:
    1. FP8 scaling state is consistent during recompute
    2. RNG state for stochastic rounding is preserved
    3. Amax history is not corrupted by recompute

    Args:
        forward_fn: Forward function to checkpoint.
        *args: Arguments to forward_fn.
        scaling_manager: Optional FP8 scaling manager.
        use_reentrant: Use reentrant checkpointing (default False).
        preserve_rng_state: Preserve RNG state for reproducibility.
        **kwargs: Additional kwargs for forward_fn.

    Returns:
        Output of forward_fn.
    """
    if scaling_manager is not None:
        fp8_ctx = _FP8ScalingContext()
        fp8_ctx.save(scaling_manager)

        original_fn = forward_fn

        def wrapped_fn(*a, **kw):
            fp8_ctx.restore(scaling_manager)
            return original_fn(*a, **kw)

        forward_fn = wrapped_fn

    return torch_checkpoint(
        forward_fn,
        *args,
        use_reentrant=use_reentrant,
        preserve_rng_state=preserve_rng_state,
        **kwargs,
    )


def lumen_checkpoint_core_attention(
    attn_forward_fn: Callable,
    *args,
    scaling_manager=None,
    **kwargs,
) -> Any:
    """Selective checkpointing for attention core computation.

    Checkpoints only the attention dot-product and softmax,
    while keeping QKV projections and output projection in memory.

    Args:
        attn_forward_fn: Attention core forward function.
        *args: Q, K, V tensors and other attention args.
        scaling_manager: Optional FP8 scaling manager.
        **kwargs: Attention kwargs (causal, softmax_scale, etc.).
    """
    return lumen_checkpoint(
        attn_forward_fn,
        *args,
        scaling_manager=scaling_manager,
        use_reentrant=False,
        **kwargs,
    )
