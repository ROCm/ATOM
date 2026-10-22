"""Avoid HIP segfault in xgrammar's CUDA bitmask kernel.

On ROCm, SGLang's default xgrammar path calls
``sgl_kernel.apply_token_bitmask_inplace_cuda``, which segfaults when
``response_format={"type":"json_object"}`` (and other constrained-decoding
requests) apply the vocab mask.

CUDA builds use a Triton kernel, but that launch requests
``num_warps`` that exceeds MI300 wavefront limits for fp32 logits
(``OutOfResources: threads Required: 2048, Hardware limit: 1024``).
CPU xgrammar apply also rejects device logits.

This patch replaces the HIP binding with a pure-Torch GPU apply
(packed int32 bitmask → ``masked_fill_(-inf)``), matching the CUDA
semantic without the broken native/Triton launches.

Disable with ``ATOM_SGLANG_XGRAMMAR_HIP_TRITON_BITMASK=0``.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional, Union

import torch

logger = logging.getLogger("atom.plugin.sglang.xgrammar_hip_bitmask")

_BITS_PER_BLOCK = 32


def _env_enabled() -> bool:
    return os.environ.get("ATOM_SGLANG_XGRAMMAR_HIP_TRITON_BITMASK", "1") != "0"


def _apply_bitmask_torch_gpu(
    logits: torch.Tensor,
    bitmask: torch.Tensor,
    indices: Optional[Union[List[int], torch.Tensor]] = None,
) -> None:
    """Apply packed int32 token bitmask on GPU via Torch (HIP-safe)."""

    if bitmask.dtype != torch.int32:
        raise TypeError(f"bitmask must be int32, got {bitmask.dtype}")

    logits_2d = logits.unsqueeze(0) if logits.ndim == 1 else logits
    bitmask_2d = bitmask.unsqueeze(0) if bitmask.ndim == 1 else bitmask

    if indices is not None:
        if not isinstance(indices, torch.Tensor):
            indices = torch.tensor(indices, dtype=torch.long, device=logits.device)
        else:
            indices = indices.to(device=logits.device, dtype=torch.long)
        logits_rows = logits_2d.index_select(0, indices)
        bitmask_rows = bitmask_2d.index_select(0, indices)
    else:
        if logits_2d.shape[0] != bitmask_2d.shape[0]:
            raise ValueError(
                f"batch size mismatch: logits {logits_2d.shape[0]} vs "
                f"bitmask {bitmask_2d.shape[0]}"
            )
        logits_rows = logits_2d
        bitmask_rows = bitmask_2d
        indices = None

    vocab_size = min(logits_rows.shape[-1], bitmask_rows.shape[-1] * _BITS_PER_BLOCK)
    bit_ids = torch.arange(_BITS_PER_BLOCK, device=logits.device, dtype=torch.int32)
    # bitmask bit==0 → masked (set to -inf), same as upstream Triton/xgrammar.
    expanded = ((bitmask_rows.unsqueeze(-1) >> bit_ids) & 1) == 0
    mask = expanded.reshape(bitmask_rows.shape[0], -1)[:, :vocab_size]
    logits_rows[:, :vocab_size].masked_fill_(mask, float("-inf"))

    if indices is not None:
        logits_2d.index_copy_(0, indices, logits_rows)


def _apply_bitmask_hip_safe(logits, vocab_mask, indices=None) -> None:
    _apply_bitmask_torch_gpu(logits, vocab_mask, indices=indices)


def apply_xgrammar_hip_bitmask_patch() -> None:
    """Replace HIP module binding so apply_vocab_mask uses Torch GPU apply."""

    if not _env_enabled():
        logger.info("Skip xgrammar HIP bitmask patch (env disabled)")
        return

    try:
        from sglang.srt.utils import is_hip
    except Exception:  # noqa: BLE001
        return
    if not is_hip():
        return

    try:
        # Importing the module binds apply_token_bitmask_inplace_cuda into
        # its globals; methods look that name up at call time, so replacing
        # the binding is enough for both XGrammarGrammar and XGrammarBackend.
        from sglang.srt.constrained import xgrammar_backend as xg
    except Exception:  # noqa: BLE001 - outlines-only installs
        logger.info("xgrammar backend not importable; skip HIP bitmask patch")
        return

    if getattr(xg, "_atom_hip_bitmask_patched", False):
        return

    if not getattr(xg, "_is_hip", False) and not hasattr(
        xg, "apply_token_bitmask_inplace_cuda"
    ):
        logger.info("xgrammar backend not on HIP CUDA bitmask path; skip")
        return

    xg.apply_token_bitmask_inplace_cuda = _apply_bitmask_hip_safe
    xg._atom_hip_bitmask_patched = True
    logger.info(
        "Patched xgrammar HIP vocab-mask apply to Torch GPU; "
        "avoids sgl_kernel CUDA bitmask segfault and Triton OutOfResources"
    )
