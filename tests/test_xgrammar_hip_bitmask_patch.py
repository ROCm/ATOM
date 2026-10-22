"""Unit tests for HIP xgrammar vocab-mask Torch fallback.

Lives under ``tests/`` (not ``tests/plugin/``) so Pre Checkin's
``run_unit_tests.sh`` picks it up: that script ``--ignore=tests/plugin``.
"""

from __future__ import annotations

import torch

from atom.plugin.sglang.patches.xgrammar_hip_bitmask_patch import (
    _apply_bitmask_torch_gpu,
)


def test_apply_bitmask_torch_gpu_masks_disallowed_tokens():
    logits = torch.zeros(2, 64, dtype=torch.float32)
    bitmask = torch.zeros(2, 2, dtype=torch.int32)
    # Allow token 0 (bit 0 of word 0) and token 33 (bit 1 of word 1).
    bitmask[:, 0] = 1 << 0
    bitmask[:, 1] = 1 << 1

    _apply_bitmask_torch_gpu(logits, bitmask)

    assert logits[0, 0] == 0
    assert logits[0, 33] == 0
    assert torch.isneginf(logits[0, 1])
    assert torch.isneginf(logits[0, 32])
    assert torch.isneginf(logits[1, 2])


def test_apply_bitmask_torch_gpu_with_indices():
    logits = torch.zeros(3, 32, dtype=torch.float32)
    bitmask = torch.zeros(3, 1, dtype=torch.int32)
    bitmask[1, 0] = 1 << 3  # only token 3 allowed on row 1

    _apply_bitmask_torch_gpu(logits, bitmask, indices=[1])

    assert logits[1, 3] == 0
    assert torch.isneginf(logits[1, 0])
    # Untouched rows stay zero.
    assert torch.all(logits[0] == 0)
    assert torch.all(logits[2] == 0)


def test_apply_bitmask_torch_gpu_masks_logits_beyond_bitmask():
    # logits padded past bitmask coverage must be -inf (not left finite).
    logits = torch.zeros(1, 40, dtype=torch.float32)
    bitmask = torch.zeros(1, 1, dtype=torch.int32)
    bitmask[0, 0] = 1 << 0  # only token 0 allowed in the first 32

    _apply_bitmask_torch_gpu(logits, bitmask)

    assert logits[0, 0] == 0
    assert torch.isneginf(logits[0, 1])
    assert torch.isneginf(logits[0, 32])
    assert torch.isneginf(logits[0, 39])
