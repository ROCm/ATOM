# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""CPU-only tests for sampling helpers."""

import sys
import types
from unittest.mock import MagicMock

import torch


def _import_apply_min_tokens_mask():
    try:
        from atom.model_ops.sampler import apply_min_tokens_mask

        return apply_min_tokens_mask
    except ImportError:
        # Either AITER is missing outright, or — as on the non-GPU CI runner —
        # an empty ``aiter`` namespace package shadows it and the kernel symbols
        # cannot be bound. Both land here.
        pass

    # ``sampler`` requires AITER in production. Stub the modules it imports;
    # these tests exercise the pure PyTorch masking helper. Every stub is
    # restored afterwards so the fake does not leak into later test modules.
    stub_names = (
        "aiter",
        "aiter.ops",
        "aiter.ops.triton",
        "aiter.ops.triton.softmax",
        "aiter.ops.triton.topk",
        "atom.model_ops.sampler",
    )
    saved = {name: sys.modules.get(name) for name in stub_names}
    for name in stub_names[:-1]:
        module = types.ModuleType(name)
        module.__path__ = []
        # Any symbol sampler.py binds from AITER resolves to a mock.
        module.__getattr__ = lambda _attr: MagicMock()
        sys.modules[name] = module
    sys.modules["aiter.ops.triton.softmax"].softmax = torch.softmax
    sys.modules["aiter.ops.triton.topk"].topk = torch.topk
    sys.modules.pop("atom.model_ops.sampler", None)
    try:
        from atom.model_ops.sampler import apply_min_tokens_mask

        return apply_min_tokens_mask
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


apply_min_tokens_mask = _import_apply_min_tokens_mask()


def test_min_tokens_masks_first_token_terminal_logits():
    logits = torch.arange(6, dtype=torch.float32).unsqueeze(0)

    result = apply_min_tokens_mask(
        logits,
        min_tokens=[1],
        num_completion_tokens=[0],
        eos_token_id=1,
        stop_token_ids=[2],
        single_token_stops=[{3}],
    )

    assert torch.isneginf(result[0, 1])
    assert torch.isneginf(result[0, 2])
    assert torch.isneginf(result[0, 3])
    torch.testing.assert_close(result[0, [0, 4, 5]], torch.tensor([0.0, 4.0, 5.0]))


def test_min_tokens_does_not_mask_at_threshold():
    logits = torch.arange(6, dtype=torch.float32).unsqueeze(0)
    original = logits.clone()

    result = apply_min_tokens_mask(
        logits,
        min_tokens=[1],
        num_completion_tokens=[1],
        eos_token_id=1,
        stop_token_ids=[2],
        single_token_stops=[{3}],
    )

    torch.testing.assert_close(result, original)
