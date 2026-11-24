# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

from collections import defaultdict, deque
from types import SimpleNamespace

import pytest
import torch
from atomrl.lumen.utils.checkpoint import (
    _FP8ScalingContext,
    lumen_checkpoint,
    lumen_checkpoint_core_attention,
)
from torch import nn


def _make_mock_sm(maxlen=16):
    """Create a mock with same state shape as real ScalingManager."""
    sm = SimpleNamespace()
    sm.amax_history = defaultdict(lambda: deque(maxlen=maxlen))
    sm.scale_cache = {}
    return sm


class TestFP8ScalingContext:

    def test_save_restore_preserves_amax_deque(self):
        sm = _make_mock_sm()
        sm.amax_history["layer1.weight"].append(torch.tensor(3.14))
        sm.amax_history["layer1.weight"].append(torch.tensor(2.71))
        sm.scale_cache["layer1.weight"] = torch.tensor(0.5)

        ctx = _FP8ScalingContext()
        ctx.save(sm)

        # Mutate: simulate checkpointed region
        sm.amax_history["layer1.weight"].append(torch.tensor(999.0))
        sm.amax_history["new_layer"].append(torch.tensor(42.0))
        sm.scale_cache["layer1.weight"] = torch.tensor(0.99)
        sm.scale_cache["new_key"] = torch.tensor(1.0)

        ctx.restore(sm)

        # amax_history should be back to snapshot (2 entries, no new_layer)
        assert len(sm.amax_history["layer1.weight"]) == 2
        assert sm.amax_history["layer1.weight"][0].item() == pytest.approx(3.14)
        assert sm.amax_history["layer1.weight"][1].item() == pytest.approx(2.71)
        assert "new_layer" not in sm.amax_history

        # scale_cache fully replaced
        assert sm.scale_cache["layer1.weight"].item() == pytest.approx(0.5)
        assert "new_key" not in sm.scale_cache

    def test_save_restore_none_manager_is_noop(self):
        ctx = _FP8ScalingContext()
        ctx.save(None)
        ctx.restore(None)  # should not raise

    def test_restore_clones_tensors(self):
        """Restored tensors are independent from snapshot."""
        sm = _make_mock_sm()
        sm.amax_history["w"].append(torch.tensor(1.0))

        ctx = _FP8ScalingContext()
        ctx.save(sm)
        ctx.restore(sm)

        # Mutate after restore — should not affect saved state
        sm.amax_history["w"][0].fill_(999.0)

        # Re-restore should give original value
        ctx.restore(sm)
        assert sm.amax_history["w"][0].item() == pytest.approx(1.0)


class TestLumenCheckpoint:

    def test_basic_checkpoint(self):
        layer = nn.Linear(16, 16)
        x = torch.randn(4, 16, requires_grad=True)

        out = lumen_checkpoint(layer, x)
        assert out.shape == (4, 16)

        out.sum().backward()
        assert x.grad is not None
        assert layer.weight.grad is not None

    def test_checkpoint_matches_no_checkpoint(self):
        torch.manual_seed(42)
        layer = nn.Linear(16, 16)
        x = torch.randn(4, 16, requires_grad=True)

        out_ckpt = lumen_checkpoint(layer, x)
        out_ckpt.sum().backward()
        grad_ckpt = x.grad.clone()

        x.grad = None
        out_direct = layer(x)
        out_direct.sum().backward()
        grad_direct = x.grad.clone()

        torch.testing.assert_close(grad_ckpt, grad_direct)

    def test_checkpoint_with_scaling_manager(self):
        mgr = _make_mock_sm()
        mgr.amax_history["w"].append(torch.tensor(1.0))
        mgr.scale_cache["w"] = torch.tensor(0.5)

        layer = nn.Linear(16, 16)
        x = torch.randn(4, 16, requires_grad=True)

        out = lumen_checkpoint(layer, x, scaling_manager=mgr)
        assert out.shape == (4, 16)


class TestLumenCheckpointCoreAttention:

    def test_basic(self):
        def simple_attn(q, k, v):
            return torch.bmm(q, k.transpose(1, 2)).softmax(-1) @ v

        B, S, D = 2, 8, 16
        q = torch.randn(B, S, D, requires_grad=True)
        k = torch.randn(B, S, D, requires_grad=True)
        v = torch.randn(B, S, D, requires_grad=True)

        out = lumen_checkpoint_core_attention(simple_attn, q, k, v)
        assert out.shape == (B, S, D)
        out.sum().backward()
        assert q.grad is not None
