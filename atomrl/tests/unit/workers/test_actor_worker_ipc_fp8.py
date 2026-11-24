# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Pre-quantized FP8 over the colocated CUDA-IPC weight-sync path.

The separated RDMA path has had ``fp8_quantize`` since it was written; the
colocated IPC path did not, which made ``weight_sync.fp8_quantize=true``
unreachable on a single node. These pin the sender half.
"""

from __future__ import annotations

import sys
import typing
from types import ModuleType, SimpleNamespace

import pytest
import torch
from atomrl.workers.actor_worker import AtomActorWorker

# The IPC send puts the weights on the device to quantize them.
pytestmark = pytest.mark.gpu


class _Sender:
    """Captures the (name, tensor) stream instead of putting it on a socket."""

    captured: typing.ClassVar[list[tuple[str, torch.Tensor]]] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        # main folds the real sender's counters into the returned timing dict.
        self.stats: dict[str, float] = {}
        _Sender.captured = []

    async def async_send_weights(self, stream):
        for name, tensor in stream:
            _Sender.captured.append((name, tensor))
        return True


@pytest.fixture
def worker(monkeypatch):
    fake_ray = ModuleType("ray")
    fake_ray.get_runtime_context = lambda: SimpleNamespace(get_job_id=lambda: "job-0")
    monkeypatch.setitem(sys.modules, "ray", fake_ray)

    fake_transfer = ModuleType("atomrl.engine.inference.bucketed_weight_transfer")
    fake_transfer.BucketedWeightSender = _Sender
    monkeypatch.setitem(
        sys.modules,
        "atomrl.engine.inference.bucketed_weight_transfer",
        fake_transfer,
    )

    # One 128x128 block per tensor keeps the scale grid at (1, 1).
    params = [
        ("model.layers.0.self_attn.q_proj.weight", torch.randn(128, 128).bfloat16()),
        ("model.layers.0.input_layernorm.weight", torch.randn(128).bfloat16()),
    ]
    w = object.__new__(AtomActorWorker)
    w.rank = 0
    w.config = SimpleNamespace()
    # main's double-buffered IPC sender accumulates into this; the real worker sets
    # it in __init__, which this fake bypasses.
    w._peak_carry_alloc = 0.0
    w._peak_carry_reserved = 0.0
    w._engine = SimpleNamespace(
        get_per_tensor_param=lambda *a, **kw: (iter(params), None)
    )
    return w


def test_ipc_send_quantizes_when_asked(worker) -> None:
    stats = worker.update_weights_ipc_send(fp8_quantize=True)
    assert isinstance(stats, dict) and "total_s" in stats
    sent = dict(_Sender.captured)

    weight = sent["model.layers.0.self_attn.q_proj.weight"]
    assert weight.dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz)
    # The dequant scale travels with it, or the rollout cannot use the weight.
    assert "model.layers.0.self_attn.q_proj.weight_scale_inv" in sent
    assert sent["model.layers.0.self_attn.q_proj.weight_scale_inv"].dtype == (
        torch.float32
    )
    # Norms stay BF16: quantizing them would be a correctness bug, not a saving.
    assert sent["model.layers.0.input_layernorm.weight"].dtype == torch.bfloat16


def test_ipc_send_stays_bf16_by_default(worker) -> None:
    stats = worker.update_weights_ipc_send()
    assert isinstance(stats, dict) and "total_s" in stats
    sent = dict(_Sender.captured)

    assert sent["model.layers.0.self_attn.q_proj.weight"].dtype == torch.bfloat16
    assert not any(name.endswith("_scale_inv") for name in sent)
