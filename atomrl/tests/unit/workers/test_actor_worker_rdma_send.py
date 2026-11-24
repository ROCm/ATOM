# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""The dtype send_weights_rdma puts on the wire for an FSDP2 trainer's weights."""

from __future__ import annotations

from types import SimpleNamespace

import atomrl.engine.inference.rdma_weight_transfer as rdma
import torch
from atomrl.workers.actor_worker import AtomActorWorker

_PARAMS = [
    ("model.embed_tokens.weight", torch.zeros(4, 2, dtype=torch.float32)),
    ("model.norm.weight", torch.zeros(2, dtype=torch.bfloat16)),
]


def _sent_dtypes(monkeypatch) -> dict[str, torch.dtype]:
    sent: dict[str, torch.dtype] = {}

    def fake_send(group, weights, *, bucket_size_bytes, version):
        sent.update((name, tensor.dtype) for name, tensor in weights)
        return {"buckets": 1.0}

    monkeypatch.setattr(rdma, "send_weight_stream", fake_send)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args, **kwargs: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    worker = SimpleNamespace(
        _engine=SimpleNamespace(get_per_tensor_param=lambda: (iter(_PARAMS), None)),
        rank=0,
        _rdma_weight_group=object(),
    )
    AtomActorWorker.send_weights_rdma(worker, version=1, bucket_size_mb=1)
    return sent


def test_fp32_weights_go_out_as_bf16(monkeypatch) -> None:
    monkeypatch.delenv("ATOMRL_SYNC_FP32", raising=False)
    assert _sent_dtypes(monkeypatch) == {
        "model.embed_tokens.weight": torch.bfloat16,
        "model.norm.weight": torch.bfloat16,
    }


def test_sync_fp32_keeps_them_fp32(monkeypatch) -> None:
    monkeypatch.setenv("ATOMRL_SYNC_FP32", "1")
    assert _sent_dtypes(monkeypatch)["model.embed_tokens.weight"] == torch.float32
