# SPDX-License-Identifier: MIT
"""ATOM_MOE_PRECOMPUTED_NVR: `batch_info` carries each count times the MoE
top-k, staged with the counts, and a pass's `scheduled_tokens_gpu` resolves to
its routes entry with no launch. Anything that is not a `batch_info` entry, or
another top-k, resolves to None (aiter multiplies, as before)."""

import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from atom.utils import CpuGpuBuffer
from atom.utils import forward_context as fc

TOPK = 8


@pytest.fixture
def registry(monkeypatch):
    monkeypatch.setattr(fc, "_batch_info_route_gpus", [])
    monkeypatch.setattr(fc, "_batch_info_route_topk", 0)


def _batch_info(topk):
    buf = CpuGpuBuffer(fc.batch_info_len(topk), dtype=torch.int32, device="cpu")
    fc.register_batch_info_routes(buf.gpu, topk)
    return buf


def test_off_keeps_three_counts(monkeypatch, registry):
    monkeypatch.delenv("ATOM_MOE_PRECOMPUTED_NVR", raising=False)
    buf = _batch_info(TOPK)
    assert buf.np.size == 3
    fc.stage_batch_info(buf.np, (5, 25, 20))
    assert buf.np.tolist() == [5, 25, 20]
    assert fc.pass_scheduled_routes_gpu(buf.gpu[1:2], TOPK) is None


def test_no_moe_keeps_three_counts(monkeypatch, registry):
    monkeypatch.setenv("ATOM_MOE_PRECOMPUTED_NVR", "1")
    buf = _batch_info(0)
    assert buf.np.size == 3
    assert fc.pass_scheduled_routes_gpu(buf.gpu[1:2], TOPK) is None


def test_routes_staged_and_resolved(monkeypatch, registry):
    monkeypatch.setenv("ATOM_MOE_PRECOMPUTED_NVR", "1")
    buf = _batch_info(TOPK)
    assert buf.np.size == 6
    for counts in ((5, 25, 20), (1, 5, 4), (0, 0, 0)):
        fc.stage_batch_info(buf.np, counts)
        assert buf.np.tolist() == list(counts) + [c * TOPK for c in counts]
        buf.copy_to_gpu()
        for k in range(3):
            count = buf.gpu[k : k + 1]
            routes = fc.pass_scheduled_routes_gpu(count, TOPK)
            # A view of the staged entry -- what a recorded forward reads in
            # place -- not a computed tensor.
            assert routes._base is buf.gpu and routes.storage_offset() == k + 3
            assert routes.dtype == torch.int32 and routes.shape == (1,)
            assert routes.item() == count.item() * TOPK


def test_unresolvable_counts_fall_back(monkeypatch, registry):
    monkeypatch.setenv("ATOM_MOE_PRECOMPUTED_NVR", "1")
    buf = _batch_info(TOPK)
    fc.stage_batch_info(buf.np, (5, 25, 20))
    buf.copy_to_gpu()
    other = CpuGpuBuffer(6, dtype=torch.int32, device="cpu")  # never registered
    assert fc.pass_scheduled_routes_gpu(None, TOPK) is None
    # Another top-k than the one staged (e.g. a fused shared expert widens it).
    assert fc.pass_scheduled_routes_gpu(buf.gpu[1:2], TOPK + 1) is None
    # A derived count (a TBO micro-batch's clamped share) is no entry.
    assert fc.pass_scheduled_routes_gpu((buf.gpu[1:2] - 3).clamp_(0, 9), TOPK) is None
    assert fc.pass_scheduled_routes_gpu(other.gpu[1:2], TOPK) is None
    # The routes entries themselves, and multi-entry slices, are not counts.
    assert fc.pass_scheduled_routes_gpu(buf.gpu[4:5], TOPK) is None
    assert fc.pass_scheduled_routes_gpu(buf.gpu[0:2], TOPK) is None


@pytest.mark.skipif(
    os.environ.get("RUN_H2D_GPU_TESTS") != "1", reason="set RUN_H2D_GPU_TESTS=1"
)
@pytest.mark.parametrize("transport", ["direct", "packed"])
def test_routes_ride_the_input_ids_publication(monkeypatch, registry, transport):
    from test_h2d_runner_publication import runner_with_buffers

    from atom.model_engine.model_runner import ModelRunner

    monkeypatch.setenv("ATOM_MOE_PRECOMPUTED_NVR", "1")
    monkeypatch.setattr(
        ModelRunner,
        "hf_text_config",
        SimpleNamespace(num_experts_per_tok=TOPK),
        raising=False,
    )
    runner = runner_with_buffers(monkeypatch, transport)
    processor = runner.tokenID_processor
    processor.draft_tokens_per_seq = 4
    runner._gate_staging_reuse()
    processor.prev_batch = None
    batch = SimpleNamespace(
        scheduled_tokens=np.array([41, 42, 43], dtype=np.int32),
        total_tokens_num=3,
        total_tokens_num_prefill=0,
        total_tokens_num_decode=3,
        total_seqs_num_prefill=0,
        total_seqs_num_decode=3,
        total_seqs_num=3,
        req_ids=[1, 2, 3],
        is_dummy_run=False,
        num_scheduled_tokens=np.ones(3, dtype=np.int32),
        num_rejected=np.zeros(3, dtype=np.int32),
        num_bonus=np.zeros(3, dtype=np.int32),
        produces_output=lambda: False,
    )
    runner.attn_metadata_builder.publish_cu_seqlens_q(
        batch, SimpleNamespace(running_bs=3)
    )
    torch.cuda._sleep(2_000_000)
    ids = processor.prepare_input_ids(batch, 1).clone()
    runner._mark_staging_h2d_enqueued()
    runner.h2d_owner.completion.synchronize()
    assert ids.cpu().tolist() == [41, 42, 43]
    gpu = processor.batch_info.gpu
    assert gpu.cpu().tolist() == [3, 3, 12, 3 * TOPK, 3 * TOPK, 12 * TOPK]
    routes = fc.pass_scheduled_routes_gpu(gpu[1:2], TOPK)
    assert routes.data_ptr() == gpu[4:5].data_ptr()
