# SPDX-License-Identifier: MIT
"""PR #2274 review's correctness claims (CPU only).

Tests named for a claim the batch-order export resolved assert the new
behavior; the rest still reproduce the claim on this checkout.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import atom.model_ops.fused_moe.routed_experts_capturer as cap_mod
from atom.model_engine.scheduler import ScheduledBatch, ScheduledBatchOutput
from atom.model_engine.sequence import SequenceStatus
from atom.model_ops.fused_moe.routed_experts_capturer import (
    UNCAPTURED,
    RoutedExpertsCapturer,
    RoutedExpertsPatch,
)


@pytest.fixture(autouse=True)
def _reset_capturer():
    RoutedExpertsCapturer.reset()
    yield
    RoutedExpertsCapturer.reset()


def test_missing_slot_mapping_returns_no_rows(monkeypatch):
    """Backends without slot_mapping (DeepSeek-V4/V4.1): the step stores
    nothing, rather than rows of zeros that read as expert 0."""
    cap = RoutedExpertsCapturer(num_slots=64, num_layers=2, top_k=2, device="cpu")
    monkeypatch.setattr(cap_mod, "_current_slot_mapping", lambda: None)

    real_ids = torch.tensor([[5, 7], [3, 9], [1, 4]], dtype=torch.int32)
    cap.capture(0, real_ids)
    cap.capture(1, real_ids)
    assert cap.store_step(None, [0], [0], [3]) is None


def test_short_slot_mapping_returns_uncaptured_tail():
    """GDN + MTP under CUDA graph: slot_mapping[:bs] while bs * q tokens run.
    The tokens it does not cover come back as -1."""
    bs, q = 2, 3
    cap = RoutedExpertsCapturer(num_slots=64, num_layers=1, top_k=2, device="cpu")
    short = torch.arange(bs)
    ids = torch.arange(bs * q * 2, dtype=torch.int32).reshape(bs * q, 2)
    cap.capture(0, ids, slot_mapping=short)
    patch = cap.take(cap.store_step(short, [0, 1], [0, 0], [q, q]))
    routes = patch.rows[:, 0, :]
    np.testing.assert_array_equal(routes[:bs], ids[:bs].numpy())
    assert (routes[bs:] == UNCAPTURED).all()


def _output(seq_id, tokens, routed=None):
    return ScheduledBatchOutput(
        req_ids=[seq_id],
        token_ids=[tuple(tokens)] if tokens else [],
        num_rejected=None,
        num_bonus=None,
        draft_token_ids=None,
        routed_experts=routed,
    )


def test_aborted_request_keeps_overlong_tail(scheduler, seq_factory):
    """ABORTED `continue`s before the per-seq truncation, so rows past
    num_tokens - 1 (e.g. rejected drafts) reach the final output."""
    seq = seq_factory([1, 2, 3, 4])
    scheduler.add(seq)
    scheduler.schedule()
    overlong = np.arange(16, dtype=np.int16).reshape(8, 1, 2)
    seq.apply_routed_expert_patch(0, overlong)
    seq.status = SequenceStatus.ABORTED

    finished = scheduler.postprocess(
        list(scheduler.running), _output(seq.id, [], routed=None)
    )
    assert finished == [seq]
    assert seq.routed_expert_rows == 8
    assert seq.routed_expert_rows > seq.num_tokens - 1


def test_eos_branch_does_trim_for_comparison(scheduler, seq_factory):
    """Same overlong rows on a normal EOS finish are trimmed."""
    seq = seq_factory([1, 2, 3, 4])
    scheduler.add(seq)
    scheduler.schedule()
    overlong = np.arange(16, dtype=np.int16).reshape(8, 1, 2)
    patch = RoutedExpertsPatch(
        np.array([seq.id]), np.array([0]), np.array([0, 8]), overlong
    )
    finished = scheduler.postprocess(
        list(scheduler.running), _output(seq.id, [2], patch)
    )
    assert finished[0].leave_reason == "eos"
    assert seq.routed_expert_rows == seq.num_tokens - 1


def test_disabled_flag_builds_no_route_plan(scheduler, seq_factory):
    """With the flag off a batch carries no routed-experts bookkeeping."""
    assert not scheduler._return_routed_experts
    seq = seq_factory([1, 2, 3, 4])
    scheduler.add(seq)
    batch, _ = scheduler.schedule()
    assert batch.routed_prefix_ranges is None
    assert not hasattr(ScheduledBatch, "routed_export_starts")


def test_routed_experts_property_copies_on_every_read(seq_factory):
    seq = seq_factory([1, 2, 3, 4])
    seq.apply_routed_expert_patch(0, np.zeros((3, 1, 1), dtype=np.int16))
    a, b = seq.routed_experts, seq.routed_experts
    assert a is not b
    assert not np.shares_memory(a, b)
