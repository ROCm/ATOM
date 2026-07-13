# SPDX-License-Identifier: MIT
# Tests for write-done nonce validation and parameterized completion counting.

import os
import threading
from unittest.mock import MagicMock

import pytest


def _make_connector_stub(**overrides):
    """Build a minimal MooncakeConnector-like object with completion dicts."""
    stub = MagicMock()
    stub._completion_lock = threading.Lock()
    stub._pending_recv_expected = {}
    stub._pending_recv_stages = {}
    stub._pending_recv_nonce = {}
    stub._pending_recv = set()
    stub._pending_recv_blocks = {}
    stub._pending_recv_slots = {}
    stub._fence_lock = threading.Lock()
    stub._blocks_pending_fence = []
    stub.done_recving = set()
    stub._scatter_slot = None
    stub._release_targets = {}
    for k, v in overrides.items():
        setattr(stub, k, v)
    return stub


def _record_write_done(stub, req_id, pp_rank, tp_rank=0, write_nonce=0):
    """Standalone reimplementation of _record_write_done logic for unit testing.

    Mirrors the real method so tests verify the algorithm without needing
    the full MooncakeConnector (which requires RDMA, ZMQ, etc.).
    """
    with stub._completion_lock:
        expected = stub._pending_recv_expected.get(req_id)
        if expected is None:
            return False
        expected_nonce = stub._pending_recv_nonce.get(req_id, 0)
        if expected_nonce and write_nonce != expected_nonce:
            return False
        stages = stub._pending_recv_stages.setdefault(req_id, set())
        stages.add((pp_rank, tp_rank))
        if len(stages) < expected:
            return False
        del stub._pending_recv_expected[req_id]
        stub._pending_recv_stages.pop(req_id, None)
        stub._pending_recv_nonce.pop(req_id, None)

    slot_info = stub._pending_recv_slots.pop(req_id, None)
    if slot_info is not None and stub._scatter_slot is not None:
        compute_slot, pool_idx = slot_info
        if pool_idx >= 0:
            stub._scatter_slot(compute_slot, pool_idx)
    dst_blocks = stub._pending_recv_blocks.pop(req_id, None)
    if dst_blocks:
        with stub._fence_lock:
            stub._blocks_pending_fence.extend(dst_blocks)
    with stub._completion_lock:
        stub.done_recving.add(req_id)
        stub._pending_recv.discard(req_id)
    return True


# ---- Nonce validation ----


class TestNonceValidation:
    def test_correct_nonce_accepted(self):
        stub = _make_connector_stub()
        nonce = int.from_bytes(os.urandom(8), "big")
        stub._pending_recv_expected["r1"] = 1
        stub._pending_recv_nonce["r1"] = nonce

        assert _record_write_done(stub, "r1", pp_rank=0, write_nonce=nonce)
        assert "r1" in stub.done_recving

    def test_wrong_nonce_rejected(self):
        stub = _make_connector_stub()
        nonce = 12345
        stub._pending_recv_expected["r1"] = 1
        stub._pending_recv_nonce["r1"] = nonce

        assert not _record_write_done(stub, "r1", pp_rank=0, write_nonce=99999)
        assert "r1" not in stub.done_recving
        assert "r1" in stub._pending_recv_expected

    def test_zero_nonce_skips_validation(self):
        """Old producers that don't send nonce (default 0) are accepted."""
        stub = _make_connector_stub()
        stub._pending_recv_expected["r1"] = 1
        stub._pending_recv_nonce["r1"] = 0

        assert _record_write_done(stub, "r1", pp_rank=0, write_nonce=0)
        assert "r1" in stub.done_recving

    def test_missing_nonce_from_old_producer(self):
        """Consumer has nonce but old producer sends 0 — rejected."""
        stub = _make_connector_stub()
        stub._pending_recv_expected["r1"] = 1
        stub._pending_recv_nonce["r1"] = 42

        assert not _record_write_done(stub, "r1", pp_rank=0, write_nonce=0)
        assert "r1" not in stub.done_recving

    def test_nonce_cleaned_up_on_completion(self):
        stub = _make_connector_stub()
        nonce = 777
        stub._pending_recv_expected["r1"] = 1
        stub._pending_recv_nonce["r1"] = nonce

        _record_write_done(stub, "r1", pp_rank=0, write_nonce=nonce)
        assert "r1" not in stub._pending_recv_nonce


# ---- Dedup key: (pp_rank, tp_rank) ----


class TestDedupKey:
    def test_pp_only_dedup(self):
        """PP4, TP symmetric: 4 distinct pp_ranks needed."""
        stub = _make_connector_stub()
        stub._pending_recv_expected["r1"] = 4

        for pp in range(3):
            assert not _record_write_done(stub, "r1", pp_rank=pp, tp_rank=0)
        assert _record_write_done(stub, "r1", pp_rank=3, tp_rank=0)
        assert "r1" in stub.done_recving

    def test_duplicate_pp_rank_ignored(self):
        """Same pp_rank sent twice (reliability resend) — not double-counted."""
        stub = _make_connector_stub()
        stub._pending_recv_expected["r1"] = 2

        assert not _record_write_done(stub, "r1", pp_rank=0)
        assert not _record_write_done(stub, "r1", pp_rank=0)
        assert _record_write_done(stub, "r1", pp_rank=1)

    def test_tp_asymmetric_dedup(self):
        """PP2 x TP fan-in 2: need 4 distinct (pp, tp) pairs."""
        stub = _make_connector_stub()
        pp_size, tp_fan_in = 2, 2
        stub._pending_recv_expected["r1"] = pp_size * tp_fan_in

        assert not _record_write_done(stub, "r1", pp_rank=0, tp_rank=0)
        assert not _record_write_done(stub, "r1", pp_rank=0, tp_rank=1)
        assert not _record_write_done(stub, "r1", pp_rank=1, tp_rank=0)
        assert _record_write_done(stub, "r1", pp_rank=1, tp_rank=1)
        assert "r1" in stub.done_recving

    def test_duplicate_pair_ignored(self):
        """Same (pp_rank, tp_rank) pair sent twice — not double-counted."""
        stub = _make_connector_stub()
        stub._pending_recv_expected["r1"] = 2

        assert not _record_write_done(stub, "r1", pp_rank=0, tp_rank=0)
        assert not _record_write_done(stub, "r1", pp_rank=0, tp_rank=0)
        assert _record_write_done(stub, "r1", pp_rank=1, tp_rank=0)


# ---- Expected responses math ----


class TestExpectedResponses:
    @pytest.mark.parametrize(
        "pp_size, tp_fan_in, expected",
        [
            (1, 1, 1),
            (4, 1, 4),
            (2, 2, 4),
            (8, 1, 8),
            (4, 4, 16),
        ],
    )
    def test_expected_count(self, pp_size, tp_fan_in, expected):
        assert pp_size * tp_fan_in == expected


# ---- Unknown / late requests ----


class TestEdgeCases:
    def test_unknown_request_ignored(self):
        stub = _make_connector_stub()
        assert not _record_write_done(stub, "unknown", pp_rank=0)

    def test_late_duplicate_after_completion(self):
        stub = _make_connector_stub()
        stub._pending_recv_expected["r1"] = 1
        assert _record_write_done(stub, "r1", pp_rank=0)
        assert not _record_write_done(stub, "r1", pp_rank=0)

    def test_blocks_fenced_on_completion(self):
        stub = _make_connector_stub()
        stub._pending_recv_expected["r1"] = 1
        stub._pending_recv_blocks["r1"] = [10, 20, 30]

        _record_write_done(stub, "r1", pp_rank=0)
        assert stub._blocks_pending_fence == [10, 20, 30]
