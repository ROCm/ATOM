# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Fusion: one dispatch, both legs, exactly one completion.

The previous implementation gave the recurrent-state leg a completion channel of
its own and then needed a park to reconcile the two reports. Every load-path
defect it shipped across six review rounds lived in that reconciliation, so the
property worth pinning down is not what either leg does but that ONE dispatch
produces exactly ONE report -- on every path, including a raise.

The second half of the file pins the asymmetry that fusion creates: a fused
verdict of "failed" may mean the KV leg, so it may not retract the state hash.
Only the state `get` actually missing may do that.

Pure Python: no torch, no aiter, no GPU.
"""

from atom.kv_transfer.offload.hybrid.kimi_k3.connector import (
    KimiK3OffloadConnector,
)
from atom.kv_transfer.offload.metadata import LMCacheReqMeta, LoadSpec, StateLoadSpec
from atom.model_engine.state_offload import StateOffloadIndex


class FakeTier:
    """Records what the state leg was asked for and what it answered."""

    def __init__(self, *, answer=True):
        self.answer = answer
        self.calls: list[tuple[int, int]] = []
        self.missed: set[int] = set()

    def load_state(self, prefix_hash, slot):
        self.calls.append((int(prefix_hash), int(slot)))
        if not self.answer:
            self.missed.add(int(prefix_hash))
        return self.answer

    def take_missed_hashes(self):
        missed, self.missed = self.missed, set()
        return missed

    def take_store_reports(self):
        return set(), set(), set()


def make_worker(*, tier, kv_ok=True):
    """A connector with only the pieces `_do_load_req` touches.

    Built with `__new__` on purpose: `__init__` reaches for an LMCache engine
    and a TP group, and none of the fusion logic under test needs either.
    """
    worker = KimiK3OffloadConnector.__new__(KimiK3OffloadConnector)
    worker._state_tier = tier
    worker._state_no_tier = []
    worker.kv_calls: list = []
    worker.finished: list[tuple] = []
    worker._load_kv_bytes = lambda req: (worker.kv_calls.append(req.req_id) or kv_ok)
    worker._finish_load = lambda req, ok: worker.finished.append((req.req_id, ok))
    return worker


def make_req(req_id="r1", *, state_hash=None, slot=3, hbm=0, lmc=0):
    return LMCacheReqMeta(
        req_id=req_id,
        token_ids=[],
        block_ids=[],
        load_spec=LoadSpec(hbm_cached_tokens=hbm, lmcache_cached_tokens=lmc),
        state_load_spec=(
            None
            if state_hash is None
            else StateLoadSpec(
                boundary_tokens=1024,
                boundary_hash=state_hash,
                destination_slot=slot,
                chunk_tokens=1024,
            )
        ),
    )


class TestOneDispatchOneCompletion:
    def test_a_joint_load_reports_once_for_both_legs(self):
        tier = FakeTier()
        worker = make_worker(tier=tier)
        worker._do_load_req(make_req(state_hash=77, slot=5))

        assert worker.kv_calls == ["r1"]
        assert tier.calls == [(77, 5)]
        assert worker.finished == [("r1", True)], "exactly one completion"

    def test_a_state_only_load_still_reports_once(self):
        """KV resident, state not: the KV leg is a no-op but the request must
        still produce the ordinary completion, or it parks forever."""
        tier = FakeTier()
        worker = make_worker(tier=tier)
        worker._do_load_req(make_req(state_hash=88, slot=1, hbm=256, lmc=256))

        assert tier.calls == [(88, 1)]
        assert worker.finished == [("r1", True)]

    def test_a_request_with_no_state_leg_is_unchanged(self):
        tier = FakeTier()
        worker = make_worker(tier=tier)
        worker._do_load_req(make_req())

        assert tier.calls == []
        assert worker.finished == [("r1", True)]

    def test_the_state_leg_does_not_run_when_the_kv_leg_failed(self):
        """State at the boundary is the compressed history of exactly the prefix
        the KV leg was asked to complete. Restoring it over a prefix whose KV
        never arrived resumes the forward on a history it does not hold."""
        tier = FakeTier()
        worker = make_worker(tier=tier, kv_ok=False)
        worker._do_load_req(make_req(state_hash=99))

        assert tier.calls == []
        assert worker.finished == [("r1", False)]

    def test_a_missed_state_leg_fails_the_whole_load(self):
        tier = FakeTier(answer=False)
        worker = make_worker(tier=tier)
        worker._do_load_req(make_req(state_hash=99))

        assert worker.finished == [("r1", False)]

    def test_no_tier_fails_the_load_rather_than_passing_it_through(self):
        """A KV leg passed through as success would have the engine count a
        state restore that never happened, and the forward would resume on
        whatever the slot already held."""
        worker = make_worker(tier=None)
        worker._do_load_req(make_req(state_hash=99))

        assert worker.finished == [("r1", False)]

    def test_a_raising_leg_still_produces_its_one_completion(self):
        """A dispatch with no report is a request parked forever."""

        class Boom(FakeTier):
            def load_state(self, prefix_hash, slot):
                raise RuntimeError("transfer exploded")

        worker = make_worker(tier=Boom())
        worker._do_load_req(make_req(state_hash=99))

        assert worker.finished == [("r1", False)]


class TestOnlyAMissRetractsTheHash:
    def test_a_fused_failure_alone_does_not_un_advertise(self):
        """`ok=False` may mean the KV leg. Retracting on it would permanently
        deny state bytes that are still present in LMCache."""
        index = StateOffloadIndex(
            can_store=True,
            can_load=True,
            chunk_tokens=1024,
            release_slot=lambda s: None,
        )
        index.note_stored(77)
        index.dispatch("r1", 77, slot=0)
        index.settle("r1", ok=False)
        assert index.could_serve(77)

    def test_only_the_state_get_missing_marks_the_hash_for_retraction(self):
        """The miss is recorded by the leg that actually missed, and drained
        once -- it is the only thing licensed to un-advertise a hash."""
        tier = FakeTier(answer=False)
        worker = make_worker(tier=tier)
        worker._do_load_req(make_req(state_hash=77))

        assert tier.take_missed_hashes() == {77}
        assert tier.take_missed_hashes() == set(), "drained, not re-reported"

    def test_a_failed_kv_leg_marks_no_hash(self):
        tier = FakeTier()
        worker = make_worker(tier=tier, kv_ok=False)
        worker._do_load_req(make_req(state_hash=77))

        assert tier.take_missed_hashes() == set()
