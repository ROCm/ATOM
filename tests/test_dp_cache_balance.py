# SPDX-License-Identifier: MIT
"""CPU tests for transactional placement; no GPU or running cache service."""

import unittest
from collections import deque
from types import SimpleNamespace as NS
from unittest.mock import patch

from atom.model_engine.dp_cache.admission import DPCacheAdmission
from atom.model_engine.dp_cache.balance import (
    CacheBalanceConfig,
    CacheBalancer,
    Placement,
)
from atom.model_engine.dp_cache.state import DPCacheState


def chain(tokens, *, fork=False):
    return NS(
        hash_block_size=128,
        cache_seed=-1,
        num_prompt_tokens=tokens,
        hashes=[i + (100000 if fork else 0) for i in range(tokens // 128)],
    )


class BalanceTest(unittest.TestCase):
    def setUp(self):
        self.b = CacheBalancer(
            4,
            CacheBalanceConfig.parse(
                {
                    "checkpoint_tokens": 128,
                    "cooldown_seconds": 10,
                }
            ),
        )

    def place(
        self,
        sid,
        session="s",
        prompt=102401,
        now=1,
        hint=None,
        loads=None,
        fork=False,
        enabled=True,
    ):
        return self.b.place(
            sid,
            session,
            prompt,
            chain(prompt, fork=fork),
            [0] * 4,
            loads or [0] * 4,
            now,
            explicit_rank=hint,
            can_migrate=enabled,
        )

    def seed(self, session="s", rank=0, prompt=102401):
        p = self.place((session, rank), session, prompt, hint=rank)
        self.b.admitted((session, rank), rank, 0, 0, 1)
        self.b.finish((session, rank), 2, succeeded=True)
        return p

    def migrate(self, request="move", now=3):
        return self.place(request, now=now, loads=[1000000, 0, 1000000, 1000000])

    def test_owner_moves_only_after_target_admits(self):
        self.seed()
        p = self.migrate()
        self.assertEqual((p.source, p.rank, p.minimum_hit), (0, 1, 102400))
        self.assertEqual(self.b.sessions["s"].owner, 0)
        self.assertFalse(self.b.admitted("move", 1, p.migration, 65536, 4))
        self.assertEqual(self.b.sessions["s"].owner, 0)
        self.assertTrue(self.b.admitted("move", 1, p.migration, 102400, 4))
        self.assertEqual(self.b.sessions["s"].owner, 1)

    def test_missing_state_or_partial_copy_falls_back(self):
        self.seed()
        p = self.migrate()
        generation = p.migration
        self.assertEqual(self.b.retry("move", 1, generation, 4), 0)
        self.assertEqual(self.b.sessions["s"].owner, 0)
        self.assertFalse(self.b.admitted("move", 1, generation, 102400, 5))
        self.assertTrue(self.b.admitted("move", 0, 0, 65536, 5))

    def test_cancellation_rejects_late_completion(self):
        self.seed()
        p = self.migrate()
        generation = p.migration
        self.b.finish("move", 4)
        self.assertIsNone(self.b.retry("move", 1, generation, 5))
        self.assertFalse(self.b.admitted("move", 1, generation, 102400, 5))
        self.assertFalse(self.b.sessions["s"].active)

    def test_partial_target_hit_is_part_of_copy_cost(self):
        self.seed(rank=1, prompt=65537)
        self.seed(rank=0)
        p = self.migrate()
        self.assertEqual(p.estimated_hit, 102400)
        self.assertAlmostEqual(p.transfer_seconds, 0.01 + 36864 / 2000000)
        self.assertEqual(
            self.b.statistics()["pending_restore_seconds_per_rank"][1],
            p.transfer_seconds,
        )
        self.b.admitted("move", 1, p.migration, 102400, 4)
        self.assertEqual(
            self.b.statistics()["pending_restore_seconds_per_rank"], [0] * 4
        )

    def test_outstanding_restore_is_not_free_target_capacity(self):
        self.seed(session="first")
        self.seed(session="second")
        first = self.place("m1", "first", now=3, loads=[1000000, 0, 1000000, 1000000])
        # Target 1 has no GPU work but does have an outstanding restore.
        # Target 2 has 10 ms of GPU work, hidden by its new restore.
        second = self.place("m2", "second", now=3, loads=[1000000, 0, 328, 1000000])
        self.assertEqual((first.rank, second.rank), (1, 2))

    def test_short_recovery_blocks_only_same_promise_until_expiry(self):
        self.seed()
        p = self.migrate()
        self.b.retry(
            "move",
            1,
            p.migration,
            4,
            reason="recoverable_prefix_too_short",
            recoverable_hit=65536,
        )
        self.b.finish("move", 5)
        blocked = self.migrate("second", now=40)
        self.assertEqual(blocked.migration, 0)
        self.assertEqual(self.b.counters["migration_recovery_miss_avoided"], 1)
        self.b.finish("second", 41)
        self.assertGreater(self.migrate("after-expiry", now=125).migration, 0)

    def test_new_checkpoint_can_retry_despite_previous_short_restore(self):
        self.seed()
        p = self.migrate()
        self.b.retry(
            "move",
            1,
            p.migration,
            4,
            reason="recoverable_prefix_too_short",
            recoverable_hit=65536,
        )
        self.b.finish("move", 5)
        self.seed(prompt=102529)
        p = self.place(
            "new-prefix", prompt=102529, now=40, loads=[1000000, 0, 1000000, 1000000]
        )
        self.assertGreater(p.migration, 0)

    def test_timeout_does_not_claim_absent_state_and_cache_clear_forgets_misses(self):
        self.seed()
        p = self.migrate()
        self.b.retry(
            "move", 1, p.migration, 4, reason="preparation_timeout", recoverable_hit=0
        )
        self.assertFalse(self.b.sessions["s"].recovery_misses)
        self.b.finish("move", 5)
        p = self.migrate("short", now=40)
        self.b.retry(
            "short",
            1,
            p.migration,
            41,
            reason="recoverable_prefix_too_short",
            recoverable_hit=0,
        )
        self.assertTrue(self.b.sessions["s"].recovery_misses)
        self.b.invalidate()
        self.assertFalse(self.b.sessions["s"].recovery_misses)

    def test_tokenizer_same_length_different_content_cannot_seed_transfer(self):
        self.seed()
        p = self.place("fork", loads=[1000000, 0, 0, 0], fork=True)
        self.assertEqual(p.rank, 0)
        self.assertEqual((p.estimated_hit, p.migration), (0, 0))

    def test_unknown_or_expired_residency_keeps_owner(self):
        self.seed()
        p = self.migrate(now=1000)
        self.assertEqual((p.rank, p.migration), (0, 0))

    def test_current_turn_cannot_be_migrated(self):
        self.seed()
        self.place("active")
        p = self.migrate()
        self.assertEqual((p.rank, p.migration), (0, 0))

    def test_concurrent_turn_fences_old_owner_commit(self):
        self.seed()
        p = self.migrate()
        newer = self.place("newer", now=3.1)
        self.assertEqual(newer.rank, 0)
        self.assertTrue(self.b.admitted("move", 1, p.migration, 102400, 4))
        self.assertEqual(self.b.sessions["s"].owner, 0)

    def test_cooldown_survives_failed_copy(self):
        self.seed()
        p = self.migrate()
        self.b.retry("move", 1, p.migration, 4)
        self.b.finish("move", 5)
        self.assertEqual(self.migrate("second", now=6).migration, 0)

    def test_retry_reason_is_bounded_and_late_replies_are_not_counted_twice(self):
        self.seed()
        p = self.migrate()
        generation = p.migration
        self.b.retry("move", 1, generation, 4, reason="preparation_timeout")
        self.b.retry("move", 1, generation, 4.1, reason="preparation_timeout")
        self.assertEqual(self.b.counters["migration_fallback"], 1)
        self.assertEqual(self.b.counters["migration_fallback_preparation_timeout"], 1)
        self.b.finish("move", 5)
        other = self.migrate("another", now=30)
        self.b.retry(
            "another", 1, other.migration, 31, reason="unbounded external text"
        )
        self.assertEqual(self.b.counters["migration_fallback_other"], 1)
        self.assertNotIn("migration_fallback_unbounded external text", self.b.counters)

    def test_profitable_move_below_old_overload_threshold(self):
        self.seed()
        # Owner wait is about 180 ms; copying the prefix costs about 61 ms.
        # The first such observation should permit the profitable move.
        p = self.place("move", now=3, loads=[5898, 0, 5898, 5898])
        self.assertEqual((p.source, p.rank), (0, 1))
        self.assertGreater(p.migration, 0)
        self.assertEqual(self.b.sessions["s"].owner, 0)

    def test_small_positive_gain_does_not_cover_absolute_margin(self):
        self.seed()
        # Around 80 ms staying versus 61 ms moving, below the 50 ms margin.
        p = self.place("move", now=3, loads=[2621, 0, 2621, 2621])
        self.assertEqual((p.rank, p.migration), (0, 0))

    def test_absolute_gain_alone_does_not_cover_relative_margin(self):
        self.seed()
        # About 100 ms saved on a 1 s wait does not meet the 15% margin.
        p = self.place("move", now=3, loads=[32768, 29491, 32768, 32768])
        self.assertEqual((p.rank, p.migration), (0, 0))

    def test_copy_cost_can_outweigh_shorter_target_queue(self):
        self.seed()
        # Owner waits 50 ms; even an empty target needs about 61 ms to copy.
        p = self.place("move", now=3, loads=[1638, 0, 1638, 1638])
        self.assertEqual((p.rank, p.migration), (0, 0))

    def test_zero_gain_never_migrates_with_zero_margins(self):
        self.b = CacheBalancer(
            4,
            CacheBalanceConfig.parse(
                {
                    "checkpoint_tokens": 128,
                    "min_gain_seconds": 0,
                    "min_gain_ratio": 0,
                }
            ),
        )
        self.seed(rank=1)
        self.seed(rank=0)
        p = self.place("move", now=3)
        self.assertEqual((p.rank, p.migration), (0, 0))

    def test_explicit_rank_and_no_transport_never_start_migration(self):
        self.seed()
        p = self.place("forced", hint=3, loads=[1000000] * 3 + [0])
        self.assertEqual((p.rank, p.migration), (3, 0))
        self.b.finish("forced", 2)
        p = self.place("no-transport", enabled=False, loads=[1000000] * 4)
        self.assertEqual((p.rank, p.migration), (3, 0))

    def test_global_and_target_migration_budgets(self):
        for i in range(4):
            self.seed(session=str(i))
        placements = [
            self.place(i, str(i), now=3, loads=[1000000, 0, 1000000, 1000000])
            for i in range(4)
        ]
        self.assertEqual(sum(bool(p.migration) for p in placements), 2)

    def test_cache_clear_invalidates_hints_and_inflight_publication(self):
        self.seed()
        self.place("old")
        self.b.admitted("old", 0, 0, 102400, 2)
        self.b.invalidate()
        self.b.finish("old", 3, succeeded=True)
        self.assertFalse(self.b.sessions["s"].hints)
        self.assertEqual(self.migrate().migration, 0)

    def test_bounded_session_directory_does_not_evict_active_owner(self):
        self.b = CacheBalancer(4, CacheBalanceConfig.parse({"max_sessions": 1}))
        self.place("active")
        self.place("other", session="t")
        self.assertEqual(list(self.b.sessions), ["s"])
        self.b.finish("active", 2)
        self.place("third", session="u")
        self.assertEqual(list(self.b.sessions), ["u"])

    def test_invalid_config(self):
        for options in (
            {"max_sessions": True},
            {"max_sessions": 1.1},
            {"prepare_timeout_seconds": 0},
            {"min_gain_ratio": 1},
            {"prefill_tokens_per_second": float("nan")},
            {"typo": 1},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                CacheBalanceConfig.parse(options)


class DecodeBalanceTest(unittest.TestCase):
    def setUp(self):
        self.b = CacheBalancer(2, CacheBalanceConfig(checkpoint_tokens=128))
        # One long request versus three short requests: the two dimensions
        # disagree, so the incoming request's length should matter.
        for i, (rank, tokens) in enumerate([(0, 900), (1, 33), (1, 33), (1, 34)]):
            self.b.requests[i] = Placement(rank, tokens, None, None)

    def test_short_request_prefers_fewer_requests_and_long_prefers_less_context(self):
        short = self.b._decode_placement_scores(10, [1, 3])
        long = self.b._decode_placement_scores(1000, [1, 3])
        self.assertLess(short[0], short[1])
        self.assertGreater(long[0], long[1])

    def test_new_sessions_use_joint_balance_only_inside_existing_time_margin(self):
        short = self.b.place("short", None, 10, chain(10), [1, 3], [0, 0], 1)
        self.assertEqual(short.rank, 0)
        self.b.requests.pop("short")
        long = self.b.place("long", None, 1000, chain(1000), [1, 3], [0, 0], 1)
        self.assertEqual(long.rank, 1)
        self.b.requests.pop("long")
        # An obvious prefill queue still takes precedence over the tie-break.
        queued = self.b.place("queued", None, 1000, chain(1000), [1, 3], [0, 100000], 1)
        self.assertEqual(queued.rank, 0)

    def test_migration_cannot_trade_prefill_relief_for_worse_decode_balance(self):
        from atom.model_engine.dp_cache.balance import PrefixHint, Session

        self.b.sessions["s"] = Session(
            owner=1, hints={1: PrefixHint(128, 128, -1, 0, 1)}
        )
        # The post-placement mean makes even this shorter request too costly
        # for rank 0's context load, despite its smaller request count.
        refused = self.b.place(
            "short", "s", 129, chain(129), [1, 3], [0, 100000], 2, can_migrate=True
        )
        self.assertEqual((refused.rank, refused.migration), (1, 0))
        self.b.finish("short", 3)
        self.b.sessions["s"].owner = 0
        self.b.sessions["s"].hints = {0: PrefixHint(896, 128, -1, 6, 4)}
        self.b.sessions["s"].last_migration = float("-inf")
        # With this long request, rank 0's larger context is the worse choice:
        # migrating from 0 to 1 is permitted, while the reverse is refused.
        allowed = self.b.place(
            "long", "s", 1025, chain(1025), [1, 3], [100000, 0], 5, can_migrate=True
        )
        self.assertEqual(allowed.rank, 1)
        self.b.finish("long", 6)
        self.b.sessions["s"].owner = 1
        self.b.sessions["s"].hints = {1: PrefixHint(896, 128, -1, 6, 7)}
        self.b.sessions["s"].last_migration = float("-inf")
        refused = self.b.place(
            "refused", "s", 1025, chain(1025), [1, 3], [0, 100000], 8, can_migrate=True
        )
        self.assertEqual(refused.rank, 1)
        self.assertEqual(refused.migration, 0)
        self.assertEqual(self.b.counters["migration_decode_balance_rejected"], 2)

    def test_finished_context_is_not_left_in_the_balance_score(self):
        self.b.finish(0, 1)
        scores = self.b._decode_placement_scores(1000, [0, 3])
        self.assertEqual(scores[0], 0)
        self.assertGreater(scores[1], 0)


class AdmissionHarness:
    def __init__(self):
        self.waiting = deque()
        self.running = deque()
        self.aborted = []
        self.events = []
        self.rejected = []
        self.kv_connector = None
        self.cache_admission = DPCacheAdmission(self)
        self.cache_admission.on_admitted = self.events.append
        self.block_manager = NS(replay_start=lambda seq, cached, hit: cached)

    def _reject_aborted_waiting(self, seq):
        self.rejected.append(seq)

    def abort_request(self, request_id):
        self.aborted.append(request_id)


class AdmissionTest(unittest.TestCase):
    def setUp(self):
        self.s = AdmissionHarness()
        self.seq = NS(
            id=7,
            dp_cache=DPCacheState(migration_id=42, minimum_hit=102400),
            num_cached_tokens=65536,
            num_tokens=102401,
        )

    def test_short_target_prefix_never_reaches_prefill(self):
        self.assertFalse(self.s.cache_admission.admit(self.seq))
        self.assertFalse(self.s.events)
        self.assertEqual(self.seq.dp_cache.retry_reason, "recoverable_prefix_too_short")
        self.assertEqual(self.seq.dp_cache.recoverable_hit, 65536)

    def test_completed_work_uses_frozen_offsets_and_preemption_fences_old_batch(self):
        self.seq.num_cached_tokens = 102400
        self.s.cache_admission.admit(self.seq)
        batch = NS(
            req_ids=[7],
            total_seqs_num_prefill=1,
            dp_cache_work_epochs={7: 0},
            num_cached_tokens=[65536],
            num_scheduled_tokens=[1024],
            is_final_chunk=[False],
        )
        self.s.cache_admission.complete_work(batch, {7: self.seq})
        self.assertEqual(self.s.cache_admission.take_work(), [(7, 42, 1, 35841)])
        self.s.cache_admission.preempt(self.seq)
        self.s.cache_admission.complete_work(batch, {7: self.seq})
        self.assertEqual(self.s.cache_admission.take_work(), [(7, 42, 2, 102401)])
        batch.dp_cache_work_epochs = {7: 1}
        batch.is_final_chunk = [True]
        self.s.cache_admission.complete_work(batch, {7: self.seq})
        self.assertEqual(self.s.cache_admission.take_work(), [(7, 42, 3, 0)])
        self.assertFalse(self.s.cache_admission.take_work())

    def test_complete_restore_emits_one_confirmation(self):
        self.seq.num_cached_tokens = 102400
        self.assertTrue(self.s.cache_admission.admit(self.seq))
        self.assertTrue(self.s.cache_admission.admit(self.seq))
        self.assertEqual(self.s.events, [(7, 42, 102400)])

    def test_replay_work_is_not_counted_as_reused_state(self):
        self.seq.num_cached_tokens = 102400
        self.s.block_manager.replay_start = lambda *args, **kw: 98304
        self.assertFalse(self.s.cache_admission.admit(self.seq))

    def test_timeout_uses_local_clock_and_existing_abort_cleanup(self):
        with patch(
            "atom.model_engine.dp_cache.admission.time.monotonic", return_value=100
        ):
            self.s.cache_admission.track(self.seq)
        self.s.waiting.append(self.seq)
        with patch(
            "atom.model_engine.dp_cache.admission.time.monotonic", return_value=106
        ):
            self.s.cache_admission.expire()
        self.assertFalse(self.s.waiting)
        self.assertEqual(self.s.rejected, [self.seq])
        self.assertFalse(self.s.cache_admission._preparing)

    def test_missing_confirmation_channel_fails_closed(self):
        self.seq.num_cached_tokens = 102400
        self.s.cache_admission.on_admitted = None
        self.assertFalse(self.s.cache_admission.admit(self.seq))

    def test_expiry_forgets_abandoned_preparation_without_reclaiming_resources(self):
        self.s.cache_admission.track(self.seq)
        # A terminal path has removed the request from both scheduler queues.
        # Its connector can still retain a DMA lease independently of admission.
        lease_owner = {self.seq.id: self.seq}
        self.s.cache_admission.expire()
        self.assertFalse(self.s.cache_admission._preparing)
        self.assertIs(lease_owner[self.seq.id], self.seq)
        self.assertFalse(self.s.aborted)
        self.assertFalse(self.s.rejected)
        self.assertFalse(self.s.events)

    def test_expiry_keeps_live_waiting_and_running_preparations(self):
        self.prepare_alignment()
        self.s.running.append(self.seq)
        waiting = NS(id=8, dp_cache=DPCacheState(migration_id=43))
        with patch(
            "atom.model_engine.dp_cache.admission.time.monotonic", return_value=100
        ):
            self.s.cache_admission.track(waiting)
        self.s.waiting.append(waiting)
        with patch(
            "atom.model_engine.dp_cache.admission.time.monotonic", return_value=101
        ):
            self.s.cache_admission.expire()
        self.assertEqual(self.s.cache_admission._preparing, {7: self.seq, 8: waiting})
        self.assertFalse(self.s.aborted)
        self.assertFalse(self.s.rejected)

    def prepare_alignment(self):
        self.seq.num_cached_tokens = 65408
        self.s.kv_connector = NS(pending_prefill_handoff=lambda seq: (65536, 102400))
        with patch(
            "atom.model_engine.dp_cache.admission.time.monotonic", return_value=100
        ):
            self.s.cache_admission.track(self.seq)
        self.assertTrue(self.s.cache_admission.admit(self.seq))
        self.assertFalse(self.s.events)

    def test_alignment_does_not_commit_until_native_restore(self):
        self.prepare_alignment()
        self.assertTrue(self.s.cache_admission.alignment_fits(self.seq, 65408, 128))
        self.assertFalse(self.s.cache_admission.alignment_fits(self.seq, 65408, 256))
        self.seq.num_cached_tokens = 102400
        self.assertTrue(self.s.cache_admission.admit(self.seq))
        self.assertEqual(self.s.events, [(7, 42, 102400)])
        self.assertFalse(self.s.cache_admission._preparing)

    def test_running_alignment_timeout_defers_release_to_postprocess(self):
        self.prepare_alignment()
        self.s.running.append(self.seq)
        with patch(
            "atom.model_engine.dp_cache.admission.time.monotonic", return_value=106
        ):
            self.s.cache_admission.expire()
        self.assertEqual(self.s.aborted, [7])
        self.assertFalse(self.s.rejected)
        self.assertFalse(self.s.events)
        self.assertFalse(self.s.cache_admission._preparing)

    def test_lost_handoff_never_continues_unconfirmed_prefill(self):
        self.prepare_alignment()
        self.s.running.append(self.seq)
        self.s.kv_connector.pending_prefill_handoff = lambda seq: None
        with patch(
            "atom.model_engine.dp_cache.admission.time.monotonic", return_value=101
        ):
            self.s.cache_admission.expire()
        self.assertEqual(self.s.aborted, [7])
        self.assertEqual(self.seq.dp_cache.retry_reason, "alignment_load_unavailable")
        self.assertFalse(self.s.rejected)

    def test_stale_boundary_attribute_is_not_a_load_reservation(self):
        self.seq.offload_handoff_boundary_tokens = 65792
        self.assertFalse(self.s.cache_admission.admit(self.seq))
        self.assertFalse(self.s.events)

    def test_expiry_reads_replaced_scheduler_queues(self):
        with patch(
            "atom.model_engine.dp_cache.admission.time.monotonic", return_value=100
        ):
            self.s.cache_admission.track(self.seq)
        old_waiting = self.s.waiting
        self.s.waiting = deque([self.seq])
        with patch(
            "atom.model_engine.dp_cache.admission.time.monotonic", return_value=106
        ):
            self.s.cache_admission.expire()
        self.assertFalse(old_waiting)
        self.assertFalse(self.s.waiting)
        self.assertEqual(self.s.rejected, [self.seq])


if __name__ == "__main__":
    unittest.main()
