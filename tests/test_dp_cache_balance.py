# SPDX-License-Identifier: MIT
"""CPU tests for transactional placement; no GPU or running cache service."""

import unittest
from collections import deque
from types import SimpleNamespace as NS
from unittest.mock import patch

from atom.model_engine.dp_cache_admission import DPCacheAdmission
from atom.model_engine.dp_cache_balance import CacheBalanceConfig, CacheBalancer


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
        self.assertEqual(p.estimated_hit, 65536)
        self.assertEqual(p.minimum_hit - p.estimated_hit, 36864)

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


class AdmissionHarness(DPCacheAdmission):
    def __init__(self):
        self.waiting = deque()
        self.running = deque()
        self.aborted = []
        self.events = []
        self.rejected = []
        self.dp_cache_admission_callback = self.events.append
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
            dp_cache_balance=True,
            dp_cache_migration=42,
            dp_cache_minimum_hit=102400,
            dp_cache_prepare_timeout=5,
            num_cached_tokens=65536,
        )

    def test_short_target_prefix_never_reaches_prefill(self):
        self.assertFalse(self.s._admit_cache_placement(self.seq))
        self.assertFalse(self.s.events)
        self.assertEqual(self.seq.dp_cache_retry_reason, "recoverable_prefix_too_short")

    def test_complete_restore_emits_one_confirmation(self):
        self.seq.num_cached_tokens = 102400
        self.assertTrue(self.s._admit_cache_placement(self.seq))
        self.assertTrue(self.s._admit_cache_placement(self.seq))
        self.assertEqual(self.s.events, [(7, 42, 102400)])

    def test_replay_work_is_not_counted_as_reused_state(self):
        self.seq.num_cached_tokens = 102400
        self.s.block_manager.replay_start = lambda *args, **kw: 98304
        self.assertFalse(self.s._admit_cache_placement(self.seq))

    def test_timeout_uses_local_clock_and_existing_abort_cleanup(self):
        with patch(
            "atom.model_engine.dp_cache_admission.time.monotonic", return_value=100
        ):
            self.s._track_cache_migration(self.seq)
        self.s.waiting.append(self.seq)
        with patch(
            "atom.model_engine.dp_cache_admission.time.monotonic", return_value=106
        ):
            self.s._expire_cache_migrations()
        self.assertFalse(self.s.waiting)
        self.assertEqual(self.s.rejected, [self.seq])
        self.assertFalse(self.s._dp_cache_preparing)

    def test_missing_confirmation_channel_fails_closed(self):
        self.seq.num_cached_tokens = 102400
        self.s.dp_cache_admission_callback = None
        self.assertFalse(self.s._admit_cache_placement(self.seq))

    def prepare_alignment(self):
        self.seq.num_cached_tokens = 65408
        self.s.kv_connector = NS(pending_prefill_handoff=lambda seq: (65536, 102400))
        with patch(
            "atom.model_engine.dp_cache_admission.time.monotonic", return_value=100
        ):
            self.s._track_cache_migration(self.seq)
        self.assertTrue(self.s._admit_cache_placement(self.seq))
        self.assertFalse(self.s.events)

    def test_alignment_does_not_commit_until_native_restore(self):
        self.prepare_alignment()
        self.assertTrue(self.s._cache_alignment_fits(self.seq, 65408, 128))
        self.assertFalse(self.s._cache_alignment_fits(self.seq, 65408, 256))
        self.seq.num_cached_tokens = 102400
        self.assertTrue(self.s._admit_cache_placement(self.seq))
        self.assertEqual(self.s.events, [(7, 42, 102400)])
        self.assertFalse(self.s._dp_cache_preparing)

    def test_running_alignment_timeout_defers_release_to_postprocess(self):
        self.prepare_alignment()
        self.s.running.append(self.seq)
        with patch(
            "atom.model_engine.dp_cache_admission.time.monotonic", return_value=106
        ):
            self.s._expire_cache_migrations()
        self.assertEqual(self.s.aborted, [7])
        self.assertFalse(self.s.rejected)
        self.assertFalse(self.s.events)
        self.assertFalse(self.s._dp_cache_preparing)

    def test_lost_handoff_never_continues_unconfirmed_prefill(self):
        self.prepare_alignment()
        self.s.running.append(self.seq)
        self.s.kv_connector.pending_prefill_handoff = lambda seq: None
        with patch(
            "atom.model_engine.dp_cache_admission.time.monotonic", return_value=101
        ):
            self.s._expire_cache_migrations()
        self.assertEqual(self.s.aborted, [7])
        self.assertEqual(self.seq.dp_cache_retry_reason, "alignment_load_unavailable")
        self.assertFalse(self.s.rejected)

    def test_stale_boundary_attribute_is_not_a_load_reservation(self):
        self.seq.offload_handoff_boundary_tokens = 65792
        self.assertFalse(self.s._admit_cache_placement(self.seq))
        self.assertFalse(self.s.events)


if __name__ == "__main__":
    unittest.main()
