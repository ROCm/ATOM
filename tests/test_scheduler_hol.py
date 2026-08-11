"""Regression tests for bounded decode-priority scheduling.

``schedule()`` returned as soon as any prefill was scheduled, so decode never ran
on a tick carrying prefill work.  A chunked prefill fills the whole token budget on
each of its ticks, so a long prompt stalled every running sequence for its entire
duration.

Scheduling decode first fixes that, but must be bounded in both directions: a
continuously busy decode batch must not block new admissions forever, and a stream
of short prompts must not freeze a long one.
"""

import time
from types import SimpleNamespace

from conftest import MockConfig

import atom.model_engine.scheduler as scheduler_mod
from atom.model_engine.scheduler import Scheduler


def _spec_config(k=3):
    return SimpleNamespace(num_speculative_tokens=k)


def _make_sched(mtp_k=3, **overrides):
    cfg = {
        "max_num_seqs": 8,
        "num_kvcache_blocks": 64,
        "kv_cache_block_size": 4,
        "max_model_len": 256,
        "max_num_batched_tokens": 256,
        "speculative_config": _spec_config(mtp_k),
    }
    cfg.update(overrides)
    return Scheduler(MockConfig(**cfg))


def _admit(sched, seq_factory, tokens):
    """Queue a sequence the way the engine does, including the arrival stamp.

    ``Sequence.__init__`` leaves ``arrive_time`` at 0.0; ``LLMEngine`` stamps it on
    entry. A test that skips the stamp makes every queued prefill look infinitely
    old to ``_oldest_waiting_prefill_age_ms``.
    """
    seq = seq_factory(tokens)
    seq.arrive_time = time.time()
    sched.add(seq)
    return seq


def _make_decode_ready(sched, seq_factory, tokens):
    """Admit a sequence and drive it to the decode-ready state."""
    seq = _admit(sched, seq_factory, tokens)
    sched.schedule()
    seq.num_cached_tokens = seq.num_prompt_tokens
    seq.is_partial_prefill = False
    seq.append_token(99)
    return seq


class TestDecodePriority:
    def test_decode_runs_before_a_queued_prefill(self, seq_factory):
        """With decode work ready and a fresh arrival queued, the tick is decode."""
        sched = _make_sched()
        _make_decode_ready(sched, seq_factory, [1, 2, 3, 4])

        _admit(sched, seq_factory, [5, 6, 7, 8])

        batch, _ = sched.schedule()

        assert batch.total_seqs_num_decode >= 1
        assert (
            batch.total_seqs_num_prefill == 0
        ), "a queued prefill preempted ready decode work"

    def test_decode_priority_yields_to_an_aged_arrival(self, seq_factory):
        """Decode-priority must yield once an arrival has waited too long.

        Without this bound a steady decode load keeps decode work ready on every
        tick, and Phase 1 / Phase 2 -- the only paths that admit new requests --
        are never reached.
        """
        sched = _make_sched()
        _make_decode_ready(sched, seq_factory, [1, 2, 3, 4])

        arrival = _admit(sched, seq_factory, [5, 6, 7, 8])

        # Fresh arrival: decode still wins.
        batch, _ = sched.schedule()
        assert batch.total_seqs_num_prefill == 0

        # Same state, but the arrival has now waited past the bound.
        arrival.arrive_time = (
            time.time() - (scheduler_mod._HOL_STARVE_MS / 1000.0) - 1.0
        )

        batch, _ = sched.schedule()
        assert batch.total_seqs_num_prefill >= 1, (
            "decode-priority starved an arrival that aged past " "_HOL_STARVE_MS"
        )


class TestLargePrefillDefer:
    def test_large_resume_is_deferred_then_released(self, monkeypatch, seq_factory):
        """A large resume steps aside for a short prompt, but only for a bounded
        time -- otherwise a stream of short prompts freezes the long one."""
        monkeypatch.setattr(scheduler_mod, "_HOL_LARGE_TOKENS", 20)
        monkeypatch.setattr(scheduler_mod, "_HOL_DEFER_MS", 1000.0)
        # Keep decode-priority out of the way so this exercises Phase 1 only:
        # any queued prefill counts as starving, so decode-priority always yields.
        monkeypatch.setattr(scheduler_mod, "_HOL_STARVE_MS", 1e-6)

        sched = _make_sched()
        large = _admit(sched, seq_factory, list(range(100, 140)))  # 40 tokens
        sched.schedule()

        # Mid-chunk with more than _HOL_LARGE_TOKENS still to do.
        large.num_cached_tokens = 10
        large.is_partial_prefill = True
        sched._partial_prefill_count = 1

        short = _admit(sched, seq_factory, list(range(200, 208)))  # 8 tokens

        cached_at_defer = large.num_cached_tokens
        _, scheduled = sched.schedule()
        assert (
            large.id not in scheduled
        ), "large prefill resumed even though a short prompt was queued"
        assert short.id in scheduled, "short prompt was not admitted"
        assert large.hol_deferred_since is not None

        # Still queued behind the deferral bound -> large must resume anyway.
        large.hol_deferred_since = (
            time.time() - (scheduler_mod._HOL_DEFER_MS / 1000.0) - 1.0
        )
        _admit(sched, seq_factory, list(range(300, 308)))  # 8 tokens

        _, scheduled = sched.schedule()
        assert large.id in scheduled, (
            "large prefill stayed frozen past _HOL_DEFER_MS while short prompts "
            "kept arriving"
        )
        assert large.num_cached_tokens >= cached_at_defer
