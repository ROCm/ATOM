"""Regression test: ``preempt()`` must not truncate a chunked prefill.

``postprocess`` appends the ``mtp_k - num_rejected`` speculative placeholders only
to sequences that are RUNNING *and not* mid-chunked-prefill, but ``preempt()``
stripped ``mtp_k + num_rejected`` tokens from every sequence.  Preempting a
sequence that was still prefilling therefore deleted real prompt tokens and pushed
``num_tokens`` below ``num_prompt_tokens`` -- which is assigned once, at
construction.  On re-admission the sequence could never satisfy
``num_cached_tokens >= num_prompt_tokens`` again: it stayed ``is_partial_prefill``
forever, pinned its KV blocks, and because Phase 1 breaks (rather than continues)
when a chunk cannot be sized, every later chunked prefill queued behind it.
"""

import time
from types import SimpleNamespace

from conftest import MockConfig

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


class TestPreemptPartialPrefill:
    def test_preempt_keeps_partial_prefill_prompt_tokens(self, seq_factory):
        """A preempted chunked prefill must keep every prompt token.

        It never received speculative placeholders, so stripping them removes
        real prompt tokens and leaves ``num_tokens < num_prompt_tokens``
        forever: the sequence stays partial on every re-admission and pins its
        KV blocks.
        """
        sched = _make_sched(mtp_k=3)
        seq = _admit(sched, seq_factory, [1, 2, 3, 4, 5, 6, 7, 8])
        sched.schedule()

        # Freeze it mid-chunk, as a real chunked prefill would be.
        seq.num_cached_tokens = 4
        seq.is_partial_prefill = True
        sched._partial_prefill_count = 1

        tokens_before = seq.num_tokens
        sched.preempt(seq)

        assert seq.num_tokens == tokens_before, (
            "preempt() stripped speculative placeholders from a partial prefill "
            "that never had any"
        )
        assert seq.num_tokens >= seq.num_prompt_tokens, (
            "prompt was truncated: the sequence can never satisfy "
            "num_cached_tokens >= num_prompt_tokens again"
        )
