"""Regression test for the MTP spec-decode IndexError caused by re-queuing a
skipped partial prefill at the head of ``running``.

When the cross-DP ``PrefillDelayer`` vetoes prefill for a tick, a partial
(chunked, prompt-not-done) prefill can be popped by the decode loop and skipped.
The scheduler used to re-insert such seqs at the HEAD of ``running``
(``extendleft``), pinning the partial at ``running[0]``. Once it finishes
prefill it becomes the batch's position-0 *deferred* seq, shifting the fresh
decode seqs to positions 1..N; ``TokenIDProcessor.prepare_input_ids`` then takes
the ``[deferred | new]`` path and indexes the compacted
``scheduled_spec_decode_tokens`` array by those shifted positions, running off
the end:

    IndexError: index N is out of bounds for axis 0 with size N

The fix re-queues skipped partial prefills at the TAIL (``extend``), so they
never occupy position 0 and the new decode seqs stay contiguous from 0 (safe
``[new | deferred]`` slice path). This test drives the real
``Scheduler.schedule()`` and asserts the skipped partial lands at the tail.
"""

from types import SimpleNamespace

from atom.model_engine.scheduler import Scheduler
from conftest import MockConfig


def _spec_config(k=3):
    return SimpleNamespace(num_speculative_tokens=k)


class _VetoDelayer:
    """Stub cross-DP delayer that always refuses prefill this tick, forcing the
    decode loop to run while a partial prefill is still sitting in `running`."""

    def should_allow_prefill(self, prefillable, pending_tokens, **kwargs):
        return False


class TestSkippedPartialPrefillGoesToTail:
    def _make_sched(self, mtp_k=3):
        return Scheduler(
            MockConfig(
                max_num_seqs=8,
                num_kvcache_blocks=64,
                kv_cache_block_size=4,
                max_model_len=256,
                max_num_batched_tokens=256,
                speculative_config=_spec_config(mtp_k),
            )
        )

    def test_skipped_partial_requeued_at_tail_not_head(self, seq_factory):
        sched = self._make_sched(mtp_k=3)

        s_decode = seq_factory([1, 2, 3, 4])  # will finish prefill -> decode-ready
        s_partial = seq_factory([5, 6, 7, 8])  # stays mid-prefill (partial)
        sched.add(s_decode)
        sched.add(s_partial)
        sched.schedule()  # prefill pass

        # s_decode finished its prompt and sampled its first token.
        s_decode.num_cached_tokens = s_decode.num_prompt_tokens
        s_decode.append_token(99)
        s_decode.is_partial_prefill = False

        # s_partial is still mid-chunk (prompt not fully prefilled).
        s_partial.num_cached_tokens = 2
        s_partial.is_partial_prefill = True
        sched._partial_prefill_count = 1

        # Delayer vetoes prefill this tick -> Phase 1/2 skipped -> num_prefill==0
        # -> no prefill-only early return -> decode loop runs and pops the
        # partial, which is skipped and re-queued.
        sched.set_prefill_delayer(_VetoDelayer())

        sched.schedule()  # decode pass (with the veto)

        ids = [s.id for s in sched.running]
        assert s_partial.id in ids, "partial must remain in running"
        # The fix: skipped partial is re-queued at the TAIL, never position 0.
        assert (
            ids[-1] == s_partial.id
        ), f"expected partial {s_partial.id} at running tail, got order {ids}"
        assert (
            ids[0] != s_partial.id
        ), f"partial {s_partial.id} must NOT be pinned at running head (order {ids})"
