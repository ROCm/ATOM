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

The original fix re-queued skipped partial prefills at the TAIL (``extend``),
so they never occupy position 0 and the new decode seqs stay contiguous from 0
(safe ``[new | deferred]`` slice path). That kept the compacted array
accidentally aligned rather than making it aligned; any other way of putting a
draft-less sequence ahead of a drafted one brings the same IndexError back — or,
short of the end of the array, silently feeds one sequence another's drafts.

``ScheduledBatch`` now builds the array with one row per sequence in batch
order, zero-filled where a sequence has no drafts, so alignment no longer
depends on queue position. Both properties are tested here: the tail placement
(still the intended scheduling behaviour) and the alignment itself.
"""

from types import SimpleNamespace

from conftest import MockConfig

from atom.model_engine.scheduler import Scheduler


def _spec_config(k=3, dspark=False):
    # `use_dspark` is part of the real SpeculativeConfig surface the scheduler
    # reads (it decides whether a drafter consumes the target's token stream),
    # so the stub carries it rather than letting the scheduler getattr around
    # a missing attribute.
    return SimpleNamespace(num_speculative_tokens=k, use_dspark=lambda: dspark)


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


class TestSpecDraftRowsAlignToBatchOrder:
    """The array must be indexable by batch position, gaps included."""

    def _batch(self, seqs, drafts, mtp_k=3):
        from atom.model_engine.scheduler import ScheduledBatch

        return ScheduledBatch(
            seqs={s.id: s for s in seqs},
            num_scheduled_tokens=[mtp_k + 1] * len(seqs),
            total_tokens_num=(mtp_k + 1) * len(seqs),
            total_tokens_num_decode=(mtp_k + 1) * len(seqs),
            total_seqs_num=len(seqs),
            total_seqs_num_decode=len(seqs),
            num_spec_step=mtp_k,
            scheduled_spec_decode_tokens=drafts,
        )

    def test_draftless_sequence_leaves_a_zero_row_in_place(self, seq_factory):
        import numpy as np

        mtp_k = 3
        seqs = []
        for _ in range(3):
            s = seq_factory([1, 2, 3, 4])
            s.num_cached_tokens = s.num_prompt_tokens
            s.append_token(99)
            seqs.append(s)
        # Middle sequence just came off prefill: no drafts proposed for it yet.
        drafts = {
            seqs[0].id: np.array([11, 12, 13], dtype=np.int32),
            seqs[2].id: np.array([31, 32, 33], dtype=np.int32),
        }

        rows = self._batch(seqs, drafts, mtp_k).scheduled_spec_decode_tokens

        assert rows.shape == (3, mtp_k)
        assert list(rows[0]) == [11, 12, 13]
        assert list(rows[1]) == [0, 0, 0]  # the gap stays a gap
        assert list(rows[2]) == [31, 32, 33]  # NOT shifted up into row 1

    def test_no_drafts_at_all_still_gives_one_row_per_sequence(self, seq_factory):
        mtp_k = 3
        seqs = []
        for _ in range(2):
            s = seq_factory([1, 2, 3, 4])
            s.num_cached_tokens = s.num_prompt_tokens
            s.append_token(99)
            seqs.append(s)

        rows = self._batch(seqs, {}, mtp_k).scheduled_spec_decode_tokens

        assert rows.shape == (2, mtp_k)
        assert not rows.any()
