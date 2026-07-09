# SPDX-License-Identifier: MIT
# CPP A·P2 — schedule-time advancement for pipeline-parallel batching.
#
# Verifies:
# 1. Back-to-back schedule() advances num_cached_tokens (no repeated chunk).
# 2. is_final_chunk frozen correctly per batch.
# 3. Decode seqs blocked while token is in flight.
# 4. pp=1 (advance_on_schedule=False) retains legacy postprocess path.
# 5. Postprocess with advance_on_schedule discards middle-chunk tokens.


from atom.model_engine.scheduler import Scheduler, ScheduledBatchOutput
from atom.model_engine.sequence import Sequence, SequenceType

# conftest.py is auto-discovered by pytest and stubs heavy imports; we access
# MockConfig through the fixture infrastructure or redefine a minimal version
# here so the file can also be collected standalone on hosts without torch/aiter.
try:
    from tests.conftest import MockConfig
except Exception:
    from unittest.mock import MagicMock

    class MockConfig:  # type: ignore[no-redef]
        def __init__(self, **overrides):
            defaults = dict(
                kv_cache_block_size=4,
                num_kvcache_blocks=10,
                enable_prefix_caching=False,
                enable_chunked_prefill=True,
                max_num_seqs=4,
                max_num_batched_tokens=64,
                long_prefill_token_threshold=0,
                max_model_len=64,
                bos_token_id=1,
                eos_token_id=2,
                stop_token_ids=[],
                scheduler_delay_factor=0.0,
                speculative_config=None,
                parallel_config=MagicMock(data_parallel_size=1),
                hf_config=MagicMock(
                    architectures=["LlamaForCausalLM"], sliding_window=128
                ),
            )
            defaults.update(overrides)
            for k, v in defaults.items():
                setattr(self, k, v)


def _pp_config(**overrides):
    defaults = dict(
        pipeline_parallel_size=4,
        max_num_seqs=8,
        max_num_batched_tokens=100,
        max_model_len=131072,
        kv_cache_block_size=16,
        num_kvcache_blocks=4096,
    )
    defaults.update(overrides)
    return MockConfig(**defaults)


def _make_seq(prompt_len, block_size=16):
    return Sequence(token_ids=list(range(prompt_len)), block_size=block_size)


class TestScheduleTimeAdvancement:
    def test_advance_enabled_for_pp_gt1(self):
        sched = Scheduler(_pp_config(pipeline_parallel_size=2))
        assert sched.advance_on_schedule is True

    def test_advance_disabled_for_pp1(self):
        sched = Scheduler(_pp_config(pipeline_parallel_size=1))
        assert sched.advance_on_schedule is False

    def test_back_to_back_schedule_advances_chunks(self):
        """Two schedule() calls without postprocess produce different chunks."""
        cfg = _pp_config(max_num_batched_tokens=100)
        sched = Scheduler(cfg)
        seq = _make_seq(200, block_size=cfg.kv_cache_block_size)
        sched.add(seq)

        batch1, _ = sched.schedule()
        assert batch1.num_cached_tokens[0] == 0
        assert int(batch1.num_scheduled_tokens[0]) == 100
        assert seq.num_cached_tokens == 100

        batch2, _ = sched.schedule()
        assert batch2.num_cached_tokens[0] == 100
        assert int(batch2.num_scheduled_tokens[0]) == 100
        assert seq.num_cached_tokens == 200

    def test_is_final_chunk_correct(self):
        cfg = _pp_config(max_num_batched_tokens=100)
        sched = Scheduler(cfg)
        seq = _make_seq(200, block_size=cfg.kv_cache_block_size)
        sched.add(seq)

        batch1, _ = sched.schedule()
        assert batch1.is_final_chunk == [False]

        batch2, _ = sched.schedule()
        assert batch2.is_final_chunk == [True]

    def test_is_final_chunk_single_shot(self):
        cfg = _pp_config(max_num_batched_tokens=1000)
        sched = Scheduler(cfg)
        seq = _make_seq(50, block_size=cfg.kv_cache_block_size)
        sched.add(seq)

        batch, _ = sched.schedule()
        assert batch.is_final_chunk == [True]

    def test_no_advancement_when_pp1(self):
        cfg = _pp_config(pipeline_parallel_size=1, max_num_batched_tokens=100)
        sched = Scheduler(cfg)
        seq = _make_seq(200, block_size=cfg.kv_cache_block_size)
        sched.add(seq)

        sched.schedule()
        # Legacy path: num_cached_tokens advanced by block allocator only
        # (num_cached_blocks * block_size), not by schedule-time chunk
        assert seq.num_cached_tokens == 0

    def test_decode_inflight_block(self):
        """Decode seq with in-flight token is skipped by the next schedule()."""
        cfg = _pp_config(max_num_batched_tokens=1024)
        sched = Scheduler(cfg)
        seq = _make_seq(32, block_size=cfg.kv_cache_block_size)
        sched.add(seq)

        batch1, _ = sched.schedule()
        assert batch1.is_final_chunk == [True]
        sched.mark_pp_inflight(batch1)

        # Simulate what postprocess does: append token, transition to decode
        seq.token_ids.append(42)
        seq.num_tokens += 1
        seq.type = SequenceType.DECODE

        # Blocked: in-flight token not yet received — scheduler returns an
        # empty batch (no eligible decode seqs). The head loop treats this as
        # "nothing to launch" via `len(batch.req_ids) == 0`.
        result2 = sched.schedule()
        assert result2 is not None
        batch2, seqs2 = result2
        assert len(batch2.req_ids) == 0

        # Release and retry
        sched.release_pp_inflight(batch1)
        result3 = sched.schedule()
        assert result3 is not None
        batch3, _ = result3
        assert seq.id in batch3.req_ids

    def test_multiple_seqs_partial_and_final(self):
        """Batch with one final and one partial chunk: is_final_chunk mixed."""
        cfg = _pp_config(max_num_batched_tokens=200)
        sched = Scheduler(cfg)
        seq_short = _make_seq(50, block_size=cfg.kv_cache_block_size)
        seq_long = _make_seq(300, block_size=cfg.kv_cache_block_size)
        sched.add(seq_short)
        sched.add(seq_long)

        batch, _ = sched.schedule()
        assert len(batch.is_final_chunk) == 2
        # Short seq (50 tokens) fits entirely in the 200-token budget
        assert batch.is_final_chunk[0] is True
        # Long seq (300 tokens) gets at most remaining budget
        assert batch.is_final_chunk[1] is False


class TestPostprocessWithAdvanceOnSchedule:
    def _make_fwd_output(self, req_ids, token_id=42):
        import numpy as _np

        n = len(req_ids)
        out = ScheduledBatchOutput(
            req_ids=req_ids,
            token_ids=[(token_id,)] * n,
            num_rejected=_np.zeros(n, dtype=_np.int32),
            num_bonus=_np.zeros(n, dtype=_np.int32),
            draft_token_ids=None,
            is_deferred_out=False,
        )
        return out

    def test_middle_chunk_discarded(self):
        """Postprocess with advance_on_schedule discards middle-chunk tokens."""
        cfg = _pp_config(max_num_batched_tokens=100)
        sched = Scheduler(cfg)
        seq = _make_seq(200, block_size=cfg.kv_cache_block_size)
        sched.add(seq)

        batch1, seqs1 = sched.schedule()
        assert batch1.is_final_chunk == [False]

        fwd_out = self._make_fwd_output([seq.id])
        initial_num_tokens = seq.num_tokens
        sched.postprocess(list(seqs1.values()), fwd_out, batch=batch1)

        # Middle chunk: token should NOT have been appended
        assert seq.num_tokens == initial_num_tokens
