"""Check the no_tail_split checkpoint-grid fix against the image's real Mamba split rule."""

import sys
import unittest
from pathlib import Path
from types import MethodType, SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    from vllm.v1.core.sched.scheduler import Scheduler
    from pd_c48_scheduler import align_tail_checkpoint_to_hash, subblock_fill
except ImportError:
    Scheduler = None


def fake_scheduler():
    s = NS(cache_config=NS(block_size=896), block_size=7168, hash_block_size=128, mamba_partial_cache_hit=False,
           use_eagle_block_drop=False, num_hash_lookahead_tokens=1, mamba_has_prefill_checkpoint_blocks=True,
           mamba_prefill_checkpoint_alignment=64, max_num_scheduled_tokens=8192,
           scheduler_config=NS(long_prefill_token_threshold=0), mamba_shared_prefix_checkpoint=False,
           kv_cache_manager=NS(coordinator=NS(enable_partial_hash_hits=True)))
    s._mamba_block_aligned_split = MethodType(Scheduler._mamba_block_aligned_split, s)
    return s


def chunks(s, start, prompt):
    out = []
    while start < prompt:
        request = NS(num_computed_tokens=start, num_prompt_tokens=prompt, num_tokens=prompt, shared_prefix_boundary=0)
        n = s._mamba_block_aligned_split(request, min(8192, prompt - start))
        assert n > 0
        out.append(n)
        start += n
    return out


@unittest.skipIf(Scheduler is None, "needs the vLLM image")
class TailCheckpointTest(unittest.TestCase):
    def test_hash_grid_keeps_tail_in_final_chunk(self):
        s = fake_scheduler()
        self.assertEqual(chunks(s, 807296, 816384), [8064, 896, 128])
        align_tail_checkpoint_to_hash(s)
        self.assertEqual(chunks(s, 807296, 816384), [8064, 1024])
        self.assertEqual(s.block_size, 7168)

    def test_short_final_chunk_runs_once(self):
        s = fake_scheduler()
        align_tail_checkpoint_to_hash(s)
        for prompt in range(250_001, 250_001 + 896 * 3, 97):
            self.assertEqual(chunks(s, (prompt - 2000) // 896 * 896, prompt), [prompt - (prompt - 2000) // 896 * 896])

    def test_requires_no_tail_split(self):
        s = fake_scheduler()
        s.mamba_partial_cache_hit = True
        with self.assertRaises(RuntimeError):
            align_tail_checkpoint_to_hash(s)


def pp8_scheduler():
    s = fake_scheduler()
    s.cache_config.block_size = s.block_size = 6400
    s.num_hash_lookahead_tokens, s.mamba_prefill_checkpoint_alignment, s.max_num_scheduled_tokens = 0, 1, 12288
    return s


def split(s, start, prompt, budget):
    request = NS(num_computed_tokens=start, num_prompt_tokens=prompt, num_tokens=prompt, shared_prefix_boundary=0)
    return s._mamba_block_aligned_split(request, budget)


@unittest.skipIf(Scheduler is None, "needs the vLLM image")
class SubblockFillTest(unittest.TestCase):
    def test_leftover_budget_advances_block_aligned_request(self):
        for tail_hash in (False, True):
            s = pp8_scheduler()
            if tail_hash:
                align_tail_checkpoint_to_hash(s)
            self.assertEqual(split(s, 0, 100_000, 5888), 0)
            subblock_fill(s)
            self.assertEqual(split(s, 0, 100_000, 12288), 6400)
            self.assertEqual(split(s, 0, 100_000, 5888), 5888)
            self.assertEqual(split(s, 5888, 100_000, 12288), 512)
            self.assertEqual(split(s, 102_400, 105_000, 2600), 2600)
            self.assertEqual(s.scheduler_config.long_prefill_token_threshold, 0)
            self.assertEqual(s.block_size, 6400)

    def test_requires_zero_long_prefill_threshold(self):
        s = pp8_scheduler()
        s.scheduler_config.long_prefill_token_threshold = 4096
        with self.assertRaises(RuntimeError):
            subblock_fill(s)


if __name__ == "__main__":
    unittest.main()
