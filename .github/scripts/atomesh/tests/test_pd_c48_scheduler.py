"""Check the no_tail_split checkpoint-grid fix against the image's real Mamba split rule."""

import sys
import unittest
from pathlib import Path
from types import MethodType, SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    from vllm.v1.core.sched.scheduler import Scheduler
    from pd_c48_scheduler import align_tail_checkpoint_to_hash, demote_superseded_states, subblock_fill
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


def tp8_kv_manager():
    """The C88 TP8 P layout: 3 align-mode Mamba groups of 896 (4 spec blocks, internal checkpoints), MLA 7168."""
    import torch
    from vllm.utils.hashing import sha256
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.kv_cache_utils import init_none_hash
    from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec, MambaSpec
    init_none_hash(sha256)
    mamba = MambaSpec(block_size=896, shapes=((1, 1),), dtypes=(torch.float32,), mamba_cache_mode="align",
                      num_speculative_blocks=4, num_prefill_checkpoint_blocks=1, prefill_checkpoint_alignment=128)
    mla = FullAttentionSpec(block_size=7168, num_kv_heads=1, head_size=1, dtype=torch.float32)
    config = KVCacheConfig(num_blocks=200, kv_cache_tensors=[], prefix_cache_retention_interval=0, kv_cache_groups=[
        *(KVCacheGroupSpec([f"kda{i}"], mamba) for i in range(3)), KVCacheGroupSpec(["mla"], mla)])
    return KVCacheManager(config, max_model_len=200_000, scheduler_block_size=7168, hash_block_size=128)


def prompt_request(rid, tokens):
    from vllm.sampling_params import SamplingParams
    from vllm.utils.hashing import sha256
    from vllm.v1.core.kv_cache_utils import get_request_block_hasher
    from vllm.v1.request import Request
    return Request(rid, tokens, SamplingParams(max_tokens=1), None, block_hasher=get_request_block_hasher(128, sha256))


def run_prompt(kv, rid, tokens, chunks):
    """Prefill tokens in the given chunks as the scheduler would (CoW copies released after each step), then finish."""
    from vllm.v1.request import RequestStatus
    r = prompt_request(rid, tokens)
    blocks, hit = kv.get_computed_blocks(r)[:2]
    for i, n in enumerate(chunks):
        assert kv.allocate_slots(r, n, hit if i == 0 else 0, blocks if i == 0 else None) is not None
        r.num_computed_tokens = (hit if i == 0 else r.num_computed_tokens) + n
        kv.block_pool.free_blocks(kv.take_kv_cache_block_copies()[1])
        kv.new_step_starts()
    assert r.num_computed_tokens == len(tokens)
    r.status = RequestStatus.FINISHED_LENGTH_CAPPED
    kv.free(r)
    return hit


def free_queue(kv):
    q, b = [], kv.block_pool.free_block_queue.fake_free_list_head.next_free_block
    while b.next_free_block is not None:
        q.append(b)
        b = b.next_free_block
    return q


@unittest.skipIf(Scheduler is None, "needs the vLLM image")
class SupersededHeadTest(unittest.TestCase):
    def setUp(self):
        import random
        rng = random.Random(0)
        self.cold = [rng.randrange(1000, 160000) for _ in range(20_000)]
        self.warm = self.cold + [rng.randrange(1000, 160000) for _ in range(2_500)]
        self.branch = self.cold + [rng.randrange(1000, 160000) for _ in range(2_500)]

    def warm_turn(self, demote):
        kv = tp8_kv_manager()
        stats = demote_superseded_states(kv) if demote else None
        self.assertEqual(run_prompt(kv, "cold", self.cold, [11_648, 8_352]), 0)  # exports the checkpoint 19,968
        self.assertEqual(run_prompt(kv, "warm", self.warm, [2_532]), 19_968)  # resumes there, exports 22,400
        q = free_queue(kv)
        hashed = [(b.block_id, b.block_hash_num_tokens) for b in q if b.block_hash is not None]
        return kv, stats, [b.block_id for b in q], hashed

    def test_hit_states_lead_the_free_queue_and_serve_branches_until_reused(self):
        kv, stats, q, hashed = self.warm_turn(demote=True)
        self.assertEqual((stats["superseded"], stats["finished"]), (3, 2))  # one hit state per Mamba group
        s = [i for i, n in hashed[:3]]
        self.assertEqual(hashed[:3], [(i, 19_968) for i in s])  # ahead of every other cached block
        depth = max(map(q.index, s))
        self.assertLess(depth, 30)  # behind only the warm turn's unhashed blocks
        branch = prompt_request("branch", self.branch)
        self.assertEqual(kv.get_computed_blocks(branch)[1], 19_968)
        kv.block_pool.get_new_blocks(depth + 1)
        self.assertEqual(kv.get_computed_blocks(branch)[1], 0)
        self.assertTrue(all(kv.block_pool.blocks[i].block_hash is None for i in s))
        self.assertEqual((stats["reused"], stats["to_head"]), (3, 3))

    def test_stock_order_keeps_hit_states_at_the_mru_end(self):
        kv, _, q, hashed = self.warm_turn(demote=False)
        self.assertEqual([i for i in q if kv.block_pool.blocks[i].block_hash is not None], q[-len(hashed):])
        self.assertEqual([n for _, n in hashed[:7]], [19_968] * 4 + [22_400] * 3)

    def test_turn_inside_its_hit_block_supersedes_nothing(self):
        kv, stats, _, _ = self.warm_turn(demote=True)
        self.assertEqual(run_prompt(kv, "short", self.warm + self.cold[:300], [400]), 22_400)
        self.assertEqual(stats["superseded"], 3)


if __name__ == "__main__":
    unittest.main()
