# SPDX-License-Identifier: MIT
# Tests for `BlockManager.can_allocate_estimate`: the PrefillDelayer's per-seq
# fit probe, memoised across coalescing hold ticks.
#
# The contract has two halves. The estimate answers what `can_allocate` would
# (same hit, same fit) for as long as the KV index answers the walk the same
# way, and re-walks the moment an entry it read changes. And it is only ever
# advice: Phase 2 admission still runs the exact probe, so an estimate that is
# wrong can move a FIRE/HOLD decision but cannot admit anything.

import random

import numpy as np
import pytest
from conftest import MockConfig
from test_dsv4_state_replay import _mp_bm, _mp_seq
from test_state_checkpoint import (
    ckpt_config,
    make_block_manager,
    publish_at_boundary,
    publisher_has_read_its_source,
    stateful_seq,
)

from atom.model_engine.block_manager import BlockManager
from atom.model_engine.prefill_delayer import PrefillDelayer
from atom.model_engine.scheduler import ScheduledBatchOutput, Scheduler
from atom.model_engine.sequence import Sequence
from atom.sampling_params import SamplingParams

BLOCK = 4


def prefix_config(**overrides):
    defaults = {
        "kv_cache_block_size": BLOCK,
        "num_kvcache_blocks": 64,
        "enable_prefix_caching": True,
        "max_num_seqs": 8,
        "max_num_batched_tokens": 256,
        "max_model_len": 256,
    }
    defaults.update(overrides)
    return MockConfig(**defaults)


def seq_of(tokens):
    return Sequence(list(tokens), BLOCK)


def publish(bm: BlockManager, seq: Sequence, free: bool = True) -> None:
    """Run `seq`'s whole prompt and leave its blocks cached (freed by default)."""
    # Unbound, so a `count_walks` spy counts only the estimate's own walks.
    hit = BlockManager.can_allocate(bm, seq)
    assert hit >= 0
    bm.allocate(seq, hit)
    bm.hash_blocks(seq, seq.num_prompt_tokens - seq.num_cached_tokens)
    if free:
        bm.deallocate(seq)


def count_walks(monkeypatch, bm: BlockManager) -> list[dict]:
    """Record every `can_allocate` call; the estimate falls back to it to walk."""
    calls = []
    exact = bm.can_allocate

    def spy(seq, *args, **kwargs):
        calls.append(kwargs)
        return exact(seq, *args, **kwargs)

    monkeypatch.setattr(bm, "can_allocate", spy)
    return calls


def exact(bm: BlockManager, seq: Sequence) -> int:
    return BlockManager.can_allocate(bm, seq, record=False)


class TestReuse:
    def test_hold_ticks_reuse_the_walk(self, monkeypatch):
        bm = BlockManager(prefix_config())
        publish(bm, seq_of(range(40)))
        head = seq_of(list(range(40)) + [900, 901, 902])
        walks = count_walks(monkeypatch, bm)

        first = bm.can_allocate_estimate(head)
        assert first == exact(bm, head) == 10
        for _ in range(5):
            assert bm.can_allocate_estimate(head) == first
        assert len(walks) == 1  # the first probe; every hold tick reused it
        assert walks[0].get("remember") is True

    def test_an_unrelated_eviction_keeps_the_walk(self, monkeypatch):
        """The steady state: decode allocations evict LRU blocks every tick."""
        bm = BlockManager(prefix_config(num_kvcache_blocks=24))
        other = seq_of(range(1000, 1016))
        publish(bm, other)  # cached first, so evicted first
        publish(bm, seq_of(range(20)))
        head = seq_of(list(range(20)) + [900])
        walks = count_walks(monkeypatch, bm)
        assert bm.can_allocate_estimate(head) == 5

        evicted = bm.total_evicted_blocks
        # 24 blocks, 9 cached: the 15 vacant go first, then two of `other`'s.
        filler = seq_of(range(2000, 2000 + 17 * BLOCK))
        bm.allocate(filler, 0)
        assert bm.total_evicted_blocks > evicted
        assert bm.kv.lookup(bm._prompt_hashes(head)[0]) != -1

        assert bm.can_allocate_estimate(head) == exact(bm, head) == 5
        assert len(walks) == 1

    def test_estimate_is_off_without_prefix_caching(self, monkeypatch):
        bm = BlockManager(prefix_config(enable_prefix_caching=False))
        head = seq_of(range(20))
        walks = count_walks(monkeypatch, bm)
        assert bm.can_allocate_estimate(head) == 0
        assert bm.can_allocate_estimate(head) == 0
        assert walks == [{"record": False}, {"record": False}]
        assert not bm._fit_estimates

    def test_env_switch_probes_every_call(self, monkeypatch):
        monkeypatch.setenv("ATOM_PREFILL_DELAYER_PROBE_MEMO", "0")
        bm = BlockManager(prefix_config())
        assert bm.fit_estimate_memo is False
        publish(bm, seq_of(range(40)))
        head = seq_of(list(range(40)) + [900])
        walks = count_walks(monkeypatch, bm)
        for _ in range(3):
            assert bm.can_allocate_estimate(head) == 10
        assert walks == [{"record": False}] * 3

    def test_admission_drops_the_memo(self):
        bm = BlockManager(prefix_config())
        publish(bm, seq_of(range(40)))
        head = seq_of(list(range(40)) + [900])
        hit = bm.can_allocate_estimate(head)
        assert head in bm._fit_estimates
        bm.allocate(head, bm.can_allocate(head))
        assert head not in bm._fit_estimates
        assert hit == 10


class TestInvalidation:
    def test_eviction_inside_the_hit_rewalks(self, monkeypatch):
        bm = BlockManager(prefix_config())
        publish(bm, seq_of(range(40)))
        head = seq_of(list(range(40)) + [900])
        walks = count_walks(monkeypatch, bm)
        assert bm.can_allocate_estimate(head) == 10

        # Evict the 4th block of the hit the way the pool does: hand its id
        # out for fresh content.
        victim = bm.kv.lookup(bm._prompt_hashes(head)[3])
        bm.kv.allocate(victim)
        assert exact(bm, head) == 3

        assert bm.can_allocate_estimate(head) == 3
        assert len(walks) == 2
        assert bm.can_allocate_estimate(head) == 3
        assert len(walks) == 2  # the new walk is reused in turn

    def test_a_publish_at_the_first_miss_extends_the_hit(self, monkeypatch):
        bm = BlockManager(prefix_config())
        publish(bm, seq_of(range(20)))
        head = seq_of(list(range(40)) + [900])
        walks = count_walks(monkeypatch, bm)
        assert bm.can_allocate_estimate(head) == 5

        # Another conversation's turn publishes the rest of the shared prefix.
        publish(bm, seq_of(range(40)))
        assert exact(bm, head) == 10

        assert bm.can_allocate_estimate(head) == 10
        assert len(walks) == 2

    def test_a_publish_elsewhere_keeps_the_walk(self, monkeypatch):
        bm = BlockManager(prefix_config())
        publish(bm, seq_of(range(20)))
        head = seq_of(list(range(20)) + [900, 901, 902, 903, 904])
        walks = count_walks(monkeypatch, bm)
        assert bm.can_allocate_estimate(head) == 5
        publish(bm, seq_of(range(3000, 3020)))
        assert bm.can_allocate_estimate(head) == exact(bm, head) == 5
        assert len(walks) == 1

    def test_evictions_past_the_log_rewalk(self, monkeypatch):
        bm = BlockManager(prefix_config())
        publish(bm, seq_of(range(40)))
        head = seq_of(list(range(40)) + [900])
        walks = count_walks(monkeypatch, bm)
        bm.can_allocate_estimate(head)
        # More evictions than the log holds: nothing to check them against.
        for h in range(bm._recent_evictions.maxlen + 1):
            bm._record_evicted(-1 - h)
        assert bm.can_allocate_estimate(head) == 10
        assert len(walks) == 2

    def test_clear_cache_rewalks(self, monkeypatch):
        bm = BlockManager(prefix_config())
        publish(bm, seq_of(range(40)))
        head = seq_of(list(range(40)) + [900])
        walks = count_walks(monkeypatch, bm)
        assert bm.can_allocate_estimate(head) == 10
        bm.clear_cache()  # drops every entry without an eviction callback
        assert bm.can_allocate_estimate(head) == exact(bm, head) == 0
        assert len(walks) == 2

    def test_a_longer_seq_rewalks(self, monkeypatch):
        """A preempted seq comes back with its generated tokens appended."""
        bm = BlockManager(prefix_config())
        publish(bm, seq_of(range(40)))
        head = seq_of(range(41))
        walks = count_walks(monkeypatch, bm)
        bm.can_allocate_estimate(head)
        head.append_token(7)
        assert bm.can_allocate_estimate(head) == exact(bm, head)
        assert len(walks) == 2

    def test_fit_is_retested_on_every_call(self, monkeypatch):
        bm = BlockManager(prefix_config(num_kvcache_blocks=12))
        head = seq_of(range(20))  # 5 fresh blocks
        walks = count_walks(monkeypatch, bm)
        assert bm.can_allocate_estimate(head) == 0

        hog = seq_of(range(5000, 5032))  # 8 blocks: 4 left
        bm.allocate(hog, 0)
        assert exact(bm, head) == -1
        assert bm.can_allocate_estimate(head) == -1

        bm.deallocate(hog)
        assert bm.can_allocate_estimate(head) == exact(bm, head) == 0
        assert len(walks) == 1  # the walk itself never went stale

    def test_the_state_gate_is_asked_again(self, monkeypatch):
        """A checkpoint can go without its KV block; the gate must notice."""
        bm = make_block_manager(ckpt_config())
        first = stateful_seq(list(range(40)))
        h = publish_at_boundary(bm, first)
        publisher_has_read_its_source(bm)
        bm.deallocate(first)
        head = stateful_seq(list(range(40)) + list(range(500, 520)))
        hit = bm.can_allocate_estimate(head)
        assert hit == exact(bm, head) > 0

        bm.state.unindex(h)  # the checkpoint goes; its KV block stays
        assert exact(bm, head) < hit
        assert bm.can_allocate_estimate(head) == exact(bm, head)

    def test_the_replay_is_asked_again(self, monkeypatch):
        """A bounded replay (DeepSeek-V4) reads the tier's answer, which lands
        between hold ticks: the estimate follows it, flags included."""
        bm = _mp_bm(monkeypatch)
        # 32 tokens in HBM with no checkpoint; LMCache holds 44 with state.
        head = _mp_seq(bm, reach_with_state=44, reach_kv_only=48)
        walks = count_walks(monkeypatch, bm)

        def estimate():
            hit = bm.can_allocate_estimate(head)
            return hit, head.replays_hit, head.replay_kv_load

        assert estimate() == (0, False, False)  # loading 44 beats replaying
        assert estimate() == (0, False, False)
        assert len(walks) == 1
        head.offload_joint.kv_prefix_tokens = 0  # only the KV-only reach left
        assert estimate() == (8, True, True)  # replay the HBM hit, load to 48
        assert exact(bm, head) == 8
        head.offload_joint.kv_prefix_tokens = 44
        assert estimate() == (0, False, False)
        assert exact(bm, head) == 0


class TestAgreesWithTheExactProbe:
    @pytest.mark.parametrize("seed", range(8))
    def test_random_churn(self, seed):
        """Under publishes, evictions and frees the estimate equals the probe.

        The pool is small enough that every allocation past the first few
        evicts cached content, and large enough that the probed prompt always
        fits, so fit -- the one input the estimate may lag on -- never decides.
        """
        rng = random.Random(seed)
        bm = BlockManager(prefix_config(num_kvcache_blocks=48))
        base = [rng.randrange(1, 50) for _ in range(40)]
        heads = [
            seq_of(base[: rng.randrange(8, 40)] + [1000 + k, 2000 + k])
            for k in range(3)
        ]
        live = []
        for _ in range(200):
            op = rng.random()
            if op < 0.4:
                n = rng.randrange(4, 41)
                tail = [rng.randrange(1, 50) for _ in range(rng.randrange(0, 8))]
                s = seq_of(base[:n] + tail)
                if bm.can_allocate(s) >= 0 and len(live) < 2:
                    publish(bm, s, free=False)
                    live.append(s)
            elif op < 0.7 and live:
                bm.deallocate(live.pop(rng.randrange(len(live))))
            else:
                s = seq_of([rng.randrange(100, 200) for _ in range(16)])
                if bm.can_allocate(s) >= 0:
                    publish(bm, s)
            for head in heads:
                assert bm.can_allocate_estimate(head) == exact(bm, head)


class _FireDelayer:
    """Local coalescer stub that releases every tick."""

    is_local = True
    max_queue_ms = None
    ttft_max_ticks = 200

    def protects_decode(self, running_decode_batch: int) -> bool:
        return False

    def should_allow_prefill(self, prefillable, pending_tokens, **kwargs):
        return True


class TestAdmissionStaysExact:
    def test_pending_work_uses_the_estimate(self, monkeypatch):
        sched = Scheduler(prefix_config())
        sched.set_prefill_delayer(_FireDelayer())
        bm = sched.block_manager
        publish(bm, seq_of(range(40)))
        head = seq_of(list(range(40)) + [900])
        sched.add(head)
        walks = count_walks(monkeypatch, bm)
        for _ in range(4):
            assert sched._local_prefill_pending_work() == (True, 1)
        assert len(walks) == 1

    def test_a_stale_estimate_cannot_admit(self, monkeypatch):
        """Phase 2 asks `can_allocate` itself and refuses on its answer."""
        sched = Scheduler(prefix_config(num_kvcache_blocks=12))
        sched.set_prefill_delayer(_FireDelayer())
        bm = sched.block_manager
        head = seq_of(range(20))
        sched.add(head)
        assert bm.can_allocate_estimate(head) == 0
        hog = seq_of(range(5000, 5032))
        bm.allocate(hog, 0)
        # Make the memo wrong the one way the estimate is allowed to lag: it
        # believes the request needs no fresh blocks, so it reports a fit.
        bm._fit_estimates[head].num_new_blocks = 0
        assert bm.can_allocate_estimate(head) == 0
        assert exact(bm, head) == -1
        free_before = bm.kv.num_free

        walks = count_walks(monkeypatch, bm)
        sched.schedule()

        assert list(sched.waiting) == [head]
        assert not head.block_table
        assert bm.kv.num_free == free_before
        # The refusal came from the exact probe, not from the memo.
        assert any(not call.get("remember") for call in walks)


class TestSameScheduleWithAndWithoutTheMemo:
    """End to end: the memo changes probe cost, not one scheduling decision.

    Two schedulers, one per setting, run the same multi-turn workload under a
    real local PrefillDelayer: conversations whose next turn resends the last
    prompt plus its output, arriving while others decode, in a pool small
    enough that every allocation past warm-up evicts cached content. Each tick
    the two must schedule the same requests for the same token counts.
    """

    @staticmethod
    def drive(memo: bool, monkeypatch, seed: int):
        monkeypatch.setenv("ATOM_PREFILL_DELAYER_PROBE_MEMO", "1" if memo else "0")
        rng = random.Random(seed)
        sched = Scheduler(
            prefix_config(
                num_kvcache_blocks=120,
                max_num_seqs=6,
                max_num_batched_tokens=64,
                max_model_len=1024,
            )
        )
        sched.set_prefill_delayer(
            PrefillDelayer(
                dp_size=1,
                cpu_group=None,
                max_num_batched_tokens=64,
                stall_ticks=3,
                prefill_decode_interval=2,
            )
        )
        bm = sched.block_manager
        walks = count_walks(monkeypatch, bm)
        convs = {
            c: [rng.randrange(1, 300) for _ in range(rng.randrange(20, 60))]
            for c in range(8)
        }
        arrivals = sorted((rng.randrange(0, 300), c) for c in convs for _ in range(6))
        next_id = 1000
        owner = {}
        log = []
        batch = seqs = None
        for tick in range(500):
            while arrivals and arrivals[0][0] <= tick:
                _, c = arrivals.pop(0)
                convs[c] = convs[c] + [rng.randrange(1, 300) for _ in range(6)]
                sp = SamplingParams(max_tokens=rng.randrange(2, 9), ignore_eos=True)
                sched.add(
                    Sequence(list(convs[c]), BLOCK, sampling_params=sp, id=next_id)
                )
                owner[next_id] = c
                next_id += 1
            if seqs:
                done = sched.postprocess(
                    list(sched.running),
                    ScheduledBatchOutput(
                        req_ids=list(seqs),
                        token_ids=[(rng.randrange(1, 300),) for _ in seqs],
                        num_rejected=np.zeros(len(seqs), dtype=np.int32),
                        num_bonus=np.zeros(len(seqs), dtype=np.int32),
                        draft_token_ids=None,
                    ),
                    batch=batch,
                )
                for s in done:
                    if s.is_finished:  # the next turn resends prompt + output
                        convs[owner[s.id]] = list(s.token_ids)
            out = sched.schedule()
            batch, seqs = out if out else (None, None)
            log.append(
                (tick, tuple(zip(batch.req_ids, batch.num_scheduled_tokens.tolist())))
                if batch is not None
                else (tick, None)
            )
        return log, len(walks), bm.total_evicted_blocks

    @pytest.mark.parametrize("seed", range(3))
    def test_identical_schedules(self, monkeypatch, seed):
        off_log, off_walks, off_ev = self.drive(False, monkeypatch, seed)
        on_log, on_walks, on_ev = self.drive(True, monkeypatch, seed)
        assert on_log == off_log
        assert on_ev == off_ev > 0  # the pool was evicting throughout
        assert 0 < on_walks < off_walks  # and the memo did save walks
