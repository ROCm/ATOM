# SPDX-License-Identifier: MIT
# `ATOM_PREFIX_FAST_WALK`: the one-pass hit walk and `allocate`'s claim of its
# ids must match the per-block walk (same hit, block tables and pool state).
# The churn harness also holds `can_allocate_estimate` to the exact probe.

import array
import random

import numpy as np
import pytest
from conftest import MockConfig
from test_dsv4_state_replay import _bm as replay_bm
from test_state_checkpoint import PAGED_COPY_RUNTIME, ckpt_config, stateful_seq

from atom.model_engine import block_manager as bm_module
from atom.model_engine.block_manager import BlockManager
from atom.model_engine.prefill_delayer import PrefillDelayer
from atom.model_engine.prefix_hash import ship_prompt_hash_chain
from atom.model_engine.scheduler import ScheduledBatchOutput, Scheduler
from atom.model_engine.sequence import Sequence
from atom.sampling_params import SamplingParams

BLOCK = 4


def prefix_config(**overrides):
    return MockConfig(
        **{
            "kv_cache_block_size": BLOCK,
            "num_kvcache_blocks": 64,
            "enable_prefix_caching": True,
            "max_num_seqs": 8,
            "max_num_batched_tokens": 256,
            "max_model_len": 1024,
            **overrides,
        }
    )


def exact(bm, seq):
    return BlockManager.can_allocate(bm, seq, record=False)


def pool_state(bm):
    kv = bm.kv
    blocks = [(b.ref_count, b.hash, list(b.token_ids)) for b in kv.blocks]
    sets = [sorted(kv._used), sorted(kv._free), sorted(kv._reused)]
    order = [list(kv._cached), list(kv._protected), bm.total_evicted_blocks]
    return [blocks, dict(kv._hash_to_block_id), sets, order]


def admit(bm, seq, chunk=None):
    """Phase 2: probe, record, allocate, forward."""
    hashes = []
    hit = bm.can_allocate(seq, record=False, block_hashes=hashes)
    if hit < 0:
        return (hit, hashes)
    bm.record_allocation(seq, hit, hashes)
    assert bm.allocate(seq, hit)
    replay = seq.replays_hit, seq.replay_kv_load
    todo = seq.num_tokens - seq.num_cached_tokens
    bm.hash_blocks(seq, todo if chunk is None else min(chunk, todo))
    bm.complete_previous_state_batch()
    claim = list(seq.block_table), list(seq.state_slots), seq.num_cached_tokens
    return hit, hashes, claim, seq.num_compressed_hit_blocks, replay


def churn(make, make_seq, seed, steps=150, ship=False, preempted=False):
    """The same admissions, frees and filler against the fast and the per-block
    walk; after each step the delayer's estimate for three waiting heads must
    equal the exact probe. Returns the managers and their logs."""
    logs, managers = [], []
    for fast in (True, False):
        rng = random.Random(seed)
        bm = make()
        bm.fast_prefix_walk = fast
        base = [rng.randrange(1, 40) for _ in range(120)]
        heads = [
            make_seq(base[: rng.randrange(8, 40)] + [41, 42 + k]) for k in range(3)
        ]
        live, log = [], []
        for _ in range(steps):
            op, r, n = rng.random(), rng.random(), rng.randrange(1, 120)
            tail = [rng.randrange(1, 40) for _ in range(rng.randrange(0, 9))]
            extra = rng.randrange(0, 7) if preempted and r < 0.3 else 0
            chunk = rng.choice([None, None, rng.randrange(1, 64)])
            if op < 0.55 or op >= 0.8:
                filler = op >= 0.8
                tokens = [100 + t for t in base[:n]] if filler else base[:n] + tail
                seq = make_seq(tokens)
                for t in range(extra):  # a preempted seq returns longer
                    seq.append_token(50 + t)
                if ship:
                    ship_prompt_hash_chain(seq, bm.hash_block_size)
                out = admit(bm, seq, None if filler else chunk)
                log.append(out)
                if out[0] >= 0 and (filler or r >= 0.4 or len(live) >= 3):
                    bm.deallocate(seq)
                elif out[0] >= 0:
                    live.append(seq)
            elif live:
                bm.deallocate(live.pop(int(r * len(live))))
            log.append(pool_state(bm))
            for head in heads:
                assert bm.can_allocate_estimate(head) == exact(bm, head)
        logs.append(log)
        managers.append(bm)
    return managers, logs


def admitted(log):
    return [e for e in log if isinstance(e, tuple) and len(e) == 5]


def replay_seq(tokens):
    """A DeepSeek-V4 seq with LMCache kv_only / kv_prefix reaches."""
    seq = stateful_seq(tokens)
    n = len(tokens) // BLOCK * BLOCK
    if len(tokens) % 3 == 0:
        seq.offload_joint.kv_only_tokens = n
    if len(tokens) % 5 == 0:
        seq.offload_joint.kv_prefix_tokens = n // 2 // BLOCK * BLOCK
    return seq


CHURNS = {  # managers, seqs, and an admission the variant must produce
    "plain": (
        lambda: BlockManager(prefix_config(num_kvcache_blocks=48)),
        lambda tokens: Sequence(tokens, BLOCK),
        lambda e: e[0] > 0,
    ),
    "page_gate": (
        lambda: BlockManager(
            ckpt_config(num_kvcache_blocks=160, pool_entries={"state": 8}),
            state_runtime=PAGED_COPY_RUNTIME,
        ),
        stateful_seq,
        lambda e: e[3] > e[0] >= 0,  # the checkpoint gate cut the hit
    ),
    "bounded_replay": (replay_bm, replay_seq, lambda e: e[4] == (True, True)),
}


@pytest.mark.parametrize("ship,preempted", [(0, 0), (1, 0), (0, 1)])
@pytest.mark.parametrize("variant", CHURNS)
@pytest.mark.parametrize("seed", range(2))
def test_random_churn(monkeypatch, seed, variant, ship, preempted):
    monkeypatch.setenv("OFFLOAD_MIN_LOAD_TOKENS", "0")
    make, make_seq, witness = CHURNS[variant]
    (fast, _), (fast_log, slow_log) = churn(
        make, make_seq, seed, ship=ship, preempted=preempted
    )
    assert fast_log == slow_log
    assert fast.total_evicted_blocks > 0
    assert any(map(witness, admitted(fast_log)))


def count_walks(monkeypatch, bm):
    """Record every `can_allocate` call; the estimate falls back to it to walk."""
    calls, real = [], bm.can_allocate
    monkeypatch.setattr(
        bm,
        "can_allocate",
        lambda seq, *a, **kw: calls.append(kw) or real(seq, *a, **kw),
    )
    return calls


def drive_scheduler(monkeypatch, seed):
    """Multi-turn conversations under a real local PrefillDelayer in an evicting
    pool; returns (per-tick schedule, can_allocate calls, evictions)."""
    rng = random.Random(seed)
    sched = Scheduler(
        prefix_config(num_kvcache_blocks=120, max_num_seqs=6, max_num_batched_tokens=64)
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
    walks = count_walks(monkeypatch, sched.block_manager)
    convs = [
        [rng.randrange(1, 300) for _ in range(rng.randrange(20, 60))] for _ in range(8)
    ]
    arrivals = sorted((rng.randrange(0, 300), c) for c in range(8) for _ in range(6))
    owner, log, out = {}, [], None
    for tick in range(500):
        while arrivals and arrivals[0][0] <= tick:
            _, c = arrivals.pop(0)
            convs[c] = convs[c] + [rng.randrange(1, 300) for _ in range(6)]
            sp = SamplingParams(max_tokens=rng.randrange(2, 9), ignore_eos=True)
            seq = Sequence(
                list(convs[c]), BLOCK, sampling_params=sp, id=1000 + len(owner)
            )
            owner[seq.id] = c
            sched.add(seq)
        if out:
            batch, seqs = out
            output = ScheduledBatchOutput(
                req_ids=list(seqs),
                token_ids=[(rng.randrange(1, 300),) for _ in seqs],
                num_rejected=np.zeros(len(seqs), dtype=np.int32),
                num_bonus=np.zeros(len(seqs), dtype=np.int32),
                draft_token_ids=None,
            )
            for s in sched.postprocess(list(sched.running), output, batch=batch):
                if s.is_finished:  # the next turn resends prompt + output
                    convs[owner[s.id]] = list(s.token_ids)
        out = sched.schedule()
        batch = out[0] if out else None
        log.append(
            batch and tuple(zip(batch.req_ids, batch.num_scheduled_tokens.tolist()))
        )
    return log, len(walks), sched.block_manager.total_evicted_blocks


@pytest.mark.parametrize("seed", range(2))
def test_identical_schedules_with_the_switch_off(monkeypatch, seed):
    logs = []
    for fast in ("0", "1"):
        monkeypatch.setenv("ATOM_PREFIX_FAST_WALK", fast)
        assert BlockManager(prefix_config()).fast_prefix_walk == (fast == "1")
        logs.append(drive_scheduler(monkeypatch, seed))
    assert logs[0] == logs[1]


def published(tokens, blocks=64):
    bm = BlockManager(prefix_config(num_kvcache_blocks=blocks))
    seq = Sequence(list(tokens), BLOCK)
    admit(bm, seq)
    bm.deallocate(seq)
    return bm


def test_allocate_claims_the_probe_ids_without_a_lookup(monkeypatch):
    bm = published(range(40))
    seq = Sequence(list(range(40)) + [900, 901], BLOCK)
    assert bm.can_allocate(seq, record=False) == 10
    monkeypatch.setattr(bm.kv, "lookup", lambda h: pytest.fail("walked again"))
    assert bm.allocate(seq, 10)
    assert seq.num_cached_tokens == 40
    assert bm._admission_walk is None


@pytest.mark.parametrize("fast", [True, False])
def test_a_stale_walk_is_rewalked(fast):
    bm = published(range(40))
    bm.fast_prefix_walk = fast
    seq = Sequence(list(range(40)) + [900, 901], BLOCK)
    assert bm.can_allocate(seq, record=False) == 10
    # Republish block 3's content elsewhere between probe and allocate.
    victim = bm.kv.block(bm.kv.lookup(bm._prompt_hashes(seq)[3]))
    other = bm._fresh_block()
    bm.kv.publish(other, victim.hash, victim.token_ids)
    assert bm.allocate(seq, 10)
    assert seq.block_table[3] == other


@pytest.mark.parametrize("block,fast_hit,slow_hit", [(9, 9, 9), (4, 10, 4)])
def test_hash_collision(block, fast_hit, slow_hit):
    """Only the last hit block's tokens are compared (the documented trade-off)."""
    probe = list(range(40)) + [900]
    for fast, want in ((True, fast_hit), (False, slow_hit)):
        bm = published(range(40))
        bm.fast_prefix_walk = fast
        chain = bm_module.chained_block_hashes(probe, BLOCK, 0, 10, -1)
        target = bm.kv.block(bm.kv.lookup(chain[block]))
        target.token_ids = array.array("i", list(target.token_ids[:-1]) + [999])
        assert bm.can_allocate(Sequence(probe, BLOCK), record=False) == want


def test_hashing_stops_within_a_round_of_the_miss(monkeypatch):
    bm = published(range(40), blocks=400)
    hashed = []
    real = bm_module.chained_block_hashes
    monkeypatch.setattr(
        bm_module,
        "chained_block_hashes",
        lambda t, hbs, s, e, p: hashed.append(e - s) or real(t, hbs, s, e, p),
    )
    seq = Sequence(list(range(40)) + list(range(500, 1100)), BLOCK)
    assert bm.can_allocate(seq, record=False) == 10
    assert sum(hashed) == len(bm._prompt_hashes(seq)) == bm_module._WALK_HASH_ROUND_MIN
    # A shipped chain is never rehashed.
    monkeypatch.setattr(bm_module, "chained_block_hashes", lambda *a: pytest.fail())
    seq = Sequence(list(range(40)) + list(range(500, 1100)), BLOCK)
    ship_prompt_hash_chain(seq, BLOCK)
    assert bm.can_allocate(seq, record=False) == 10
    assert bm.allocate(seq, 10)
