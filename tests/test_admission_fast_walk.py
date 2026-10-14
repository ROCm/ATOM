# SPDX-License-Identifier: MIT
# Tests for the C-speed admission walk (`ATOM_PREFIX_FAST_WALK`):
# `can_allocate` resolves the prompt chain with one `map` and hands the hit's
# block ids to `allocate`, which claims them instead of walking the prompt
# again. The contract is that nothing a caller can observe moves: same hit,
# same fit, same block tables, same pool state, against the per-block walk
# kept behind the flag as the reference.

import array
import random

import pytest
from conftest import MockConfig
from test_delayer_probe_memo import TestSameScheduleWithAndWithoutTheMemo
from test_dsv4_state_replay import _bm as replay_bm
from test_state_checkpoint import PAGED_COPY_RUNTIME, ckpt_config, stateful_seq

from atom.model_engine import block_manager as bm_module
from atom.model_engine.block_manager import BlockManager
from atom.model_engine.prefix_hash import (
    chained_block_hashes,
    ship_prompt_hash_chain,
)
from atom.model_engine.sequence import Sequence

BLOCK = 4


def prefix_config(**overrides):
    defaults = {
        "kv_cache_block_size": BLOCK,
        "num_kvcache_blocks": 64,
        "enable_prefix_caching": True,
        "max_num_seqs": 8,
        "max_num_batched_tokens": 256,
        "max_model_len": 512,
    }
    defaults.update(overrides)
    return MockConfig(**defaults)


def pair(make):
    """The same manager twice: the fast walk, and the per-block reference."""
    fast, slow = make(), make()
    fast.fast_prefix_walk = True
    slow.fast_prefix_walk = False
    return fast, slow


def pool_state(bm: BlockManager):
    kv = bm.kv
    return (
        [(b.ref_count, b.hash, list(b.token_ids)) for b in kv.blocks],
        dict(kv._hash_to_block_id),
        sorted(kv._used),
        sorted(kv._free),
        list(kv._cached),
        list(kv._protected),
        sorted(kv._reused),
        bm.total_evicted_blocks,
    )


def admit(bm: BlockManager, seq: Sequence, chunk: int | None = None):
    """Phase 2 as the scheduler runs it: probe, record, allocate, forward."""
    hashes: list[int] = []
    hit = bm.can_allocate(seq, record=False, block_hashes=hashes)
    if hit < 0:
        return (hit, hashes)
    bm.record_allocation(seq, hit, hashes)
    assert bm.allocate(seq, hit)
    todo = seq.num_tokens - seq.num_cached_tokens
    bm.hash_blocks(seq, todo if chunk is None else min(chunk, todo))
    return (
        hit,
        hashes,
        list(seq.block_table),
        seq.num_cached_tokens,
        seq.num_compressed_hit_blocks,
        list(seq.state_slots),
    )


def churn(make, make_seq, rng, steps=150, ship=False, preempted=False):
    """Admissions over shared prefixes, frees and filler in a pool that evicts.

    Run against both walks with the same choices; every observable is logged.
    """
    managers = pair(make)
    base = [rng.randrange(1, 40) for _ in range(120)]
    plan = []
    for _ in range(steps):
        op = rng.random()
        if op < 0.55:
            n = rng.randrange(1, 120)
            tail = [rng.randrange(1, 40) for _ in range(rng.randrange(0, 9))]
            extra = rng.randrange(0, 7) if preempted and rng.random() < 0.3 else 0
            chunk = rng.choice([None, None, rng.randrange(1, 64)])
            plan.append(("admit", base[:n] + tail, extra, chunk, rng.random()))
        elif op < 0.8:
            plan.append(("free", rng.random()))
        else:
            filler = [rng.randrange(100, 200) for _ in range(rng.randrange(4, 60))]
            plan.append(("filler", filler))

    logs = []
    for bm in managers:
        live, log = [], []
        for step in plan:
            if step[0] == "admit":
                _, tokens, extra, chunk, keep = step
                seq = make_seq(tokens)
                for t in range(extra):  # a preempted seq returns longer
                    seq.append_token(50 + t)
                if ship:
                    ship_prompt_hash_chain(seq, bm.hash_block_size)
                out = admit(bm, seq, chunk)
                log.append(out)
                if out[0] >= 0:
                    bm.complete_previous_state_batch()
                    if keep < 0.4 and len(live) < 3:
                        live.append(seq)
                    else:
                        bm.deallocate(seq)
            elif step[0] == "free":
                if live:
                    bm.deallocate(live.pop(int(step[1] * len(live))))
            else:
                seq = make_seq(step[1])
                out = admit(bm, seq)
                log.append(("filler", out[0]))
                if out[0] >= 0:
                    bm.deallocate(seq)
            log.append(pool_state(bm))
        logs.append(log)
    return managers, logs


class TestSameAsThePerBlockWalk:
    @pytest.mark.parametrize("ship", [False, True])
    @pytest.mark.parametrize("seed", range(6))
    def test_random_churn(self, seed, ship):
        rng = random.Random(seed)
        (fast, _slow), (fast_log, slow_log) = churn(
            lambda: BlockManager(prefix_config(num_kvcache_blocks=48)),
            lambda t: Sequence(t, BLOCK),
            rng,
            ship=ship,
        )
        assert fast_log == slow_log
        hits = [e[0] for e in fast_log if isinstance(e, tuple) and len(e) == 6]
        assert any(h > 0 for h in hits)  # prefixes did hit
        assert fast.total_evicted_blocks > 0  # and the pool did evict

    @pytest.mark.parametrize("seed", range(4))
    def test_preempted_seqs_walk_past_the_prompt(self, seed):
        rng = random.Random(100 + seed)
        _, (fast_log, slow_log) = churn(
            lambda: BlockManager(prefix_config(num_kvcache_blocks=48)),
            lambda t: Sequence(t, BLOCK),
            rng,
            preempted=True,
        )
        assert fast_log == slow_log

    @pytest.mark.parametrize("seed", range(4))
    def test_random_churn_under_the_page_checkpoint_gate(self, seed):
        """PAGE checkpoints gate the hit (`_gated_hit`); chunked forwards leave
        checkpoints at chunk ends, so gated hits sit below the walk's."""
        rng = random.Random(200 + seed)
        (fast, _), (fast_log, slow_log) = churn(
            lambda: BlockManager(
                ckpt_config(num_kvcache_blocks=160, pool_entries={"state": 8}),
                state_runtime=PAGED_COPY_RUNTIME,
            ),
            stateful_seq,
            rng,
            steps=100,
        )
        assert fast_log == slow_log
        admits = [e for e in fast_log if isinstance(e, tuple) and len(e) == 6]
        # Some hit was cut by the gate: compressed hit above the admitted one.
        assert any(e[4] > e[0] >= 0 for e in admits)
        assert fast.paged_state_checkpoints is not None

    @pytest.mark.parametrize("seed", range(4))
    def test_random_churn_under_a_bounded_replay(self, monkeypatch, seed):
        """DeepSeek-V4 bounded replay (`_replay_hit`): with no checkpoint the
        gate admits nothing, and the replay widens the hit back to the walk's,
        loading LMCache's KV-only reach past it when that is further; a
        state-carrying tier hit (`kv_prefix_tokens`) widens the claim too."""
        monkeypatch.setenv("OFFLOAD_MIN_LOAD_TOKENS", "0")
        rng = random.Random(300 + seed)
        replayed = []

        def make_seq(tokens):
            seq = stateful_seq(tokens)
            # The tier's answer, a function of the prompt so both walks see
            # the same one.
            n = len(tokens) // BLOCK * BLOCK
            if len(tokens) % 3 == 0:
                seq.offload_joint.kv_only_tokens = n
            if len(tokens) % 5 == 0:
                seq.offload_joint.kv_prefix_tokens = n // 2 // BLOCK * BLOCK
            return seq

        real_allocate = BlockManager.allocate

        def allocate(bm, seq, num_cached_blocks=0):
            ok = real_allocate(bm, seq, num_cached_blocks)
            replayed.append((seq.replays_hit, seq.replay_kv_load))
            return ok

        monkeypatch.setattr(BlockManager, "allocate", allocate)
        (fast, _), (fast_log, slow_log) = churn(replay_bm, make_seq, rng, steps=100)
        assert fast_log == slow_log
        half = len(replayed) // 2
        assert replayed[:half] == replayed[half:]
        assert fast.state_replay_tokens > 0
        assert fast.total_evicted_blocks > 0  # the pool evicted throughout
        assert any(r for r, _ in replayed)  # some hit was replayed
        assert any(load for _, load in replayed)  # and some loaded KV first

    @pytest.mark.parametrize("seed", range(3))
    def test_identical_schedules(self, monkeypatch, seed):
        """End to end through the Scheduler and a real PrefillDelayer."""
        logs = []
        for fast in ("0", "1"):
            monkeypatch.setenv("ATOM_PREFIX_FAST_WALK", fast)
            logs.append(
                TestSameScheduleWithAndWithoutTheMemo.drive(True, monkeypatch, seed)
            )
        assert logs[0] == logs[1]


def published(make, tokens):
    bm = make()
    seq = Sequence(list(tokens), BLOCK)
    admit(bm, seq)
    bm.deallocate(seq)
    return bm


class TestHandoff:
    def test_allocate_claims_the_probe_ids_without_a_lookup(self, monkeypatch):
        bm = published(lambda: BlockManager(prefix_config()), range(40))
        seq = Sequence(list(range(40)) + [900, 901], BLOCK)
        hit = bm.can_allocate(seq, record=False)
        assert hit == 10
        monkeypatch.setattr(
            bm.kv, "lookup", lambda h: pytest.fail("allocate walked the prompt")
        )
        assert bm.allocate(seq, hit)
        assert seq.num_cached_tokens == 40
        assert bm._admission_walk is None  # consumed

    def test_a_stale_walk_is_rewalked(self):
        """An index change between probe and allocate: the ids are rechecked."""
        fast, slow = pair(
            lambda: published(lambda: BlockManager(prefix_config()), range(40))
        )
        for bm in (fast, slow):
            seq = Sequence(list(range(40)) + [900, 901], BLOCK)
            assert bm.can_allocate(seq, record=False) == 10
            # Something else runs and republishes block 3's content elsewhere.
            victim = bm.kv.lookup(bm._prompt_hashes(seq)[3])
            other = bm._fresh_block()
            bm.kv.publish(
                other, bm.kv.block(victim).hash, bm.kv.block(victim).token_ids
            )
            assert bm.allocate(seq, 10)
            assert seq.block_table[3] == other
        assert pool_state(fast) == pool_state(slow)

    def test_a_walk_of_another_seq_is_not_used(self, monkeypatch):
        bm = published(lambda: BlockManager(prefix_config()), range(40))
        a = Sequence(list(range(40)) + [900], BLOCK)
        b = Sequence(list(range(20)) + [901], BLOCK)
        assert bm.can_allocate(a, record=False) == 10
        a_ids = [bm.kv.lookup(h) for h in bm._prompt_hashes(a)[:5]]
        assert bm.allocate(b, 5)
        assert list(b.block_table[:5]) == a_ids
        assert b.num_cached_tokens == 20
        assert bm._admission_walk is None

    def test_off_switch(self, monkeypatch):
        monkeypatch.setenv("ATOM_PREFIX_FAST_WALK", "0")
        bm = BlockManager(prefix_config())
        assert bm.fast_prefix_walk is False
        assert bm.can_allocate(Sequence(list(range(9)), BLOCK)) == 0
        assert bm._admission_walk is None


class TestTokenCheck:
    def _collide(self, bm, tokens, block):
        """Make indexed block `block` hold other tokens: a hash collision."""
        chain = chained_block_hashes(
            array.array("i", tokens), BLOCK, 0, len(tokens) // BLOCK, -1
        )
        target = bm.kv.block(bm.kv.lookup(chain[block]))
        target.token_ids = array.array("i", list(target.token_ids[:-1]) + [999])

    def test_a_collision_on_the_last_hit_block_cuts_the_hit_like_before(self):
        tokens = list(range(40))
        fast, slow = pair(
            lambda: published(lambda: BlockManager(prefix_config()), tokens)
        )
        for bm in (fast, slow):
            self._collide(bm, tokens, 9)
        probe = list(range(40)) + [900]
        assert fast.can_allocate(Sequence(probe, BLOCK), record=False) == 9
        assert slow.can_allocate(Sequence(probe, BLOCK), record=False) == 9

    def test_an_interior_collision_is_what_the_fast_walk_gives_up(self):
        """The documented relaxation: only the last hit block is compared."""
        tokens = list(range(40))
        fast, slow = pair(
            lambda: published(lambda: BlockManager(prefix_config()), tokens)
        )
        for bm in (fast, slow):
            self._collide(bm, tokens, 4)
        probe = list(range(40)) + [900]
        assert slow.can_allocate(Sequence(probe, BLOCK), record=False) == 4
        assert fast.can_allocate(Sequence(probe, BLOCK), record=False) == 10


class TestHashRounds:
    def test_hashing_stops_within_a_round_of_the_miss(self, monkeypatch):
        bm = published(
            lambda: BlockManager(prefix_config(num_kvcache_blocks=400)), range(40)
        )
        hashed = []
        real = bm_module.chained_block_hashes

        def spy(tokens, hbs, start, end, prev):
            hashed.append(end - start)
            return real(tokens, hbs, start, end, prev)

        monkeypatch.setattr(bm_module, "chained_block_hashes", spy)
        seq = Sequence(list(range(40)) + list(range(500, 1100)), BLOCK)
        assert bm.can_allocate(seq, record=False) == 10
        assert sum(hashed) == bm_module._WALK_HASH_ROUND_MIN  # one round
        # Only full prompt blocks are kept in the chain.
        assert len(bm._prompt_hashes(seq)) == bm_module._WALK_HASH_ROUND_MIN

    def test_a_shipped_chain_is_never_rehashed(self, monkeypatch):
        bm = published(
            lambda: BlockManager(prefix_config(num_kvcache_blocks=400)), range(40)
        )
        monkeypatch.setattr(
            bm_module,
            "chained_block_hashes",
            lambda *a: pytest.fail("hashed a shipped prompt block"),
        )
        seq = Sequence(list(range(40)) + list(range(500, 1100)), BLOCK)
        ship_prompt_hash_chain(seq, BLOCK)
        assert bm.can_allocate(seq, record=False) == 10
        assert bm.allocate(seq, 10)
