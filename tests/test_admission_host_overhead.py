# SPDX-License-Identifier: MIT
# Admission host-overhead switches must not change answers: the fast prefix walk
# (`ATOM_PREFIX_FAST_WALK`), the per-tick walk memo (`ATOM_PREFIX_WALK_MEMO`) and
# the frontend-hashed prompt chain (`ATOM_FRONTEND_PREFIX_HASH`).

import array
import importlib
import pickle
import queue
import random
import sys
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
from aiter_stub import stubbed_aiter
from conftest import MockConfig
from test_dsv4_state_replay import _bm, _mp_bm, _mp_seq
from test_state_checkpoint import (
    PAGED_COPY_RUNTIME,
    ckpt_config,
    make_block_manager,
    publish_at_boundary,
    publisher_has_read_its_source,
    stateful_seq,
)

from atom.model_engine.block_manager import BlockManager
from atom.model_engine.prefix_hash import (
    attach_shipped_chains,
    chained_block_hashes,
    ship_prompt_hash_chain,
    take_shipped_chains,
)
from atom.model_engine.sequence import Sequence
from atom.sampling_params import SamplingParams

BLOCK = 4


def prefix_config(**kw):
    kw = {"num_kvcache_blocks": 64, "enable_prefix_caching": True, **kw}
    return MockConfig(kv_cache_block_size=BLOCK, **kw)


def exact(bm, seq):
    return BlockManager.can_allocate(bm, seq, record=False)


SIDE_EFFECTS = (
    "checkpoint_demand_pos",
    "checkpoint_end_pos",
    "num_wanted_hit_blocks",
    "num_compressed_hit_blocks",
    "replays_hit",
    "replay_kv_load",
)


def probed(probe, bm, seq):
    """`probe`'s answer and what it wrote on `seq`, over poisoned fields."""
    for name in SIDE_EFFECTS:
        setattr(seq, name, -7)
    return probe(bm, seq), [getattr(seq, name) for name in SIDE_EFFECTS]


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


def published(tokens, bm=None):
    bm = BlockManager(prefix_config()) if bm is None else bm
    admit(bm, seq := Sequence(list(tokens), BLOCK))
    bm.deallocate(seq)
    return bm


def churn(make, make_seq, seed, steps=150, ship=False, preempted=False):
    """The same admissions, frees and filler against the fast and the per-block
    walk; after each step the memoised probe of three waiting heads must equal
    the exact probe. Returns the managers and their logs."""
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
                memoized = probed(BlockManager.can_allocate_memoized, bm, head)
                assert memoized == probed(exact, bm, head)
        logs.append(log)
        managers.append(bm)
    return managers, logs


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
    "bounded_replay": (_bm, replay_seq, lambda e: e[4] == (True, True)),
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
    admitted = [e for e in fast_log if isinstance(e, tuple) and len(e) == 5]
    assert any(map(witness, admitted))


@pytest.mark.parametrize("fast,stale", [(1, 0), (1, 1), (0, 1)])
def test_allocate_claims_the_probe_ids_unless_stale(monkeypatch, fast, stale):
    bm = published(range(40))
    bm.fast_prefix_walk = fast
    seq = Sequence(list(range(40)) + [900, 901], BLOCK)
    assert bm.can_allocate(seq, record=False) == 10
    want = list(map(bm.kv.lookup, bm._prompt_hashes(seq)))
    if stale:  # republish block 3's content elsewhere between probe and allocate
        victim = bm.kv.block(want[3])
        want[3] = bm._fresh_block()
        bm.kv.publish(want[3], victim.hash, victim.token_ids)
    else:
        monkeypatch.setattr(bm.kv, "lookup", lambda h: pytest.fail("walked again"))
    assert bm.allocate(seq, 10)
    assert list(seq.block_table[:10]) == want and seq not in bm._admission_walks


def test_the_walk_is_handed_to_its_own_allocate():
    """Probes of other seqs, and memoised ones, do not take A's walk."""
    bm = published(range(40))
    a, b = (Sequence(list(range(40)) + [900 + k], BLOCK) for k in range(2))
    assert exact(bm, a) == bm.can_allocate_memoized(a) == 10
    assert bm.can_allocate_memoized(b) == 10 and b not in bm._admission_walks
    assert exact(bm, b) == 10
    assert bm.can_allocate_memoized(a) == 10
    bm.kv.lookup = lambda h: pytest.fail("walked again")
    assert bm.allocate(a, 10) and a.num_cached_tokens == 40


def test_allocate_checks_only_the_ids_it_claims(monkeypatch):
    """An eviction past the claim keeps the walk; one inside it does not."""
    for victim, kept in ((8, True), (3, False)):
        bm = published(range(40))
        seq = Sequence(list(range(40)) + [900, 901], BLOCK)
        assert bm.can_allocate(seq, record=False) == 10
        want = list(map(bm.kv.lookup, bm._prompt_hashes(seq)))
        bm.kv.allocate(want[victim])  # unindexes it
        if kept:
            monkeypatch.setattr(bm.kv, "lookup", lambda h: pytest.fail("walked"))
        assert bm.allocate(seq, 5)
        claimed = 5 if kept else victim
        assert list(seq.block_table[:claimed]) == want[:claimed]
        assert seq.num_cached_tokens == claimed * BLOCK


class CountingIndex(dict):
    lookups = 0

    def get(self, *a):
        self.lookups += 1
        return super().get(*a)


@pytest.mark.parametrize("hit", [0, 1, 8, 40])
def test_a_walk_looks_up_near_its_hit_not_the_whole_chain(hit):
    """The first block alone, then rounds of 16, 64, ...: a cold 400-block
    prompt costs one lookup."""
    tokens = list(range(BLOCK * 400 + 1))
    bm = BlockManager(prefix_config(num_kvcache_blocks=1024))
    if hit:
        published(tokens[: hit * BLOCK], bm)
    bm.kv._hash_to_block_id = index = CountingIndex(bm.kv._hash_to_block_id)
    assert bm.can_allocate(shipped_seq(tokens), record=False) == hit
    assert index.lookups <= max(1, 4 * hit + 16)


def test_publish_revokes_the_hash_a_block_held():
    """Indexed content never changes: a new hash on an indexed block drops
    its old one, as an eviction does, for the memo and `allocate` too."""
    bm = published(range(40))
    seq = Sequence(list(range(40)) + [900], BLOCK)
    assert bm.can_allocate_memoized(seq) == exact(bm, seq) == 10
    old = bm._prompt_hashes(seq)[3]
    block_id = bm.kv.lookup(old)
    bm.kv.publish(block_id, 12345, array.array("i", [7] * BLOCK))
    assert bm.kv.lookup(old) == -1 and bm.total_evicted_blocks == 1
    assert bm.can_allocate_memoized(seq) == 3
    assert bm.allocate(seq, 10) and seq.num_cached_tokens == 3 * BLOCK


def test_only_a_moved_or_dropped_hash_bumps_the_index_version():
    """`allocate` skips revalidation while it holds: not bumped by a
    republish in place or a new hash."""
    bm = published(range(40))
    kv, version = bm.kv, bm.kv.index_version
    h = chained_block_hashes(list(range(8)), BLOCK, 0, 2, -1)[1]
    block = kv.block(kv.lookup(h))
    kv.publish(block.block_id, block.hash, block.token_ids)
    published(range(100, 140), bm)
    assert kv.index_version == version
    kv.publish(bm._fresh_block(), block.hash, block.token_ids)  # remapped
    assert kv.index_version == version + 1


def test_chain_to_continues_only_a_prefix_of_the_prompt_chain():
    bm = BlockManager(prefix_config())
    seq = Sequence(PROMPT, BLOCK)
    want = reference_chain(PROMPT, BLOCK, -1, 10)
    assert bm._chain_to(seq, [], 10) == want == bm._prompt_hashes(seq)
    assert bm._chain_to(seq, want[:4], 10) == want
    other = reference_chain(PROMPT, BLOCK, 7, 10)  # another seed's chain
    assert bm._chain_to(seq, other[:4], 10) == other
    assert bm._prompt_hashes(seq) == want


@pytest.mark.parametrize("block,fast_hit,slow_hit", [(9, 9, 9), (4, 10, 4)])
def test_hash_collision(block, fast_hit, slow_hit):
    """Only the last hit block's tokens are compared (the documented trade-off)."""
    probe = list(range(40)) + [900]
    for fast, want in ((True, fast_hit), (False, slow_hit)):
        bm = published(range(40))
        bm.fast_prefix_walk = fast
        chain = chained_block_hashes(probe, BLOCK, 0, 10, -1)
        target = bm.kv.block(bm.kv.lookup(chain[block]))
        target.token_ids = array.array("i", list(target.token_ids[:-1]) + [999])
        assert bm.can_allocate(Sequence(probe, BLOCK), record=False) == want


@pytest.mark.parametrize(
    "change",
    [
        lambda bm, head: bm.kv.allocate(bm.kv.lookup(bm._prompt_hashes(head)[3])),
        lambda bm, head: published(range(60), bm),  # at the first miss
        lambda bm, head: bm.clear_cache(),  # no eviction callback
        lambda bm, head: bm.allocate(Sequence(list(range(5000, 5216)), BLOCK), 0),
    ],
)
def test_the_memo_follows_a_change_to_what_its_walk_read(change):
    bm = published(range(40))
    head = Sequence(list(range(56)) + [900], BLOCK)
    assert bm.can_allocate_memoized(head) == 10
    change(bm, head)
    assert bm.can_allocate_memoized(head) == exact(bm, head) != 10


@pytest.mark.parametrize("watched", [0, 1])
def test_the_memo_follows_only_remaps_of_what_its_walk_read(watched):
    bm = published(range(500, 540), published(range(40)))
    head = Sequence(list(range(56)) + [900], BLOCK)
    assert bm.can_allocate_memoized(head) == 10
    moved = chained_block_hashes(
        list(range(0 if watched else 500, 540)), BLOCK, 0, 4, -1
    )
    victim = bm.kv.block(bm.kv.lookup(moved[3]))
    fresh = bm._fresh_block()
    bm.kv.publish(fresh, moved[3], victim.token_ids)  # remapped, nothing evicted
    assert bm.total_evicted_blocks == 0
    walks = []
    walk = bm._walk_prefix
    bm._walk_prefix = lambda s: walks.append(s) or walk(s)
    assert bm.can_allocate_memoized(head) == exact(bm, head) == 10
    assert len(walks) == 1 + watched
    assert bm._cached_walks[head].walk.ids == list(
        map(bm.kv.lookup, bm._prompt_hashes(head)[:10])
    )


def test_the_memo_recounts_hit_blocks_another_seq_claimed():
    """No eviction, but the hit's blocks went from free to used."""
    bm = published(range(40), BlockManager(prefix_config(num_kvcache_blocks=24)))
    head = Sequence(list(range(44)), BLOCK)
    assert bm.can_allocate_memoized(head) == 10
    other = Sequence(list(range(40)) + list(range(500, 520)), BLOCK)
    assert bm.can_allocate(other) == 10 and bm.allocate(other, 10)
    assert bm.total_evicted_blocks == 0 and bm.kv.num_free == 9
    assert bm.can_allocate_memoized(head) == exact(bm, head) == 10


def test_the_state_gate_is_asked_again():
    bm = make_block_manager(ckpt_config())
    first = stateful_seq(list(range(40)))
    h = publish_at_boundary(bm, first)
    publisher_has_read_its_source(bm)
    bm.deallocate(first)
    head = stateful_seq(list(range(40)) + list(range(500, 520)))
    hit = bm.can_allocate_memoized(head)
    assert hit == exact(bm, head) > 0
    bm.state.unindex(h)  # the checkpoint goes; its KV block stays
    assert bm.can_allocate_memoized(head) == exact(bm, head) < hit


def test_the_replay_is_asked_again(monkeypatch):
    """The tier's answer lands between ticks; the memoised probe follows, flags too."""
    bm = _mp_bm(monkeypatch)
    # 32 tokens in HBM with no checkpoint; LMCache holds 44 with state.
    head = _mp_seq(bm, reach_with_state=44, reach_kv_only=48)

    def memoized():
        return bm.can_allocate_memoized(head), head.replays_hit, head.replay_kv_load

    assert memoized() == memoized() == (0, False, False)  # loading 44 wins
    head.offload_joint.kv_prefix_tokens = 0  # only the KV-only reach left
    assert memoized() == (8, True, True) and exact(bm, head) == 8
    head.offload_joint.kv_prefix_tokens = 44
    assert memoized() == (0, False, False) and exact(bm, head) == 0


def reference_chain(tokens, hbs, seed, n):
    out, h = [], seed
    for i in range(n):
        h = BlockManager.compute_hash(
            array.array("i", tokens[i * hbs : (i + 1) * hbs]), h
        )
        out.append(h)
    return out


def shipped_seq(tokens, seed=-1, hbs=BLOCK):
    seq = Sequence(list(tokens), BLOCK)
    seq.cache_seed = seed
    ship_prompt_hash_chain(seq, hbs)
    return seq


@pytest.mark.parametrize("seed", [-1, 2**63 + 12345])
@pytest.mark.parametrize("dcp", [1, 2])
@pytest.mark.parametrize("length", [1, 3, 4, 5, 17, 40, 203])
def test_shipped_chain_equals_engine_chain(seed, dcp, length):
    tokens = [(i * 7919) % 50021 for i in range(length)]
    engine_bm = BlockManager(prefix_config(decode_context_parallel_size=dcp))
    hbs = engine_bm.hash_block_size
    n = -(-length // hbs)  # includes the partial last block
    whole = chained_block_hashes(array.array("i", tokens), hbs, 0, n, seed)
    assert whole == reference_chain(tokens, hbs, seed, n)
    head = chained_block_hashes(tokens, hbs, 0, n // 2, seed)
    tail = chained_block_hashes(tokens, hbs, n // 2, n, head[-1] if head else seed)
    assert head + tail == whole
    engine_seq = Sequence(tokens, BLOCK)
    engine_seq.cache_seed = seed
    engine_bm._chain_to(engine_seq, [], length // hbs)
    engine_chain = engine_bm._prompt_hashes(engine_seq)
    assert engine_chain == whole[: length // hbs]
    [seq] = over_ipc([shipped_seq(tokens, seed, hbs)])
    assert seq.prompt_hash_chain[:3] == (hbs, seed, length)
    assert seq.prompt_hash_chain.hashes.tolist() == engine_chain
    bm = BlockManager(prefix_config(decode_context_parallel_size=dcp))
    assert bm._prompt_hashes(seq) == engine_chain
    assert (bm.shipped_chains_adopted, bm.shipped_chains_rejected) == (1, 0)
    assert seq.prompt_hash_chain is None  # consumed


def over_ipc(seqs):
    """The ADD frame's trip: seqs pickled, their chains beside them."""
    seqs, chains = pickle.loads(pickle.dumps((seqs, take_shipped_chains(seqs))))
    attach_shipped_chains(seqs, chains)
    return seqs


PROMPT = list(range(40)) + [900, 901, 902]


@pytest.mark.parametrize(
    "corrupt",
    [
        lambda c: c._replace(hash_block_size=c.hash_block_size * 2),
        lambda c: c._replace(cache_seed=c.cache_seed + 1),
        lambda c: c._replace(hashes=c.hashes[:-1]),  # too short
        lambda c: c._replace(hashes=array.array("Q", [x ^ 1 for x in c.hashes])),
        lambda c: c._replace(hashes=c.hashes[:-1] + array.array("Q", [5])),
        lambda c: c._replace(hashes=relinked(c.hashes, 5)),  # the middle link
        lambda c: c._replace(num_prompt_tokens=c.num_prompt_tokens - 1),
        # Another prompt that differs at block 2 only: every sampled link
        # (0, 5, 9) holds, only the fingerprint tells.
        lambda c: shipped_seq(PROMPT[:9] + [777] + PROMPT[10:]).prompt_hash_chain,
    ],
)
def test_a_bad_chain_is_rehashed_with_the_same_hit(corrupt):
    bm = published(range(40))
    seq = shipped_seq(PROMPT)
    seq.prompt_hash_chain = corrupt(seq.prompt_hash_chain)
    assert bm.can_allocate(seq, record=False) == 10
    assert (bm.shipped_chains_adopted, bm.shipped_chains_rejected) == (0, 1)
    assert bm._prompt_hashes(seq) == reference_chain(PROMPT, BLOCK, -1, 10)


def relinked(chain, i):
    chain = array.array("Q", chain)
    chain[i] ^= 1
    return chain


@pytest.mark.parametrize("ship", [0, 1])
def test_prompt_blocks_are_published_under_the_chain(monkeypatch, ship):
    """A preempted seq: 10 prompt blocks from the chain, 2 hashed at publish."""
    seq = shipped_seq(PROMPT) if ship else Sequence(PROMPT, BLOCK)
    for t in range(5):
        seq.append_token(50 + t)
    bm = BlockManager(prefix_config())
    hit = bm.can_allocate(seq, record=False)
    assert hit == 0 and bm.allocate(seq, hit)
    hashed = []
    real = bm.compute_hash
    monkeypatch.setattr(bm, "compute_hash", lambda *a: hashed.append(a) or real(*a))
    bm.hash_blocks(seq, seq.num_tokens)
    names = [bm.kv.block(b).hash for b in seq.block_table[:12]]
    assert names == reference_chain(PROMPT + list(range(50, 55)), BLOCK, -1, 12)
    assert len(hashed) == 2  # each prompt block was hashed once, before


def _processor(**config):
    with stubbed_aiter():
        from atom.model_engine.llm_engine import InputOutputProcessor

    cfg = SimpleNamespace(
        hf_config=SimpleNamespace(model_type="qwen3_5_moe_text"),
        speculative_config=None,
        **{"enable_prefix_caching": True, "decode_context_parallel_size": 1, **config},
    )
    return InputOutputProcessor(cfg, tokenizer=None, block_size=BLOCK)


def test_frontend_preprocess_ships_the_chain():
    tokens = list(range(41))
    chain = _processor().preprocess(tokens, SamplingParams()).prompt_hash_chain
    assert chain.hash_block_size == BLOCK
    assert chain.hashes.tolist() == reference_chain(tokens, BLOCK, -1, 10)
    seqs = _processor().preprocess_fanout(tokens, SamplingParams(n=3, temperature=0.8))
    assert all(s.prompt_hash_chain is seqs[0].prompt_hash_chain for s in seqs)
    dcp = _processor(decode_context_parallel_size=2)
    assert dcp.preprocess(tokens, SamplingParams()).prompt_hash_chain[0] == 2 * BLOCK
    off = _processor(enable_prefix_caching=False)
    assert off.preprocess(tokens, SamplingParams()).prompt_hash_chain is None


def _real(name, monkeypatch):
    """`name` imported for real, even over a stub (no `__file__`) another test
    file left in `sys.modules`."""
    if getattr(sys.modules.get(name), "__file__", None) is None:
        monkeypatch.delitem(sys.modules, name, raising=False)
    with stubbed_aiter():
        return importlib.import_module(name)


@pytest.mark.parametrize("manager", ["CoreManager", "DisaggCoreManager"])
def test_the_chain_rides_beside_the_seqs(monkeypatch, manager):
    """Frontend to engine in the ADD frame, next to the seqs; the frontend
    keeps no copy, and the disaggregated prefill engine (no BlockManager)
    gets none."""
    tokens = list(range(41))
    seqs = _processor().preprocess_fanout(tokens, SamplingParams(n=3, temperature=0.8))
    mgr = object.__new__(
        getattr(_real("atom.model_engine.engine_core_mgr", monkeypatch), manager)
    )
    mgr.label, mgr.pp_size, mgr.local_engine_count = "test", 1, 1
    mgr._seq_id_to_callback, frames = {}, []
    mgr._send_request = lambda rank, payload: frames.append((rank, payload))
    mgr.add_request(seqs)
    assert all(s.prompt_hash_chain is None for s in seqs)
    if manager == "DisaggCoreManager":
        (rank, frame), (prefill, prefill_frame) = frames
        assert (rank, prefill, len(pickle.loads(prefill_frame))) == (1, 0, 2)
    else:
        [(rank, frame)] = frames
    core = _real("atom.model_engine.engine_core", monkeypatch)
    feed = iter([frame, pickle.dumps((core.EngineCoreRequestType.SHUTDOWN, None))])
    sock = SimpleNamespace(send=lambda _: None, recv=lambda **_: next(feed))
    monkeypatch.setattr(core, "make_zmq_socket", lambda *a, **kw: nullcontext(sock))
    poller = SimpleNamespace(register=lambda *a: None, poll=lambda: [(sock, 1)])
    monkeypatch.setattr(core.zmq, "Poller", lambda: poller)
    engine = object.__new__(core.EngineCore)
    engine.label, engine.input_queue = "test", queue.Queue()
    engine.process_input_sockets("input", "control")
    want = reference_chain(tokens, BLOCK, -1, 10)
    got = engine.input_queue.get_nowait()
    assert [s.prompt_hash_chain.hashes.tolist() for s in got] == [want] * 3


def test_a_pickled_seq_never_carries_a_chain():
    """Neither to the engine (the ADD frame carries it) nor back."""
    seq = shipped_seq(PROMPT)
    assert pickle.loads(pickle.dumps(seq)).prompt_hash_chain is None
    assert seq.prompt_hash_chain is not None


def test_the_chain_counters_reach_the_periodic_stats(monkeypatch):
    from atom.model_engine import engine_stats

    lines = []
    monkeypatch.setattr(engine_stats.logger, "info", lines.append)
    stats = engine_stats.EngineStats(enable_prefix_caching=True)
    stats.block_manager = bm = published(range(40))
    bm.shipped_chains_adopted, bm.shipped_chains_rejected = 3, 2
    stats._log_cache()
    assert any("adopted 3, rejected 2" in line for line in lines)


@pytest.mark.parametrize(
    "var", ["PREFIX_FAST_WALK", "PREFIX_WALK_MEMO", "FRONTEND_PREFIX_HASH"]
)
def test_off_switch(monkeypatch, var):
    monkeypatch.setenv("ATOM_" + var, "0")
    bm = published(range(40))
    assert bm.fast_prefix_walk == (var != "PREFIX_FAST_WALK")
    assert bm.walk_memo == (var != "PREFIX_WALK_MEMO")
    seq = shipped_seq(list(range(40)) + [900])
    assert (seq.prompt_hash_chain is None) == (var == "FRONTEND_PREFIX_HASH")
    assert bm.can_allocate_memoized(seq) == exact(bm, seq) == 10
    assert bm.allocate(seq, 10) and seq.num_cached_tokens == 40
