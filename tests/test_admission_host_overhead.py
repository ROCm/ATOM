# SPDX-License-Identifier: MIT
# Admission host-overhead switches must not change answers: the fast prefix walk
# (`ATOM_PREFIX_FAST_WALK`), the delayer's probe memo (`..._PROBE_MEMO`) and the
# frontend-hashed prompt chain (`ATOM_FRONTEND_PREFIX_HASH`).

import array
import pickle
import random
import sys
import types
from types import SimpleNamespace

import pytest
from conftest import MockConfig
from test_dsv4_state_replay import _bm, _mp_bm, _mp_seq
from test_state_checkpoint import PAGED_COPY_RUNTIME, ckpt_config, stateful_seq
from test_state_checkpoint import make_block_manager, publish_at_boundary
from test_state_checkpoint import publisher_has_read_its_source

from atom.model_engine.block_manager import BlockManager
from atom.model_engine.prefix_hash import chained_block_hashes, ship_prompt_hash_chain
from atom.model_engine.sequence import Sequence
from atom.sampling_params import SamplingParams

BLOCK = 4


def prefix_config(**kw):
    kw = {"num_kvcache_blocks": 64, "enable_prefix_caching": True, **kw}
    return MockConfig(kv_cache_block_size=BLOCK, **kw)


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


def published(tokens, bm=None):
    bm = BlockManager(prefix_config()) if bm is None else bm
    admit(bm, seq := Sequence(list(tokens), BLOCK))
    bm.deallocate(seq)
    return bm


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
    assert list(seq.block_table[:10]) == want and bm._admission_walk is None


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
    assert bm.can_allocate_estimate(head) == 10
    change(bm, head)
    assert bm.can_allocate_estimate(head) == exact(bm, head) != 10


def test_the_state_gate_is_asked_again():
    bm = make_block_manager(ckpt_config())
    first = stateful_seq(list(range(40)))
    h = publish_at_boundary(bm, first)
    publisher_has_read_its_source(bm)
    bm.deallocate(first)
    head = stateful_seq(list(range(40)) + list(range(500, 520)))
    hit = bm.can_allocate_estimate(head)
    assert hit == exact(bm, head) > 0
    bm.state.unindex(h)  # the checkpoint goes; its KV block stays
    assert bm.can_allocate_estimate(head) == exact(bm, head) < hit


def test_the_replay_is_asked_again(monkeypatch):
    """The tier's answer lands between ticks; the estimate follows, flags too."""
    bm = _mp_bm(monkeypatch)
    # 32 tokens in HBM with no checkpoint; LMCache holds 44 with state.
    head = _mp_seq(bm, reach_with_state=44, reach_kv_only=48)

    def estimate():
        return bm.can_allocate_estimate(head), head.replays_hit, head.replay_kv_load

    assert estimate() == estimate() == (0, False, False)  # loading 44 wins
    head.offload_joint.kv_prefix_tokens = 0  # only the KV-only reach left
    assert estimate() == (8, True, True) and exact(bm, head) == 8
    head.offload_joint.kv_prefix_tokens = 44
    assert estimate() == (0, False, False) and exact(bm, head) == 0


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
    seq = pickle.loads(pickle.dumps(shipped_seq(tokens, seed, hbs)))  # the IPC
    assert seq.prompt_hash_chain[:2] == (hbs, seed)
    assert seq.prompt_hash_chain[2].tolist() == engine_chain
    bm = BlockManager(prefix_config(decode_context_parallel_size=dcp))
    assert bm._prompt_hashes(seq) == engine_chain
    assert (bm.shipped_chains_adopted, bm.shipped_chains_rejected) == (1, 0)
    assert seq.prompt_hash_chain is None  # consumed


@pytest.mark.parametrize(
    "corrupt",
    [
        lambda c: (c[0] * 2, c[1], c[2]),  # another hash block size
        lambda c: (c[0], c[1] + 1, c[2]),  # another seed
        lambda c: (c[0], c[1], c[2][:-1]),  # too short
        lambda c: (c[0], c[1], array.array("Q", [x ^ 1 for x in c[2]])),
        lambda c: (c[0], c[1], c[2][:-1] + array.array("Q", [5])),  # last link
    ],
)
def test_a_bad_chain_is_rehashed_with_the_same_hit(corrupt):
    tokens = list(range(40)) + [900, 901, 902]
    bm = published(range(40))
    seq = shipped_seq(tokens)
    seq.prompt_hash_chain = corrupt(seq.prompt_hash_chain)
    assert bm.can_allocate(seq, record=False) == 10
    assert (bm.shipped_chains_adopted, bm.shipped_chains_rejected) == (0, 1)
    assert bm._prompt_hashes(seq) == reference_chain(tokens, BLOCK, -1, 10)


def _processor(**config):
    if "atom.model_engine.engine_core_mgr" not in sys.modules:
        stub = types.ModuleType("atom.model_engine.engine_core_mgr")
        stub.CoreManager = stub.DisaggCoreManager = object
        sys.modules["atom.model_engine.engine_core_mgr"] = stub
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
    assert chain[0] == BLOCK
    assert chain[2].tolist() == reference_chain(tokens, BLOCK, -1, 10)
    seqs = _processor().preprocess_fanout(tokens, SamplingParams(n=3, temperature=0.8))
    assert all(s.prompt_hash_chain is seqs[0].prompt_hash_chain for s in seqs)
    dcp = _processor(decode_context_parallel_size=2)
    assert dcp.preprocess(tokens, SamplingParams()).prompt_hash_chain[0] == 2 * BLOCK
    off = _processor(enable_prefix_caching=False)
    assert off.preprocess(tokens, SamplingParams()).prompt_hash_chain is None


@pytest.mark.parametrize(
    "var", ["PREFIX_FAST_WALK", "PREFILL_DELAYER_PROBE_MEMO", "FRONTEND_PREFIX_HASH"]
)
def test_off_switch(monkeypatch, var):
    monkeypatch.setenv("ATOM_" + var, "0")
    bm = published(range(40))
    assert bm.fast_prefix_walk == (var != "PREFIX_FAST_WALK")
    assert bm.fit_estimate_memo == (var != "PREFILL_DELAYER_PROBE_MEMO")
    seq = shipped_seq(list(range(40)) + [900])
    assert (seq.prompt_hash_chain is None) == (var == "FRONTEND_PREFIX_HASH")
    assert bm.can_allocate_estimate(seq) == exact(bm, seq) == 10
    assert bm.allocate(seq, 10) and seq.num_cached_tokens == 40
