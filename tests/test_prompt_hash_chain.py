# SPDX-License-Identifier: MIT
# The frontend-hashed prompt chain (`prefix_hash`) must equal the engine's bit
# for bit, and a chain that does not check out is rehashed with the same hits.

import array
import pickle
import sys
import types
from types import SimpleNamespace

import pytest
from test_admission_fast_walk import prefix_config

from atom.model_engine.block_manager import BlockManager
from atom.model_engine.prefix_hash import chained_block_hashes, ship_prompt_hash_chain
from atom.model_engine.sequence import Sequence
from atom.sampling_params import SamplingParams

BLOCK = 4
SEEDS = [-1, 2**63 + 12345]


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


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("hbs", [1, 4, 16])
@pytest.mark.parametrize("length", [0, 1, 5, 17, 64, 65])
def test_bulk_chain_matches_compute_hash(seed, hbs, length):
    tokens = [(i * 7919 + hbs) % 50021 for i in range(length)]
    n = (length + hbs - 1) // hbs  # includes the partial last block
    whole = chained_block_hashes(array.array("i", tokens), hbs, 0, n, seed)
    assert whole == reference_chain(tokens, hbs, seed, n)
    head = chained_block_hashes(tokens, hbs, 0, n // 2, seed)
    assert (
        head + chained_block_hashes(tokens, hbs, n // 2, n, head[-1] if head else seed)
        == whole
    )


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("dcp", [1, 2])
@pytest.mark.parametrize("length", [1, 3, 4, 5, 17, 40, 203])
def test_shipped_chain_equals_engine_chain(seed, dcp, length):
    tokens = [(i * 7919) % 50021 for i in range(length)]
    engine_bm = BlockManager(prefix_config(decode_context_parallel_size=dcp))
    hbs = engine_bm.hash_block_size
    engine_seq = Sequence(tokens, BLOCK)
    engine_seq.cache_seed = seed
    engine_bm._chain_to(engine_seq, [], length // hbs)
    engine_chain = engine_bm._prompt_hashes(engine_seq)
    assert engine_chain == reference_chain(tokens, hbs, seed, length // hbs)

    seq = pickle.loads(pickle.dumps(shipped_seq(tokens, seed, hbs)))  # the IPC
    assert seq.prompt_hash_chain[:2] == (hbs, seed)
    assert seq.prompt_hash_chain[2].tolist() == engine_chain

    bm = BlockManager(prefix_config(decode_context_parallel_size=dcp))
    assert bm._prompt_hashes(seq) == engine_chain
    assert (bm.shipped_chains_adopted, bm.shipped_chains_rejected) == (1, 0)
    assert seq.prompt_hash_chain is None  # consumed


def published(tokens):
    bm = BlockManager(prefix_config())
    seq = Sequence(list(tokens), BLOCK)
    bm.allocate(seq, bm.can_allocate(seq))
    bm.hash_blocks(seq, seq.num_prompt_tokens)
    bm.deallocate(seq)
    return bm


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


def test_an_undetected_interior_corruption_only_loses_hits():
    bm = published(range(40))
    seq = shipped_seq(list(range(40)) + [900, 901, 902])
    seq.prompt_hash_chain[2][4] ^= 1  # the spot checks see first and last only
    assert bm.can_allocate(seq, record=False) == 4


def test_off_switch(monkeypatch):
    monkeypatch.setenv("ATOM_FRONTEND_PREFIX_HASH", "0")
    seq = shipped_seq(range(40))
    assert seq.prompt_hash_chain is None
    assert published(range(40)).can_allocate(seq, record=False) == 9


def _processor(**config):
    if "atom.model_engine.engine_core_mgr" not in sys.modules:
        stub = types.ModuleType("atom.model_engine.engine_core_mgr")
        stub.CoreManager = stub.DisaggCoreManager = object
        sys.modules["atom.model_engine.engine_core_mgr"] = stub
    from atom.model_engine.llm_engine import InputOutputProcessor

    cfg = SimpleNamespace(
        hf_config=SimpleNamespace(model_type="qwen3_5_moe_text"),
        speculative_config=None,
        enable_prefix_caching=True,
        decode_context_parallel_size=1,
    )
    cfg.__dict__.update(config)
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
