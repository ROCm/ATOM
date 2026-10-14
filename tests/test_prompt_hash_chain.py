# SPDX-License-Identifier: MIT
# Tests for the frontend-hashed prompt chain (`prefix_hash`): the chain the
# request carries must be bit-identical to the one the engine would compute,
# and anything that does not check out falls back to engine-side hashing with
# the same prefix hits.

import array
import pickle
import random
import sys
import types
from types import SimpleNamespace

import pytest
from conftest import MockConfig

from atom.model_engine.block_manager import BlockManager
from atom.model_engine.prefix_hash import (
    chained_block_hashes,
    ship_prompt_hash_chain,
)
from atom.model_engine.sequence import Sequence

BLOCK = 4
SEEDS = [-1, 0, 7, 2**63 + 12345, 2**64 - 1]


def reference_chain(tokens, hbs, seed, n):
    """Block by block through `BlockManager.compute_hash`, the engine's form."""
    out, h = [], seed
    for i in range(n):
        h = BlockManager.compute_hash(
            array.array("i", tokens[i * hbs : (i + 1) * hbs]), h
        )
        out.append(h)
    return out


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


def seeded_seq(tokens, seed=-1):
    seq = Sequence(list(tokens), BLOCK)
    seq.cache_seed = seed
    return seq


def shipped_seq(tokens, seed=-1, hbs=BLOCK):
    seq = seeded_seq(tokens, seed)
    ship_prompt_hash_chain(seq, hbs)
    return seq


def publish(bm, seq):
    hit = bm.can_allocate(seq)
    assert hit >= 0
    bm.allocate(seq, hit)
    bm.hash_blocks(seq, seq.num_prompt_tokens - seq.num_cached_tokens)
    bm.deallocate(seq)


class TestBitIdentical:
    @pytest.mark.parametrize("seed", SEEDS)
    @pytest.mark.parametrize("hbs", [1, 4, 16])
    @pytest.mark.parametrize("length", [0, 1, 3, 4, 5, 16, 17, 63, 64, 65, 1000])
    def test_bulk_chain_matches_compute_hash(self, seed, hbs, length):
        rng = random.Random(length * 31 + hbs)
        tokens = [rng.randrange(0, 2**31 - 1) for _ in range(length)]
        # Include the partial last block: `_chain_to` hashes it too.
        n = (length + hbs - 1) // hbs
        assert chained_block_hashes(
            array.array("i", tokens), hbs, 0, n, seed
        ) == reference_chain(tokens, hbs, seed, n)

    @pytest.mark.parametrize("seed", SEEDS)
    def test_a_resumed_range_continues_the_chain(self, seed):
        tokens = array.array("i", range(100, 150))
        whole = chained_block_hashes(tokens, BLOCK, 0, 13, seed)
        head = chained_block_hashes(tokens, BLOCK, 0, 5, seed)
        tail = chained_block_hashes(tokens, BLOCK, 5, 13, head[-1])
        assert head + tail == whole

    @pytest.mark.parametrize("seed", SEEDS)
    @pytest.mark.parametrize("length", [1, 3, 4, 5, 17, 40, 203])
    def test_shipped_chain_equals_engine_chain(self, seed, length):
        tokens = [(i * 7919) % 50021 for i in range(length)]
        engine_bm = BlockManager(prefix_config())
        engine_seq = seeded_seq(tokens, seed)
        # The engine's own chain, filled the way a full-length walk fills it.
        engine_bm._chain_to(engine_seq, [], length // BLOCK)
        engine_chain = engine_bm._prompt_hashes(engine_seq)

        seq = shipped_seq(tokens, seed)
        hbs, shipped_seed, packed = seq.prompt_hash_chain
        assert (hbs, shipped_seed) == (BLOCK, seed)
        assert len(packed) == length // BLOCK  # full prompt blocks only
        assert packed.tolist() == engine_chain

        bm = BlockManager(prefix_config())
        assert bm._prompt_hashes(seq) == engine_chain
        assert bm.shipped_chains_adopted == 1
        assert bm.shipped_chains_rejected == 0
        assert seq.prompt_hash_chain is None  # consumed

    def test_survives_the_ipc_pickle(self):
        seq = shipped_seq(range(37), seed=2**63 + 5)
        clone = pickle.loads(pickle.dumps(seq))
        assert clone.prompt_hash_chain[2].tolist() == seq.prompt_hash_chain[2].tolist()
        bm = BlockManager(prefix_config())
        assert bm._prompt_hashes(clone) == reference_chain(
            list(range(37)), BLOCK, 2**63 + 5, 9
        )

    def test_dcp_hash_block(self):
        """Under DCP the hash block is `block_size * dcp`, on both sides."""
        tokens = list(range(50))
        seq = shipped_seq(tokens, hbs=2 * BLOCK)
        assert seq.prompt_hash_chain[2].tolist() == reference_chain(
            tokens, 2 * BLOCK, -1, 50 // (2 * BLOCK)
        )


class TestEngineUsesIt:
    def test_no_prompt_block_is_hashed_on_the_engine(self, monkeypatch):
        bm = BlockManager(prefix_config())
        publish(bm, seeded_seq(range(40)))
        seq = shipped_seq(list(range(40)) + [900, 901, 902])

        calls = []
        real = BlockManager.compute_hash
        monkeypatch.setattr(
            BlockManager,
            "compute_hash",
            classmethod(lambda cls, t, p=-1: calls.append(1) or real(t, p)),
        )
        monkeypatch.setattr(
            "atom.model_engine.block_manager.chained_block_hashes",
            lambda *a: pytest.fail("the engine hashed a shipped prompt block"),
            raising=False,
        )
        hit = bm.can_allocate(seq, record=False)
        assert hit == 10
        bm.allocate(seq, hit)
        assert seq.num_cached_tokens == 40
        # The adoption's two spot checks, nothing else.
        assert len(calls) == 2

    @pytest.mark.parametrize("seed", [-1, 2**63 + 1])
    def test_same_hits_with_and_without_the_shipped_chain(self, seed):
        rng = random.Random(seed & 0xFFFF)
        base = [rng.randrange(1, 60) for _ in range(80)]
        prompts = [
            base[: rng.randrange(4, 80)] + [rng.randrange(1, 60) for _ in range(5)]
            for _ in range(30)
        ]
        results = []
        for ship in (False, True):
            bm = BlockManager(prefix_config(num_kvcache_blocks=60))
            hits = []
            for p in prompts:
                seq = shipped_seq(p, seed) if ship else seeded_seq(p, seed)
                hit = bm.can_allocate(seq)
                if hit < 0:
                    hits.append(None)
                    continue
                bm.allocate(seq, hit)
                bm.hash_blocks(seq, seq.num_prompt_tokens - seq.num_cached_tokens)
                hits.append((hit, list(seq.block_table)))
                bm.deallocate(seq)
            results.append(hits)
            assert bm.shipped_chains_rejected == 0
        assert results[0] == results[1]
        assert any(h and h[0] for h in results[0])  # there were prefix hits


class TestFallback:
    @pytest.mark.parametrize(
        "corrupt",
        [
            lambda c: (c[0] * 2, c[1], c[2]),  # another hash block size
            lambda c: (c[0], c[1] + 1, c[2]),  # another seed
            lambda c: (c[0], c[1], c[2][:-1]),  # too short
            lambda c: (c[0], c[1], c[2] + array.array("Q", [1])),  # too long
            lambda c: (c[0], c[1], array.array("Q", [x ^ 1 for x in c[2]])),
            lambda c: (c[0], c[1], c[2][:-1] + array.array("Q", [5])),  # last link
        ],
    )
    def test_a_bad_chain_is_rehashed_with_the_same_hit(self, corrupt):
        tokens = list(range(40)) + [900, 901, 902]
        bm = BlockManager(prefix_config())
        publish(bm, seeded_seq(range(40)))
        seq = shipped_seq(tokens)
        seq.prompt_hash_chain = corrupt(seq.prompt_hash_chain)
        hit = bm.can_allocate(seq, record=False)
        if bm.shipped_chains_rejected:
            assert bm.shipped_chains_adopted == 0
        assert hit == 10
        assert bm._prompt_hashes(seq) == reference_chain(tokens, BLOCK, -1, 10)

    def test_the_interior_corruption_case_is_caught_or_harmless(self):
        """A middle link the spot checks cannot see is adopted -- and then only
        loses hits past it, never invents one (the boundary token check)."""
        tokens = list(range(40)) + [900, 901, 902]
        bm = BlockManager(prefix_config())
        publish(bm, seeded_seq(range(40)))
        seq = shipped_seq(tokens)
        hbs, seed, packed = seq.prompt_hash_chain
        packed[4] ^= 1
        seq.prompt_hash_chain = (hbs, seed, packed)
        hit = bm.can_allocate(seq, record=False)
        assert hit <= 10

    def test_no_chain_without_prefix_caching_or_with_the_switch_off(self, monkeypatch):
        monkeypatch.setenv("ATOM_FRONTEND_PREFIX_HASH", "0")
        seq = shipped_seq(range(40))
        assert seq.prompt_hash_chain is None
        bm = BlockManager(prefix_config())
        assert bm.can_allocate(seq, record=False) == 0
        assert bm.shipped_chains_adopted == bm.shipped_chains_rejected == 0


def _stub_core_manager():
    if "atom.model_engine.engine_core_mgr" not in sys.modules:
        stub = types.ModuleType("atom.model_engine.engine_core_mgr")
        stub.CoreManager = stub.DisaggCoreManager = object
        sys.modules["atom.model_engine.engine_core_mgr"] = stub


class TestFrontend:
    def _processor(self, **config):
        _stub_core_manager()
        from atom.model_engine.llm_engine import InputOutputProcessor

        cfg = SimpleNamespace(
            hf_config=SimpleNamespace(model_type="qwen3_5_moe_text"),
            speculative_config=None,
            enable_prefix_caching=True,
            decode_context_parallel_size=1,
        )
        for k, v in config.items():
            setattr(cfg, k, v)
        return InputOutputProcessor(cfg, tokenizer=None, block_size=BLOCK)

    def test_preprocess_ships_the_chain(self):
        from atom.sampling_params import SamplingParams

        proc = self._processor()
        seq = proc.preprocess(list(range(41)), SamplingParams())
        assert seq.prompt_hash_chain[0] == BLOCK
        assert seq.prompt_hash_chain[2].tolist() == reference_chain(
            list(range(41)), BLOCK, -1, 10
        )

    def test_fanout_siblings_share_one_chain(self):
        from atom.sampling_params import SamplingParams

        proc = self._processor()
        seqs = proc.preprocess_fanout(
            list(range(41)), SamplingParams(n=3, temperature=0.8)
        )
        assert all(s.prompt_hash_chain is seqs[0].prompt_hash_chain for s in seqs)

    def test_dcp_and_off_switches(self):
        from atom.sampling_params import SamplingParams

        dcp = self._processor(decode_context_parallel_size=2)
        assert dcp.preprocess(list(range(41)), SamplingParams()).prompt_hash_chain[
            0
        ] == (2 * BLOCK)
        off = self._processor(enable_prefix_caching=False)
        assert (
            off.preprocess(list(range(41)), SamplingParams()).prompt_hash_chain is None
        )
