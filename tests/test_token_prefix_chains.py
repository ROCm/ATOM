# SPDX-License-Identifier: MIT
"""Token-id prompts reuse the block-hash chain of the stored prompt they share
the most leading tokens with (`TokenPrefixChains`)."""

from concurrent.futures import ThreadPoolExecutor
from itertools import pairwise
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from aiter_stub import stubbed_aiter
from conftest import MockConfig
from test_admission_host_overhead import over_ipc

from atom.model_engine import prefix_block_hashes as pb
from atom.model_engine import prefix_hash as ph
from atom.model_engine.block_manager import BlockManager
from atom.sampling_params import SamplingParams

with stubbed_aiter():
    from atom.model_engine.llm_engine import InputOutputProcessor, LLMEngine

COUNTS = ("prompts", "blocks_hashed", "blocks_reused", "blocks_rehashed")
GEN = [3, 4]  # a turn's generation prompt, rewritten once the turn is answered


@pytest.fixture(autouse=True)
def env(monkeypatch):
    monkeypatch.delenv("ATOM_TOKENIZE_CACHE_MB", raising=False)
    monkeypatch.delenv("ATOM_TOKENIZE_CACHE_BLOCK_HASHES", raising=False)
    monkeypatch.setenv("ATOM_FRONTEND_PREFIX_HASH", "1")


def processor(block_size):
    config = SimpleNamespace(
        hf_config=SimpleNamespace(model_type="llama"),
        speculative_config=None,
        enable_prefix_caching=True,
        decode_context_parallel_size=1,
    )
    return InputOutputProcessor(config, MagicMock(), block_size)


def served(proc):
    """The token-id chain counts `/debug/cache_stats` serves for `proc`."""
    engine = LLMEngine.__new__(LLMEngine)
    engine.io_processor = proc
    engine.core_mgr = SimpleNamespace(broadcast_utility_command_sync=lambda *a, **k: [])
    return engine.get_cache_statistics()["token_prefix_chains"]


def full_chain(prompt, block):
    return ph.chained_block_hashes(prompt, block, 0, len(prompt) // block, -1)


def tokens(rng, n):
    return rng.integers(10, 150_000, int(n)).tolist()


def conversation(rng, system, turns):
    """A token-id conversation's prompts: each turn rewrites the generation
    prompt, then adds an answer, a user message and the generation prompt."""
    prompt = system + tokens(rng, rng.integers(50, 3000)) + GEN
    for _ in range(turns):
        yield prompt
        prompt = prompt[: -len(GEN)] + [5] + tokens(rng, rng.integers(50, 3000)) + GEN


def scenario():
    rng = np.random.default_rng(0)
    system = tokens(rng, 6000)
    a, b = conversation(rng, system, 7), conversation(rng, system, 7)
    prompts = [p for pair in zip(a, b) for p in pair]
    fork = prompts[6]
    # a fork, an edit inside a turn, a compaction, no complete block, a retry
    prompts += [fork + tokens(rng, 900), fork[:-500] + tokens(rng, 700)]
    return prompts + [system + tokens(rng, 400), tokens(rng, 7), list(prompts[3])]


def shared_blocks(earlier, prompt, block):
    """Complete blocks `prompt` shares with the earlier prompt sharing the most."""
    best, p = 0, np.array(prompt)
    for other in earlier:
        q = np.array(other)
        m = min(len(p), len(q))
        diff = np.flatnonzero(p[:m] != q[:m])
        best = max(best, int(diff[0] if len(diff) else m) // block)
    return best


@pytest.mark.parametrize("block", [16, 48, 256])
def test_each_turn_ships_a_full_recompute_hashing_only_unshared_blocks(block):
    proc, prompts = processor(block), scenario()
    bm = BlockManager(MockConfig(kv_cache_block_size=block, enable_prefix_caching=True))
    before, reuses = served(proc), 0
    for i, prompt in enumerate(prompts):
        with patch.object(ph, "_link", wraps=ph._link) as link:
            [seq] = proc.preprocess_fanout(prompt, SamplingParams())
        n, reused = len(prompt) // block, shared_blocks(prompts[:i], prompt, block)
        assert list(seq.prompt_hash_chain.hashes) == full_chain(prompt, block)
        after = served(proc)
        checked = reused > 0 and reuses < pb._VERIFY_FIRST
        assert {k: after[k] - before[k] for k in COUNTS} == {
            "prompts": 1,
            "blocks_hashed": n - reused,
            "blocks_reused": reused,
            "blocks_rehashed": n if checked else 0,
        }
        assert link.call_count == n - reused + (n if checked else 0)
        [seq] = over_ipc([seq])
        assert bm._prompt_hashes(seq) == full_chain(prompt, block)
        before, reuses = after, reuses + (reused > 0)
    assert after["enabled"] and reuses == len(prompts) - 2
    assert (bm.shipped_chains_adopted, bm.shipped_chains_rejected) == (len(prompts), 0)
    assert proc.tokenizer.encode.call_count == 0
    proc.tokenizer.encode.side_effect = lambda text, **_: list(range(1, 99))
    proc.preprocess_fanout("a prompt sent as text", SamplingParams())
    assert served(proc)["prompts"] == len(prompts)


def test_each_conversation_finds_its_own_turn_among_many_sharing_a_start():
    # 40 conversations share a 6,000-token system prompt; each one's next turn,
    # oldest first, reuses every block it shares with its own stored turn.
    proc, rng = processor(16), np.random.default_rng(3)
    system = tokens(rng, 6000)
    firsts = [system + tokens(rng, rng.integers(500, 1500)) + GEN for _ in range(40)]
    for prompt in firsts:
        proc.preprocess_fanout(prompt, SamplingParams())
    for first in firsts:
        prompt = first[: -len(GEN)] + [5] + tokens(rng, 300) + GEN
        before = served(proc)["blocks_reused"]
        [seq] = proc.preprocess_fanout(prompt, SamplingParams())
        assert list(seq.prompt_hash_chain.hashes) == full_chain(prompt, 16)
        assert served(proc)["blocks_reused"] - before == (len(first) - len(GEN)) // 16


def test_reuse_is_sized_by_a_token_compare_from_the_first():
    # Past the first-uses check, stored prompts that sort next to a new one and
    # differ from it only at its first token, or inside a block, size its reuse.
    proc, rng = processor(16), np.random.default_rng(4)
    prompt = tokens(rng, 2000)
    for _ in range(1 + pb._VERIFY_FIRST):
        proc.preprocess_fanout(prompt, SamplingParams())
        prompt = prompt + tokens(rng, 100)
    stored, cut = prompt[:-100], 37 * 16 + 5
    for edit, reused in ((0, 0), (cut, cut // 16)):
        new = list(stored)
        new[edit] += 1
        new += tokens(rng, 50)
        before = served(proc)
        [seq] = proc.preprocess_fanout(new, SamplingParams())
        assert list(seq.prompt_hash_chain.hashes) == full_chain(new, 16)
        after = served(proc)
        assert after["blocks_reused"] - before["blocks_reused"] == reused
        assert after["blocks_rehashed"] == before["blocks_rehashed"]


@pytest.mark.parametrize(
    "env, kept",
    [
        ({}, "a1"),
        # One 124K-token prompt fits in 1 MB, two do not.
        ({"ATOM_TOKENIZE_CACHE_MB": "1"}, "b1"),
        ({"ATOM_TOKENIZE_CACHE_MB": "0"}, None),
        ({"ATOM_TOKENIZE_CACHE_BLOCK_HASHES": "0"}, None),
    ],
)
def test_reuse_follows_the_cache_budget_and_its_switch(monkeypatch, env, kept):
    for var, value in env.items():
        monkeypatch.setenv(var, value)
    rng = np.random.default_rng(1)
    system = tokens(rng, 4096)
    a1, b1 = system + tokens(rng, 120_000), system + tokens(rng, 120_000)
    proc, reused = processor(16), []
    for prompt in (a1, b1, a1 + tokens(rng, 3000)):
        before = served(proc)["blocks_reused"]
        [seq] = proc.preprocess_fanout(prompt, SamplingParams())
        assert list(seq.prompt_hash_chain.hashes) == full_chain(prompt, 16)
        reused.append(served(proc)["blocks_reused"] - before)
    assert served(proc)["enabled"] == (kept is not None)
    if kept is None:
        assert reused == [0, 0, 0]
    else:  # b1 shares the system prompt with a1; a1's next turn finds a1 or b1
        assert reused == [0, 256, len(a1) // 16 if kept == "a1" else 256]


def test_a_mismatch_in_the_first_uses_check_turns_reuse_off(monkeypatch, caplog):
    real = pb.extend_block_hashes

    def wrong_when_reused(tokens, block_size, seed=-1, *, base=None, shared_tokens=0):
        out = real(tokens, block_size, seed, base=base, shared_tokens=shared_tokens)
        if base is None:
            return out
        bad = bytes([out.hashes[0] ^ 1]) + out.hashes[1:]
        return pb.PrefixBlockHashes(block_size, seed, out.num_tokens, bad)

    monkeypatch.setattr(pb, "extend_block_hashes", wrong_when_reused)
    proc, reused = processor(16), []
    with caplog.at_level("WARNING", logger="atom"):
        for prompt in scenario()[:6]:
            before = served(proc)["blocks_reused"]
            [seq] = proc.preprocess_fanout(prompt, SamplingParams())
            assert list(seq.prompt_hash_chain.hashes) == full_chain(prompt, 16)
            reused.append(served(proc)["blocks_reused"] - before)
    # The second prompt reuses the first's system prompt; nothing reuses after it.
    assert reused == [0, 6000 // 16, 0, 0, 0, 0]
    assert not served(proc)["enabled"] and "reuse is off" in caplog.text


def test_concurrent_forks_each_ship_a_full_recompute():
    proc, rng = processor(16), np.random.default_rng(2)
    base = tokens(rng, 9000)
    proc.preprocess_fanout(base, SamplingParams())
    forks = [
        base[: rng.integers(1, 9000)] + tokens(rng, rng.integers(1, 3000))
        for _ in range(48)
    ]

    def ships_full_chain(prompt):
        [seq] = proc.preprocess_fanout(prompt, SamplingParams())
        return list(seq.prompt_hash_chain.hashes) == full_chain(prompt, 16)

    with ThreadPoolExecutor(8) as pool:
        assert all(pool.map(ships_full_chain, forks))
    stats, chains = served(proc), proc.token_prefix_chains
    assert stats["enabled"] and stats["prompts"] == 49 and stats["blocks_reused"]
    assert chains._lru.bytes <= chains._lru.budget
    order = chains._order
    assert len(order) == len(chains._lru) and all(r.rid in chains._lru for r in order)
    assert all(a.tokens <= b.tokens for a, b in pairwise(order))
