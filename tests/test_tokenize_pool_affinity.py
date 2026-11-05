# SPDX-License-Identifier: MIT
"""TokenizePool session routing (ATOM_API_OFFLOAD_TOKENIZE_CACHE_MB) and the
encoder hook of chat_offload.prepare_chat, without spawning worker processes."""

import itertools

from atom.entrypoints.openai import chat_offload
from atom.entrypoints.openai.tokenize_pool import TokenizePool


def _pool(n):
    p = TokenizePool.__new__(TokenizePool)
    p._pools = [object() for _ in range(n)]
    p._pool = p._pools[0]
    p._next = itertools.count()
    return p


def test_equal_keys_reach_the_same_worker():
    p = _pool(8)
    for key in ("sim_000329", "textual-kitty-key-phases", 17):
        assert len({id(p._executor(key)) for _ in range(20)}) == 1


def test_keys_spread_over_the_workers():
    p = _pool(8)
    hit = {id(p._executor(f"conv-{i}")) for i in range(400)}
    assert len(hit) == 8


def test_no_key_round_robins():
    p = _pool(4)
    assert [p._pools.index(p._executor()) for _ in range(8)] == [0, 1, 2, 3] * 2


def test_one_shared_pool_ignores_the_key():
    p = _pool(1)
    assert p._executor("a") is p._executor("b") is p._executor() is p._pool


def test_prepare_chat_uses_the_given_encoder(monkeypatch):
    seen = []

    class Tok:
        def encode(self, text):
            raise AssertionError("the plain encoder ran")

    monkeypatch.setattr(chat_offload, "apply_chat_template", lambda *a, **k: "PROMPT")
    body = b'{"model": "m", "messages": [{"role": "user", "content": "hi"}]}'
    settings = chat_offload.ChatPrepareSettings(None, {}, None, 0)
    prepared = chat_offload.prepare_chat(
        body, Tok(), None, settings, encode=lambda t: seen.append(t) or [1, 2, 3]
    )
    assert seen == ["PROMPT"]
    assert prepared is not None and list(prepared.prompt_ids) == [1, 2, 3]
