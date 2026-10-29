# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""LMCache chunk keys carry a multimodal prompt's media identity.

A prompt's image placeholder ids are the same for every image. Keyed on the
ids alone, two prompts that differ only in their pictures share every KV chunk
in LMCache, while a hybrid's state images (filed under `BlockManager` hashes
seeded with `Sequence.cache_seed`) do not.
"""

from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest

from atom.kv_transfer.offload import config as offcfg
from atom.kv_transfer.offload._offload_common import tier_token_ids
from atom.kv_transfer.offload.metadata import LoadSpec

CHUNK = 256
PROMPT = 3 * CHUNK


class _Tier:
    """LMCache's bookkeeping: chunk keys chained over the ids, deduplicated."""

    def __init__(self):
        self.chunks: dict[bytes, str] = {}

    @staticmethod
    def _keys(token_ids):
        link = b""
        for start in range(0, len(token_ids) - len(token_ids) % CHUNK, CHUNK):
            chunk = token_ids[start : start + CHUNK]
            assert all(0 <= t < 1 << 32 for t in chunk)  # the MP server packs ">I"
            link = hashlib.blake2b(link + repr(chunk).encode()).digest()
            yield link

    def store(self, token_ids, owner):
        for key in self._keys(token_ids):
            self.chunks.setdefault(key, owner)

    def hit(self, token_ids) -> tuple[int, set]:
        owners, tokens = set(), 0
        for key in self._keys(token_ids):
            if key not in self.chunks:
                break
            owners.add(self.chunks[key])
            tokens += CHUNK
        return tokens, owners


def _seq(sid, cache_seed, tokens=None):
    tokens = list(range(100, 100 + PROMPT)) if tokens is None else tokens
    return SimpleNamespace(
        id=sid,
        token_ids=tokens,
        num_prompt_tokens=len(tokens),
        num_tokens=len(tokens),
        cache_seed=cache_seed,
        block_table=list(range(len(tokens) // 64)),
    )


@pytest.fixture
def scheduler(monkeypatch):
    from atom.kv_transfer.offload.connector import _build_scheduler

    monkeypatch.setattr(
        offcfg,
        "build_lmcache_config",
        lambda _config=None: SimpleNamespace(chunk_size=CHUNK),
    )
    monkeypatch.setattr(offcfg, "build_lmcache_metadata", lambda *_args: object())
    return _build_scheduler(
        SimpleNamespace(
            hf_config=SimpleNamespace(
                model_type="qwen4_exp",
                text_config=SimpleNamespace(
                    model_type="qwen4_exp_text", num_attention_heads=16
                ),
            ),
            kv_transfer_config={"kv_role": "offload"},
            kv_cache_block_size=64,
            decode_context_parallel_size=1,
            tensor_parallel_size=1,
        )
    )


def _save(scheduler, tier, seq):
    request = scheduler._build_save_request(
        seq, 0, PROMPT, SimpleNamespace(), list(seq.block_table), True
    )
    tier.store(request.token_ids, seq.id)


def test_another_image_with_the_same_text_misses_the_first_ones_kv(scheduler):
    """Review 5698904199, must change 3. A = image1 + T stores its KV; B =
    image2 + T must not find it, or C = image2 + T would decode from image1's
    QSA K/V/index under image2's GDN/PLE state."""
    tier = _Tier()
    a, b, c = _seq("A", 11), _seq("B", 22), _seq("C", 22)
    _save(scheduler, tier, a)

    assert tier.hit(scheduler._lookup_token_ids(b)) == (0, set())
    _save(scheduler, tier, b)
    assert tier.hit(scheduler._lookup_token_ids(c)) == (PROMPT, {"B"})
    assert tier.hit(scheduler._lookup_token_ids(_seq("A2", 11))) == (PROMPT, {"A"})


def test_the_load_keys_what_the_lookup_and_save_keyed(scheduler):
    tier = _Tier()
    b, c = _seq("B", 22), _seq("C", 22)
    _save(scheduler, tier, b)
    load = scheduler._build_load_request(
        c,
        LoadSpec(hbm_cached_tokens=0, lmcache_cached_tokens=PROMPT),
        SimpleNamespace(),
        PROMPT,
    )
    assert load.token_ids == scheduler._lookup_token_ids(c)
    assert tier.hit(load.token_ids) == (PROMPT, {"B"})


def test_a_text_prompt_keys_on_its_ids_alone(scheduler):
    text = _seq("T", -1)
    assert scheduler._lookup_token_ids(text) == text.token_ids
    tier = _Tier()
    _save(scheduler, tier, text)
    assert tier.hit(list(text.token_ids)) == (PROMPT, {"T"})
    assert tier.hit(scheduler._lookup_token_ids(_seq("I", 11)))[0] == 0


def test_media_keys_touch_only_the_leading_ids_and_keep_prefixes():
    seq = _seq("A", 11)
    keyed = tier_token_ids(seq, PROMPT)
    assert keyed[4:] == seq.token_ids[4:]
    assert all(1 << 31 <= t < 1 << 32 for t in keyed[:4])
    for end in (1, 3, 4, CHUNK, PROMPT - 1):
        assert tier_token_ids(seq, end) == keyed[:end]
    assert tier_token_ids(_seq("B", 22), PROMPT)[:4] != keyed[:4]
    other_text = _seq("A", 11, [7] + seq.token_ids[1:])
    assert tier_token_ids(other_text, PROMPT)[0] != keyed[0]
    assert seq.token_ids[0] == 100  # the request's own ids are untouched


def test_lmcache_mp_native_state_lookup_is_keyed_the_same_way():
    from atom.kv_transfer.offload.mp.native_state_scheduler import (
        NativeStateLMCacheMPConnectorScheduler,
    )

    stub = SimpleNamespace(_chunk_floor=lambda tokens: tokens // CHUNK * CHUNK)
    seq = _seq("A", 11)
    ids = NativeStateLMCacheMPConnectorScheduler._lookup_token_ids(stub, seq)
    assert ids == tier_token_ids(seq, PROMPT - CHUNK)
    text = _seq("T", -1)
    assert NativeStateLMCacheMPConnectorScheduler._lookup_token_ids(stub, text) == list(
        text.token_ids[: PROMPT - CHUNK]
    )
