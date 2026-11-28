# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Prehashed LMCache saves (M3 offload) against the token path they replace."""

import array
import json
import random
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

pytest.importorskip("lmcache")

from lmcache.v1.token_database import ChunkedTokenDatabase

from atom.kv_transfer.disaggregation.types import SaveOperationId
from atom.kv_transfer.offload import config as offcfg
from atom.kv_transfer.offload._offload_common import CACHE_SALT_TAG, tokens_to_tensor
from atom.kv_transfer.offload.hybrid.m3.connector import (
    M3OffloadConnector,
    M3OffloadScheduler,
)
from atom.kv_transfer.offload.lookup_client import as_lookup_client

CHUNK = 8


def _tdb():
    cfg = SimpleNamespace(
        chunk_size=CHUNK,
        save_unfull_chunk=False,
        pre_caching_hash_algorithm="builtin",
        remote_url=None,
        enable_pd=False,
        get_extra_config_value=lambda _name, default=None: default,
    )
    meta = SimpleNamespace(
        model_name="m3",
        world_size=4,
        worker_id=1,
        kv_dtype=torch.uint8,
        use_mla=False,
        is_first_rank=lambda: False,
    )
    return ChunkedTokenDatabase(cfg, meta)


def _scheduler(monkeypatch, hash_seed="0"):
    monkeypatch.setenv("PYTHONHASHSEED", hash_seed)
    monkeypatch.setattr(
        offcfg,
        "build_lmcache_config",
        lambda _c=None: SimpleNamespace(chunk_size=CHUNK, enable_kv_events=False),
    )
    monkeypatch.setattr(offcfg, "build_lmcache_metadata", lambda *_a: object())
    monkeypatch.setattr(offcfg, "lmcache_replica_world_size", lambda _c: 4)
    scheduler = M3OffloadScheduler(
        SimpleNamespace(
            kv_transfer_config={"kv_role": "offload"},
            kv_cache_block_size=4,
            decode_context_parallel_size=1,
            tensor_parallel_size=4,
            offload_min_load_tokens=8192,
        )
    )
    transport = MagicMock()
    transport.send_and_recv_all.return_value = [(3 * CHUNK).to_bytes(4, "big")] * 4
    client = SimpleNamespace(
        token_database=_tdb(), transport=transport, enable_blending=False
    )
    scheduler._lookup_client = as_lookup_client(client)
    scheduler._init_prehashed_save()
    return scheduler


def _seq(n_tokens, cache_seed=-1):
    rng = random.Random(n_tokens)
    return SimpleNamespace(
        id=7,
        token_ids=array.array("i", [rng.randrange(200000) for _ in range(n_tokens)]),
        num_prompt_tokens=n_tokens,
        num_cached_tokens=0,
        block_table=list(range(n_tokens // 4 + 1)),
        cache_seed=cache_seed,
    )


@pytest.mark.parametrize("cache_seed", [-1, 42])
def test_prehashed_lookup_and_save_match_the_token_path(monkeypatch, cache_seed):
    scheduler = _scheduler(monkeypatch)
    seq = _seq(6 * CHUNK + 1, cache_seed=cache_seed)
    tokens, tdb = list(seq.token_ids), _tdb()
    salt = {CACHE_SALT_TAG: str(cache_seed)} if cache_seed >= 0 else None
    # Lookup: the message LMCacheLookupClient.lookup sends for these tokens.
    hashes = [h for _s, _e, h in tdb.process_tokens(tokens, make_key=False)]
    assert scheduler._tier_lookup(seq, "7") == 3 * CHUNK
    sent = scheduler._lookup_client.transport.send_and_recv_all.call_args.args[0]
    assert sent == [hashes, [CHUNK] * 6, "7", json.dumps(salt) if salt else ""]
    # Save past a stored prefix: the worker stores the token path's keys.
    req = scheduler._build_save_request(
        seq, 2 * CHUNK, 5 * CHUNK, SaveOperationId(7, 0), list(range(12)), False
    )
    assert req.token_ids == [] and req.prehashed_save is not None
    engine = MagicMock(token_database=tdb, kv_events_enabled=False)
    engine.is_frozen.return_value = engine._is_passive.return_value = False
    worker = M3OffloadConnector.__new__(M3OffloadConnector)
    worker._init_worker_common(SimpleNamespace(kv_transfer_config={}))
    worker.chunk_size, worker._engine, worker._early_release = CHUNK, engine, False
    fence = MagicMock()
    worker._do_save_req(req, producer_event=fence)
    mask = torch.arange(5 * CHUNK) >= 2 * CHUNK
    want = list(
        tdb.process_tokens(
            tokens_to_tensor(tokens[: 5 * CHUNK]), mask=mask, request_configs=salt
        )
    )
    keys = engine.storage_manager.batched_put.call_args.args[0]
    _objs, starts, ends = engine.gpu_connector.batched_from_gpu.call_args.args
    assert keys == [k for _s, _e, k in want]
    assert (starts, ends) == ([s for s, _e, _k in want], [e for _s, e, _k in want])
    fence.synchronize.assert_called_once()
    assert worker.get_finished().finished_saving == {SaveOperationId(7, 0)}


@pytest.mark.parametrize("why", ["unseeded-builtin-hash", "chain-mismatch"])
def test_token_path_is_kept_when_prehashing_is_unsafe(monkeypatch, why):
    from atom.kv_transfer.offload import prehashed

    scheduler = _scheduler(monkeypatch, hash_seed="random" if why[0] == "u" else "0")
    if why == "chain-mismatch":
        real = prehashed._chunk_list
        # Corrupt the per-chunk slices only, not the one-time check's prefix.
        monkeypatch.setattr(
            prehashed,
            "_chunk_list",
            lambda t, s, e: real(t, s, e) if e - s > CHUNK else real(t, s + 1, e + 1),
        )
    seq = _seq(4 * CHUNK)
    req = scheduler._build_save_request(
        seq, 0, 4 * CHUNK, SaveOperationId(7, 0), [0] * 8, True
    )
    assert not scheduler._prehashed and req.prehashed_save is None
    assert req.token_ids == list(seq.token_ids)
