# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Shared parsing contracts used by routing and the offload connector."""

import pytest

from atom.kv_transfer.offload import config as offcfg
from atom.kv_transfer.offload.mooncake_store.config import (
    parse_mooncake_store_config,
    store_chunk_tokens,
)


@pytest.mark.parametrize("value", [512, "512", " 512 "])
def test_both_chunk_readers_accept_integer_settings(value):
    assert offcfg.lmcache_chunk_tokens({"lmcache.chunk_size": value}) == 512
    assert store_chunk_tokens({"mooncake_store.chunk_tokens": value}) == 512


@pytest.mark.parametrize("value", [True, 512.5, "512.5", "", 0, -1])
def test_both_chunk_readers_reject_invalid_geometry(value):
    for reader, key in (
        (offcfg.lmcache_chunk_tokens, "lmcache.chunk_size"),
        (store_chunk_tokens, "mooncake_store.chunk_tokens"),
    ):
        with pytest.raises(ValueError):
            reader({"kv_connector_extra_config": {key: value}})


@pytest.mark.parametrize("extra", [[], "512", 512])
def test_both_chunk_readers_reject_nonmapping_extras(extra):
    for reader in (offcfg.lmcache_chunk_tokens, store_chunk_tokens):
        with pytest.raises(ValueError, match="JSON object"):
            reader({"kv_connector_extra_config": extra})


def test_store_parser_and_routing_agree_on_integer_text():
    config = {
        "kv_connector_extra_config": {
            "mooncake_store.master": "127.0.0.1:19871",
            "mooncake_store.metadata": "http://127.0.0.1:19872",
            "mooncake_store.protocol": "tcp",
            "mooncake_store.local_hostname": "127.0.0.1",
            "mooncake_store.chunk_tokens": "512",
        }
    }
    assert parse_mooncake_store_config(config).chunk_tokens == store_chunk_tokens(
        config
    )


def test_lmcache_env_integer_default_and_invalid_input(monkeypatch):
    monkeypatch.setenv("LMCACHE_CHUNK_SIZE", "512")
    assert offcfg.lmcache_chunk_tokens({}) == 512
    monkeypatch.setenv("LMCACHE_CHUNK_SIZE", "")
    assert offcfg.lmcache_chunk_tokens({}) == 256
    monkeypatch.setenv("LMCACHE_CHUNK_SIZE", "512.5")
    with pytest.raises(ValueError, match="LMCACHE_CHUNK_SIZE"):
        offcfg.lmcache_chunk_tokens({})


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("environment", [None, "", "1024", "bad", "0", "512.5"])
def test_lmcache_explicit_chunk_overrides_environment(monkeypatch, nested, environment):
    if environment is None:
        monkeypatch.delenv("LMCACHE_CHUNK_SIZE", raising=False)
    else:
        monkeypatch.setenv("LMCACHE_CHUNK_SIZE", environment)
    extras = {"lmcache.chunk_size": "512"}
    config = {"kv_connector_extra_config": extras} if nested else extras
    assert offcfg.lmcache_chunk_tokens(config) == 512


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("value", [None, True, "bad", "512.5", 0])
def test_lmcache_invalid_override_does_not_fall_back(monkeypatch, nested, value):
    monkeypatch.setenv("LMCACHE_CHUNK_SIZE", "1024")
    extras = {"lmcache.chunk_size": value}
    config = {"kv_connector_extra_config": extras} if nested else extras
    with pytest.raises(ValueError, match="lmcache.chunk_size"):
        offcfg.lmcache_chunk_tokens(config)
