# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Which transports the FP4 sparse indexer refuses, and which it serves.

`--index_cache_dtype fp4` stores the indexer keys as two planes -- packed E2M1
and a separate e8m0 exponent plane. `KVTransferRegion`'s role vocabulary has a
single `INDEX_CACHE_ROLE` for the indexer, so a transport that addresses the
cache through the region map cannot describe the second plane and is refused.

An offload connector does not read the region map at all: `DenseOffloadConnector
.register_kv_caches` takes `transfer_tensors` and ignores it, building its codec
from the `KVCacheTensor`s, which carry both planes. Refusing it as well -- which
a blanket `if config.kv_transfer_config` does -- costs FP4 the offload path for a
reason that is not about it.

The gate is exercised as an unbound method on a stub supplying the three
attributes it reads before deciding, so the predicate under test is the shipped
predicate. `aiter_mla` imports aiter at load, so this file skips whole on a
plain runner.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

AiterMLAMetadataBuilder = pytest.importorskip(
    "atom.model_ops.attentions.aiter_mla",
    reason="the MLA builder's module imports aiter at load",
    exc_type=ImportError,
).AiterMLAMetadataBuilder


def _fp4_builder(kv_transfer_config):
    """A stub carrying only what `get_kv_transfer_tensors` reads before the gate.

    `kv_pool` is a sentinel rather than a pool: reaching it would mean the gate
    fell through, and every later line needs a real one -- so an attribute error
    past this point is the test failing loudly rather than passing by accident.
    """
    return SimpleNamespace(
        model_runner=SimpleNamespace(
            config=SimpleNamespace(kv_transfer_config=kv_transfer_config)
        ),
        kv_pool=object(),
        _indexer_fp4=True,
    )


@pytest.mark.parametrize(
    "kv_transfer_config",
    [
        pytest.param({"kv_connector": "lmcache_offload"}, id="lmcache_offload"),
        pytest.param({"kv_connector": "LMCacheConnectorV1"}, id="lmcache-v1-alias"),
        pytest.param({"kv_connector": "lmcache_mp"}, id="lmcache_mp"),
        pytest.param({}, id="no-connector"),
        pytest.param(None, id="unset"),
    ],
)
def test_fp4_indexer_serves_offload_connectors(kv_transfer_config):
    """Offload never reads these regions, so FP4 hands it None instead of raising."""
    builder = _fp4_builder(kv_transfer_config)

    assert (
        AiterMLAMetadataBuilder.get_kv_transfer_tensors(builder) is None
    ), "an offload topology must not be refused the FP4 indexer"


@pytest.mark.parametrize(
    "connector",
    ["mooncake", "moriio", "multi"],
)
def test_fp4_indexer_refuses_pd_connectors(connector):
    """A transport that addresses the cache by region still cannot see plane two.

    The message has to name P/D: an operator reading it decides between dropping
    to FP8 and dropping P/D, and the old wording ("KV transfer ... unsupported")
    sent someone using only offload to FP8 for nothing.
    """
    builder = _fp4_builder({"kv_connector": connector})

    with pytest.raises(NotImplementedError, match="P/D KV transfer"):
        AiterMLAMetadataBuilder.get_kv_transfer_tensors(builder)


def test_fp4_gate_reads_the_shared_connector_predicate():
    """Pinned to the factory's own answer, not to a second list of names here.

    A connector registered later gets classified once, by the registry it
    declared `requires_pd_staging` to -- not by a copy of that judgement kept in
    the attention backend, which is how the two drift apart.
    """
    from atom.kv_transfer.disaggregation.factory import KVConnectorFactory

    for connector in ("lmcache_offload", "mooncake", "moriio", "multi"):
        cfg = {"kv_connector": connector}
        refused = KVConnectorFactory.topology_uses_pd_staging(cfg)
        builder = _fp4_builder(cfg)
        if refused:
            with pytest.raises(NotImplementedError):
                AiterMLAMetadataBuilder.get_kv_transfer_tensors(builder)
        else:
            assert AiterMLAMetadataBuilder.get_kv_transfer_tensors(builder) is None
