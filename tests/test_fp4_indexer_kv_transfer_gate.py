# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Which transports the FP4 sparse indexer refuses, and which it serves.

`--index_cache_dtype fp4` stores the indexer keys as two planes -- packed E2M1
and a separate e8m0 exponent plane. `KVTransferRegion`'s role vocabulary has a
single `INDEX_CACHE_ROLE` for the indexer, so a transport that addresses the
cache through the region map cannot describe the second plane and is refused.

Dense offload does not read the region map at all: `DenseOffloadConnector
.register_kv_caches` takes `transfer_tensors` and ignores it, building its codec
from the `KVCacheTensor`s, which carry both planes. Refusing it as well -- which
a blanket `if config.kv_transfer_config` does -- costs FP4 the offload path for a
reason that is not about it.

"Offload" is not the line, though: `lmcache_mp` is offload and `_build_cache_views`
raises on a None `KVTransferTensors`, and the hybrid/m3/kimi_k3 layouts source
their PAGE bytes from `block_regions`. The line is whether the region map is read,
which `topology_uses_pd_staging` does not answer -- it is about compressor P/D
staging, and `lmcache_mp` declares that False. These pin the real predicate.

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


def _fp4_builder(kv_transfer_config, *, hf_config=None):
    """A stub carrying only what `get_kv_transfer_tensors` reads before the gate.

    `kv_pool` is a sentinel rather than a pool: reaching it would mean the gate
    fell through, and every later line needs a real one -- so an attribute error
    past this point is the test failing loudly rather than passing by accident.

    `hf_config` reaches `select_offload_layout`, which decides the offload family
    and so whether the regions are read. None is the dense layout.
    """
    return SimpleNamespace(
        model_runner=SimpleNamespace(
            config=SimpleNamespace(
                kv_transfer_config=kv_transfer_config,
                hf_config=hf_config,
            )
        ),
        kv_pool=object(),
        _indexer_fp4=True,
    )


@pytest.mark.parametrize(
    "kv_transfer_config",
    [
        pytest.param({"kv_connector": "lmcache_offload"}, id="lmcache_offload"),
        pytest.param({"kv_connector": "LMCacheConnectorV1"}, id="lmcache-v1-alias"),
        pytest.param({}, id="no-connector"),
        pytest.param(None, id="unset"),
    ],
)
def test_fp4_indexer_serves_dense_offload(kv_transfer_config):
    """Dense offload never reads these regions, so FP4 hands it None, not a raise."""
    builder = _fp4_builder(kv_transfer_config)

    assert (
        AiterMLAMetadataBuilder.get_kv_transfer_tensors(builder) is None
    ), "an offload topology must not be refused the FP4 indexer"


@pytest.mark.parametrize(
    "connector",
    ["mooncake", "moriio", "multi"],
)
def test_fp4_indexer_refuses_region_map_transports(connector):
    """A transport that addresses the cache by region still cannot see plane two.

    The message has to say which transports it means: an operator reading it
    decides between dropping to FP8 and dropping the transport, and a bare "KV
    transfer ... unsupported" sent someone using only dense offload to FP8 for
    nothing.
    """
    builder = _fp4_builder({"kv_connector": connector})

    with pytest.raises(NotImplementedError, match="region map"):
        AiterMLAMetadataBuilder.get_kv_transfer_tensors(builder)


def test_fp4_indexer_refuses_lmcache_mp():
    """Offload that DOES read the regions is still refused.

    `lmcache_mp` registers with `requires_pd_staging=False`, so a gate keyed on
    that flag lets it through -- and then `_build_cache_views` raises
    "lmcache_mp requires KVTransferTensors" during cache registration, which is
    a worse failure than the refusal it skipped.
    """
    builder = _fp4_builder({"kv_connector": "lmcache_mp"})

    with pytest.raises(NotImplementedError, match="region map"):
        AiterMLAMetadataBuilder.get_kv_transfer_tensors(builder)


def test_fp4_indexer_refuses_offload_layouts_that_read_regions():
    """The connector name is not the line; the offload LAYOUT is.

    `lmcache_offload` on a model with `compress_ratios` resolves to the hybrid
    family, whose codec sources its PAGE bytes from `block_regions`. Same name,
    opposite answer.
    """
    builder = _fp4_builder(
        {"kv_connector": "lmcache_offload"},
        hf_config=SimpleNamespace(compress_ratios=[1, 2], architectures=[]),
    )

    with pytest.raises(NotImplementedError, match="region map"):
        AiterMLAMetadataBuilder.get_kv_transfer_tensors(builder)


def test_fp4_indexer_serves_multi_wrapping_only_dense_offload():
    """A `multi` is whatever it wraps, not a name to refuse on sight.

    `MultiConnector.register_kv_caches` forwards `transfer_tensors` to each sub
    unchanged, so a `multi` holding only dense offload hands it to a connector
    that ignores it. Refusing the wrapper would deny a topology that works.
    """
    builder = _fp4_builder(
        {
            "kv_connector": "multi",
            "connectors": [{"kv_connector": "lmcache_offload", "kv_role": "offload"}],
        }
    )

    assert AiterMLAMetadataBuilder.get_kv_transfer_tensors(builder) is None


def test_fp4_indexer_refuses_multi_with_any_region_reader():
    """One region-reading sub is enough: they all get the same `None`."""
    builder = _fp4_builder(
        {
            "kv_connector": "multi",
            "connectors": [
                {"kv_connector": "lmcache_offload", "kv_role": "offload"},
                {"kv_connector": "mooncake", "kv_role": "kv_producer"},
            ],
        }
    )

    with pytest.raises(NotImplementedError, match="region map"):
        AiterMLAMetadataBuilder.get_kv_transfer_tensors(builder)


def test_region_map_verdict_comes_from_the_registration():
    """A connector declares this where it is registered, not here.

    The default is "reads them", so a backend that never says otherwise is
    refused rather than quietly handed a `None` its `register_kv_caches` may
    not survive. A connector that consumes only `KVCacheTensor`s declares
    `reads_block_regions=False` at registration and is served without anyone
    editing the attention backend.
    """
    from atom.kv_transfer.disaggregation.factory import KVConnectorFactory

    KVConnectorFactory.register(
        "fp4gate_probe_tensors_only",
        worker_module="atom.kv_transfer.offload.connector",
        worker_class="LMCacheOffloadConnector",
        scheduler_module="atom.kv_transfer.offload.connector",
        scheduler_class="LMCacheOffloadConnectorScheduler",
        reads_block_regions=False,
    )
    KVConnectorFactory.register(
        "fp4gate_probe_default",
        worker_module="atom.kv_transfer.offload.connector",
        worker_class="LMCacheOffloadConnector",
        scheduler_module="atom.kv_transfer.offload.connector",
        scheduler_class="LMCacheOffloadConnectorScheduler",
    )
    try:
        served = _fp4_builder({"kv_connector": "fp4gate_probe_tensors_only"})
        assert AiterMLAMetadataBuilder.get_kv_transfer_tensors(served) is None

        refused = _fp4_builder({"kv_connector": "fp4gate_probe_default"})
        with pytest.raises(NotImplementedError, match="region map"):
            AiterMLAMetadataBuilder.get_kv_transfer_tensors(refused)
    finally:
        for name in ("fp4gate_probe_tensors_only", "fp4gate_probe_default"):
            KVConnectorFactory._registry.pop(name, None)
            KVConnectorFactory._requires_pd_staging.pop(name, None)
            KVConnectorFactory._reads_block_regions.pop(name, None)


def test_fp4_gate_reads_the_shared_connector_predicate():
    """Pinned to the factory's own answer, not to a second list of names here.

    A connector registered later gets classified once, by the registry it
    declared `requires_pd_staging` to -- not by a copy of that judgement kept in
    the attention backend, which is how the two drift apart.
    """
    from atom.kv_transfer.disaggregation.factory import KVConnectorFactory

    for connector in ("lmcache_offload", "lmcache_mp", "mooncake", "moriio", "multi"):
        cfg = {"kv_connector": connector}
        builder = _fp4_builder(cfg)
        refused = KVConnectorFactory.topology_reads_block_regions(
            builder.model_runner.config
        )
        if refused:
            with pytest.raises(NotImplementedError):
                AiterMLAMetadataBuilder.get_kv_transfer_tensors(builder)
        else:
            assert AiterMLAMetadataBuilder.get_kv_transfer_tensors(builder) is None
