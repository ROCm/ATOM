# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""KV blocks through a real Mooncake Store and back, on a real GPU.

Skipped unless a ROCm or CUDA GPU, Triton and ``mooncake.store`` are present
and a running Store is named by

* ``ATOM_TEST_MOONCAKE_STORE_MASTER`` -- its master, ``host:port``;
* ``ATOM_TEST_MOONCAKE_STORE_METADATA`` -- its metadata server,
  ``http://host:port/metadata``;
* ``ATOM_TEST_MOONCAKE_STORE_RDMA_DEVICE`` (optional) -- an RDMA device to use
  instead of TCP; the test then sets ``MC_NUM_QP_PER_EP=1`` for itself;
* ``ATOM_TEST_MOONCAKE_STORE_LOCAL_HOSTNAME`` (optional) -- this host's
  reachable address, when ``ATOM_HOST_IP`` does not give it.

The worker is the real one -- dense codec, block GPU connector, a transfer
pool registered in HBM, the startup probe -- with only the TP group faked.
"""

from __future__ import annotations

import importlib.util
import os
import time
import uuid
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from atom.kv_transfer.disaggregation.types import (
    ConnectorCompletion,
    LoadOperationId,
    SaveOperationId,
    SaveSourceGroupId,
)
from atom.kv_transfer.offload.chunked_scheduler import (
    DENSE_PAGE_SOURCE_QUIESCENT_CHANNEL,
    DENSE_PAGE_SOURCE_SAFE_CHANNEL,
    DENSE_PAGE_STORE_CHANNEL,
)
from atom.kv_transfer.offload.metadata import (
    LMCacheOffloadMetadata,
    LMCacheReqMeta,
    LoadSpec,
    SaveSpec,
)
from atom.kv_transfer.offload.mooncake_store import keys
from atom.kv_transfer.offload.mooncake_store import worker as worker_mod
from atom.kv_transfer.offload.mooncake_store.worker import (
    MooncakeStoreOffloadConnector,
)

_MASTER = os.environ.get("ATOM_TEST_MOONCAKE_STORE_MASTER", "")
_METADATA = os.environ.get("ATOM_TEST_MOONCAKE_STORE_METADATA", "")
_RDMA_DEVICE = os.environ.get("ATOM_TEST_MOONCAKE_STORE_RDMA_DEVICE", "")
_LOCAL_HOSTNAME = os.environ.get("ATOM_TEST_MOONCAKE_STORE_LOCAL_HOSTNAME", "")


def _skip_reason() -> str:
    if not (
        hasattr(torch, "cuda")
        and torch.cuda.is_available()
        and (
            getattr(torch.version, "hip", None) or getattr(torch.version, "cuda", None)
        )
    ):
        return "a ROCm or CUDA GPU is required"
    if importlib.util.find_spec("triton") is None:
        return "Triton is required for the dense staging kernels"
    if (
        importlib.util.find_spec("mooncake") is None
        or importlib.util.find_spec("mooncake.store") is None
    ):
        return "mooncake.store is required"
    if not (_MASTER and _METADATA):
        return (
            "set ATOM_TEST_MOONCAKE_STORE_MASTER and "
            "ATOM_TEST_MOONCAKE_STORE_METADATA to a running Mooncake Store"
        )
    return ""


_SKIP_REASON = _skip_reason()
pytestmark = pytest.mark.skipif(bool(_SKIP_REASON), reason=_SKIP_REASON)

BLOCK = 16
CHUNK = 256
PROMPT = 4 * CHUNK
# Source blocks [0, 64), destination blocks [64, 128).
NUM_BLOCKS = 2 * PROMPT // BLOCK
SOURCE = list(range(PROMPT // BLOCK))
DESTINATION = list(range(PROMPT // BLOCK, NUM_BLOCKS))


def _config():
    extra = {
        "mooncake_store.master": _MASTER,
        "mooncake_store.metadata": _METADATA,
        "mooncake_store.protocol": "rdma" if _RDMA_DEVICE else "tcp",
        "mooncake_store.rdma_devices": _RDMA_DEVICE,
        "mooncake_store.chunk_tokens": CHUNK,
        "mooncake_store.save_pool_mib": 16,
        "mooncake_store.load_pool_mib": 16,
    }
    if _LOCAL_HOSTNAME:
        extra["mooncake_store.local_hostname"] = _LOCAL_HOSTNAME
    return SimpleNamespace(
        kv_transfer_config={
            "kv_connector": "mooncake_store",
            "kv_role": "offload",
            "kv_connector_extra_config": extra,
        },
        kv_cache_block_size=BLOCK,
        decode_context_parallel_size=1,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        hf_config=SimpleNamespace(num_hidden_layers=2, model_type="glm_moe_dsa"),
        # A fresh namespace per run: a key put by an earlier run "already
        # exists" and would keep that run's bytes.
        model=f"mooncake-store-e2e-{uuid.uuid4().hex}",
        kv_cache_dtype="fp8",
        speculative_config=None,
    )


def _kv_caches(device):
    """Two GLM-5.2-like layers: 576-byte MLA rows and 144-byte indexer rows."""
    return {
        f"layer_{layer}": SimpleNamespace(
            k_cache=torch.randint(
                0, 256, (NUM_BLOCKS * BLOCK, 1, 576), dtype=torch.uint8, device=device
            ),
            index_cache=torch.randint(
                0, 256, (NUM_BLOCKS, BLOCK, 144), dtype=torch.uint8, device=device
            ),
        )
        for layer in range(2)
    }


def _drain(worker, until, *, timeout_s=60.0):
    """Collect worker reports until ``until(collected)`` holds."""
    collected = {"saved": set(), "loaded": set(), "failed": set(), "completions": set()}
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        output = worker.get_finished()
        collected["saved"] |= output.finished_saving
        collected["loaded"] |= output.finished_loading
        collected["failed"] |= output.failed_loading
        collected["completions"] |= output.connector_completions
        if until(collected):
            return collected
        time.sleep(0.01)
    raise AssertionError(f"worker reports incomplete after {timeout_s}s: {collected}")


def test_kv_round_trip_through_the_store(monkeypatch):
    if _RDMA_DEVICE:
        monkeypatch.setenv("MC_NUM_QP_PER_EP", "1")
    monkeypatch.setattr(
        worker_mod, "_tp_group", lambda: SimpleNamespace(world_size=1, rank_in_group=0)
    )
    device = torch.device("cuda", torch.cuda.current_device())
    kv_caches = _kv_caches(device)
    worker = MooncakeStoreOffloadConnector(_config())
    stored_keys: list[str] = []
    try:
        # Includes the startup probe: one chunk through the Store into HBM.
        worker.register_kv_caches(kv_caches, num_blocks=NUM_BLOCKS)
        for cache in kv_caches.values():
            cache.k_cache[DESTINATION[0] * BLOCK :].zero_()
            cache.index_cache[DESTINATION[0] :].zero_()
        torch.cuda.synchronize(device)

        tokens = torch.randint(0, 150_000, (PROMPT,)).tolist()
        hashes = keys.chunk_hash_chain(tokens, CHUNK)
        save = LMCacheReqMeta(
            req_id="e2e",
            token_ids=[],
            block_ids=SOURCE,
            save_spec=SaveSpec(skip_leading_tokens=0),
            save_operation=SaveOperationId("e2e", 0),
            chunk_hashes=hashes,
        )
        metadata = LMCacheOffloadMetadata()
        metadata.add_request(save)
        worker.start_load_kv(metadata)
        operation = save.save_operation
        source_safe = {
            ConnectorCompletion(
                DENSE_PAGE_SOURCE_SAFE_CHANNEL,
                SaveSourceGroupId(operation, ((i * CHUNK, (i + 1) * CHUNK),)),
                True,
            )
            for i in range(PROMPT // CHUNK)
        }
        saved = _drain(
            worker,
            lambda got: operation in got["saved"] and source_safe <= got["completions"],
        )
        assert ConnectorCompletion(DENSE_PAGE_STORE_CHANNEL, operation, True) in (
            saved["completions"]
        )
        assert (
            ConnectorCompletion(DENSE_PAGE_SOURCE_QUIESCENT_CHANNEL, operation, True)
            in saved["completions"]
        )
        stored_keys = keys.chunk_keys(
            worker._namespace, 0, 1, hashes, 0, PROMPT // CHUNK
        )
        assert worker._client.exists(stored_keys) == [1] * len(stored_keys)

        load = LMCacheReqMeta(
            req_id="e2e",
            token_ids=[],
            block_ids=DESTINATION,
            load_spec=LoadSpec(
                hbm_cached_tokens=0, lmcache_cached_tokens=PROMPT, can_load=True
            ),
            load_operation=LoadOperationId("e2e", 0),
            chunk_hashes=hashes,
        )
        metadata = LMCacheOffloadMetadata()
        metadata.add_request(load)
        worker.start_load_kv(metadata)
        loaded = _drain(
            worker,
            lambda got: got["loaded"] or got["failed"],
        )
        assert loaded["loaded"] == {load.load_operation}
        assert loaded["failed"] == set()

        torch.cuda.synchronize(device)
        rows = PROMPT
        for name, cache in kv_caches.items():
            assert torch.equal(
                cache.k_cache[rows : 2 * rows], cache.k_cache[:rows]
            ), f"{name} MLA rows differ after the Store round trip"
            assert torch.equal(
                cache.index_cache[DESTINATION[0] :],
                cache.index_cache[: DESTINATION[0]],
            ), f"{name} indexer rows differ after the Store round trip"
        assert worker._pool.quarantined("save") == 0
        assert worker._pool.quarantined("load") == 0
    finally:
        client = worker._client
        if client is not None:
            for key in stored_keys:
                client.remove(key, force=True)
        worker.close()
